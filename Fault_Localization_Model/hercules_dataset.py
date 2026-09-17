"""Aeva/Continental inputs for the maintained reconstruction pipeline.

Only dataset I/O lives here; faults, rasterization and models remain shared.
Sensor GT translations are sensor positions; IMU extrinsics rotate their axes.
"""
from bisect import bisect_right
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .vod_dataset.vod_io import VODFrame, _named_transform
from .vod_dataset.radar_accumulation import (
    RadarTemporalFilterConfig, filter_accumulated_radar_points,
)
from .hercules_radar_types import AdaptiveStackConfig, DopplerTrackingConfig, ProcessedFrame
from .hercules_tracking import compensate_doppler, dbscan_labels, make_observations, associate_tracks

ALIGNMENT_POLICY = 'hercules_v2_adaptive_tracked_raw_points_v1'

CONTINENTAL_DTYPE = np.dtype({
    'names': ['x', 'y', 'z', 'velocity', 'range', 'rcs', 'azimuth', 'elevation'],
    'formats': ['<f4'] * 5 + ['u1', '<f4', '<f4'],
    'offsets': [0, 4, 8, 12, 16, 20, 21, 25], 'itemsize': 29,
})

def load_hercules_lidar(path):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 29:
        raise ValueError(f'Malformed 29-byte Aeva records: {path}')
    points = raw.reshape(-1, 29)[:, :16].copy().view('<f4').reshape(-1, 4)
    return points[np.isfinite(points).all(axis=1)]

@lru_cache(maxsize=128)
def load_continental(path):
    if Path(path).stat().st_size % 29:
        raise ValueError(f'Malformed Continental records: {path}')
    records = np.fromfile(path, dtype=CONTINENTAL_DTYPE)
    return np.column_stack([records[key] for key in CONTINENTAL_DTYPE.names]).astype(np.float32)

@lru_cache(maxsize=128)
def _session_text_index(root):
    """Walk directories once; never stat each of the scene's raw bin files."""
    index = {}
    for directory, _subdirectories, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith('.txt'):
                index.setdefault(filename.lower(), []).append(Path(directory) / filename)
    return index


@lru_cache(maxsize=128)
def unique_file(root, name):
    matches = _session_text_index(str(root)).get(name.lower(), [])
    if len(matches) != 1:
        raise ValueError(f'Expected one {name} in session {root}, found {len(matches)}')
    return matches[0]

@lru_cache(maxsize=64)
def poses(path):
    data = np.loadtxt(path, dtype=str, ndmin=2)
    if data.shape[1] != 8:
        raise ValueError(f'Invalid timestamp/pose schema: {path}')
    times = np.asarray([int(v) for v in data[:, 0]], dtype=np.int64)
    values = data[:, 1:].astype(float)
    if len(times) < 2 or np.any(np.diff(times) <= 0) or not np.isfinite(values).all():
        raise ValueError(f'Invalid pose sequence: {path}')
    return times, values

def sensor_pose(path, timestamp, max_gap_s=.2):
    times, values = poses(str(path))
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError('Pose interpolation gap limit must be finite and positive')
    exact = int(np.searchsorted(times, timestamp, side='left'))
    if exact < len(times) and int(times[exact]) == timestamp:
        matrix = np.eye(4)
        matrix[:3, 3] = values[exact, :3]
        matrix[:3, :3] = Rotation.from_quat(values[exact, 3:]).as_matrix()
        # Position/orientation need no interpolation. Velocity still requires a
        # supported measured interval; select the shorter adjacent one.
        pairs = [(i, i+1) for i in (exact-1, exact) if 0 <= i < len(times)-1]
        left, right = min(pairs, key=lambda p: int(times[p[1]])-int(times[p[0]]))
        dt = (int(times[right])-int(times[left])) / 1e9
        if dt > max_gap_s:
            raise ValueError(f'Exact pose exists, but velocity interval {dt*1000:.3f} ms '
                             f'exceeds {max_gap_s*1000:g} ms: {path}, timestamp={timestamp}')
        return matrix, (values[right, :3]-values[left, :3]) / dt
    right = int(np.searchsorted(times, timestamp, side='right'))
    left = max(0, right - 1)
    right = min(right, len(times) - 1)
    if timestamp < times[0] or timestamp > times[-1]:
        raise ValueError('Pose extrapolation is forbidden')
    dt = (int(times[right]) - int(times[left])) / 1e9
    if left == right:
        left -= 1
        dt = (int(times[right]) - int(times[left])) / 1e9
    if dt > max_gap_s:
        raise ValueError(f'Pose interpolation gap {dt*1000:.3f} ms exceeds '
                         f'{max_gap_s*1000:g} ms: {path}, timestamp={timestamp}')
    alpha = (timestamp - int(times[left])) / 1e9 / dt
    matrix = np.eye(4)
    matrix[:3, 3] = values[left, :3] * (1-alpha) + values[right, :3] * alpha
    matrix[:3, :3] = Slerp([0., dt], Rotation.from_quat(values[[left, right], 3:]))([alpha * dt]).as_matrix()[0]
    velocity = (values[right, :3] - values[left, :3]) / dt
    return matrix, velocity

def discover_hercules_frames(root, split, radar_variant='radar', split_manifest=None):
    root = Path(root)
    manifest = json.loads(Path(split_manifest).read_text()) if split_manifest else None
    if manifest is not None and manifest.get('version') != 1:
        raise ValueError('Unsupported HeRCULES split manifest version')
    frames = []
    frame_index = 0
    requested = {'train', 'val'} if split == 'train_val' else {'train', 'val', 'test'} if split == 'full' else {split}
    for aeva in sorted(root.rglob('Aeva')):
        paths = sorted(aeva.glob('*.bin'), key=lambda p: int(p.stem))
        if not paths:
            continue
        # Standard HeRCULES layout: session/LiDAR/Aeva.
        if aeva.parent.name.lower() != 'lidar':
            raise ValueError(f'Expected session/LiDAR/Aeva, got {aeva}')
        session = aeva.parent.parent
        scene = session.relative_to(root).as_posix()
        assignment = None
        if manifest is not None:
            if scene not in manifest['scenes']:
                raise ValueError(f'Scene absent from split manifest: {scene}')
            assignment = manifest['scenes'][scene]
        continental = unique_file(session, 'Continental_LiDAR.txt')
        imu = unique_file(session, 'IMU_LiDAR.txt')
        for index, path in enumerate(paths):
            selected = 'train' if index < int(.7*len(paths)) else 'val' if index < int(.85*len(paths)) else 'test'
            if assignment is not None:
                selected = assignment['split']
                if selected == 'val_test':
                    timestamp = int(path.stem)
                    boundary = int(assignment['boundary_ns'])
                    gap = int(manifest['boundary_buffer_ns'])
                    selected = 'val' if timestamp < boundary-gap else 'test' if timestamp >= boundary+gap else None
                elif selected not in {'train', 'val', 'test'}:
                    raise ValueError(f'Invalid scene split: {selected}')
            if selected in requested:
                frames.append(VODFrame(str(frame_index), selected, path, session, imu, continental, radar_variant))
            frame_index += 1
    if not frames:
        raise FileNotFoundError(f'No HeRCULES Aeva frames for {split} in {root}')
    return frames

@lru_cache(maxsize=64)
def radar_paths(session_text):
    directories = []
    for directory, _subdirectories, filenames in os.walk(session_text):
        path = Path(directory)
        if path.name.lower() == 'continental' and any(name.endswith('.bin') for name in filenames):
            directories.append(path)
    if len(directories) != 1:
        raise ValueError(f'Ambiguous/missing Continental directory: {session_text}')
    paths = sorted(directories[0].glob('*.bin'), key=lambda p: int(p.stem))
    return tuple(int(p.stem) for p in paths), paths

def load_frame_radar(frame, config):
    session = frame.radar_path
    timestamp = int(frame.lidar_path.stem)
    times, paths = radar_paths(str(session))
    stop = bisect_right(times, timestamp)
    max_age_ms = float(config.get('hercules_max_radar_age_ms', 30.0))
    if not np.isfinite(max_age_ms) or max_age_ms <= 0:
        raise ValueError('hercules_max_radar_age_ms must be finite and positive')
    if not stop:
        raise ValueError(f'No radar at/before {timestamp}')
    newest_age_ms = (timestamp - times[stop-1]) / 1e6
    if newest_age_ms > max_age_ms:
        raise ValueError(f'Newest causal radar is {newest_age_ms:.2f} ms old; '
                         f'limit is {max_age_ms:g} ms at {timestamp}')
    stack = AdaptiveStackConfig(**config.get('hercules_stack', {
        'max_frames': config['hercules_radar_frames'] or None,
    }))
    tracking = DopplerTrackingConfig(**config.get('hercules_tracking', {}))
    stack.validate()
    tracking.validate()
    radar_gt = unique_file(session, 'Continental_gt.txt')
    max_pose_gap_s = float(config.get('hercules_max_pose_gap_ms', 200.0)) / 1000
    reference, _ = sensor_pose(radar_gt, timestamp, max_pose_gap_s)
    selected = []
    for index in range(stop-1, -1, -1):
        if stack.max_frames is not None and len(selected) >= stack.max_frames:
            break
        age = (timestamp - times[index]) / 1e9
        if age > stack.max_age_s:
            break
        pose, velocity = sensor_pose(radar_gt, times[index], max_pose_gap_s)
        relative = np.linalg.inv(reference) @ pose
        distance = float(np.linalg.norm(relative[:3, 3]))
        angle = float(np.degrees(Rotation.from_matrix(relative[:3, :3]).magnitude()))
        if distance > stack.max_translation_m or angle > stack.max_rotation_deg:
            break
        weight = float(np.exp(-age/stack.weight_time_s - distance/stack.weight_translation_m - angle/stack.weight_rotation_deg))
        selected.append((paths[index], pose, velocity, age, distance, angle, weight))
    selected.reverse()
    if not selected:
        raise ValueError(f'V2 pose gates selected no radar for {timestamp}')
    lidar_to_imu = _named_transform(frame.lidar_calibration_path, 'Tr_lidar_to_imu')
    radar_to_lidar = np.linalg.inv(_named_transform(frame.radar_calibration_path, 'Tr_lidar_to_radar'))
    imu_from_radar = lidar_to_imu[:3, :3] @ radar_to_lidar[:3, :3]
    current, _ = sensor_pose(unique_file(session, 'Aeva_gt.txt'), timestamp, max_pose_gap_s)
    current[:3, :3] = current[:3, :3] @ lidar_to_imu[:3, :3]
    processed, raw_frames, rotations, alignment_rows = [], [], [], []
    for path, pose, velocity, age, distance, angle, weight in selected:
        native = load_continental(path)
        native = native[np.isfinite(native).all(axis=1) & (native[:, 4] > 0)]
        world = pose.copy()
        world[:3, :3] = world[:3, :3] @ imu_from_radar
        transform = np.linalg.inv(current) @ world
        xyz = native[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        compensated, sign, _ = compensate_doppler(native, world[:3, :3].T @ velocity,
            tracking.doppler_sign, tracking.sign_inference_min_speed_mps)
        dynamic = np.abs(compensated) > tracking.dynamic_threshold_mps
        labels = dbscan_labels(native[:, :2], dynamic, tracking.cluster_eps_m, tracking.cluster_min_samples)
        aligned_native = native.copy()
        aligned_native[:, :3] = xyz
        processed.append(ProcessedFrame(int(path.stem), aligned_native, compensated, dynamic, labels,
            weight, sign, float(np.linalg.norm(velocity)), 0.))
        raw_frames.append(native)
        rotations.append(transform[:3, :3])
        alignment_rows.append({'source': str(path), 'timestamp_ns': int(path.stem),
            'age_s': age, 'translation_m': distance, 'rotation_deg': angle,
            'weight': weight, 'doppler_sign': sign,
            'radar_to_current_lidar': transform.tolist()})
    observations = make_observations(processed, raw_frames, rotations)
    tracks = associate_tracks(observations, tracking.association_distance_m, tracking.velocity_smoothing)
    stacks = []
    compensated_points = 0
    for i, (scan, scan_observations) in enumerate(zip(processed, observations)):
        points = scan.points.copy()
        age = (timestamp - scan.timestamp) / 1e9
        for observation in scan_observations:
            track = tracks[observation.track_id]
            if track.hits >= tracking.min_track_hits:
                speed = np.linalg.norm(track.velocity)
                if np.isfinite(speed) and speed <= tracking.max_abs_velocity_mps:
                    points[observation.point_indices, :2] += track.velocity * age
                    compensated_points += len(observation.point_indices)
        stacks.append(np.column_stack([points[:, :3], points[:, 5], points[:, 3],
            scan.doppler_residual_mps, np.full(len(points), i-len(processed)+1)]))
    aligned = np.concatenate(stacks).astype(np.float32)
    # Filter after both ego and confirmed-object motion compensation.
    filtered, stats = filter_accumulated_radar_points(aligned, RadarTemporalFilterConfig(
        temporal_radius_m=config['hercules_temporal_radius'],
    ))
    # Weights are scan-constant; recover them from preserved scan IDs.
    scan_weights = np.array([scan.weight for scan in processed], dtype=np.float32)
    point_weights = scan_weights[filtered[:, 6].astype(int) + len(processed)-1]
    aligned = filtered
    config['_hercules_alignment'] = {
        'policy': ALIGNMENT_POLICY, 'stack': asdict(stack), 'tracking': asdict(tracking),
        'alignment_rows': alignment_rows, 'filter_counts': stats,
        'confirmed_tracks': sum(track.hits >= tracking.min_track_hits for track in tracks.values()),
        'motion_compensated_points': compensated_points,
        'effective_frame_support': sum(scan.weight for scan in processed),
        'newest_radar_age_ms': newest_age_ms,
        'max_radar_age_ms': max_age_ms,
        'max_pose_gap_ms': max_pose_gap_s * 1000,
    }
    config['_hercules_point_weights'] = point_weights
    return aligned, aligned, radar_to_lidar
