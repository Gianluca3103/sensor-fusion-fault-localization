"""Offline, split-isolated clean-LiDAR reference construction.

Uses actual dataset pose/calibration readers. No reference arrays are sensor
inputs. Boxes conservatively exclude all annotated objects from neighbor scans.
Without boxes the default is central-only, never silently accepted motion trails.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .io_utils import atomic_savez_compressed
from .vod_dataset.vod_io import load_vod_lidar, load_vod_split_ids, resolve_vod_public_root, _named_transform
from .vod_dataset.radar_accumulation import load_vod_odom_from_camera
from .hercules_dataset import discover_hercules_frames, load_hercules_lidar, sensor_pose, unique_file

REFERENCE_VERSION = 1


@dataclass(frozen=True)
class GeometricReferenceConfig:
    enabled: bool = False
    cache_root: str = ''
    past_frames: int = 5
    future_frames: int = 5
    dynamic_handling: str = 'boxes_or_central_only'
    dynamic_box_margin_m: float = .5
    persistence_radius_m: float = .10
    persistence_min_scans: int = 3
    voxel_size_m: float = .05
    max_step_translation_m: float = 3.
    max_step_rotation_deg: float = 15.
    max_step_time_s: float = .2

    def validate(self):
        if self.enabled and not self.cache_root:
            raise ValueError('Enabled geometric_reference requires cache_root')
        if self.past_frames < 0 or self.future_frames < 0:
            raise ValueError('Reference frame windows must be nonnegative')
        if self.dynamic_handling not in {'boxes_or_central_only', 'boxes', 'central_only', 'persistence'}:
            raise ValueError('Invalid dynamic_handling strategy')
        if self.voxel_size_m < 0 or self.dynamic_box_margin_m < 0 or self.persistence_radius_m <= 0:
            raise ValueError('Invalid reference voxel/filter distances')
        if self.persistence_min_scans < 2:
            raise ValueError('Persistence needs at least two distinct scans')
        if min(self.max_step_translation_m, self.max_step_rotation_deg, self.max_step_time_s) <= 0:
            raise ValueError('Reference recording-boundary gates must be positive')


def reference_cache_path(sample_path, data_root, config):
    return Path(config.cache_root) / Path(sample_path).relative_to(Path(data_root))


def fingerprint(paths):
    return [{'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths]


def exclude_box_points(points, annotation_path, calibration, margin):
    """KITTI camera boxes: bottom center, dimensions h,w,l and rotation_y.

    Reuses camera-from-LiDAR calibration rather than creating a BEV transform.
    There is no maintained annotation parser after detector removal.
    """
    camera = points[:, :3] @ calibration[:3, :3].T + calibration[:3, 3]
    inside = np.zeros(len(points), dtype=bool)
    for line in Path(annotation_path).read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] == 'DontCare':
            continue
        if len(fields) < 15:
            raise ValueError(f'Malformed KITTI object box: {annotation_path}')
        h, w, length, x, y, z, yaw = map(float, fields[8:15])
        if min(h, w, length) <= 0 or not np.isfinite([h, w, length, x, y, z, yaw]).all():
            raise ValueError(f'Invalid object dimensions: {annotation_path}')
        delta = camera-np.array([x, y, z])
        local_x = np.cos(yaw)*delta[:, 0]-np.sin(yaw)*delta[:, 2]
        local_z = np.sin(yaw)*delta[:, 0]+np.cos(yaw)*delta[:, 2]
        inside |= ((np.abs(local_x) <= length/2+margin) & (np.abs(local_z) <= w/2+margin)
                   & (delta[:, 1] >= -h-margin) & (delta[:, 1] <= margin))
    return points[~inside]


def downsample(points, voxel_size):
    if not len(points) or voxel_size == 0:
        return points
    _, indices = np.unique(np.floor(points/voxel_size).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(indices)]


class DatasetReferenceSource:
    """Dataset-specific lookup; all neighboring IDs must belong to central split."""
    def __init__(self, raw_root, dataset, split):
        self.root, self.dataset, self.split = Path(raw_root), dataset.lower(), split
        if self.dataset in {'vod', 'view-of-delft', 'view of delft'}:
            self.root = resolve_vod_public_root(self.root)
            self.ids = sorted(load_vod_split_ids(self.root, split), key=int)
            self.frames = None
        elif self.dataset == 'hercules':
            frames = discover_hercules_frames(self.root, split)
            self.frames = {f.frame_id: f for f in frames}
            self.ids = list(self.frames)
        else:
            raise ValueError(f'Unsupported reference dataset {dataset}')

    def lidar_path(self, identifier):
        return self.frames[identifier].lidar_path if self.frames is not None else self.root / 'lidar/training/velodyne' / f'{int(identifier):05d}.bin'

    def dependencies(self, identifier):
        if self.frames is None:
            return [self.lidar_path(identifier), self.root / 'lidar/training/pose' / f'{int(identifier):05d}.json',
                    self.root / 'lidar/training/calib' / f'{int(identifier):05d}.txt']
        f = self.frames[identifier]
        return [f.lidar_path, f.lidar_calibration_path, unique_file(f.radar_path, 'Aeva_gt.txt')]

    def load(self, identifier):
        return (load_hercules_lidar if self.frames is not None else load_vod_lidar)(self.lidar_path(identifier))[:, :3]

    def world_from_lidar(self, identifier):
        if self.frames is None:
            _, pose_path, calibration = self.dependencies(identifier)
            return load_vod_odom_from_camera(pose_path) @ _named_transform(calibration)
        frame = self.frames[identifier]
        world, _ = sensor_pose(unique_file(frame.radar_path, 'Aeva_gt.txt'), int(frame.lidar_path.stem))
        world[:3, :3] = world[:3, :3] @ _named_transform(frame.lidar_calibration_path, 'Tr_lidar_to_imu')[:3, :3]
        return world

    def label(self, identifier):
        return self.root / 'lidar/training/label_2' / f'{int(identifier):05d}.txt' if self.frames is None else None

    def static_points(self, points, identifier, margin):
        return exclude_box_points(points, self.label(identifier),
            _named_transform(self.root / 'lidar/training/calib' / f'{int(identifier):05d}.txt'), margin)

    def adjacent(self, left, right, config):
        if self.frames is None:
            if int(right) != int(left)+1:
                return False
        else:
            a, b = self.lidar_path(left), self.lidar_path(right)
            if a.parent != b.parent or (int(b.stem)-int(a.stem))/1e9 > config.max_step_time_s:
                return False
        relative = np.linalg.inv(self.world_from_lidar(right)) @ self.world_from_lidar(left)
        return (np.linalg.norm(relative[:3, 3]) <= config.max_step_translation_m
                and np.degrees(Rotation.from_matrix(relative[:3, :3]).magnitude()) <= config.max_step_rotation_deg)


def build_reference(sample_path, data_root, source, config):
    config.validate()
    with np.load(sample_path, allow_pickle=False) as sample:
        central_metadata = json.loads(str(sample['metadata_json'].item()))
    if central_metadata['split'] != source.split:
        raise ValueError('Cross-split reference source is forbidden')
    identifier = str(central_metadata['frame_id'])
    # VoD metadata may omit leading zeros whereas official ImageSets retain them.
    if source.frames is None:
        identifier = next((s for s in source.ids if int(s) == int(identifier)), identifier)
    if identifier not in source.ids:
        raise ValueError(f'Central frame {identifier} does not belong to {source.split}')
    center_index = source.ids.index(identifier)
    selected = [identifier]
    central_only = config.dynamic_handling == 'central_only' or (
        config.dynamic_handling == 'boxes_or_central_only' and
        (source.label(identifier) is None or not source.label(identifier).is_file()))
    for direction, count in ((-1, 0 if central_only else config.past_frames),
                             (1, 0 if central_only else config.future_frames)):
        previous = identifier
        for offset in range(1, count+1):
            index = center_index+direction*offset
            if not 0 <= index < len(source.ids):
                break
            candidate = source.ids[index]
            a, b = (candidate, previous) if direction == -1 else (previous, candidate)
            if not source.adjacent(a, b, config):
                break
            selected.append(candidate)
            previous = candidate
    selected.sort(key=lambda s: source.ids.index(s))
    boxes_available = all(source.label(i) is not None and source.label(i).is_file() for i in selected)
    strategy = config.dynamic_handling
    if strategy == 'boxes_or_central_only':
        strategy = 'boxes' if boxes_available else 'central_only'
    if strategy == 'boxes' and not boxes_available:
        raise FileNotFoundError('Box-only reference strategy requires annotations for every selected frame')
    if strategy == 'central_only':
        selected = [identifier]
    dependencies = [path for i in selected for path in source.dependencies(i)]
    dependencies += [source.label(i) for i in selected if source.label(i) is not None and source.label(i).is_file()]
    metadata = {'version': REFERENCE_VERSION, 'config': asdict(config), 'central_frame_id': str(central_metadata['frame_id']),
                'split': source.split, 'frames_used': selected, 'dependencies': fingerprint(dependencies),
                'dynamic_strategy': strategy}
    destination = reference_cache_path(sample_path, data_root, config)
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as existing:
            old = json.loads(str(existing['metadata_json'].item()))
        if all(old.get(k) == v for k, v in metadata.items()):
            return destination, True
    central = source.load(identifier)
    current_from_world = np.linalg.inv(source.world_from_lidar(identifier))
    scans, rows = [], []
    for i in selected:
        points = source.load(i)
        count = len(points)
        transform = current_from_world @ source.world_from_lidar(i)
        if i != identifier and strategy == 'boxes':
            points = source.static_points(points, i, config.dynamic_box_margin_m)
        aligned = points @ transform[:3, :3].T+transform[:3, 3]
        if i != identifier and strategy == 'boxes':
            aligned = source.static_points(aligned, identifier, config.dynamic_box_margin_m)
        scans.append(aligned.astype(np.float32))
        rows.append({'frame_id': i, 'split': source.split, 'source_points': count,
                     'retained_points': len(aligned), 'current_from_source': transform.tolist()})
    if strategy == 'persistence':
        trees = [cKDTree(scan) for scan in scans]
        for index, i in enumerate(selected):
            if i == identifier:
                continue
            points = scans[index]
            support = sum(tree.query(points)[0] <= config.persistence_radius_m for tree in trees)
            scans[index] = points[support >= config.persistence_min_scans]
        # This conservative strategy is explicitly opt-in: persistence is not
        # reliable object tracking and may still retain slowly moving geometry.
    reference = downsample(np.concatenate(scans), config.voxel_size_m).astype(np.float32)
    # Ensure the measured central points survive reference voxelization exactly.
    reference = np.unique(np.concatenate([reference, central]), axis=0)
    metadata.update({'alignment_rows': rows, 'central_points': len(central),
                     'final_reference_points': len(reference), 'input_access': 'supervision_only',
                     'observability': 'not inferred; far predictions remain unknown'})
    atomic_savez_compressed(destination, reference_points=reference, central_points=central,
                           metadata_json=np.asarray(json.dumps(metadata)))
    return destination, False


def load_reference(sample_path, data_root, config):
    destination = reference_cache_path(sample_path, data_root, config)
    with np.load(sample_path, allow_pickle=False) as sample:
        sample_metadata = json.loads(str(sample['metadata_json'].item()))
    with np.load(destination, allow_pickle=False) as cache:
        metadata = json.loads(str(cache['metadata_json'].item()))
        if (metadata['version'] != REFERENCE_VERSION or metadata['config'] != asdict(config)
                or metadata['central_frame_id'] != str(sample_metadata['frame_id'])
                or metadata['split'] != sample_metadata['split']
                or any(row['split'] != metadata['split'] for row in metadata['alignment_rows'])):
            raise ValueError(f'Stale or cross-split geometric reference: {destination}')
        return np.asarray(cache['reference_points'], dtype=np.float32)
