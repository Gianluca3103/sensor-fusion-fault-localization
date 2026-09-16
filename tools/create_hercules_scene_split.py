"""Create a reproducible three-scene held-out split, without copying scans."""
import argparse
import random
from pathlib import Path
from Fault_Localization_Model.io_utils import atomic_write_json
from Fault_Localization_Model.hercules_dataset import discover_hercules_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hercules-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--boundary-buffer-s', type=float, default=1.0)
    args = parser.parse_args()
    if args.boundary_buffer_s < 1 or not args.boundary_buffer_s < float('inf'):
        raise ValueError('Buffer must be finite and at least the 1-second radar history')
    if args.output.exists():
        raise FileExistsError(f'Manifest already exists; reuse it: {args.output}')
    frames = discover_hercules_frames(args.hercules_root, 'full')
    scenes = {}
    for frame in frames:
        name = frame.radar_path.relative_to(args.hercules_root).as_posix()
        scenes.setdefault(name, []).append(int(frame.lidar_path.stem))
    if len(scenes) < 4:
        raise ValueError('At least four scenes are required to retain training scenes')
    names = sorted(scenes)
    random.Random(args.seed).shuffle(names)
    validation, testing, divided = names[:3]
    assignments = {name: {'split': 'train'} for name in sorted(scenes)}
    assignments[validation] = {'split': 'val'}
    assignments[testing] = {'split': 'test'}
    timestamps = sorted(scenes[divided])
    boundary = timestamps[len(timestamps)//2]
    buffer = int(args.boundary_buffer_s * 1e9)
    if not any(t < boundary-buffer for t in timestamps) or not any(t >= boundary+buffer for t in timestamps):
        raise ValueError('Divided scene is too short for the requested boundary buffer')
    assignments[divided] = {'split': 'val_test', 'boundary_ns': boundary}
    atomic_write_json(args.output, {'version': 1, 'seed': args.seed,
        'boundary_buffer_ns': buffer, 'scenes': assignments})
    print(f'Validation scene: {validation}\nTest scene: {testing}\nDivided scene: {divided}')
    print(f'Saved: {args.output}')

if __name__ == '__main__':
    main()
