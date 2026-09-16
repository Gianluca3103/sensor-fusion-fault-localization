"""Build supervision-only dense references from actual clean dataset poses."""
import argparse
import json
import time
from pathlib import Path
from Fault_Localization_Model.io_utils import atomic_write_json
from Fault_Localization_Model.geometric_reference import (
    GeometricReferenceConfig, DatasetReferenceSource, build_reference,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--limit-samples', type=int)
    args = parser.parse_args()
    config = GeometricReferenceConfig(**json.loads(args.config.read_text())['geometric_reference'])
    config.validate()
    paths = sorted((args.data_root / args.split).glob('*.npz'))
    if args.limit_samples:
        paths = paths[:args.limit_samples]
    if not paths:
        raise FileNotFoundError('No reconstruction samples')
    import numpy as np
    with np.load(paths[0], allow_pickle=False) as sample:
        dataset = json.loads(str(sample['metadata_json'].item()))['dataset']
    source = DatasetReferenceSource(args.raw_root, dataset, args.split)
    started = time.perf_counter()
    timings = []
    cache_hits = 0
    for index, path in enumerate(paths, 1):
        frame_started = time.perf_counter()
        _, cached = build_reference(path, args.data_root, source, config)
        timings.append(time.perf_counter() - frame_started)
        cache_hits += int(cached)
        if index % 10 == 0 or index == len(paths):
            print(f'References {index}/{len(paths)} | {time.perf_counter()-started:.1f}s | last cached={cached}', flush=True)
    atomic_write_json(Path(config.cache_root) / f'build_timing_{args.split}.json', {
        'split': args.split, 'frames': len(paths), 'cache_hits': cache_hits,
        'elapsed_s': time.perf_counter() - started,
        'mean_frame_s': float(np.mean(timings)), 'p95_frame_s': float(np.percentile(timings, 95)),
        'includes_cache_lookup': True,
    })

if __name__ == '__main__':
    main()
