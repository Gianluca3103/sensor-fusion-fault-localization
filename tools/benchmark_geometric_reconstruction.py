"""Geometry-loss forward/backward timing and peak CUDA allocation.

Real-batch mode uses generated artifacts/references; synthetic mode is explicitly
labeled and is not evidence of real-dataset preprocessing time.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from Fault_Localization_Model.io_utils import atomic_write_json
from Fault_Localization_Model.geometric_reference import GeometricReferenceConfig
from models.two_stage_reconstruction_head import CoarseReconstructionDataset, coarse_reconstruction_collate
from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry
from models.two_stage_reconstruction_head.geometric_reconstruction import (
    GeometricLossConfig, SoftGeometricReconstructionLoss, expected_xyz,
)
from models.two_stage_reconstruction_head.coarse_reconstruction.evaluate_coarse_by_fault import _load_selector_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--selector-config', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--radar-root', type=Path)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text()) if args.config else {}
    device = torch.device(args.device)
    config = GeometricLossConfig(**payload.get('geometric_loss', {'enabled': True}))
    if args.data_root:
        if not args.radar_root or not args.selector_config or not args.config:
            parser.error('Real batch needs --radar-root --selector-config --config')
        dataset = CoarseReconstructionDataset(sorted((args.data_root / 'train').glob('*.npz'))[:args.batch_size],
            args.radar_root, data_root=args.data_root, selector_config=_load_selector_config(args.selector_config),
            use_pointpillars=True, geometric_reference_config=GeometricReferenceConfig(**payload['geometric_reference']))
        batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, collate_fn=coarse_reconstruction_collate)))
        prediction = batch['clean_bev'].to(device).float().requires_grad_()
        references = batch['geometric_reference_points']
        mask = batch['reconstruction_mask'].to(device).float()
        geometry = dataset.grid_geometry
        mode = 'real cached training references, clean BEV surrogate prediction'
    else:
        geometry = BEVGridGeometry(0., 64., -32., 32., 320, 320)
        prediction = torch.zeros(args.batch_size, 3, 320, 320, device=device)
        prediction[:, 0] = .3
        prediction[:, 2] = 3/8
        mask = torch.zeros(args.batch_size, 1, 320, 320, device=device)
        mask[:, :, 100:180, 100:180] = 1.
        xyz = expected_xyz(prediction[0], geometry)
        points = xyz.reshape(320, 320, 3)[100:180:2, 100:180:2].reshape(-1, 3).detach()
        references = tuple(points for _ in range(args.batch_size))
        prediction.requires_grad_()
        mode = 'synthetic 320x320 BEV; 1600 reference points/sample; 80x80 repair'
    loss = SoftGeometricReconstructionLoss(config, geometry)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    timings = []
    peak = None
    for iteration in range(args.iterations+1):
        prediction.grad = None
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        sync()
        start = time.perf_counter()
        values = loss(prediction, references, mask)
        sync()
        forward = time.perf_counter()-start
        start = time.perf_counter()
        values['geometric_loss'].backward()
        sync()
        backward = time.perf_counter()-start
        if iteration:
            timings.append({'forward_s': forward, 'backward_s': backward})
        if device.type == 'cuda':
            peak = max(peak or 0, torch.cuda.max_memory_allocated(device))
    result = {'mode': mode, 'device': str(device), 'batch_size': len(prediction),
        'reference_points': [len(p) for p in references], 'peak_cuda_allocated_bytes': peak,
        'timings': timings, 'mean_forward_s': sum(t['forward_s'] for t in timings)/len(timings),
        'mean_backward_s': sum(t['backward_s'] for t in timings)/len(timings)}
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
