"""Evaluate saved coarse/fine models against supervision-only dense geometry."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial import cKDTree
from Fault_Localization_Model.io_utils import atomic_write_json, write_csv_rows
from Fault_Localization_Model.geometric_reference import GeometricReferenceConfig, reference_cache_path
from models.Fault_Localization.training_utils import _split_paths, resolve_device
from models.two_stage_reconstruction_head import (
    CoarseReconstructionDataset, coarse_reconstruction_collate, load_frozen_coarse_model,
    BEVChannelNormalization, FineDiffusionRefiner, FrozenCoarseFineDiffusionPipeline,
    coarse_reconstruction_metrics, validate_fine_diffusion_checkpoint_compatibility,
)
from models.two_stage_reconstruction_head.coarse_reconstruction.evaluate_coarse_by_fault import _load_selector_config
from models.two_stage_reconstruction_head.coarse_reconstruction.train_coarse_reconstruction import _move_batch
from models.two_stage_reconstruction_head.diffusion_process.evaluate_fine_diffusion_by_fault import (
    _diffusion_config_from_checkpoint, _normalizer_from_fine_config,
)
from models.two_stage_reconstruction_head.diffusion_process.train_fine_diffusion import _residual_normalizer
from models.two_stage_reconstruction_head.geometric_reconstruction import (
    GeometricLossConfig, expected_xyz, select_reference, region_masks, point_metrics,
)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(v) for key, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def visualization(path, conditions, maximum=30000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    nonempty = [np.asarray(p) for p in conditions.values() if len(p)]
    bounds = np.concatenate(nonempty) if nonempty else np.zeros((1, 3))
    for ax, (name, points) in zip(axes.flatten(), conditions.items()):
        points = np.asarray(points)
        if len(points) > maximum:
            points = points[np.linspace(0, len(points)-1, maximum, dtype=int)]
        ax.scatter(points[:, 1], points[:, 0], s=.5, c='cyan' if name != 'dense_reference' else 'white')
        ax.set_facecolor('black')
        ax.set_title(name)
        ax.set_aspect('equal')
    for ax in axes.flatten():
        ax.set_xlim(bounds[:, 1].min()-1, bounds[:, 1].max()+1)
        ax.set_ylim(bounds[:, 0].min()-1, bounds[:, 0].max()+1)
    # Last panel: geometrically supported (green), far/unknown (orange).
    fine = conditions.get('fine', conditions['coarse'])
    reference = conditions['dense_reference']
    if len(fine) and len(reference):
        distance = cKDTree(reference).query(fine)[0]
        ax = axes.flatten()[-1]
        ax.clear()
        ax.set_facecolor('black')
        colors = np.where(distance <= .2, 'lime', 'orange')
        ids = np.linspace(0, len(fine)-1, min(maximum, len(fine)), dtype=int)
        ax.scatter(fine[ids, 1], fine[ids, 0], s=1, c=colors[ids])
        ax.set_title('supported <=0.2m / far UNKNOWN')
        ax.set_aspect('equal')
    fig.tight_layout()
    for ax in axes.flatten():
        ax.set_xlim(bounds[:, 1].min()-1, bounds[:, 1].max()+1)
        ax.set_ylim(bounds[:, 0].min()-1, bounds[:, 0].max()+1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    for name in ('data-root', 'radar-root', 'config', 'selector-config', 'coarse-checkpoint', 'output-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--fine-checkpoint', type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--limit-samples', type=int)
    parser.add_argument('--visualize-samples', type=int, default=10)
    parser.add_argument('--occupancy-threshold', type=float, default=.5)
    parser.add_argument('--sampling-steps', type=int)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text())
    reference_config = GeometricReferenceConfig(**payload['geometric_reference'])
    config = GeometricLossConfig(**payload.get('geometric_loss', {}))
    reference_config.validate()
    config.validate()
    device = resolve_device(args.device)
    coarse, _ = load_frozen_coarse_model(args.coarse_checkpoint, device, allow_pointpillars=True)
    if not coarse.config.pointpillars_enabled:
        raise ValueError('This raw-point diagnostic requires a PointPillars coarse checkpoint')
    dataset = CoarseReconstructionDataset(_split_paths(args.data_root, args.split, args.limit_samples, 0),
        args.radar_root, data_root=args.data_root, selector_config=_load_selector_config(args.selector_config),
        use_pointpillars=coarse.config.pointpillars_enabled, geometric_reference_config=reference_config)
    geometry = dataset.grid_geometry
    if getattr(coarse, 'grid_geometry', None) is not None and coarse.grid_geometry != geometry:
        raise ValueError('Checkpoint and geometric reference grid differ')
    pipeline = None
    if args.fine_checkpoint:
        checkpoint = torch.load(args.fine_checkpoint, map_location='cpu', weights_only=False)
        fine_config = _diffusion_config_from_checkpoint(checkpoint['diffusion_config'], checkpoint.get('fine_diffusion_architecture'))
        validate_fine_diffusion_checkpoint_compatibility(checkpoint, fine_config)
        normalizer = _normalizer_from_fine_config(None, fine_config)
        if checkpoint.get('bev_normalization'):
            meta = checkpoint['bev_normalization']
            normalizer = BEVChannelNormalization(meta['means'], meta['stds'], epsilon=meta['epsilon'], source=meta.get('source', 'checkpoint'))
        refiner = FineDiffusionRefiner(fine_config, normalizer,
            _residual_normalizer(checkpoint['residual_normalization'], fine_config)).to(device)
        refiner.load_state_dict(checkpoint['diffusion_state_dict'], strict=True)
        pipeline = FrozenCoarseFineDiffusionPipeline(coarse, refiner).to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=coarse_reconstruction_collate)
    records = []
    tolerances = tuple(payload.get('geometric_metrics', {}).get('tolerances_m', [.1, .2, .5]))
    frames = 0
    with torch.no_grad():
        for batch in loader:
            inputs = _move_batch(batch, device)  # deliberately excludes dense reference
            output = coarse(inputs['faulty_bev'], inputs['radar_bev'], inputs['reconstruction_mask'],
                inputs['healthy_context_mask'], inputs['halo_mask'],
                faulty_lidar_points=inputs.get('faulty_lidar_points'), radar_points=inputs.get('radar_points'))
            fine_output = None
            if pipeline:
                fine_output = pipeline.sample(inputs['faulty_bev'], inputs['radar_bev'], inputs['reconstruction_mask'],
                    inputs['healthy_context_mask'], inputs['halo_mask'],
                    faulty_lidar_points=inputs.get('faulty_lidar_points'), radar_points=inputs.get('radar_points'),
                    coarse_lidar_bev=output['coarse_lidar_bev'], coarse_output=output,
                    sampling_steps=args.sampling_steps or pipeline.diffusion.config.sampling_steps)
            core, match = region_masks(inputs['reconstruction_mask'], geometry, config.halo_m, config.repair_only)
            for index, sample_path in enumerate(batch['sample_path']):
                reference = batch['geometric_reference_points'][index].numpy()
                with np.load(reference_cache_path(sample_path, args.data_root, reference_config)) as cache:
                    central = cache['central_points'].copy()
                conditions = {'clean_single_frame': central,
                    'faulty': batch['faulty_lidar_points'][index][:, :3].numpy(), 'dense_reference': reference}
                bevs = {'coarse': output['coarse_lidar_bev'][index]}
                if fine_output:
                    bevs['fine'] = fine_output['final_lidar_bev'][index]
                for name, bev in bevs.items():
                    xyz = expected_xyz(bev, geometry).cpu().numpy()
                    conditions[name] = xyz[bev[0].cpu().numpy().flatten() >= args.occupancy_threshold]
                ref_query = reference[select_reference(reference, core[index], geometry)]
                ref_match = reference[select_reference(reference, match[index], geometry)]
                for name, points in conditions.items():
                    query = points[select_reference(points, core[index], geometry)]
                    candidates = points[select_reference(points, match[index], geometry)]
                    record = {'sample_path': sample_path, 'condition': name,
                        **point_metrics(query, ref_query, candidates, ref_match, tolerances)}
                    if name in bevs:
                        historical = coarse_reconstruction_metrics({'coarse_lidar_bev': bevs[name][None],
                            'occupancy_logits': torch.logit(bevs[name][None, 0:1].clamp(1e-6, 1-1e-6)),
                            'reconstruction_mask': inputs['reconstruction_mask'][index:index+1]},
                            inputs['faulty_bev'][index:index+1], inputs['clean_bev'][index:index+1])
                        record.update({f'historical/{key}': float(value) for key, value in historical.items()})
                    records.append(record)
                if frames < args.visualize_samples:
                    visualization(args.output_root / 'visualizations' / f'{frames:05d}.png', conditions)
                frames += 1
            print(f'Geometry evaluated {frames}/{len(dataset)}', flush=True)
    summary = {}
    for condition in {row['condition'] for row in records}:
        rows = [row for row in records if row['condition'] == condition]
        numeric = {k for row in rows for k, v in row.items() if isinstance(v, (int, float))}
        summary[condition] = {k: float(np.mean([row[k] for row in rows if k in row])) for k in numeric}
    write_csv_rows(args.output_root / 'per_frame_metrics.csv', records)
    atomic_write_json(args.output_root / 'geometric_metrics.json', json_safe({'frames': frames,
        'aggregation': 'per-frame macro means; empty one-sided distances serialized as null',
        'observability': 'not available; far predictions UNKNOWN, not proven hallucinations',
        'summary': summary, 'per_frame': records}))

if __name__ == '__main__':
    main()
