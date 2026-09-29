"""Create PV-RCNN configs for the matched VoD detector export.

Requires PyYAML and an already-cloned OpenPCDet checkout. Does not install or
modify OpenPCDet source code; only writes new experiment YAML files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configure(openpcdet: Path, export_root: Path, *, with_radar: bool) -> list[Path]:
    cfg_root = openpcdet / "tools" / "cfgs"
    source_dataset = _read(cfg_root / "dataset_configs" / "kitti_dataset.yaml")
    source_model = _read(cfg_root / "kitti_models" / "pv_rcnn.yaml")
    outputs = []
    modes = ("lidar", "lidar_radar") if with_radar else ("lidar",)
    for mode in modes:
        for condition in ("clean", "faulty", "reconstructed"):
            root = (export_root / mode / condition).resolve()
            if not (root / "export_manifest.json").is_file():
                continue
            dataset = dict(source_dataset)
            dataset["DATA_PATH"] = str(root)
            dataset["FOV_POINTS_ONLY"] = False
            dataset["GET_ITEM_LIST"] = ["points"]
            # Train and eval must use identical camera/point visibility rules.
            dataset_path = cfg_root / "dataset_configs" / f"vod_pvrcnn_{mode}_{condition}.yaml"
            model = dict(source_model)
            model["DATA_CONFIG"] = {
                "_BASE_CONFIG_": f"cfgs/dataset_configs/{dataset_path.name}",
                "DATA_AUGMENTOR": {
                    "DISABLE_AUG_LIST": [],
                    "AUG_CONFIG_LIST": [
                        {"NAME": "random_world_flip", "ALONG_AXIS_LIST": ["x"]},
                        {"NAME": "random_world_rotation", "WORLD_ROT_ANGLE": [-0.78539816, 0.78539816]},
                        {"NAME": "random_world_scaling", "WORLD_SCALE_RANGE": [0.95, 1.05]},
                    ],
                },
            }
            model_path = cfg_root / "kitti_models" / f"vod_pvrcnn_{mode}_{condition}.yaml"
            for path, document in ((dataset_path, dataset), (model_path, model)):
                if path.exists():
                    raise FileExistsError(f"Refusing to overwrite existing config: {path}")
                path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            outputs.append(model_path)
    if not outputs:
        raise FileNotFoundError(f"No exported detector variants under {export_root}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpcdet-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--with-radar", action="store_true")
    args = parser.parse_args()
    for path in configure(args.openpcdet_root, args.export_root,
                          with_radar=args.with_radar):
        print(path)


if __name__ == "__main__":
    main()
