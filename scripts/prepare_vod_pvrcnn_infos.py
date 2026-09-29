"""Create shared OpenPCDet KITTI infos for clean-trained VoD experiments.

Run with the Python environment in which OpenPCDet is installed. This avoids
OpenPCDet's CLI hard-coded `data/kitti` path and omits the unused GT database.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import shutil

import yaml


def prepare(export_root: Path, openpcdet_root: Path, *, with_radar: bool,
            workers: int) -> None:
    from easydict import EasyDict
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset

    modes = ("lidar", "lidar_radar") if with_radar else ("lidar",)
    for mode in modes:
        clean_root = (export_root / mode / "clean").resolve()
        cfg_path = openpcdet_root / "tools" / "cfgs" / "dataset_configs" / f"vod_pvrcnn_{mode}_clean.yaml"
        with cfg_path.open(encoding="utf-8") as handle:
            cfg = EasyDict(yaml.safe_load(handle))
        dataset = KittiDataset(dataset_cfg=cfg,
                               class_names=["Car", "Pedestrian", "Cyclist"],
                               root_path=clean_root, training=False)
        for split in ("train", "val"):
            dataset.set_split(split)
            infos = dataset.get_infos(num_workers=workers, has_label=True,
                                      count_inside_pts=False)
            expected = len((clean_root / "ImageSets" / f"{split}.txt").read_text().split())
            if len(infos) != expected:
                raise RuntimeError(f"{mode}/{split}: {len(infos)} infos, expected {expected}")
            target = clean_root / f"kitti_infos_{split}.pkl"
            with target.open("wb") as handle:
                pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"{mode}/{split}: {len(infos)} frames -> {target}", flush=True)
            if split == "val":
                for condition in ("faulty", "reconstructed"):
                    other = export_root / mode / condition
                    if other.is_dir():
                        shutil.copy2(target, other / target.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--openpcdet-root", type=Path, required=True)
    parser.add_argument("--with-radar", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    prepare(args.export_root, args.openpcdet_root, with_radar=args.with_radar,
            workers=args.workers)


if __name__ == "__main__":
    main()
