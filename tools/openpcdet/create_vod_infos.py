"""Generate official OpenPCDet CustomDataset infos and GT database for VoD."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcdet_integration.openpcdet_eval import load_openpcdet_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpcdet-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sys.path.insert(0, str(args.openpcdet_root))
    from pcdet.datasets.custom import custom_dataset as custom_dataset_module
    from pcdet.utils import common_utils

    # The upstream CustomDataset implementation at our pinned OpenPCDet
    # revision references ``Path`` in create_groundtruth_database without
    # importing it.  Supply the standard pathlib symbol without modifying the
    # isolated upstream dependency.
    custom_dataset_module.Path = Path
    CustomDataset = custom_dataset_module.CustomDataset

    cfg = load_openpcdet_config(args.openpcdet_root, args.dataset_config)
    dataset_cfg = cfg.DATA_CONFIG if "DATA_CONFIG" in cfg else cfg
    classes = ["Car", "Pedestrian", "Cyclist"]
    dataset = CustomDataset(
        dataset_cfg=dataset_cfg,
        class_names=classes,
        root_path=args.data_root,
        training=False,
        logger=common_utils.create_logger(),
    )
    for split in ("train", "val"):
        dataset.set_split(split)
        destination = args.data_root / f"vod_infos_{split}.pkl"
        if destination.is_file():
            with destination.open("rb") as handle:
                infos = pickle.load(handle)
            print(f"Reusing {len(infos)} {split} infos: {destination}")
        else:
            infos = dataset.get_infos(
                classes,
                num_workers=args.workers,
                has_label=True,
                num_features=4,
            )
            with destination.open("wb") as handle:
                pickle.dump(infos, handle)
            print(f"Saved {len(infos)} {split} infos: {destination}")
        if split == "train":
            dataset.create_groundtruth_database(destination, split="train")
            generated = args.data_root / "custom_dbinfos_train.pkl"
            target = args.data_root / "vod_dbinfos_train.pkl"
            if generated.is_file():
                generated.replace(target)
                print(f"Renamed GT database index: {target}")


if __name__ == "__main__":
    main()
