"""Thin runtime wrapper around upstream OpenPCDet evaluation APIs."""

from __future__ import annotations

import copy
import pickle
import sys
from argparse import Namespace
from contextlib import contextmanager
import os
from pathlib import Path


def add_openpcdet_to_path(openpcdet_root: str | Path) -> Path:
    root = Path(openpcdet_root).resolve()
    if not (root / "pcdet").is_dir():
        raise FileNotFoundError(
            f"Official OpenPCDet checkout is missing under {root}; run "
            "tools/openpcdet/bootstrap_openpcdet.sh"
        )
    for path in (root, root / "tools"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _trusted_checkpoint_loading(torch_module):
    """Restore the pre-2.6 torch.load default for our own checkpoints only."""

    original_load = torch_module.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch_module.load = trusted_load
    try:
        yield
    finally:
        torch_module.load = original_load


def load_openpcdet_config(openpcdet_root: str | Path, config_path: str | Path):
    """Load YAML from the cwd expected by upstream `_BASE_CONFIG_` handling."""

    root = add_openpcdet_to_path(openpcdet_root)
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file

    config = EasyDict()
    config.ROOT_DIR = root
    config.LOCAL_RANK = 0
    with _working_directory(root / "tools"):
        cfg_from_yaml_file(str(Path(config_path).resolve()), config)
    return config


def create_custom_infos(
    openpcdet_root: str | Path,
    dataset_config,
    class_names: list[str],
    data_root: str | Path,
    split: str,
    *,
    workers: int,
) -> Path:
    add_openpcdet_to_path(openpcdet_root)
    from pcdet.datasets.custom.custom_dataset import CustomDataset
    from pcdet.utils import common_utils

    data_root = Path(data_root)
    dataset_cfg = copy.deepcopy(dataset_config)
    dataset_cfg.DATA_PATH = str(data_root.resolve())
    dataset = CustomDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=data_root,
        training=False,
        logger=common_utils.create_logger(),
    )
    dataset.set_split(split)
    infos = dataset.get_infos(
        class_names,
        num_workers=workers,
        has_label=True,
        num_features=4,
    )
    destination = data_root / f"vod_infos_{split}.pkl"
    with destination.open("wb") as handle:
        pickle.dump(infos, handle)
    return destination


def evaluate_checkpoint_on_condition(
    openpcdet_root: str | Path,
    cfg,
    checkpoint: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    *,
    split: str,
    batch_size: int,
    workers: int,
    model=None,
):
    """Run official ``eval_one_epoch`` and return metrics/predictions/model."""

    add_openpcdet_to_path(openpcdet_root)
    import torch
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network
    from pcdet.utils import common_utils
    from eval_utils import eval_utils

    condition_cfg = copy.deepcopy(cfg)
    condition_cfg.DATA_CONFIG.DATA_PATH = str(Path(data_root).resolve())
    condition_cfg.DATA_CONFIG.DATA_SPLIT.test = split
    condition_cfg.DATA_CONFIG.INFO_PATH.test = [f"vod_infos_{split}.pkl"]
    logger = common_utils.create_logger()
    dataset, loader, _ = build_dataloader(
        dataset_cfg=condition_cfg.DATA_CONFIG,
        class_names=condition_cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        logger=logger,
        training=False,
    )
    if model is None:
        model = build_network(
            model_cfg=condition_cfg.MODEL,
            num_class=len(condition_cfg.CLASS_NAMES),
            dataset=dataset,
        )
        # OpenPCDet checkpoints include optimizer/NumPy metadata and therefore
        # are not weights-only archives.  These checkpoints are produced by
        # this project's own trusted training run.
        with _trusted_checkpoint_loading(torch):
            model.load_params_from_file(
                filename=str(checkpoint), logger=logger, to_cpu=False
            )
        model.cuda().eval().requires_grad_(False)
    else:
        model.dataset = dataset
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics = eval_utils.eval_one_epoch(
        condition_cfg,
        Namespace(save_to_file=True, infer_time=False),
        model,
        loader,
        epoch_id="frozen_best",
        logger=logger,
        dist_test=False,
        save_to_file=True,
        result_dir=output_root,
    )
    prediction_path = output_root / "result.pkl"
    if not prediction_path.is_file():
        candidates = list(output_root.rglob("result.pkl"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"OpenPCDet did not produce one result.pkl under {output_root}"
            )
        prediction_path = candidates[0]
    with prediction_path.open("rb") as handle:
        predictions = pickle.load(handle)
    return metrics, predictions, model


def validation_3d_map(metrics: dict) -> float:
    """Select moderate R40 3D mAP from OpenPCDet's KITTI metric dictionary."""

    preferred = [
        float(value)
        for key, value in metrics.items()
        if "3d" in key.lower()
        and "moderate" in key.lower()
        and "r40" in key.lower()
        and isinstance(value, (int, float))
    ]
    fallback = [
        float(value)
        for key, value in metrics.items()
        if "3d" in key.lower() and isinstance(value, (int, float))
    ]
    values = preferred or fallback
    if not values:
        raise KeyError(
            "OpenPCDet evaluation returned no numeric 3D AP keys: "
            + ", ".join(sorted(metrics))
        )
    return sum(values) / len(values)
