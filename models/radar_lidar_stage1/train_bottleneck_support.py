"""Train and validate the two-output bottleneck Stage I without legacy metrics."""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .bottleneck_support import BottleneckSupportStage1
from .data import VoDStage1Dataset, collate_stage1


def _move(batch, device):
    return {key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()}


def _checkpoint(path, model, optimizer, scheduler, epoch, step, validation, data):
    torch.save({"config": model.config.as_dict(), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch": epoch, "step": step, "validation": validation,
                "data": data, "stage1_interface": "bottleneck_support_v1"}, path)


def train(args, config):
    if args.selection_metric != "support_f1":
        raise ValueError("Bottleneck Stage I selects checkpoints using support_f1")
    if args.tensorboard:
        raise ValueError("The bottleneck-support trainer does not write TensorBoard summaries")
    if args.object_instances or args.corruption_every:
        print("Bottleneck Stage I evaluates support and correspondence; object and corruption diagnostics are unavailable in this trainer.", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "config.json").write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
    train_set = VoDStage1Dataset(args.vod_root, "train", radar_variant=args.radar_variant)
    val_set = VoDStage1Dataset(args.vod_root, "val", radar_variant=args.radar_variant)
    if args.train_limit:
        train_set.frames = train_set.frames[:args.train_limit]
    if args.val_limit:
        val_set.frames = val_set.frames[:args.val_limit]
    loaders = {
        "train": DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_stage1,
                            pin_memory=args.device.startswith("cuda")),
        "val": DataLoader(val_set, batch_size=1, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_stage1,
                          pin_memory=args.device.startswith("cuda")),
    }
    model = BottleneckSupportStage1(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch, step, best = 1, 0, float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        if checkpoint.get("stage1_interface") != "bottleneck_support_v1" or checkpoint["config"] != config.as_dict():
            raise ValueError("Only a matching bottleneck-support checkpoint can be resumed")
        if checkpoint.get("data", {}).get("radar_variant") != args.radar_variant:
            raise ValueError("Resume radar variant differs from the original training data")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, step = checkpoint["epoch"] + 1, checkpoint["step"]
        selected_path = args.output_root / "best_selected.ckpt"
        selected = (torch.load(selected_path, map_location="cpu", weights_only=False)
                    if selected_path.is_file() else checkpoint)
        if selected.get("stage1_interface") != "bottleneck_support_v1":
            raise ValueError("Existing best checkpoint is from another Stage-I interface")
        best = selected.get("validation", {}).get("support_f1", float("-inf"))
    provenance = {"vod_root": str(args.vod_root.resolve()), "radar_variant": args.radar_variant,
                  "train_frames": len(train_set), "val_frames": len(val_set)}
    print(f"Bottleneck Stage I: train={len(train_set)} val={len(val_set)} support stride={config.support_stride}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {"loss/total": 0., "loss/corr": 0., "loss/distill": 0., "loss/support": 0.}
        progress = tqdm(loaders["train"], desc=f"Bottleneck train {epoch:03d}/{args.epochs:03d}", unit="batch", dynamic_ncols=True)
        samples = 0
        for batch_index, raw in enumerate(progress):
            batch = _move(raw, args.device)
            loss, _ = model.forward_train(batch["radar"], batch["radar_valid"],
                                          batch["clean_lidar"], batch["clean_lidar_valid"],
                                          defer_diagnostics=True, trusted_inputs=True)
            group_start = batch_index - batch_index % args.grad_accum_steps
            group_size = min(args.grad_accum_steps, len(loaders["train"]) - group_start)
            (loss["loss/total"] / group_size).backward()
            if (batch_index + 1) % args.grad_accum_steps == 0 or batch_index + 1 == len(loaders["train"]):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            count = len(batch["frame_id"])
            samples += count
            for key in sums:
                sums[key] += float(loss[key].detach()) * count
            if batch_index % 20 == 0:
                progress.set_postfix(loss=f"{sums['loss/total']/samples:.4f}")
        scheduler.step()
        training = {key: value / max(samples, 1) for key, value in sums.items()}
        validation = {}
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            model.eval()
            tp = fp = positives = unknown_selected = support_sites = clean_covered = clean_total = 0
            val_loss = 0.
            with torch.no_grad():
                for raw in tqdm(loaders["val"], desc=f"Bottleneck val {epoch:03d}", unit="frame", dynamic_ncols=True):
                    batch = _move(raw, args.device)
                    loss, diagnostics = model.forward_train(batch["radar"], batch["radar_valid"],
                        batch["clean_lidar"], batch["clean_lidar_valid"],
                        defer_diagnostics=True, trusted_inputs=True)
                    val_loss += float(loss["loss/total"])
                    output = diagnostics["output"]
                    selected = output.confidence_region.features[:, 0] > config.support_eval_threshold
                    occupied = diagnostics["occupied"]
                    free = diagnostics["known_free"]
                    tp += int((selected & occupied).sum())
                    fp += int((selected & free).sum())
                    positives += int(occupied.sum())
                    unknown_selected += int((selected & ~(occupied | free)).sum())
                    support_sites += len(selected)
                    clean_covered += diagnostics["clean_points_in_support"]
                    clean_total += diagnostics["clean_points_in_grid"]
            precision = tp / (tp + fp) if tp + fp else 0.
            recall = tp / positives if positives else 0.
            validation = {"loss": val_loss / max(len(loaders["val"]), 1),
                          "support_threshold": config.support_eval_threshold,
                          "support_precision_on_observed": precision,
                          "support_recall_on_eligible_clean": recall,
                          "support_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
                          "unverified_selected_sites": unknown_selected,
                          "support_sites": support_sites,
                          "clean_point_coverage_before_score": clean_covered / max(clean_total, 1)}
            (args.output_root / f"validation_epoch_{epoch:03d}.json").write_text(
                json.dumps(validation, indent=2), encoding="utf-8")
        with (args.output_root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch, "step": step, "train": training,
                                      "validation": validation}) + "\n")
        _checkpoint(args.output_root / "last.ckpt", model, optimizer, scheduler,
                    epoch, step, validation, provenance)
        if validation and validation["support_f1"] > best:
            best = validation["support_f1"]
            _checkpoint(args.output_root / "best_selected.ckpt", model, optimizer, scheduler,
                        epoch, step, validation, provenance)
        state = model.radar_only_state_dict()
        state["data"] = provenance
        torch.save(state, args.output_root / "radar_only.pth")
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | train {training['loss/total']:.4f}"
              + (f" | val {validation['loss']:.4f} support F1 {validation['support_f1']:.3f}"
                 if validation else ""), flush=True)
