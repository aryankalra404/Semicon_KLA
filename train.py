#!/usr/bin/env python3
"""Train KLARestoreNet on the leakage-safe training split."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from kla_restore.data import (
    PairedNpyDataset,
    all_pair_names,
    deterministic_split_names,
    names_for_split,
)
from kla_restore.losses import RestorationLoss
from kla_restore.metrics import psnr, ssim
from kla_restore.model import (
    KLARestoreNet,
    build_model,
    initialize_v3_from_v2,
    initialize_v4a_from_v2,
    model_config,
    parameter_count,
)
from kla_restore.runtime import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("weights"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--variant", choices=("v2", "v3", "v4a"), default="v2")
    parser.add_argument("--condition-dim", type=int, default=32)
    parser.add_argument("--hr-width", type=int, default=48)
    parser.add_argument("--hr-blocks", type=int, default=2)
    parser.add_argument("--disable-degradation-conditioning", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-train", type=int, help="Optional smoke-test sample limit")
    parser.add_argument("--limit-val", type=int, help="Optional smoke-test sample limit")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--split-seed",
        type=int,
        help="Use a reproducible random 90/10 split instead of fixed IDs",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train on all 3,200 pairs after configuration selection; validation is disabled",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path, help="Resume model and optimizer state")
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Initialize v2, v3, or v4a from a v2 inference/training checkpoint",
    )
    parser.add_argument("--synthetic-probability", type=float, default=0.0)
    parser.add_argument(
        "--synthetic-policy", choices=("fixed", "randomized"), default="fixed"
    )
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--pixel-weight", type=float, default=0.7)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--skip-data-audit", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model: KLARestoreNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    psnr_values, ssim_values = [], []
    for lr, gt, _ in loader:
        lr, gt = lr.to(device), gt.to(device)
        prediction = model(lr).clamp(0.0, 1.0)
        psnr_values.extend(psnr(prediction, gt).cpu().tolist())
        ssim_values.extend(ssim(prediction, gt).cpu().tolist())
    return {
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
    }


def main() -> None:
    args = parse_args()
    if args.resume and args.initialize_from:
        raise SystemExit("Use only one of --resume and --initialize-from")
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_all:
        train_names = all_pair_names()
        val_names: list[str] = []
    elif args.split_seed is not None:
        train_names, val_names = deterministic_split_names(args.split_seed)
    else:
        train_names = names_for_split("train")
        val_names = names_for_split("val")
    if args.limit_train is not None:
        train_names = train_names[: args.limit_train]
    if args.limit_val is not None:
        val_names = val_names[: args.limit_val]
    train_set = PairedNpyDataset(
        args.data_root / "NoisyLR",
        args.data_root / "GT",
        train_names,
        augment=True,
        synthetic_probability=args.synthetic_probability,
        synthetic_policy=args.synthetic_policy,
        audit=not args.skip_data_audit,
    )
    val_set = (
        PairedNpyDataset(
            args.data_root / "NoisyLR",
            args.data_root / "GT",
            val_names,
            audit=not args.skip_data_audit,
        )
        if val_names
        else None
    )
    pin_memory = device.type == "cuda"
    data_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
        generator=data_generator,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=pin_memory,
            persistent_workers=args.workers > 0,
        )
        if val_set is not None
        else None
    )

    model = KLARestoreNet(
        args.width,
        args.blocks,
        variant=args.variant,
        condition_dim=args.condition_dim,
        hr_width=args.hr_width,
        hr_blocks=args.hr_blocks,
        degradation_conditioning=not args.disable_degradation_conditioning,
    ).to(device)
    if args.initialize_from:
        source = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
        source_config = source.get("model_config", {})
        if source_config.get("variant", "v2") != "v2":
            raise ValueError("--initialize-from checkpoint must contain a v2 model")
        if int(source_config.get("width", 48)) != args.width or int(
            source_config.get("blocks", 12)
        ) != args.blocks:
            raise ValueError(
                "--initialize-from width/blocks must match the requested model "
                f"configuration; checkpoint={source_config}, "
                f"requested width={args.width}/blocks={args.blocks}"
            )
        if args.variant == "v2":
            model.load_state_dict(source["model"], strict=True)
            copied_tensors = len(source["model"])
            copied_parameters = sum(value.numel() for value in source["model"].values())
        else:
            initializer = (
                initialize_v3_from_v2
                if args.variant == "v3"
                else initialize_v4a_from_v2
            )
            copied_tensors, copied_parameters = initializer(model, source["model"])
        source_parameters = sum(value.numel() for value in source["model"].values())
        if copied_parameters != source_parameters:
            raise ValueError(
                "Warm-start did not copy the complete v2 model: "
                f"copied {copied_parameters:,}/{source_parameters:,} parameters. "
                "For exact v3 transfer, set --hr-width equal to --width."
            )
        print(
            f"initialized_from={args.initialize_from} tensors={copied_tensors} "
            f"parameters={copied_parameters:,}"
        )
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    loss_fn = RestorationLoss(
        pixel_weight=args.pixel_weight,
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
        consistency_weight=args.consistency_weight,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_ssim = -1.0
    best_psnr = -1.0
    best_balanced = -1.0
    history = []
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        expected_config = model_config(model)
        checkpoint_config = model_config(build_model(checkpoint.get("model_config")))
        if checkpoint_config != expected_config:
            raise ValueError(
                f"Checkpoint architecture {checkpoint_config} does not "
                f"match requested architecture {expected_config}"
            )
        model.load_state_dict(checkpoint.get("train_model", checkpoint["model"]))
        ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_ssim = float(checkpoint.get("best_ssim", checkpoint["metrics"]["ssim"]))
        best_psnr = float(checkpoint.get("best_psnr", checkpoint["metrics"]["psnr"]))
        best_balanced = float(
            checkpoint.get(
                "best_balanced",
                checkpoint["metrics"]["psnr"] + 10 * checkpoint["metrics"]["ssim"],
            )
        )
        history_path = args.output_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text())

    print(
        f"device={device} train={len(train_set)} "
        f"val={len(val_set) if val_set is not None else 0} "
        f"parameters={parameter_count(model):,}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        running_loss = 0.0
        running_parts: dict[str, float] = {}
        for lr, gt, _ in train_loader:
            lr, gt = lr.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(lr)
                loss, parts = loss_fn(prediction, gt, lr)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    ema_model.parameters(), model.parameters(), strict=True
                ):
                    ema_parameter.lerp_(parameter, 1.0 - args.ema_decay)
            running_loss += float(loss.detach()) * lr.shape[0]
            for name, value in parts.items():
                running_parts[name] = running_parts.get(name, 0.0) + value * lr.shape[0]
        scheduler.step()

        metrics = (
            validate(ema_model, val_loader, device)
            if val_loader is not None
            else {"psnr": float("nan"), "ssim": float("nan")}
        )
        record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_set),
            "train_parts": {
                name: value / len(train_set) for name, value in running_parts.items()
            },
            "val_psnr": metrics["psnr"],
            "val_ssim": metrics["ssim"],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record))

        checkpoint = {
            "model": ema_model.state_dict(),
            "train_model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "model_config": model_config(model),
            "training_config": {
                "synthetic_probability": args.synthetic_probability,
                "synthetic_policy": args.synthetic_policy,
                "consistency_weight": args.consistency_weight,
                "ema_decay": args.ema_decay,
                "gradient_clip": args.gradient_clip,
                "seed": args.seed,
                "data_order_seed": args.seed,
                "split_seed": args.split_seed,
                "train_all": args.train_all,
                "loss_weights": {
                    "pixel": args.pixel_weight,
                    "ssim": args.ssim_weight,
                    "edge": args.edge_weight,
                    "consistency": args.consistency_weight,
                },
                "initialized_from": (
                    str(args.initialize_from) if args.initialize_from else None
                ),
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "best_ssim": max(best_ssim, metrics["ssim"]),
            "best_psnr": max(best_psnr, metrics["psnr"]),
            "best_balanced": max(
                best_balanced, metrics["psnr"] + 10 * metrics["ssim"]
            ),
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_loader is None:
            torch.save(checkpoint, args.output_dir / "final_all_data.pt")
            (args.output_dir / "history.json").write_text(
                json.dumps(history, indent=2)
            )
            continue
        if metrics["ssim"] > best_ssim:
            best_ssim = metrics["ssim"]
            torch.save(checkpoint, args.output_dir / "best_ssim.pt")
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            torch.save(checkpoint, args.output_dir / "best_psnr.pt")
        balanced = metrics["psnr"] + 10 * metrics["ssim"]
        if balanced > best_balanced:
            best_balanced = balanced
            torch.save(checkpoint, args.output_dir / "best_balanced.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
