#!/usr/bin/env python3
"""Train KLARestoreNet on the leakage-safe training split."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from kla_restore.data import PairedNpyDataset, names_for_split
from kla_restore.losses import RestorationLoss
from kla_restore.metrics import psnr, ssim
from kla_restore.model import KLARestoreNet, parameter_count
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
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-train", type=int, help="Optional smoke-test sample limit")
    parser.add_argument("--limit-val", type=int, help="Optional smoke-test sample limit")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path, help="Resume model and optimizer state")
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
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    )
    val_set = PairedNpyDataset(
        args.data_root / "NoisyLR",
        args.data_root / "GT",
        val_names,
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )

    model = KLARestoreNet(args.width, args.blocks).to(device)
    loss_fn = RestorationLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_ssim = -1.0
    history = []
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        expected_config = {"width": args.width, "blocks": args.blocks}
        if checkpoint.get("model_config") != expected_config:
            raise ValueError(
                f"Checkpoint architecture {checkpoint.get('model_config')} does not "
                f"match requested architecture {expected_config}"
            )
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_ssim = float(checkpoint.get("best_ssim", checkpoint["metrics"]["ssim"]))
        history_path = args.output_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text())

    print(f"device={device} train={len(train_set)} val={len(val_set)} parameters={parameter_count(model):,}")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        running_loss = 0.0
        for lr, gt, _ in train_loader:
            lr, gt = lr.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(lr)
                loss, _ = loss_fn(prediction, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * lr.shape[0]
        scheduler.step()

        metrics = validate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_set),
            "val_psnr": metrics["psnr"],
            "val_ssim": metrics["ssim"],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record))

        checkpoint = {
            "model": model.state_dict(),
            "model_config": {"width": args.width, "blocks": args.blocks},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "best_ssim": max(best_ssim, metrics["ssim"]),
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["ssim"] > best_ssim:
            best_ssim = metrics["ssim"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
