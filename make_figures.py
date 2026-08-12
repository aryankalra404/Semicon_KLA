#!/usr/bin/env python3
"""Generate learning curves and honest qualitative validation comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from kla_restore.data import load_npy_tensor
from kla_restore.runtime import choose_device, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("weights/v2/best_balanced.pt"))
    parser.add_argument("--history", type=Path, default=Path("weights/v2/history.json"))
    parser.add_argument("--per-image", type=Path, default=Path("outputs/v2_per_image.csv"))
    parser.add_argument("--lr-dir", type=Path, default=Path("data/train/NoisyLR"))
    parser.add_argument("--gt-dir", type=Path, default=Path("data/train/GT"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def save_learning_curve(history_path: Path, output: Path) -> None:
    history = json.loads(history_path.read_text())
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot(epochs, [row["val_psnr"] for row in history], color="#2563eb", linewidth=2)
    axes[0].set(title="Validation PSNR", xlabel="Epoch", ylabel="PSNR (dB)")
    axes[1].plot(epochs, [row["val_ssim"] for row in history], color="#7c3aed", linewidth=2)
    axes[1].set(title="Validation SSIM", xlabel="Epoch", ylabel="SSIM")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_case(
    name: str, label: str, lr_dir: Path, gt_dir: Path, model: torch.nn.Module,
    device: torch.device, output: Path,
) -> None:
    lr = load_npy_tensor(lr_dir / name).unsqueeze(0).to(device)
    gt = load_npy_tensor(gt_dir / name).unsqueeze(0).to(device)
    with torch.inference_mode():
        bicubic = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
        restored = model(lr).clamp(0, 1)
    panels = [
        ("Degraded input", lr[0, 0].cpu().numpy()),
        ("Bicubic", bicubic[0, 0].cpu().numpy()),
        ("KLA-RestoreNet v2", restored[0, 0].cpu().numpy()),
        ("Ground truth", gt[0, 0].cpu().numpy()),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    for axis, (title, image) in zip(axes, panels, strict=True):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle(f"{label} validation case — {name}", fontsize=12)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_learning_curve(args.history, args.output_dir / "v2_learning_curves.png")

    with args.per_image.open() as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: float(row["ssim"]))
    selected = (
        (rows[-1], "Best"),
        (rows[len(rows) // 2], "Median"),
        (rows[0], "Worst"),
    )
    device = choose_device(args.device)
    model = load_model(args.weights, device)
    for row, label in selected:
        plot_case(
            row["filename"], label, args.lr_dir, args.gt_dir, model, device,
            args.output_dir / f"v2_{label.lower()}_{Path(row['filename']).stem}.png",
        )
    (args.output_dir / "selected_cases.json").write_text(
        json.dumps([{**row, "selection": label} for row, label in selected], indent=2)
    )
    print(f"wrote learning curve and {len(selected)} comparison panels to {args.output_dir}")


if __name__ == "__main__":
    main()
