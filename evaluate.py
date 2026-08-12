#!/usr/bin/env python3
"""Evaluate bicubic or a trained model against paired ground truth."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from kla_restore.data import PairedNpyDataset, names_for_split
from kla_restore.metrics import mean_and_ci95, psnr, ssim
from kla_restore.runtime import choose_device, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/test/NoisyLR"))
    parser.add_argument("--gt-dir", type=Path, default=Path("data/train/GT"))
    parser.add_argument("--split", choices=("val", "all"), default="val")
    parser.add_argument("--method", choices=("bicubic", "model"), default="bicubic")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = (
        names_for_split("val")
        if args.split == "val"
        else sorted(path.name for path in args.input_dir.glob("*.npy"))
    )
    dataset = PairedNpyDataset(args.input_dir, args.gt_dir, names)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    device = choose_device(args.device)
    model = None
    if args.method == "model":
        if args.weights is None:
            raise SystemExit("--weights is required when --method=model")
        model = load_model(args.weights, device)

    psnr_values, ssim_values, elapsed, images = [], [], 0.0, 0
    with torch.inference_mode():
        for lr, gt, _ in loader:
            lr, gt = lr.to(device), gt.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            if model is None:
                prediction = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
            else:
                prediction = model(lr).clamp(0.0, 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            images += lr.shape[0]
            psnr_values.extend(psnr(prediction, gt).cpu().tolist())
            ssim_values.extend(ssim(prediction, gt).cpu().tolist())

    psnr_mean, psnr_ci = mean_and_ci95(psnr_values)
    ssim_mean, ssim_ci = mean_and_ci95(ssim_values)
    result = {
        "method": args.method,
        "images": images,
        "device": str(device),
        "psnr": psnr_mean,
        "psnr_ci95": psnr_ci,
        "ssim": ssim_mean,
        "ssim_ci95": ssim_ci,
        "milliseconds_per_image": 1000 * elapsed / images,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
