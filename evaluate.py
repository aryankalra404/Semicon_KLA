#!/usr/bin/env python3
"""Evaluate bicubic or a trained model against paired ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from kla_restore.data import PairedNpyDataset, names_for_split
from kla_restore.ensemble import restore
from kla_restore.metrics import mean_and_ci95, psnr, ssim
from kla_restore.runtime import choose_device, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/test/NoisyLR"))
    parser.add_argument("--gt-dir", type=Path, default=Path("data/train/GT"))
    parser.add_argument("--split", choices=("val", "all"), default="val")
    parser.add_argument(
        "--names-manifest",
        type=Path,
        help="Optional JSON manifest containing the filenames to evaluate",
    )
    parser.add_argument("--manifest-key", default="val_names")
    parser.add_argument("--method", choices=("bicubic", "model"), default="bicubic")
    parser.add_argument(
        "--weights", type=Path, nargs="+", help="One or more model checkpoints"
    )
    parser.add_argument(
        "--self-ensemble", choices=("x1", "x4", "x8"), default="x1"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--per-image-output", type=Path)
    parser.add_argument("--lpips", action="store_true", help="Compute LPIPS (requires lpips)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.names_manifest is not None:
        manifest = json.loads(args.names_manifest.read_text())
        names = list(manifest.get(args.manifest_key, ()))
        if not names:
            raise ValueError(
                f"Manifest {args.names_manifest} has no names under {args.manifest_key!r}"
            )
    else:
        names = (
            names_for_split("val")
            if args.split == "val"
            else sorted(path.name for path in args.input_dir.glob("*.npy"))
        )
    dataset = PairedNpyDataset(args.input_dir, args.gt_dir, names)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    device = choose_device(args.device)
    models = None
    lpips_model = None
    if args.method == "model":
        if args.weights is None:
            raise SystemExit("--weights is required when --method=model")
        models = [load_model(weights, device) for weights in args.weights]
    if args.lpips:
        try:
            import lpips
        except ImportError as error:
            raise SystemExit("Install the optional 'lpips' package to use --lpips") from error
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    psnr_values, ssim_values, lpips_values, rows = [], [], [], []
    elapsed, images = 0.0, 0
    with torch.inference_mode():
        for lr, gt, batch_names in loader:
            lr, gt = lr.to(device), gt.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            if models is None:
                prediction = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
            else:
                prediction = restore(models, lr, args.self_ensemble).clamp(0.0, 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            images += lr.shape[0]
            batch_psnr = psnr(prediction, gt).cpu().tolist()
            batch_ssim = ssim(prediction, gt).cpu().tolist()
            if lpips_model is not None:
                pred_rgb = prediction.repeat(1, 3, 1, 1) * 2.0 - 1.0
                gt_rgb = gt.repeat(1, 3, 1, 1) * 2.0 - 1.0
                batch_lpips = lpips_model(pred_rgb, gt_rgb, normalize=False).flatten().cpu().tolist()
            else:
                batch_lpips = [None] * len(batch_names)
            psnr_values.extend(batch_psnr)
            ssim_values.extend(batch_ssim)
            lpips_values.extend(value for value in batch_lpips if value is not None)
            rows.extend(
                {
                    "filename": name,
                    "psnr": p,
                    "ssim": s,
                    "lpips": perceptual,
                }
                for name, p, s, perceptual in zip(
                    batch_names, batch_psnr, batch_ssim, batch_lpips, strict=True
                )
            )

    psnr_mean, psnr_ci = mean_and_ci95(psnr_values)
    ssim_mean, ssim_ci = mean_and_ci95(ssim_values)
    result = {
        "method": args.method,
        "evaluation_manifest": (
            str(args.names_manifest) if args.names_manifest is not None else None
        ),
        "images": images,
        "device": str(device),
        "psnr": psnr_mean,
        "psnr_ci95": psnr_ci,
        "ssim": ssim_mean,
        "ssim_ci95": ssim_ci,
        "milliseconds_per_image": 1000 * elapsed / images,
        "self_ensemble": args.self_ensemble,
        "models": len(models) if models is not None else 0,
    }
    if lpips_values:
        lpips_mean, lpips_ci = mean_and_ci95(lpips_values)
        result["lpips"] = lpips_mean
        result["lpips_ci95"] = lpips_ci
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")
    if args.per_image_output:
        args.per_image_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_image_output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("filename", "psnr", "ssim", "lpips"))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
