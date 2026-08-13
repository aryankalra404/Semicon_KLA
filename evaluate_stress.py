#!/usr/bin/env python3
"""Evaluate restoration robustness on deterministic compound degradations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from kla_restore.data import load_npy_tensor, names_for_split
from kla_restore.metrics import mean_and_ci95, psnr, ssim
from kla_restore.runtime import choose_device, load_model


SCENARIOS = (
    "downsample_only",
    "gaussian_high",
    "speckle_high",
    "blur_high",
    "noise_before_downsample",
    "downsample_before_noise",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=Path, default=Path("data/train/GT"))
    parser.add_argument("--weights", type=Path, default=Path("weights/final.pt"))
    parser.add_argument("--method", choices=("bicubic", "model"), default="model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=260813)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    coordinates = torch.arange(7, dtype=image.dtype, device=image.device) - 3
    kernel = torch.exp(-coordinates.square() / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel).view(1, 1, 7, 7)
    return F.conv2d(F.pad(image, (3, 3, 3, 3), mode="reflect"), window)


def noise(
    image: torch.Tensor,
    generator: torch.Generator,
    gaussian_sigma: float,
    speckle_sigma: float,
) -> torch.Tensor:
    gaussian = torch.randn(
        image.shape, dtype=image.dtype, device=image.device, generator=generator
    )
    speckle = torch.randn(
        image.shape, dtype=image.dtype, device=image.device, generator=generator
    )
    return image + gaussian * gaussian_sigma + image.abs() * speckle * speckle_sigma


def degrade(
    gt: torch.Tensor, scenario: str, generator: torch.Generator
) -> torch.Tensor:
    """Apply a named deterministic stress degradation without clipping."""
    if scenario == "downsample_only":
        return F.interpolate(gt, scale_factor=0.5, mode="bicubic", align_corners=False)
    if scenario == "gaussian_high":
        lr = F.interpolate(gt, scale_factor=0.5, mode="area")
        return noise(lr, generator, gaussian_sigma=0.13, speckle_sigma=0.0)
    if scenario == "speckle_high":
        lr = F.interpolate(gt, scale_factor=0.5, mode="area")
        return noise(lr, generator, gaussian_sigma=0.0, speckle_sigma=0.22)
    if scenario == "blur_high":
        return F.interpolate(blur(gt, 1.6), scale_factor=0.5, mode="area")
    if scenario == "noise_before_downsample":
        degraded = noise(blur(gt, 0.9), generator, 0.10, 0.16)
        return F.interpolate(degraded, scale_factor=0.5, mode="area")
    if scenario == "downsample_before_noise":
        degraded = F.interpolate(blur(gt, 0.9), scale_factor=0.5, mode="area")
        return noise(degraded, generator, 0.08, 0.14)
    raise ValueError(f"Unknown scenario: {scenario}")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model = load_model(args.weights, device) if args.method == "model" else None
    names = names_for_split("val")
    if args.limit is not None:
        names = names[: args.limit]
    results: dict[str, dict[str, float | int]] = {}

    with torch.inference_mode():
        for scenario_index, scenario in enumerate(SCENARIOS):
            psnr_values: list[float] = []
            ssim_values: list[float] = []
            for image_index, name in enumerate(names):
                gt = load_npy_tensor(args.gt_dir / name).unsqueeze(0).to(device)
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + scenario_index * 10_000 + image_index
                )
                lr = degrade(gt, scenario, generator)
                if model is None:
                    prediction = F.interpolate(
                        lr, scale_factor=2, mode="bicubic", align_corners=False
                    )
                else:
                    prediction = model(lr)
                prediction = prediction.clamp(0.0, 1.0)
                psnr_values.append(float(psnr(prediction, gt)[0]))
                ssim_values.append(float(ssim(prediction, gt)[0]))
            psnr_mean, psnr_ci = mean_and_ci95(psnr_values)
            ssim_mean, ssim_ci = mean_and_ci95(ssim_values)
            results[scenario] = {
                "images": len(names),
                "psnr": psnr_mean,
                "psnr_ci95": psnr_ci,
                "ssim": ssim_mean,
                "ssim_ci95": ssim_ci,
            }

    result = {
        "method": args.method,
        "weights": str(args.weights) if model is not None else None,
        "device": str(device),
        "seed": args.seed,
        "scenarios": results,
        "macro_average": {
            "psnr": float(np.mean([value["psnr"] for value in results.values()])),
            "ssim": float(np.mean([value["ssim"] for value in results.values()])),
        },
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
