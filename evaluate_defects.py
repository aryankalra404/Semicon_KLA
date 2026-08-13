#!/usr/bin/env python3
"""Measure preservation and hallucination of controlled defect-like structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_stress import degrade
from kla_restore.data import load_npy_tensor, names_for_split
from kla_restore.metrics import mean_and_ci95
from kla_restore.robustness import binary_dilation, precision_recall
from kla_restore.runtime import choose_device, load_model


DEFECTS = ("bright_dot", "dark_dot", "thin_line", "line_break", "edge_notch")
SCENARIOS = ("noise_before_downsample", "downsample_before_noise")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=Path, default=Path("data/train/GT"))
    parser.add_argument("--weights", type=Path, default=Path("weights/final.pt"))
    parser.add_argument("--method", choices=("bicubic", "model"), default="model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--seed", type=int, default=260813)
    parser.add_argument("--response-threshold", type=float, default=0.035)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def inject_defect(
    image: torch.Tensor, kind: str, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject one localized deterministic probe and return image plus support mask."""
    if kind not in DEFECTS:
        raise ValueError(f"Unknown defect kind: {kind}")
    result = image.clone()
    mask = torch.zeros_like(image, dtype=torch.bool)
    height, width = image.shape[-2:]
    generator = torch.Generator(device=image.device).manual_seed(seed)
    y = int(torch.randint(height // 4, 3 * height // 4, (), generator=generator, device=image.device))
    x = int(torch.randint(width // 4, 3 * width // 4, (), generator=generator, device=image.device))

    if kind in {"bright_dot", "dark_dot"}:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=image.device),
            torch.arange(width, device=image.device),
            indexing="ij",
        )
        support = (yy - y).square() + (xx - x).square() <= 4
        mask[..., support] = True
        delta = 0.32 if kind == "bright_dot" else -0.32
        result = torch.where(mask, (result + delta).clamp(0.0, 1.0), result)
    elif kind == "thin_line":
        length = 15
        mask[..., y, max(0, x - length // 2) : min(width, x + length // 2 + 1)] = True
        local = result[..., y : y + 1, max(0, x - 1) : min(width, x + 2)].mean()
        value = torch.tensor(0.9 if local < 0.5 else 0.1, device=image.device)
        result = torch.where(mask, value, result)
    elif kind == "line_break":
        # Remove a short segment by replacing it with its immediate vertical context.
        mask[..., y - 1 : y + 2, x - 3 : x + 4] = True
        replacement = 0.5 * (result[..., y - 4 : y - 3, x - 3 : x + 4] + result[..., y + 3 : y + 4, x - 3 : x + 4])
        result[..., y - 1 : y + 2, x - 3 : x + 4] = replacement.expand_as(result[..., y - 1 : y + 2, x - 3 : x + 4])
    else:
        # A small rectangular notch with contrast opposite to its local neighborhood.
        mask[..., y - 2 : y + 3, x - 2 : x + 3] = True
        local = result[..., y - 5 : y + 6, x - 5 : x + 6].mean()
        value = torch.tensor(0.85 if local < 0.5 else 0.15, device=image.device)
        result = torch.where(mask, value, result)
    return result.clamp(0.0, 1.0), mask


def restore(model: torch.nn.Module | None, lr: torch.Tensor) -> torch.Tensor:
    if model is None:
        return F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    return model(lr).clamp(0.0, 1.0)


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    device = choose_device(args.device)
    model = load_model(args.weights, device) if args.method == "model" else None
    names = names_for_split("val")[: args.limit]
    rows: list[dict[str, float | str]] = []

    with torch.inference_mode():
        for image_index, name in enumerate(names):
            base = load_npy_tensor(args.gt_dir / name).unsqueeze(0).to(device)
            for defect_index, defect in enumerate(DEFECTS):
                seed = args.seed + image_index * 100 + defect_index
                modified, support = inject_defect(base, defect, seed=seed)
                truth_response = (modified - base).abs()
                region = binary_dilation(support, radius=3)
                target_mask = binary_dilation(truth_response > 0.02, radius=2)
                for scenario_index, scenario in enumerate(SCENARIOS):
                    noise_seed = seed + scenario_index * 1_000_000
                    base_lr = degrade(base, scenario, torch.Generator(device=device).manual_seed(noise_seed))
                    defect_lr = degrade(modified, scenario, torch.Generator(device=device).manual_seed(noise_seed))
                    restored_base = restore(model, base_lr)
                    restored_defect = restore(model, defect_lr)
                    response = (restored_defect - restored_base).abs()
                    predicted_mask = response > args.response_threshold
                    precision, recall, f1 = precision_recall(predicted_mask, target_mask)
                    recovered = float(response[region].sum() / truth_response[region].sum().clamp_min(1e-8))
                    outside = ~binary_dilation(region, radius=2)
                    false_rate = float(predicted_mask[outside].float().mean())
                    rows.append(
                        {
                            "filename": name,
                            "defect": defect,
                            "scenario": scenario,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "contrast_recovery": recovered,
                            "false_pattern_rate": false_rate,
                        }
                    )

    metrics = ("precision", "recall", "f1", "contrast_recovery", "false_pattern_rate")
    summary = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        mean, ci = mean_and_ci95(values)
        summary[metric] = {"mean": mean, "ci95": ci}
    by_defect = {
        defect: {
            metric: float(np.mean([float(row[metric]) for row in rows if row["defect"] == defect]))
            for metric in metrics
        }
        for defect in DEFECTS
    }
    result = {
        "method": args.method,
        "weights": str(args.weights) if model is not None else None,
        "images": len(names),
        "probes": len(rows),
        "defects": list(DEFECTS),
        "scenarios": list(SCENARIOS),
        "response_threshold": args.response_threshold,
        "summary": summary,
        "by_defect": by_defect,
        "limitations": "Synthetic probes measure controlled sensitivity; they are not real defect labels.",
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
