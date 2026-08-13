#!/usr/bin/env python3
"""Validate whether geometric disagreement ranks restoration error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kla_restore.data import PairedNpyDataset, names_for_split
from kla_restore.robustness import restoration_distribution
from kla_restore.runtime import choose_device, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/train/NoisyLR"))
    parser.add_argument("--gt-dir", type=Path, default=Path("data/train/GT"))
    parser.add_argument("--weights", type=Path, default=Path("weights/final.pt"))
    parser.add_argument("--self-ensemble", choices=("x4", "x8"), default="x8")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = rankdata(left), rankdata(right)
    if left_rank.std() == 0 or right_rank.std() == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model = load_model(args.weights, device)
    names = names_for_split("val")
    if args.limit is not None:
        names = names[: args.limit]
    dataset = PairedNpyDataset(args.input_dir, args.gt_dir, names)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    uncertainty_scores, errors = [], []
    with torch.inference_mode():
        for inputs, targets, _ in loader:
            targets = targets.to(device)
            mean, uncertainty = restoration_distribution(
                [model], inputs.to(device), args.self_ensemble
            )
            uncertainty_scores.extend(
                torch.quantile(uncertainty.flatten(1), 0.95, dim=1).cpu().tolist()
            )
            errors.extend((mean.clamp(0.0, 1.0) - targets).abs().flatten(1).mean(1).cpu().tolist())
    uncertainty_array = np.asarray(uncertainty_scores)
    error_array = np.asarray(errors)
    order = np.argsort(uncertainty_array)
    quartile = max(1, len(order) // 4)
    result = {
        "images": len(dataset),
        "weights": str(args.weights),
        "self_ensemble": args.self_ensemble,
        "uncertainty_error_spearman": spearman(uncertainty_array, error_array),
        "lowest_uncertainty_mae": float(error_array[order[:quartile]].mean()),
        "highest_uncertainty_mae": float(error_array[order[-quartile:]].mean()),
        "uncertainty_score_mean": float(uncertainty_array.mean()),
        "limitations": "Geometric disagreement ranks risk but is not a calibrated probability.",
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
