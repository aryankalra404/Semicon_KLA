#!/usr/bin/env python3
"""Optional restoration reliability mode using geometric-consistency uncertainty."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from kla_restore.data import UnpairedNpyDataset
from kla_restore.robustness import restoration_distribution
from kla_restore.runtime import choose_device, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uncertainty-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, nargs="+", default=[Path("weights/final.pt")])
    parser.add_argument("--self-ensemble", choices=("x4", "x8"), default="x8")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.uncertainty_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    root = Path(__file__).resolve().parent
    models = [load_model(path if path.is_absolute() else root / path, device) for path in args.weights]
    dataset = UnpairedNpyDataset(args.input_dir)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, path in enumerate(dataset.paths):
        shape = np.load(path, mmap_mode="r", allow_pickle=False).shape
        if len(shape) != 2:
            raise ValueError(f"Expected a 2D grayscale array at {path}, got {shape}")
        groups[tuple(shape)].append(index)

    scores: dict[str, float] = {}
    with torch.inference_mode():
        for indices in groups.values():
            loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.workers)
            for inputs, names in loader:
                mean, uncertainty = restoration_distribution(models, inputs.to(device), args.self_ensemble)
                mean = mean.clamp(0.0, 1.0).cpu().numpy()[:, 0]
                uncertainty = uncertainty.cpu().numpy()[:, 0]
                for name, restored, heatmap in zip(names, mean, uncertainty, strict=True):
                    np.save(args.output_dir / name, restored.astype(np.float32), allow_pickle=False)
                    np.save(args.uncertainty_dir / name, heatmap.astype(np.float32), allow_pickle=False)
                    scores[name] = float(np.quantile(heatmap, 0.95))

    ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    result = {
        "images": len(dataset),
        "method": "geometric_consistency",
        "self_ensemble": args.self_ensemble,
        "score": "95th percentile pixelwise standard deviation",
        "highest_uncertainty": [
            {"filename": name, "score": score} for name, score in ranking[:20]
        ],
        "mean_score": float(np.mean(list(scores.values()))),
        "limitations": "Disagreement is a reliability indicator, not calibrated failure probability.",
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
