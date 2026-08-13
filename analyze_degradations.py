#!/usr/bin/env python3
"""Measure reproducible degradation statistics from the official paired data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from kla_restore.data import names_for_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/train"))
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument(
        "--output", type=Path, default=Path("results/degradation_statistics.json")
    )
    return parser.parse_args()


def summarize(values: np.ndarray) -> dict[str, float]:
    levels = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    return {
        f"p{int(level * 100):02d}": float(value)
        for level, value in zip(levels, np.quantile(values, levels), strict=True)
    }


def main() -> None:
    args = parse_args()
    if args.split == "all":
        names = names_for_split("train") + names_for_split("val")
    else:
        names = names_for_split(args.split)

    rows: list[tuple[float, ...]] = []
    for name in names:
        lr = np.load(args.data_root / "NoisyLR" / name, allow_pickle=False).astype(
            np.float64
        )
        gt = np.load(args.data_root / "GT" / name, allow_pickle=False).astype(
            np.float32
        )
        baseline = F.interpolate(
            torch.from_numpy(gt)[None, None],
            size=lr.shape,
            mode="bicubic",
            align_corners=False,
        )[0, 0].numpy().astype(np.float64)
        design = np.column_stack((baseline.ravel(), np.ones(baseline.size)))
        gain, offset = np.linalg.lstsq(design, lr.ravel(), rcond=None)[0]
        residual = lr - (gain * baseline + offset)
        rows.append(
            (
                gain,
                offset,
                residual.std(),
                lr.min(),
                lr.max(),
                np.mean((lr < 0.0) | (lr > 1.0)),
            )
        )

    values = np.asarray(rows)
    result = {
        "split": args.split,
        "images": len(names),
        "method": (
            "Per-image least-squares fit LR ~= gain*bicubic(GT)+offset; residual "
            "includes noise and unmodeled resampling/blur differences."
        ),
        "quantiles": {
            name: summarize(values[:, index])
            for index, name in enumerate(
                ("gain", "offset", "residual_std", "lr_min", "lr_max", "outside_fraction")
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2)
    args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
