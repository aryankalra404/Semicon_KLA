#!/usr/bin/env python3
"""Paired statistical comparison for KLA per-image evaluation CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METRICS = {"psnr": 1.0, "ssim": 1.0, "lpips": -1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bootstrap-runs", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=260813)
    parser.add_argument(
        "--candidate-name",
        default="candidate",
        help="Label used for the experimental model in comparison keys",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        name = row["filename"]
        if name in result:
            raise ValueError(f"Duplicate filename {name} in {path}")
        result[name] = {metric: float(row[metric]) for metric in METRICS}
    return result


def exact_sign_pvalue(wins: int, losses: int) -> float:
    """Two-sided exact binomial sign-test p-value; ties are excluded."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, index) for index in range(0, min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def paired_summary(
    left: np.ndarray,
    right: np.ndarray,
    direction: float,
    bootstrap_runs: int,
    rng: np.random.Generator,
) -> dict[str, float | int | list[float]]:
    raw_delta = right - left
    utility_delta = direction * raw_delta
    indices = rng.integers(0, len(raw_delta), size=(bootstrap_runs, len(raw_delta)))
    boot = raw_delta[indices].mean(axis=1)
    wins = int(np.count_nonzero(utility_delta > 1e-12))
    losses = int(np.count_nonzero(utility_delta < -1e-12))
    ties = int(len(raw_delta) - wins - losses)
    return {
        "left_mean": float(left.mean()),
        "right_mean": float(right.mean()),
        "delta_right_minus_left": float(raw_delta.mean()),
        "delta_ci95": [float(x) for x in np.quantile(boot, (0.025, 0.975))],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_test_pvalue": exact_sign_pvalue(wins, losses),
    }


def compare(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    bootstrap_runs: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    if set(left) != set(right):
        raise ValueError("Paired CSV files contain different filenames")
    names = sorted(left)
    return {
        metric: paired_summary(
            np.asarray([left[name][metric] for name in names]),
            np.asarray([right[name][metric] for name in names]),
            direction,
            bootstrap_runs,
            rng,
        )
        for metric, direction in METRICS.items()
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_runs < 1:
        raise SystemExit("--bootstrap-runs must be positive")
    baseline = read_rows(args.baseline)
    control = read_rows(args.control)
    candidate = read_rows(args.candidate)
    candidate_name = args.candidate_name.strip().replace(" ", "_")
    if not candidate_name:
        raise SystemExit("--candidate-name must not be empty")
    rng = np.random.default_rng(args.seed)
    result = {
        "images": len(baseline),
        "seed": args.seed,
        "bootstrap_runs": args.bootstrap_runs,
        "comparisons": {
            "control_vs_frozen_v2": compare(
                baseline, control, args.bootstrap_runs, rng
            ),
            f"{candidate_name}_vs_frozen_v2": compare(
                baseline, candidate, args.bootstrap_runs, rng
            ),
            f"{candidate_name}_vs_matched_v2_control": compare(
                control, candidate, args.bootstrap_runs, rng
            ),
        },
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
