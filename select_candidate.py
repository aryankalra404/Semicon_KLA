#!/usr/bin/env python3
"""Rank KLA candidates against the frozen v2 control using explicit gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONTROL = {
    "name": "v2",
    "psnr": 26.296219098567963,
    "ssim": 0.7004095890559257,
    "lpips": 0.3737956412602216,
    "stress_psnr": 24.996988656620186,
    "stress_ssim": 0.6053215542517137,
    "latency_p50": 11.218712003028486,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=("seed3407", "seed8119", "width64", "pixelheavy"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/candidate_selection.json"))
    return parser.parse_args()


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    rows = []
    for name in args.candidates:
        metrics = read(args.experiments_dir / f"{name}.json")
        stress = read(args.experiments_dir / f"{name}_stress.json")
        benchmark = read(args.experiments_dir / f"{name}_benchmark.json")
        row = {
            "name": name,
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "lpips": metrics.get("lpips"),
            "stress_psnr": stress["macro_average"]["psnr"],
            "stress_ssim": stress["macro_average"]["ssim"],
            "latency_p50": benchmark["milliseconds_per_image"]["p50"],
            "parameters": benchmark.get(
                "parameters",
                benchmark.get("parameters_per_model", [None])[0],
            ),
        }
        row["delta"] = {
            key: row[key] - CONTROL[key]
            for key in ("psnr", "ssim", "stress_psnr", "stress_ssim", "latency_p50")
        }
        fidelity_win = row["psnr"] > CONTROL["psnr"] and row["ssim"] > CONTROL["ssim"]
        robustness_win = (
            row["stress_psnr"] > CONTROL["stress_psnr"]
            and row["stress_ssim"] > CONTROL["stress_ssim"]
            and row["psnr"] >= CONTROL["psnr"] - 0.10
            and row["ssim"] >= CONTROL["ssim"] - 0.002
        )
        latency_ok = row["latency_p50"] <= 15.0
        row["passes_gate"] = bool((fidelity_win or robustness_win) and latency_ok)
        row["gate_reason"] = {
            "fidelity_win": fidelity_win,
            "robustness_win": robustness_win,
            "latency_ok": latency_ok,
        }
        rows.append(row)

    eligible = [row for row in rows if row["passes_gate"]]
    selected = (
        max(eligible, key=lambda row: row["psnr"] + 10 * row["ssim"])["name"]
        if eligible
        else "v2"
    )
    result = {"control": CONTROL, "candidates": rows, "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
