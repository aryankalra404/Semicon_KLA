#!/usr/bin/env python3
"""Measure warmed inference latency and peak CUDA memory."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from kla_restore.data import load_npy_tensor
from kla_restore.ensemble import restore
from kla_restore.runtime import choose_device, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=Path, nargs="+", default=[Path("weights/final.pt")]
    )
    parser.add_argument(
        "--self-ensemble", choices=("x1", "x4", "x8"), default="x1"
    )
    parser.add_argument("--sample", type=Path, default=Path("data/test/NoisyLR/000000.npy"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.warmup, args.runs) < 1:
        raise SystemExit("batch-size, warmup, and runs must be positive")
    device = choose_device(args.device)
    models = [load_model(weights, device) for weights in args.weights]
    sample = load_npy_tensor(args.sample).unsqueeze(0).to(device)
    inputs = sample.repeat(args.batch_size, 1, 1, 1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            restore(models, inputs, args.self_ensemble)
        synchronize(device)
        timings = []
        for _ in range(args.runs):
            started = time.perf_counter()
            restore(models, inputs, args.self_ensemble)
            synchronize(device)
            timings.append(1000 * (time.perf_counter() - started) / args.batch_size)

    result = {
        "weights": [str(path) for path in args.weights],
        "self_ensemble": args.self_ensemble,
        "device": str(device),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "runs": args.runs,
        "milliseconds_per_image": {
            "mean": float(np.mean(timings)),
            "p50": float(np.quantile(timings, 0.50)),
            "p95": float(np.quantile(timings, 0.95)),
        },
        "models": len(models),
        "parameters_per_model": [
            sum(parameter.numel() for parameter in model.parameters())
            for model in models
        ],
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else None
        ),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
