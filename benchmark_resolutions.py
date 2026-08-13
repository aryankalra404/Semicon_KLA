#!/usr/bin/env python3
"""Benchmark both official 2x spatial contracts, including mixed-size folders."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from kla_restore.runtime import choose_device, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("weights/final.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values), q))


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model = load_model(args.weights, device)
    results = {}
    for size in (128, 256):
        inputs = torch.rand(1, 1, size, size, device=device) * 1.4 - 0.2
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(inputs)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(args.runs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                output = model(inputs).clamp(0.0, 1.0)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append(1000 * (time.perf_counter() - started))
        expected = (1, 1, 2 * size, 2 * size)
        if tuple(output.shape) != expected or not torch.isfinite(output).all():
            raise RuntimeError(f"Invalid output for {size}: {tuple(output.shape)}")
        results[f"{size}x{size}_to_{2*size}x{2*size}"] = {
            "input_shape": list(inputs.shape),
            "output_shape": list(output.shape),
            "p50_ms": percentile(timings, 0.50),
            "p95_ms": percentile(timings, 0.95),
            "peak_vram_mib": (
                torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
            ),
            "finite": True,
            "output_range": [float(output.min()), float(output.max())],
        }
    result = {
        "weights": str(args.weights),
        "device": str(device),
        "batch_size": 1,
        "runs": args.runs,
        "results": results,
        "quality_scope": "Only 128-to-256 has official paired validation metrics.",
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
