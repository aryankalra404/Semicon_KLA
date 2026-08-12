#!/usr/bin/env python3
"""Standalone KLA evaluator entry point: input directory to restored arrays."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kla_restore.data import UnpairedNpyDataset
from kla_restore.runtime import choose_device, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("weights/final.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model = load_model(args.weights, device)
    dataset = UnpairedNpyDataset(args.input_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    elapsed = 0.0

    with torch.inference_mode():
        for inputs, names in loader:
            inputs = inputs.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = model(inputs).clamp(0.0, 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            arrays = outputs[:, 0].float().cpu().numpy()
            for name, array in zip(names, arrays, strict=True):
                np.save(args.output_dir / name, array.astype(np.float32), allow_pickle=False)

    print(
        f"restored={len(dataset)} device={device} "
        f"milliseconds_per_image={1000 * elapsed / len(dataset):.3f} "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
