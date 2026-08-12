#!/usr/bin/env python3
"""Export a compact inference-only checkpoint from a training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("weights/final.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    compact = {
        "model": checkpoint["model"],
        "model_config": checkpoint["model_config"],
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics"),
        "format": "kla-restorenet-inference-v1",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"output={args.output} bytes={args.output.stat().st_size} sha256={digest}")


if __name__ == "__main__":
    main()
