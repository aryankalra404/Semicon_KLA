#!/usr/bin/env python3
"""Audit the repository's compact checkpoint and standalone inference contract."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from kla_restore.runtime import choose_device, load_model


EXPECTED_SHA256 = "c1e67ad4400b1c899ef30a2bb6748a086c036661fef41932fbef548e5998bacd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("weights/final.pt"))
    parser.add_argument("--sample-dir", type=Path, default=Path("data/test/NoisyLR"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    digest = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"Checkpoint SHA-256 mismatch: {digest}")
    device = choose_device(args.device)
    model = load_model(args.weights, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 580_609:
        raise SystemExit(f"Unexpected parameter count: {parameter_count}")

    source_paths = sorted(args.sample_dir.glob("*.npy"))[:2]
    if len(source_paths) != 2:
        raise SystemExit(f"Need at least two .npy samples in {args.sample_dir}")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        inputs, outputs = root / "inputs", root / "outputs"
        inputs.mkdir()
        expected_shapes = {}
        for source in source_paths:
            array = np.load(source, allow_pickle=False)
            expected_shapes[source.name] = (array.shape[0] * 2, array.shape[1] * 2)
            (inputs / source.name).write_bytes(source.read_bytes())
        # Exercise KLA's second stated resolution pair in the same invocation.
        large_name = "__contract_256.npy"
        np.save(
            inputs / large_name,
            np.zeros((256, 256), dtype=np.float32),
            allow_pickle=False,
        )
        expected_shapes[large_name] = (512, 512)
        subprocess.run(
            [
                sys.executable, "inference.py", "--input-dir", str(inputs),
                "--output-dir", str(outputs), "--weights", str(args.weights),
                "--device", str(device), "--batch-size", "2",
            ],
            check=True,
        )
        produced = sorted(outputs.glob("*.npy"))
        if [path.name for path in produced] != sorted(expected_shapes):
            raise SystemExit("Inference did not preserve input filenames")
        for path in produced:
            array = np.load(path, allow_pickle=False)
            if array.shape != expected_shapes[path.name] or array.dtype != np.float32:
                raise SystemExit(f"Invalid output {path}: {array.shape}/{array.dtype}")
            if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                raise SystemExit(f"Invalid values in {path}")
    print(
        f"OK: checkpoint={args.weights} sha256={digest} parameters={parameter_count} "
        f"device={device} inference_contract=passed"
    )


if __name__ == "__main__":
    main()
