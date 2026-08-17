#!/usr/bin/env python3
"""Standalone KLA evaluator entry point: input directory to restored arrays."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from kla_restore.data import UnpairedNpyDataset
from kla_restore.ensemble import restore
from kla_restore.runtime import choose_device, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        nargs="+",
        default=[Path("weights/final.pt")],
        help="One or more checkpoints; multiple models are output-averaged",
    )
    parser.add_argument(
        "--self-ensemble",
        choices=("x1", "x4", "x8"),
        default="x1",
        help="Invertible geometric test-time transforms",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def run_inference(
    input_dir: Path,
    output_dir: Path,
    *,
    weights: list[Path] | None = None,
    self_ensemble: str = "x1",
    batch_size: int = 8,
    workers: int = 0,
    device_name: str = "auto",
) -> None:
    """Restore every .npy file in *input_dir* and save it under the same name."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(device_name)
    script_root = Path(__file__).resolve().parent
    weight_paths = [
        path if path.is_absolute() else script_root / path
        for path in (weights or [Path("models/final.pt")])
    ]
    missing_weights = [path for path in weight_paths if not path.is_file()]
    if missing_weights:
        raise FileNotFoundError(f"Missing model checkpoint(s): {missing_weights}")
    models = [load_model(weight_path, device) for weight_path in weight_paths]
    dataset = UnpairedNpyDataset(input_dir)
    # KLA may provide both 128->256 and 256->512 samples. Group by input shape
    # so mixed-resolution directories remain batched without padding artifacts.
    shape_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, path in enumerate(dataset.paths):
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(
                f"Expected a grayscale (H, W) or (H, W, 1) array at {path}, "
                f"got {array.shape}"
            )
        shape_groups[tuple(array.shape)].append(index)
    elapsed = 0.0

    with torch.inference_mode():
        for indices in shape_groups.values():
            loader = DataLoader(
                Subset(dataset, indices),
                batch_size=batch_size,
                num_workers=workers,
            )
            for inputs, names in loader:
                inputs = inputs.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                outputs = torch.nan_to_num(
                    restore(models, inputs, self_ensemble), nan=0.0, posinf=1.0, neginf=0.0
                ).clamp(0.0, 1.0)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed += time.perf_counter() - started
                arrays = outputs[:, 0].float().cpu().numpy()
                for name, array in zip(names, arrays, strict=True):
                    np.save(
                        output_dir / name,
                        array.astype(np.float32),
                        allow_pickle=False,
                    )

    print(
        f"restored={len(dataset)} device={device} "
        f"models={len(models)} self_ensemble={self_ensemble} "
        f"milliseconds_per_image={1000 * elapsed / len(dataset):.3f} "
        f"output_dir={output_dir}"
    )


def main() -> None:
    args = parse_args()
    run_inference(
        args.input_dir,
        args.output_dir,
        weights=args.weights,
        self_ensemble=args.self_ensemble,
        batch_size=args.batch_size,
        workers=args.workers,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
