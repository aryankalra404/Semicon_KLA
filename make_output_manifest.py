#!/usr/bin/env python3
"""Create a reproducible manifest for submitted restored NumPy arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/restored"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/output_manifest.json")
    )
    parser.add_argument("--expected-count", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.output_dir.glob("*.npy"))
    if len(paths) != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count} outputs, found {len(paths)}")
    files = []
    aggregate = hashlib.sha256()
    for path in paths:
        array = np.load(path, allow_pickle=False)
        if array.shape != (256, 256) or array.dtype != np.float32:
            raise SystemExit(f"Invalid output {path}: {array.shape}/{array.dtype}")
        if not np.isfinite(array).all() or array.min() < 0.0 or array.max() > 1.0:
            raise SystemExit(f"Invalid values in {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        aggregate.update(f"{path.name} {digest}\n".encode())
        files.append(
            {
                "filename": path.name,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "min": float(array.min()),
                "max": float(array.max()),
            }
        )
    manifest = {
        "format": "KLA restored float32 NumPy arrays",
        "count": len(files),
        "shape": [256, 256],
        "dtype": "float32",
        "aggregate_sha256": aggregate.hexdigest(),
        "files": files,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"OK: count={len(files)} aggregate_sha256={manifest['aggregate_sha256']} "
        f"manifest={args.manifest}"
    )


if __name__ == "__main__":
    main()
