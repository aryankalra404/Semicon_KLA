#!/usr/bin/env python3
"""Fail-fast integrity audit for the complete official KLA release."""

from pathlib import Path

import numpy as np


EXPECTED = {
    Path("data/train/GT"): ((256, 256), 3200),
    Path("data/train/NoisyLR"): ((128, 128), 3200),
    Path("data/test/NoisyLR"): ((128, 128), 400),
}


def main() -> None:
    problems: list[str] = []
    checked = 0
    for folder, (shape, count) in EXPECTED.items():
        paths = sorted(folder.glob("*.npy"))
        if len(paths) != count:
            problems.append(f"{folder}: expected {count} files, found {len(paths)}")
        for path in paths:
            checked += 1
            try:
                array = np.load(path, allow_pickle=False)
                if array.shape != shape or array.dtype != np.float32:
                    problems.append(
                        f"{path}: expected {shape}/float32, got {array.shape}/{array.dtype}"
                    )
                elif not np.isfinite(array).all():
                    problems.append(f"{path}: contains NaN or infinity")
            except Exception as error:
                problems.append(f"{path}: {type(error).__name__}: {error}")
    if problems:
        print("\n".join(problems))
        raise SystemExit(f"FAILED: {len(problems)} problem(s) across {checked} files")
    print(f"OK: {checked} arrays passed shape, dtype, readability, and finiteness checks")


if __name__ == "__main__":
    main()
