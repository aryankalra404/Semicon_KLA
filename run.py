#!/usr/bin/env python3
"""Required KLA submission entry point.

Usage: python run.py <input-dir> <output-dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inference import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore all grayscale .npy files in an input directory."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing input .npy files")
    parser.add_argument("output_dir", type=Path, help="Directory to receive restored .npy files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
