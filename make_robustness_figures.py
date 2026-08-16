#!/usr/bin/env python3
"""Render submission-ready figures from robustness evidence without changing scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/robustness"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/test/NoisyLR"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/robustness"))
    return parser.parse_args()


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required to render robustness figures") from error
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed = read(args.results_dir / "ood_fixed.json")
    randomized = read(args.results_dir / "ood_randomized.json")
    labels = ["Fixed order", "Randomized order"]
    psnr_values = [fixed["psnr"], randomized["psnr"]]
    ssim_values = [fixed["ssim"], randomized["ssim"]]
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    axes[0].bar(labels, psnr_values, color=("#6b7280", "#6d28d9"))
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Cluster-disjoint OOD proxy")
    axes[1].bar(labels, ssim_values, color=("#6b7280", "#6d28d9"))
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("Matched policy ablation")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", rotation=12)
    figure.tight_layout()
    figure.savefig(args.output_dir / "ood_policy_ablation.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    bicubic = read(args.results_dir / "bicubic_defects.json")
    model = read(args.results_dir / "final_defects.json")
    defects = list(model["by_defect"])
    positions = np.arange(len(defects))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9.2, 3.8))
    axis.bar(
        positions - width / 2,
        [bicubic["by_defect"][name]["f1"] for name in defects],
        width,
        label="Bicubic",
        color="#9ca3af",
    )
    axis.bar(
        positions + width / 2,
        [model["by_defect"][name]["f1"] for name in defects],
        width,
        label="Our Model",
        color="#6d28d9",
    )
    axis.set_xticks(positions, [name.replace("_", " ").title() for name in defects])
    axis.set_ylabel("Localized response F1")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Synthetic defect-preservation probes (both degradation orders)")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(args.output_dir / "defect_preservation.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    uncertainty = read(args.results_dir / "uncertainty_summary.json")
    top = uncertainty["highest_uncertainty"][0]["filename"]
    degraded = np.load(args.input_dir / top, allow_pickle=False)
    restored = np.load(args.results_dir / "restored_x8" / top, allow_pickle=False)
    heatmap = np.load(args.results_dir / "uncertainty" / top, allow_pickle=False)
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    axes[0].imshow(degraded, cmap="gray")
    axes[0].set_title(f"Degraded input\n{degraded.shape[0]}×{degraded.shape[1]}")
    axes[1].imshow(restored, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Restored\n{restored.shape[0]}×{restored.shape[1]}")
    image = axes[2].imshow(heatmap, cmap="magma")
    axes[2].set_title("Geometric disagreement\n(higher = inspect)")
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.axis("off")
    figure.suptitle("Optional reliability view—not a calibrated probability", fontsize=11)
    figure.tight_layout()
    figure.savefig(args.output_dir / "uncertainty_example.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"rendered=3 output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
