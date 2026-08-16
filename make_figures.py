#!/usr/bin/env python3
"""Generate reproducible, presentation-quality restoration evidence.

The original figure script selected the best/median/worst cases using model
SSIM alone.  That is useful for failure analysis, but it can choose a visually
uninformative smooth image as the "best" case and an over-smoothed texture as
the "median" case.  This script instead evaluates model *and bicubic* on every
validation pair, measures edge fidelity and scene information, and applies a
documented deterministic rule to select a representative improvement.

It deliberately writes the known over-smoothing case as a separate limitation
figure; the presentation visual is not allowed to hide that failure mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from kla_restore.data import PairedNpyDataset, load_npy_tensor, names_for_split
from kla_restore.metrics import psnr, ssim
from kla_restore.runtime import choose_device, load_model


@dataclass(frozen=True)
class CaseMetrics:
    filename: str
    model_psnr: float
    model_ssim: float
    bicubic_psnr: float
    bicubic_ssim: float
    classical_psnr: float
    classical_ssim: float
    psnr_gain: float
    ssim_gain: float
    gt_std: float
    gt_edge_energy: float
    gt_coherent_edge_energy: float
    gt_directional_coherence: float
    model_edge_error: float
    bicubic_edge_error: float
    edge_error_reduction: float
    selection_score: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("weights/final.pt"))
    parser.add_argument("--history", type=Path, default=Path("weights/v2/history.json"))
    parser.add_argument("--lr-dir", type=Path, default=Path("data/train/NoisyLR"))
    parser.add_argument("--gt-dir", type=Path, default=Path("data/train/GT"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--limitation-case",
        default="002994.npy",
        help="Known texture case retained as an honest over-smoothing example",
    )
    return parser.parse_args()


def image_edge_energy(images: torch.Tensor) -> torch.Tensor:
    """Mean absolute first derivative per image."""
    dx = images[..., :, 1:] - images[..., :, :-1]
    dy = images[..., 1:, :] - images[..., :-1, :]
    return 0.5 * (
        dx.abs().flatten(1).mean(1) + dy.abs().flatten(1).mean(1)
    )


def image_edge_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute gradient mismatch per image (lower is better)."""
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * (
        (pred_x - target_x).abs().flatten(1).mean(1)
        + (pred_y - target_y).abs().flatten(1).mean(1)
    )


def coherent_edge_energy(images: torch.Tensor) -> torch.Tensor:
    """Edge energy after low-pass filtering, suppressing stochastic texture."""
    filtered = F.avg_pool2d(images, kernel_size=5, stride=1, padding=2)
    return image_edge_energy(filtered)


def gaussian_denoise(images: torch.Tensor, sigma: float = 0.8) -> torch.Tensor:
    """Fixed classical denoiser used before bicubic 2x interpolation.

    Sigma 0.8 is selected once on the validation aggregate from a small,
    documented fixed grid; it is not adapted per image.
    """
    size = 5
    coordinates = torch.arange(size, dtype=images.dtype, device=images.device)
    coordinates = coordinates - (size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, size, size)
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(padded, kernel)


def directional_structure_coherence(images: torch.Tensor) -> torch.Tensor:
    """Return energy-weighted structure-tensor coherence per image.

    Random or isotropic texture can have high raw edge energy while containing
    little stable geometry.  Coherence rewards persistent lines, corners and
    boundaries—the structures that make a restoration visually interpretable.
    """
    filtered = F.avg_pool2d(images, kernel_size=5, stride=1, padding=2)
    dx = filtered[..., :, 1:] - filtered[..., :, :-1]
    dy = filtered[..., 1:, :] - filtered[..., :-1, :]
    # Bring both derivatives to the shared interior support.
    dx = dx[..., :-1, :]
    dy = dy[..., :, :-1]
    jxx = F.avg_pool2d(dx.square(), kernel_size=5, stride=1, padding=2)
    jyy = F.avg_pool2d(dy.square(), kernel_size=5, stride=1, padding=2)
    jxy = F.avg_pool2d(dx * dy, kernel_size=5, stride=1, padding=2)
    energy = jxx + jyy
    coherence = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square()) / (
        energy + 1e-9
    )
    return (coherence * energy).flatten(1).sum(1) / (
        energy.flatten(1).sum(1) + 1e-9
    )


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def select_representative_case(rows: list[CaseMetrics]) -> CaseMetrics:
    """Choose a structured, honest case where the model improves on bicubic.

    Selection is deterministic and uses only validation targets/predictions:

    1. require GT contrast and raw edge energy above the 35th percentile, and
       both coherent edge energy and directional coherence above the 60th;
    2. require positive PSNR and SSIM gains over both bicubic and the fixed
       Gaussian-denoise + bicubic classical pipeline, plus improved gradients;
    3. reject the top 5% model PSNR to avoid a cherry-picked easiest example;
    4. rank remaining cases by robust percentile ranks of PSNR gain (20%),
       SSIM gain (25%), edge-error reduction (10%), coherent edges (20%), and
       directional structure (25%).
    """
    if not rows:
        raise ValueError("Cannot select a presentation case from an empty list")
    contrast_floor = _percentile([row.gt_std for row in rows], 35)
    raw_edge_floor = _percentile([row.gt_edge_energy for row in rows], 35)
    edge_floor = _percentile([row.gt_coherent_edge_energy for row in rows], 60)
    direction_floor = _percentile([row.gt_directional_coherence for row in rows], 60)
    easy_ceiling = _percentile([row.model_psnr for row in rows], 95)
    eligible = [
        row
        for row in rows
        if row.gt_std >= contrast_floor
        and row.gt_edge_energy >= raw_edge_floor
        and row.gt_coherent_edge_energy >= edge_floor
        and row.gt_directional_coherence >= direction_floor
        and row.model_psnr <= easy_ceiling
        and row.psnr_gain > 0.0
        and row.ssim_gain > 0.0
        and row.model_psnr > row.classical_psnr
        and row.model_ssim > row.classical_ssim
        and row.edge_error_reduction > 0.0
    ]
    # The fallback remains deterministic and still requires measurable benefit.
    if not eligible:
        eligible = [
            row
            for row in rows
            if row.psnr_gain > 0.0
            and row.ssim_gain > 0.0
            and row.model_psnr > row.classical_psnr
            and row.model_ssim > row.classical_ssim
        ]
    if not eligible:
        raise ValueError("No validation case improves on bicubic in PSNR and SSIM")

    def percentile_rank(value: float, values: np.ndarray) -> float:
        return float(np.mean(values <= value))

    psnr_gains = np.asarray([row.psnr_gain for row in eligible])
    ssim_gains = np.asarray([row.ssim_gain for row in eligible])
    edge_gains = np.asarray([row.edge_error_reduction for row in eligible])
    edge_energy = np.asarray([row.gt_coherent_edge_energy for row in eligible])
    direction = np.asarray([row.gt_directional_coherence for row in eligible])
    scored = []
    for row in eligible:
        score = (
            0.20 * percentile_rank(row.psnr_gain, psnr_gains)
            + 0.25 * percentile_rank(row.ssim_gain, ssim_gains)
            + 0.10 * percentile_rank(row.edge_error_reduction, edge_gains)
            + 0.20 * percentile_rank(row.gt_coherent_edge_energy, edge_energy)
            + 0.25 * percentile_rank(row.gt_directional_coherence, direction)
        )
        scored.append(CaseMetrics(**{**asdict(row), "selection_score": score}))
    return max(scored, key=lambda row: (row.selection_score, row.filename))


def best_detail_crop(
    gt: np.ndarray,
    model: np.ndarray,
    bicubic: np.ndarray,
    crop_size: int = 96,
) -> tuple[int, int, int, int]:
    """Find a detailed crop where model gradient error improves over bicubic."""
    height, width = gt.shape
    crop_size = min(crop_size, height, width)
    grad_y, grad_x = np.gradient(gt)
    detail = np.abs(grad_x) + np.abs(grad_y)
    model_y, model_x = np.gradient(model)
    bic_y, bic_x = np.gradient(bicubic)
    model_error = np.abs(model_x - grad_x) + np.abs(model_y - grad_y)
    bic_error = np.abs(bic_x - grad_x) + np.abs(bic_y - grad_y)
    evidence = detail + 0.75 * np.maximum(bic_error - model_error, 0.0)

    stride = max(8, crop_size // 4)
    best = (0, 0, crop_size, crop_size)
    best_score = -np.inf
    for top in range(0, height - crop_size + 1, stride):
        for left in range(0, width - crop_size + 1, stride):
            score = float(evidence[top : top + crop_size, left : left + crop_size].mean())
            if score > best_score:
                best_score = score
                best = (left, top, left + crop_size, top + crop_size)
    return best


def save_learning_curve(history_path: Path, output: Path) -> None:
    if not history_path.is_file():
        return
    history = json.loads(history_path.read_text())
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot(epochs, [row["val_psnr"] for row in history], color="#2563eb", linewidth=2)
    axes[0].set(title="Validation PSNR", xlabel="Epoch", ylabel="PSNR (dB)")
    axes[1].plot(epochs, [row["val_ssim"] for row in history], color="#7c3aed", linewidth=2)
    axes[1].set(title="Validation SSIM", xlabel="Epoch", ylabel="SSIM")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def evaluate_cases(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[CaseMetrics]:
    rows: list[CaseMetrics] = []
    for lr, gt, names in loader:
        lr, gt = lr.to(device), gt.to(device)
        bicubic = F.interpolate(
            lr, scale_factor=2, mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
        classical = F.interpolate(
            gaussian_denoise(lr),
            scale_factor=2,
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
        restored = model(lr).clamp(0.0, 1.0)
        model_psnr = psnr(restored, gt).cpu().tolist()
        model_ssim = ssim(restored, gt).cpu().tolist()
        bicubic_psnr = psnr(bicubic, gt).cpu().tolist()
        bicubic_ssim = ssim(bicubic, gt).cpu().tolist()
        classical_psnr = psnr(classical, gt).cpu().tolist()
        classical_ssim = ssim(classical, gt).cpu().tolist()
        gt_std = gt.flatten(1).std(dim=1, unbiased=False).cpu().tolist()
        gt_edges = image_edge_energy(gt).cpu().tolist()
        coherent_edges = coherent_edge_energy(gt).cpu().tolist()
        directional_coherence = directional_structure_coherence(gt).cpu().tolist()
        model_errors = image_edge_error(restored, gt).cpu().tolist()
        bicubic_errors = image_edge_error(bicubic, gt).cpu().tolist()
        for values in zip(
            names,
            model_psnr,
            model_ssim,
            bicubic_psnr,
            bicubic_ssim,
            classical_psnr,
            classical_ssim,
            gt_std,
            gt_edges,
            coherent_edges,
            directional_coherence,
            model_errors,
            bicubic_errors,
            strict=True,
        ):
            name, mp, ms, bp, bs, cp, cs, contrast, edge, coherent, direction, me, be = values
            rows.append(
                CaseMetrics(
                    filename=name,
                    model_psnr=mp,
                    model_ssim=ms,
                    bicubic_psnr=bp,
                    bicubic_ssim=bs,
                    classical_psnr=cp,
                    classical_ssim=cs,
                    psnr_gain=mp - bp,
                    ssim_gain=ms - bs,
                    gt_std=contrast,
                    gt_edge_energy=edge,
                    gt_coherent_edge_energy=coherent,
                    gt_directional_coherence=direction,
                    model_edge_error=me,
                    bicubic_edge_error=be,
                    edge_error_reduction=be - me,
                )
            )
    return rows


@torch.inference_mode()
def restore_case(
    name: str,
    lr_dir: Path,
    gt_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, np.ndarray]:
    lr = load_npy_tensor(lr_dir / name).unsqueeze(0).to(device)
    gt = load_npy_tensor(gt_dir / name).unsqueeze(0).to(device)
    bicubic = F.interpolate(
        lr, scale_factor=2, mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)
    classical = F.interpolate(
        gaussian_denoise(lr),
        scale_factor=2,
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)
    restored = model(lr).clamp(0.0, 1.0)
    return {
        "input": lr[0, 0].cpu().numpy(),
        "bicubic": bicubic[0, 0].cpu().numpy(),
        "classical": classical[0, 0].cpu().numpy(),
        "model": restored[0, 0].cpu().numpy(),
        "gt": gt[0, 0].cpu().numpy(),
    }


def plot_evidence_case(
    arrays: dict[str, np.ndarray],
    metrics: CaseMetrics,
    output: Path,
    *,
    limitation: bool = False,
) -> None:
    left, top, right, bottom = best_detail_crop(
        arrays["gt"], arrays["model"], arrays["bicubic"]
    )
    panels = (
        ("Degraded input", arrays["input"], None),
        (
            f"Bicubic\n{metrics.bicubic_psnr:.2f} dB  |  {metrics.bicubic_ssim:.3f} SSIM",
            arrays["bicubic"],
            "bicubic",
        ),
        (
            f"Gaussian denoise + bicubic\n{metrics.classical_psnr:.2f} dB  |  {metrics.classical_ssim:.3f} SSIM",
            arrays["classical"],
            "classical",
        ),
        (f"Our Model\n{metrics.model_psnr:.2f} dB  |  {metrics.model_ssim:.3f} SSIM", arrays["model"], "model"),
        ("Ground truth", arrays["gt"], "gt"),
    )
    fig, axes = plt.subplots(2, 5, figsize=(16.2, 6.7))
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.055, top=0.82, wspace=0.14, hspace=0.14)
    for index, (title, image, key) in enumerate(panels):
        axes[0, index].imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[0, index].set_title(title, fontsize=10, fontweight="bold" if key == "model" else None)
        axes[0, index].axis("off")
        if key is not None:
            crop = image[top:bottom, left:right]
            axes[1, index].imshow(crop, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[1, index].set_title("Detail crop", fontsize=9)
        else:
            # Upscale the same physical field for the LR input crop.
            scale_x = image.shape[1] / arrays["gt"].shape[1]
            scale_y = image.shape[0] / arrays["gt"].shape[0]
            lr_crop = image[
                round(top * scale_y) : round(bottom * scale_y),
                round(left * scale_x) : round(right * scale_x),
            ]
            axes[1, index].imshow(lr_crop, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[1, index].set_title("Same field (LR)", fontsize=9)
        axes[1, index].axis("off")

    if limitation:
        headline = "Known limitation — stochastic fine textures can be over-smoothed"
        subtitle = (
            "The model removes noise but can suppress thin, unpredictable detail; "
            "retained as an explicit failure case."
        )
        color = "#9b1c1c"
    else:
        headline = "Representative validation improvement (deterministic selection)"
        subtitle = (
            f"{metrics.filename}  •  +{metrics.psnr_gain:.2f} dB PSNR  •  "
            f"+{metrics.ssim_gain:.3f} SSIM vs bicubic  •  improved gradient fidelity"
        )
        color = "#12355b"
    fig.suptitle(headline, y=0.965, fontsize=16, fontweight="bold", color=color)
    fig.text(0.5, 0.895, subtitle, ha="center", va="top", fontsize=10, color="#333333")
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_slide_case(
    arrays: dict[str, np.ndarray], metrics: CaseMetrics, output: Path
) -> None:
    """Render a compact one-row comparison that remains legible on a slide."""
    panels = (
        ("Degraded input", arrays["input"]),
        (
            f"Bicubic\n{metrics.bicubic_psnr:.2f} dB  |  {metrics.bicubic_ssim:.3f} SSIM",
            arrays["bicubic"],
        ),
        (
            f"Gaussian + bicubic\n{metrics.classical_psnr:.2f} dB  |  {metrics.classical_ssim:.3f} SSIM",
            arrays["classical"],
        ),
        (f"Our Model\n{metrics.model_psnr:.2f} dB  |  {metrics.model_ssim:.3f} SSIM", arrays["model"]),
        ("Ground truth", arrays["gt"]),
    )
    fig, axes = plt.subplots(1, 5, figsize=(16.2, 3.45))
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.04, top=0.78, wspace=0.10)
    for index, (title, image) in enumerate(panels):
        axes[index].imshow(
            image, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
        )
        axes[index].set_title(
            title,
            fontsize=11,
            fontweight="bold" if index == 3 else None,
            pad=6,
        )
        axes[index].axis("off")
    fig.suptitle(
        "Representative Validation Comparison",
        y=0.97,
        fontsize=15,
        fontweight="bold",
        color="#12355b",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_case_csv(rows: list[CaseMetrics], output: Path) -> None:
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def mean_ci95(values: list[float]) -> dict[str, float]:
    """Return the sample mean and normal-approximation 95% CI half-width."""
    mean = statistics.fmean(values)
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "ci95_half_width": half_width}


def aggregate_validation(rows: list[CaseMetrics]) -> dict[str, object]:
    """Summarize all validation images, including paired win counts."""
    model_psnr = [row.model_psnr for row in rows]
    model_ssim = [row.model_ssim for row in rows]
    bicubic_psnr = [row.bicubic_psnr for row in rows]
    bicubic_ssim = [row.bicubic_ssim for row in rows]
    classical_psnr = [row.classical_psnr for row in rows]
    classical_ssim = [row.classical_ssim for row in rows]

    def paired(left: list[float], right: list[float]) -> dict[str, object]:
        differences = [a - b for a, b in zip(left, right, strict=True)]
        return {
            "delta": mean_ci95(differences),
            "wins": sum(a > b for a, b in zip(left, right, strict=True)),
            "losses": sum(a < b for a, b in zip(left, right, strict=True)),
        }

    return {
        "images": len(rows),
        "bicubic": {"psnr": mean_ci95(bicubic_psnr), "ssim": mean_ci95(bicubic_ssim)},
        "classical": {"psnr": mean_ci95(classical_psnr), "ssim": mean_ci95(classical_ssim)},
        "model": {"psnr": mean_ci95(model_psnr), "ssim": mean_ci95(model_ssim)},
        "model_vs_bicubic": {
            "psnr": paired(model_psnr, bicubic_psnr),
            "ssim": paired(model_ssim, bicubic_ssim),
        },
        "model_vs_classical": {
            "psnr": paired(model_psnr, classical_psnr),
            "ssim": paired(model_ssim, classical_ssim),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_learning_curve(args.history, args.output_dir / "v2_learning_curves.png")

    device = choose_device(args.device)
    model = load_model(args.weights, device)
    dataset = PairedNpyDataset(
        args.lr_dir, args.gt_dir, names_for_split("val"), audit=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    rows = evaluate_cases(model, loader, device)
    representative = select_representative_case(rows)
    by_name = {row.filename: row for row in rows}
    if args.limitation_case not in by_name:
        raise ValueError(f"Limitation case is outside validation: {args.limitation_case}")
    limitation = by_name[args.limitation_case]

    representative_arrays = restore_case(
        representative.filename, args.lr_dir, args.gt_dir, model, device
    )
    plot_slide_case(
        representative_arrays,
        representative,
        args.output_dir / "presentation_representative.png",
    )
    plot_evidence_case(
        representative_arrays,
        representative,
        args.output_dir / "presentation_representative_detailed.png",
    )
    plot_evidence_case(
        restore_case(limitation.filename, args.lr_dir, args.gt_dir, model, device),
        limitation,
        args.output_dir / "limitation_oversmoothing_002994.png",
        limitation=True,
    )
    write_case_csv(rows, args.output_dir / "validation_case_metrics.csv")
    manifest = {
        "selection_policy": {
            "eligibility": (
                "GT contrast and raw edge energy >= 35th percentile; coherent edge "
                "energy and directional coherence >= 60th percentile; positive PSNR "
                "and SSIM gains over bicubic and Gaussian-denoise + bicubic; positive "
                "edge-fidelity gain; model PSNR below 95th percentile"
            ),
            "ranking": (
                "20% PSNR-gain rank + 25% SSIM-gain rank + 10% edge-error-reduction "
                "rank + 20% coherent-edge-energy rank + 25% directional-coherence rank"
            ),
            "purpose": "representative presentation evidence, not aggregate evaluation",
        },
        "representative": asdict(representative),
        "known_limitation": asdict(limitation),
        "classical_baseline": {
            "method": "5x5 Gaussian denoise (sigma=0.8) followed by bicubic 2x",
            "selection": "best aggregate PSNR/SSIM among fixed classical pilot variants",
            "validation_psnr": float(np.mean([row.classical_psnr for row in rows])),
            "validation_ssim": float(np.mean([row.classical_ssim for row in rows])),
        },
        "aggregate_validation": aggregate_validation(rows),
        "validation_images_scored": len(rows),
    }
    (args.output_dir / "presentation_cases.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    print(f"wrote presentation evidence to {args.output_dir}")


if __name__ == "__main__":
    main()
