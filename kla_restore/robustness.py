"""Deterministic robustness utilities for KLA restoration experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .ensemble import _apply_transform, _invert_transform, transform_indices


def image_descriptor(image: np.ndarray) -> np.ndarray:
    """Return an interpretable appearance descriptor for a 2D grayscale image."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got {array.shape}")
    array = np.clip(array, 0.0, 1.0)
    intensity, _ = np.histogram(array, bins=16, range=(0.0, 1.0), density=False)
    gy, gx = np.gradient(array)
    magnitude = np.sqrt(gx * gx + gy * gy)
    gradients, _ = np.histogram(
        np.clip(magnitude, 0.0, 1.0), bins=12, range=(0.0, 1.0), density=False
    )

    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(array - array.mean()))))
    yy, xx = np.indices(array.shape)
    radius = np.sqrt((yy - (array.shape[0] - 1) / 2) ** 2 + (xx - (array.shape[1] - 1) / 2) ** 2)
    edges = np.linspace(0.0, radius.max() + 1e-6, 9)
    radial = np.array(
        [spectrum[(radius >= lo) & (radius < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:], strict=True)],
        dtype=np.float32,
    )

    height = array.shape[0] - array.shape[0] % 8
    width = array.shape[1] - array.shape[1] % 8
    cropped = array[:height, :width]
    coarse = cropped.reshape(8, height // 8, 8, width // 8).mean(axis=(1, 3))
    moments = np.array(
        [array.mean(), array.std(), np.quantile(array, 0.1), np.quantile(array, 0.5), np.quantile(array, 0.9)],
        dtype=np.float32,
    )
    descriptor = np.concatenate(
        (
            intensity.astype(np.float32) / max(1, array.size),
            gradients.astype(np.float32) / max(1, array.size),
            radial,
            coarse.ravel(),
            moments,
        )
    )
    if not np.isfinite(descriptor).all():
        raise ValueError("Descriptor contains non-finite values")
    return descriptor


def standardize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (features - mean) / scale, mean, scale


def deterministic_kmeans(
    features: np.ndarray, clusters: int, *, seed: int = 260813, iterations: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster standardized features with deterministic k-means++ initialization."""
    points = np.asarray(features, dtype=np.float64)
    if points.ndim != 2 or not 1 < clusters <= len(points):
        raise ValueError("clusters must be between 2 and the number of samples")
    generator = np.random.default_rng(seed)
    centers = [points[int(generator.integers(len(points)))]]
    closest = np.sum((points - centers[0]) ** 2, axis=1)
    for _ in range(1, clusters):
        total = closest.sum()
        index = int(generator.integers(len(points))) if total <= 0 else int(generator.choice(len(points), p=closest / total))
        centers.append(points[index])
        closest = np.minimum(closest, np.sum((points - centers[-1]) ** 2, axis=1))
    centers_array = np.asarray(centers)
    labels = np.full(len(points), -1, dtype=np.int64)
    for _ in range(iterations):
        distances = np.sum((points[:, None, :] - centers_array[None, :, :]) ** 2, axis=2)
        updated_labels = distances.argmin(axis=1)
        if np.array_equal(updated_labels, labels):
            break
        labels = updated_labels
        for cluster in range(clusters):
            members = points[labels == cluster]
            if len(members):
                centers_array[cluster] = members.mean(axis=0)
            else:
                farthest = int(np.argmax(distances.min(axis=1)))
                centers_array[cluster] = points[farthest]
    return labels, centers_array


def choose_holdout_cluster(labels: np.ndarray, target_size: int) -> int:
    """Choose the cluster closest to a requested validation size."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64))
    return min(range(len(counts)), key=lambda index: (abs(int(counts[index]) - target_size), index))


@torch.inference_mode()
def restoration_distribution(
    models: Sequence[torch.nn.Module], inputs: torch.Tensor, mode: str = "x8"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return geometric-ensemble mean and per-pixel standard deviation."""
    predictions = []
    for index in transform_indices(mode):
        transformed = _apply_transform(inputs, index)
        model_predictions = [
            _invert_transform(model(transformed), index) for model in models
        ]
        predictions.append(torch.stack(model_predictions).mean(dim=0))
    stacked = torch.stack(predictions)
    return stacked.mean(dim=0), stacked.std(dim=0, unbiased=False)


def binary_dilation(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return mask.bool()
    kernel = 2 * radius + 1
    return torch.nn.functional.max_pool2d(mask.float(), kernel, stride=1, padding=radius).bool()


def precision_recall(predicted: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    predicted, target = predicted.bool(), target.bool()
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1
