"""Torch-native image quality metrics used by training and evaluation."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _as_batch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        image = image[None, None]
    elif image.ndim == 3:
        image = image[None]
    if image.ndim != 4:
        raise ValueError(f"Expected 2D, CHW, or NCHW image, got {image.shape}")
    return image


def psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    prediction, target = _as_batch(prediction), _as_batch(target)
    mse = (prediction - target).square().flatten(1).mean(1)
    peak = torch.tensor(data_range, device=mse.device, dtype=mse.dtype)
    return 20 * torch.log10(peak) - 10 * torch.log10(mse.clamp_min(1e-12))


def _gaussian_window(
    size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel)
    return window.expand(channels, 1, size, size).contiguous()


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    prediction, target = _as_batch(prediction), _as_batch(target)
    if prediction.shape != target.shape:
        raise ValueError(f"Metric shapes differ: {prediction.shape} vs {target.shape}")
    channels = prediction.shape[1]
    window = _gaussian_window(
        window_size, sigma, channels, prediction.device, prediction.dtype
    )
    padding = window_size // 2
    mu_x = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x = F.conv2d(prediction.square(), window, padding=padding, groups=channels) - mu_x2
    sigma_y = F.conv2d(target.square(), window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_xy
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return score.flatten(1).mean(1)


def mean_and_ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = float(tensor.mean())
    if len(values) == 1:
        return mean, 0.0
    ci95 = 1.96 * float(tensor.std(unbiased=True)) / math.sqrt(len(values))
    return mean, ci95

