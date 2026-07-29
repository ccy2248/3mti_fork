"""Minimal SSIM for PyTorch tensors.

Adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim
"""

import torch
import torch.nn.functional as F


def _gaussian_kernel(size=11, sigma=1.5, channels=1):
    """Create a 2D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    kernel = kernel_2d[None, None, :, :].repeat(channels, 1, 1, 1)
    return kernel


def ssim(img1, img2, window_size=11, size_average=True):
    """Compute SSIM between two image tensors.

    Args:
        img1, img2: (B, C, H, W) tensors, range [0, 1].
        window_size: Gaussian window size.
        size_average: If True, return scalar mean; else per-sample.

    Returns:
        SSIM value (scalar or (B,) tensor).
    """
    C = img1.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5, C).to(img1.device).to(img1.dtype)

    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=C) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=[1, 2, 3])
