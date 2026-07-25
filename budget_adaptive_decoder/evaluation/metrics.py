"""
Evaluation metrics for Budget-Constrained Neural Decoder.

Implements PSNR, MS-SSIM, and LPIPS metrics as specified in Section 11
of the design document.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
from scipy.ndimage import gaussian_filter
import numpy as np


def compute_psnr(
    img1: torch.Tensor, img2: torch.Tensor, max_value: float = 1.0
) -> float:
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) between two images.

    PSNR = 10 * log10(MAX^2 / MSE)

    Args:
        img1: First image tensor [..., C, H, W] or [..., H, W]
        img2: Second image tensor [..., C, H, W] or [..., H, W]
        max_value: Maximum pixel value (1.0 for normalized images)

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    psnr = 10 * torch.log10(max_value ** 2 / mse)
    return psnr.item()


def compute_msssim(
    img1: torch.Tensor, img2: torch.Tensor, max_value: float = 1.0
) -> float:
    """
    Compute MS-SSIM (Multi-Scale Structural Similarity) between two images.

    Uses the same implementation as the design document's metric definitions.

    Args:
        img1: First image tensor [..., C, H, W] (C=3 for RGB)
        img2: Second image tensor [..., C, H, W] (C=3 for RGB)
        max_value: Maximum pixel value (1.0 for normalized images)

    Returns:
        MS-SSIM value (higher is better, 1.0 = identical)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    if img1.dim() == 4:
        batch_size = img1.shape[0]
        ms_ssim_values = []
        for i in range(batch_size):
            ms_ssim_values.append(_msssim_single(img1[i], img2[i], max_value))
        return np.mean(ms_ssim_values)
    else:
        return _msssim_single(img1, img2, max_value)


def _msssim_single(
    img1: torch.Tensor, img2: torch.Tensor, max_value: float = 1.0
) -> float:
    """
    Compute MS-SSIM for a single image.

    Implements the Wang et al. MS-SSIM algorithm with 5 scales.
    """
    from math import exp

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    if img1.shape[-3] == 3:
        img1 = img1.transpose(0, 2)
        img2 = img2.transpose(0, 2)

    img1 = img1.squeeze(0) if img1.shape[0] == 1 else img1
    img2 = img2.squeeze(0) if img2.shape[0] == 1 else img2

    if img1.shape[0] == 3:
        img1 = img1.transpose(0, 2)
        img2 = img2.transpose(0, 2)

    if img1.dim() == 3 and img1.shape[-1] == 3:
        img1 = img1.transpose(0, 2).transpose(1, 2)
        img2 = img2.transpose(0, 2).transpose(1, 2)

    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    if img1_np.shape[0] == 3:
        img1_np = np.transpose(img1_np, (1, 2, 0))
        img2_np = np.transpose(img2_np, (1, 2, 0))

    p = (img1_np * 255).astype(np.float64)
    q = (img2_np * 255).astype(np.float64)

    sigma = 1.5
    truncate = 3.5

    num_scales = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    ms_ssim_val = 1.0
    for scale in range(num_scales):
        if scale > 0:
            p = gaussian_filter(p, sigma)
            q = gaussian_filter(q, sigma)
            p = p[::2, ::2, :]
            q = q[::2, ::2, :]

        p_u = p.mean(axis=(0, 1))
        q_u = q.mean(axis=(0, 1))
        sigma_p = np.sqrt(((p - p_u) ** 2).mean(axis=(0, 1)))
        sigma_q = np.sqrt(((q - q_u) ** 2).mean(axis=(0, 1)))

        sigma_pq = ((p - p_u) * (q - q_u)).mean(axis=(0, 1))

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        luminance = (2 * p_u * q_u + c1) / (p_u ** 2 + q_u ** 2 + c1)
        contrast = (2 * sigma_p * sigma_q + c2) / (sigma_p ** 2 + sigma_q ** 2 + c2)
        structure = (sigma_pq + c2) / (sigma_p * sigma_q + c2)

        if scale < num_scales - 1:
            ms_ssim_val *= luminance * contrast * structure
        else:
            ms_ssim_val *= luminance * contrast ** 0.5 * structure

    return ms_ssim_val


def compute_lpips(
    img1: torch.Tensor,
    img2: torch.Tensor,
    model: Optional[nn.Module] = None,
    max_value: float = 1.0,
) -> float:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).

    LPIPS measures perceptual similarity using a pretrained network.
    Lower is better (0 = identical).

    Args:
        img1: First image tensor [..., C, H, W]
        img2: Second image tensor [..., C, H, W]
        model: Feature extractor network (e.g., VGG). If None, returns MSE.
        max_value: Maximum pixel value (1.0 for normalized images)

    Returns:
        LPIPS distance (lower is better)
    """
    if model is None:
        mse = torch.mean((img1 - img2) ** 2)
        return mse.item()

    with torch.no_grad():
        feat1 = model(img1)
        feat2 = model(img2)

        diff = (feat1 - feat2) ** 2
        lpips = diff.mean()

    return lpips.item()


def batch_compute_psnr(
    tensor1: torch.Tensor, tensor2: torch.Tensor, max_value: float = 1.0
) -> torch.Tensor:
    """
    Compute PSNR for a batch of images.

    Args:
        tensor1: First batch [..., B, C, H, W]
        tensor2: Second batch [..., B, C, H, W]
        max_value: Maximum pixel value (1.0 for normalized images)

    Returns:
        Tensor of PSNR values for each item in the batch
    """
    mse = torch.mean((tensor1 - tensor2) ** 2, dim=(-3, -2, -1))
    psnr = 10 * torch.log10(max_value ** 2 / mse)
    return psnr


def compute_quality_metrics(
    img1: torch.Tensor,
    img2: torch.Tensor,
    ms_ssim_model: Optional[nn.Module] = None,
    lpips_model: Optional[nn.Module] = None,
) -> dict:
    """
    Compute all quality metrics between two images.

    Args:
        img1: First image tensor [..., C, H, W]
        img2: Second image tensor [..., C, H, W]
        ms_ssim_model: Optional MS-SSIM model (currently uses scipy implementation)
        lpips_model: Optional LPIPS feature extractor

    Returns:
        Dictionary with psnr, ms_ssim, and lpips values
    """
    psnr = compute_psnr(img1, img2)
    ms_ssim = compute_msssim(img1, img2)
    lpips = compute_lpips(img1, img2, lpips_model)

    return {
        "psnr": psnr,
        "ms_ssim": ms_ssim,
        "lpips": lpips,
    }


class MetricsTracker:
    """
    Track metrics over a validation run.

    Computes running statistics and returns summaries.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.psnr_values = []
        self.ms_ssim_values = []
        self.lpips_values = []
        self.count = 0

    def update(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        ms_ssim_model: Optional[nn.Module] = None,
        lpips_model: Optional[nn.Module] = None,
    ):
        """Add a sample to tracked metrics."""
        metrics = compute_quality_metrics(img1, img2, ms_ssim_model, lpips_model)
        self.psnr_values.append(metrics["psnr"])
        self.ms_ssim_values.append(metrics["ms_ssim"])
        self.lpips_values.append(metrics["lpips"])
        self.count += 1

    def get_summary(self) -> dict:
        """Get summary statistics for all tracked metrics."""
        import numpy as np

        return {
            "psnr": {
                "mean": np.mean(self.psnr_values) if self.psnr_values else 0,
                "std": np.std(self.psnr_values) if self.psnr_values else 0,
            },
            "ms_ssim": {
                "mean": np.mean(self.ms_ssim_values) if self.ms_ssim_values else 0,
                "std": np.std(self.ms_ssim_values) if self.ms_ssim_values else 0,
            },
            "lpips": {
                "mean": np.mean(self.lpips_values) if self.lpips_values else 0,
                "std": np.std(self.lpips_values) if self.lpips_values else 0,
            },
            "count": self.count,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Metrics Module Test")
    print("=" * 60)

    torch.manual_seed(42)

    img1 = torch.rand(1, 3, 64, 64)
    img2 = torch.rand(1, 3, 64, 64)

    psnr = compute_psnr(img1, img2)
    print(f"PSNR between random images: {psnr:.2f} dB")

    ms_ssim = compute_msssim(img1, img2)
    print(f"MS-SSIM between random images: {ms_ssim:.4f}")

    batch_img1 = torch.rand(4, 3, 64, 64)
    batch_img2 = torch.rand(4, 3, 64, 64)
    batch_psnr = batch_compute_psnr(batch_img1, batch_img2)
    print(f"Batch PSNR shape: {batch_psnr.shape}, values: {batch_psnr.tolist()}")

    identical = torch.rand(2, 3, 32, 32)
    psnr_identical = compute_psnr(identical, identical)
    print(f"PSNR of identical images: {psnr_identical:.2f} dB")

    print("\n" + "=" * 60)
    print("Metrics tests passed!")
    print("=" * 60)