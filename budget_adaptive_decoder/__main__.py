"""
Budget-Adaptive Decoder - Main Entry Point

This module provides a simple interface to the training phases
and verification scripts.
"""

import torch
from pathlib import Path
from typing import Optional, Dict

from .models.decoder import BudgetAdaptiveDecoder
from .models.teacher import DCVCWrapper
from .models.policy import PolicyNetwork
from .models.extractor import BitstreamContentExtractor
from .models.scheduler import Scheduler

from .training.phase1_distill import train_phase1
from .training.phase2_precompute import precompute_phase2
from .training.phase3_policy import train_phase3
from .training.phase4_adapt import train_phase4

from .data.vimeo90k import Vimeo90kDataset, Vimeo90kCollator
from .data.augmentation import ReferenceFrameCorruption

from .evaluation.metrics import compute_psnr, compute_msssim, batch_compute_psnr


__version__ = "0.1.0"


def create_models(
    K: int = 4,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Create all models for the Budget-Adaptive Decoder.

    Args:
        K: Number of refinement stages (default 4)
        device: Device to load models on

    Returns:
        Dictionary of model instances
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = DCVCWrapper(load_pretrained=True).to(device)
    decoder = BudgetAdaptiveDecoder().to(device)
    extractor = BitstreamContentExtractor().to(device)
    policy = PolicyNetwork(K=K).to(device)
    scheduler = Scheduler(stage_costs=[1.0, 4.0, 2.0, 2.0, 1.0]).to(device)

    return {
        "teacher": teacher,
        "decoder": decoder,
        "extractor": extractor,
        "policy": policy,
        "scheduler": scheduler,
    }


def load_checkpoint(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on

    Returns:
        Dictionary with model state dicts and metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    return checkpoint


__all__ = [
    "BudgetAdaptiveDecoder",
    "DCVCWrapper",
    "PolicyNetwork",
    "BitstreamContentExtractor",
    "Scheduler",
    "train_phase1",
    "precompute_phase2",
    "train_phase3",
    "train_phase4",
    "Vimeo90kDataset",
    "Vimeo90kCollator",
    "ReferenceFrameCorruption",
    "compute_psnr",
    "compute_msssim",
    "batch_compute_psnr",
    "create_models",
    "load_checkpoint",
]


if __name__ == "__main__":
    print("Budget-Adaptive Decoder Package")
    print(f"Version: {__version__}")
    print("\nAvailable components:")
    for name in __all__:
        print(f"  - {name}")