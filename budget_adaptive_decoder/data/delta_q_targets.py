"""
Delta Q Targets Module for Budget-Constrained Neural Decoder.

Handles storage and loading of precomputed ΔQ targets from Phase 2
as specified in Section 8, Phase 2.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class DeltaQTargetStorage:
    """
    Storage and retrieval for precomputed ΔQ targets.

    Phase 2 computes:
        ΔQ(k) = PSNR(k) - PSNR(k-1)

    for each frame at each depth k=1,...,K.

    These targets are used for Phase 3 regression training.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Args:
            cache_dir: Directory for delta_q cache
        """
        self.cache_dir = cache_dir or Path("./delta_q_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_targets(
        self,
        targets: Dict[str, torch.Tensor],
        metadata: Dict,
        filename: str = "delta_q_targets.pt",
    ):
        """
        Save ΔQ targets to disk.

        Args:
            targets: Dictionary with keys 'delta_q' and 'frame_ids'
            metadata: Metadata about the computation (dataset, model, etc.)
            filename: Output filename
        """
        output_path = self.cache_dir / filename

        save_dict = {
            "delta_q": targets["delta_q"],
            "frame_ids": targets.get("frame_ids", []),
            "metadata": metadata,
        }

        torch.save(save_dict, output_path)
        print(f"Saved ΔQ targets to {output_path}")

    def load_targets(
        self,
        filename: str = "delta_q_targets.pt",
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Load ΔQ targets from disk.

        Returns:
            Tuple of (targets_dict, metadata_dict)
        """
        input_path = self.cache_dir / filename

        if not input_path.exists():
            raise FileNotFoundError(f"ΔQ targets not found at {input_path}")

        loaded = torch.load(input_path)

        targets = {
            "delta_q": loaded["delta_q"],
            "frame_ids": loaded.get("frame_ids", []),
        }

        metadata = loaded.get("metadata", {})

        return targets, metadata

    def targets_exist(self, filename: str = "delta_q_targets.pt") -> bool:
        """Check if targets exist on disk."""
        return (self.cache_dir / filename).exists()

    def get_target_statistics(self, delta_q: torch.Tensor) -> Dict:
        """
        Compute statistics over ΔQ targets.

        Useful for verifying Phase 2 output before Phase 3.
        """
        stats = {
            "mean": delta_q.mean(dim=0).tolist(),
            "std": delta_q.std(dim=0).tolist(),
            "min": delta_q.min(dim=0).values.tolist(),
            "max": delta_q.max(dim=0).values.tolist(),
            "shape": list(delta_q.shape),
        }
        return stats


def compute_delta_q_from_psnr(
    psnr_by_depth: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ΔQ from PSNR values at each depth.

    Args:
        psnr_by_depth: PSNR at each depth [N, K+1] where depth 0 is base

    Returns:
        ΔQ values [N, K] where ΔQ(k) = PSNR(k) - PSNR(k-1)
    """
    delta_q = psnr_by_depth[:, 1:] - psnr_by_depth[:, :-1]
    return delta_q


if __name__ == "__main__":
    print("Delta Q Target Storage module loaded")
    print("Use via import from data package")