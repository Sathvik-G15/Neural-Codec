"""
Data augmentation for Budget-Constrained Neural Decoder.

Implements reference frame corruption as specified in Section 8
of the design document.

IMPORTANT: Reference frame corruption is ONLY applied during Phase 3 and Phase 4.
It is NOT applied during Phase 1.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ReferenceFrameCorruption(nn.Module):
    """
    Corrupts the previous frame to simulate temporal error propagation
    during Phase 3 and Phase 4 training.

    prev_frame_corrupted = prev_frame + ε,  ε ~ N(0, σ²)

    where σ = min(0.005 * frame_position_in_sequence, 0.02)

    The cap at σ_max = 0.02 keeps corruption within realistic decoded
    frame quality range (~34 dB PSNR degradation).

    IMPORTANT: This is only applied during Phase 3 and Phase 4.
    Phase 1 uses clean references to match teacher outputs.
    """

    SIGMA_SLOPE = 0.005
    SIGMA_MAX = 0.02

    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: If True, apply corruption. Set to False for Phase 1.
        """
        super().__init__()
        self.enabled = enabled

    def forward(
        self,
        prev_frame: torch.Tensor,
        frame_position: Optional[int] = None,
        frame_position_in_sequence: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply corruption to the previous frame.

        Args:
            prev_frame: Previous frame tensor [B, C, H, W]
            frame_position: (Deprecated) Position in the sequence (0-indexed).
                          Use frame_position_in_sequence instead.
            frame_position_in_sequence: Position in the sequence (0-indexed).
                                       If None, uses frame_position.

        Returns:
            Corrupted previous frame (same shape as input)
        """
        if not self.enabled:
            return prev_frame

        if frame_position_in_sequence is None:
            frame_position_in_sequence = frame_position if frame_position is not None else 0

        sigma = self._compute_sigma(frame_position_in_sequence)

        noise = torch.randn_like(prev_frame) * sigma
        corrupted = prev_frame + noise

        return corrupted

    def _compute_sigma(self, frame_position: int) -> float:
        """
        Compute σ for the given frame position.

        σ = min(0.005 * frame_position, 0.02)
        """
        sigma = self.SIGMA_SLOPE * frame_position
        return min(sigma, self.SIGMA_MAX)

    def get_sigma(self, frame_position: int) -> float:
        """Get the sigma value for a given frame position (for logging)."""
        return self._compute_sigma(frame_position)


class TemporalAugmentation:
    """
    Additional temporal augmentation strategies for video data.

    This is a placeholder for future augmentation strategies mentioned
    in the design document (Section 8, Content distribution augmentation).
    """

    def __init__(self, bvi_dvc_root: Optional[str] = None):
        """
        Args:
            bvi_dvc_root: Root directory for BVI-DVC augmentation sequences.
                         If None, BVI-DVC augmentation is disabled.
        """
        self.bvi_dvc_root = bvi_dvc_root
        self.enabled = bvi_dvc_root is not None

    def get_augmentation_weight(self, use_bvi_dvc: bool = False) -> float:
        """
        Return probability of using BVI-DVC augmentation.

        As per design doc: 70% Vimeo90k + 30% BVI-DVC.
        """
        if not self.enabled:
            return 0.0
        return 0.3 if use_bvi_dvc else 0.7


def verify_corruption_sigma():
    """Verify sigma computation matches design document."""
    corruption = ReferenceFrameCorruption()

    print("=" * 60)
    print("ReferenceFrameCorruption Sigma Test")
    print("=" * 60)

    test_positions = [0, 1, 2, 3, 4, 5, 6, 10, 20]
    expected_sigmas = [
        0.0,
        0.005,
        0.01,
        0.015,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
    ]

    print(f"\n{'Position':<10} {'Computed σ':<15} {'Expected σ':<15} {'PASS':<10}")
    print("-" * 50)

    all_passed = True
    for pos, expected in zip(test_positions, expected_sigmas):
        sigma = corruption.get_sigma(pos)
        passed = abs(sigma - expected) < 1e-6
        all_passed = all_passed and passed
        print(f"{pos:<10} {sigma:<15.6f} {expected:<15.6f} {'✓' if passed else '✗':<10}")

    print("-" * 50)

    prev_frame = torch.rand(2, 3, 64, 64)
    corrupted = corruption(prev_frame, frame_position_in_sequence=3)

    print(f"\nInput shape: {prev_frame.shape}")
    print(f"Output shape: {corrupted.shape}")
    print(f"Noise std: {(corrupted - prev_frame).std().item():.6f}")
    print(f"Expected σ for position 3: 0.015")

    assert all_passed, "Sigma computation test failed"
    assert corrupted.shape == prev_frame.shape, "Shape mismatch"

    print("\nAll corruption tests passed!")

    print("\n" + "=" * 60)
    print("Augmentation tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    verify_corruption_sigma()