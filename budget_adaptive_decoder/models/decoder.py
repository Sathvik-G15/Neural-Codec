"""
BudgetAdaptiveDecoder module for Budget-Constrained Neural Decoder.

Implements the decoder with K=4 sequential refinement stages as specified
in Section 6.5 of the design document.

The student decoder learns to refine a base reconstruction.
Since the original DCVC teacher only provides a single reconstruction
(and context), the student learns to improve upon the base reconstruction
through self-supervised refinement stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


class ResidualBlock(nn.Module):
    """Residual block with pre-activation structure."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class BaseReconstruction(nn.Module):
    """
    Produces initial reconstruction from context and previous frame.

    DCVC original outputs:
        - context: [B, 64, H/4, W/4] motion-compensated features
        - recon_image: [B, 3, H, W] base reconstruction

    We upsample context to H and combine with prev_frame for base recon.
    """

    def __init__(self, context_channels=64, feature_channels=64):
        super().__init__()

        self.feature_channels = feature_channels

        # Context upsampling: H/4 -> H (2 upsamples, each 2x)
        self.context_upsample = nn.Sequential(
            nn.ConvTranspose2d(context_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, feature_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Combine with prev_frame at H resolution
        self.combine = nn.Sequential(
            nn.Conv2d(feature_channels + 3, feature_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Final reconstruction head
        self.recon_head = nn.Conv2d(feature_channels, 3, kernel_size=3, padding=1)

    def forward(self, context, prev_frame):
        """
        Args:
            context:    [B, 64, H/4, W/4] motion-compensated context from DCVC
            prev_frame: [B, 3, H, W] previous decoded frame

        Returns:
            recon:    [B, 3, H, W]
            features: [B, feature_channels, H, W]
        """
        # Upsample context from H/4 to H
        features = self.context_upsample(context)

        # Match features to prev_frame spatial size
        if features.shape[2:] != prev_frame.shape[2:]:
            features = F.interpolate(features, size=prev_frame.shape[2:],
                                    mode='bilinear', align_corners=False)

        # Combine with prev_frame
        combined = torch.cat([features, prev_frame], dim=1)
        features_out = self.combine(combined)

        # Reconstruct
        recon = self.recon_head(features_out)
        recon = torch.sigmoid(recon)

        return recon, features_out


class RefinementStage(nn.Module):
    """
    Single refinement stage.
    Input/output: same spatial resolution H×W.
    Uses residual connection: output = input + delta
    """

    def __init__(self, feature_channels=64, n_residual_blocks=4):
        super().__init__()

        blocks = [ResidualBlock(feature_channels) for _ in range(n_residual_blocks)]
        self.net = nn.Sequential(*blocks)

        self.feature_head = nn.Conv2d(feature_channels, feature_channels,
                                       kernel_size=3, padding=1)
        self.recon_head   = nn.Conv2d(feature_channels, 3,
                                       kernel_size=3, padding=1)

    def forward(self, recon, features):
        """
        Args:
            recon:    [B, 3, H, W]       current reconstruction
            features: [B, C_feat, H, W]  current feature map

        Returns:
            recon_refined:    [B, 3, H, W]
            features_updated: [B, C_feat, H, W]
        """
        feat_out = self.net(features)

        # Update features
        features_updated = features + self.feature_head(feat_out)

        # Compute reconstruction delta and add residually
        delta = self.recon_head(feat_out)
        recon_refined = recon + delta
        recon_refined = torch.clamp(recon_refined, 0.0, 1.0)

        return recon_refined, features_updated


class BudgetAdaptiveDecoder(nn.Module):
    """
    Budget-Adaptive Decoder with K=4 refinement stages.

    Takes context from DCVC teacher and previous frame to produce
    base reconstruction, then applies K refinement stages.

    Stage cost ratios (verified): C1:C2:C3:C4 = 4:2:2:1
    Stage boundaries: 0.444, 0.667, 0.889, 1.000
    """

    BLOCKS_PER_STAGE = [8, 4, 4, 2]  # Achieves 4:2:2:1 ratio

    def __init__(self, context_channels=64, feature_channels=64, K=4):
        super().__init__()
        self.K = K
        self.feature_channels = feature_channels

        # Base reconstruction from context + prev_frame
        self.base = BaseReconstruction(
            context_channels=context_channels,
            feature_channels=feature_channels
        )

        # Refinement stages — decreasing block counts for decreasing cost
        self.stages = nn.ModuleList([
            RefinementStage(feature_channels,
                           n_residual_blocks=self.BLOCKS_PER_STAGE[k])
            for k in range(K)
        ])

    def base_reconstruct(self, context, prev_frame):
        """Always runs. Returns (recon, features) at full resolution."""
        return self.base(context, prev_frame)

    def run_to_depth(self, context, prev_frame, k_star):
        """Inference path: run exactly k_star stages."""
        recon, features = self.base_reconstruct(context, prev_frame)

        for k in range(k_star):
            recon, features = self.stages[k](recon, features)

        return recon

    def decode_all_stages(self, context, prev_frame):
        """
        DEPRECATED: Use decode_all_stages_phase1 or decode_all_stages_phase2.

        Kept for backwards compatibility. Returns full unrolled gradient chain
        (used in Phase 3+ when all stages share a unified supervision signal).
        """
        recon, features = self.base_reconstruct(context, prev_frame)

        stage_outputs = []
        for k in range(self.K):
            recon, features = self.stages[k](recon, features)
            stage_outputs.append(recon)

        return stage_outputs

    def decode_all_stages_phase1(self, context, prev_frame):
        """
        Phase 1 training path (CRITICAL FIX for C1).

        Stage 1: supervised by teacher. Gradient signal from teacher MSE only.
        Stages 2-4: supervised by ground truth. Gradient signal from GT MSE.

        Architectural requirement: Stage 1 output is DETACHED before being
        passed to Stage 2, so that the Stage 2-4 GT-loss gradient does not
        flow back into Stage 1's weights. This isolates Stage 1's training
        signal to match the teacher, and Stages 2-4's signal to surpass it.

        Without this detach, Stage 1 receives conflicting gradient signals
        (teacher MSE + propagated GT MSE), and degenerates to a compromise
        that cannot cleanly match teacher quality.

        Returns:
            stage_outputs: list of K reconstructions [R1, R2, R3, R4]
        """
        recon, features = self.base_reconstruct(context, prev_frame)

        # Stage 1: teacher-supervised (gradient signal: teacher MSE only)
        recon, features = self.stages[0](recon, features)
        stage1_output = recon

        # CRITICAL: detach Stage 1 output before feeding into Stage 2
        # Stage 2-4 gradients must NOT flow back into Stage 1
        recon_detached = recon.detach()
        features_detached = features.detach()

        stage_outputs = [stage1_output]

        # Stages 2-4: GT-supervised (gradient signal: GT MSE only within these stages)
        recon, features = recon_detached, features_detached
        for k in range(1, self.K):
            recon, features = self.stages[k](recon, features)
            stage_outputs.append(recon)

        return stage_outputs

    def get_stage_states(self, context, prev_frame):
        """
        Phase 2 path: run all stages, return recon at each depth.
        Returns list of K+1 reconstructions [R0, R1, R2, R3, R4].
        """
        recon, features = self.base_reconstruct(context, prev_frame)
        all_recons = [recon]

        for k in range(self.K):
            recon, features = self.stages[k](recon, features)
            all_recons.append(recon)

        return all_recons


def verify_decoder_stages():
    """Verify decoder produces correct number of stage outputs."""
    import torch

    decoder = BudgetAdaptiveDecoder()

    print("=" * 60)
    print("BudgetAdaptiveDecoder Stage Verification")
    print("=" * 60)
    print(f"Number of stages (K): {decoder.K}")
    print(f"Blocks per stage: {decoder.BLOCKS_PER_STAGE}")

    batch_size = 2
    H, W = 256, 448

    # Context at H/4, prev_frame at H
    context = torch.randn(batch_size, 64, H//4, W//4)
    prev_frame = torch.randn(batch_size, 3, H, W)

    reconstructions = decoder.decode_all_stages(context, prev_frame)

    print(f"decode_all_stages returned: {len(reconstructions)} outputs")
    print(f"Expected: K = {decoder.K}")

    assert len(reconstructions) == decoder.K

    for i, r in enumerate(reconstructions):
        print(f"  R_{i+1} shape: {r.shape}")

    # Test base reconstruction
    recon, features = decoder.base_reconstruct(context, prev_frame)
    print(f"\nbase_reconstruct returned:")
    print(f"  recon shape: {recon.shape} (expected [B, 3, H, W])")
    print(f"  features shape: {features.shape} (expected [B, 64, H, W])")

    assert recon.shape == (batch_size, 3, H, W)
    assert features.shape == (batch_size, 64, H, W)

    print("\nAll decoder stage outputs verified!")
    print("=" * 60)


if __name__ == "__main__":
    verify_decoder_stages()