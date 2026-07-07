# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Progressive extension — replaces contextualDecoder_part1 + contextualDecoder_part2
# with a multi-stage decoder that supports variable compute depth.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .video_net import ResBlock, GDN
from ..layers.layers import subpel_conv3x3


# ---------------------------------------------------------------------------
# Tiered residual refinement block (3 ResBlocks per stage)
# ---------------------------------------------------------------------------

class TieredRefinement(nn.Module):
    """Residual refinement block used for progressive depth tiers.

    Architecture
    ------------
        x               : (B, 3,  H, W)
        proj_in         : 3 -> N  (3x3)
        body            : 3 x ResBlock(N)
        proj_out        : N -> 3  (3x3, ZERO-INIT weight & bias)
        out = x + damping * proj_out(body(proj_in(x)))

    Key property: with `proj_out` zeroed at construction the block is the
    identity function. Gradient flow through the body allows the model to
    grow corrections smoothly during training, guaranteeing Tier 1 (no
    block applied) matches the original DCVC reconstruction exactly.

    `damping` (default 1.0) is a per-instance attribute that can be set
    externally (e.g. `block.damping = 0.8`) to softly damp overshoot
    without changing the structure.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.channels = channels
        self.proj_in = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(
            ResBlock(channels, channels, 3),
            ResBlock(channels, channels, 3),
            ResBlock(channels, channels, 3),
        )
        self.proj_out = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj_out(self.body(self.proj_in(x)))
        damping = getattr(self, "damping", 1.0)
        return x + damping * residual


# ---------------------------------------------------------------------------
# Progressive Contextual Decoder
# Replaces contextualDecoder_part1 + contextualDecoder_part2 in DCVC_net.py
# ---------------------------------------------------------------------------

class ProgressiveContextualDecoder(nn.Module):
    """Compute-scalable contextual decoder for DCVC.

    This is a **drop-in replacement** for DCVC's two-part fixed decoder:
        contextualDecoder_part1  (y_hat → upsampled features)
        contextualDecoder_part2  (cat(features, context) → RGB)

    Instead of a single reconstruction, it produces a list of reconstructions
    at increasing quality levels (depth 1 = fastest, depth 4 = full quality).
    All tiers decode the **same bitstream** — only the number of refinement
    stages applied post-reconstruction varies.

    Architecture
    ------------
    Part 1  (base upsample — identical to original DCVC part1):
        y_hat  (B, out_channel_M=96, H/16, W/16)
        → 4× subpel_conv3x3 upsample with IGDN
        → features  (B, out_channel_N=64, H, W)

    Part 2  (context fusion — identical to original DCVC part2):
        cat(features, context)  (B, 128, H, W)
        → 2× ResBlock + Conv2d(3)
        → recon_0  (B, 3, H, W)       ← Tier 1 output

    Refinement stages R_1 … R_k  (new — each adds incremental quality):
        recon_{i-1}  (B, 3, H, W)
        → project to N channels, ResBlock, project back to 3 channels
        → recon_i  (B, 3, H, W)       ← Tier i+1 output

    Parameters
    ----------
    out_channel_M          : latent channels from entropy decoder (default 96)
    out_channel_N          : decoder internal channels (default 64)
    num_refinement_blocks  : number of progressive refinement stages (default 3)
    """

    def __init__(
        self,
        out_channel_M: int = 96,
        out_channel_N: int = 64,
        num_refinement_blocks: int = 3,
    ):
        super().__init__()
        self.out_channel_M = out_channel_M
        self.out_channel_N = out_channel_N
        self.num_refinement_blocks = num_refinement_blocks

        # -----------------------------------------------------------------
        # Part 1: same architecture as original contextualDecoder_part1
        # y_hat (B, M, H/16, W/16) -> features (B, N, H, W)  [4x upsample]
        # -----------------------------------------------------------------
        self.part1 = nn.Sequential(
            subpel_conv3x3(out_channel_M, out_channel_N, 2),   # -> H/8
            GDN(out_channel_N, inverse=True),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),   # -> H/4
            GDN(out_channel_N, inverse=True),
            ResBlock(out_channel_N, out_channel_N, 3),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),   # -> H/2
            GDN(out_channel_N, inverse=True),
            ResBlock(out_channel_N, out_channel_N, 3),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),   # -> H
        )

        # -----------------------------------------------------------------
        # Part 2: same architecture as original contextualDecoder_part2
        # cat(features, context) (B, 2N, H, W) -> RGB (B, 3, H, W)
        # -----------------------------------------------------------------
        self.part2 = nn.Sequential(
            nn.Conv2d(out_channel_N * 2, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, 3, 3, stride=1, padding=1),
        )

        # -----------------------------------------------------------------
        # Progressive refinement stages (new contribution).
        # Each stage is a TieredRefinement block: 4x FLOPs cost over base.
        #   Tier 1 = part1+part2 (DCVC baseline) -> no refinement blocks
        #   Tier i+1 = Tier i + refinement_blocks[i-1]
        # -----------------------------------------------------------------
        self.refinement_blocks = nn.ModuleList([
            TieredRefinement(out_channel_N)
            for _ in range(num_refinement_blocks)
        ])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        y_hat: torch.Tensor,
        context: torch.Tensor,
        depth: int | None = None,
    ) -> list[torch.Tensor]:
        """Decode y_hat + context to a list of progressive reconstructions.

        Args:
            y_hat:   Quantised latent from entropy decoder
                     shape (B, out_channel_M, H/16, W/16)
            context: Motion-compensated context features
                     shape (B, out_channel_N, H, W)
            depth:   Number of output stages to produce (1 … num_refinement_blocks+1).
                     None means produce all stages.

        Returns:
            List of (B, 3, H, W) tensors.
            outputs[0]  = Tier 1 (base, same as original DCVC)
            outputs[-1] = Tier 4 (full quality, all refinements applied)
        """
        if depth is None:
            depth = self.num_refinement_blocks + 1
        depth = max(1, min(depth, self.num_refinement_blocks + 1))

        features = self.part1(y_hat)                                   # (B, N, H, W)
        x = self.part2(torch.cat([features, context], dim=1))          # (B, 3, H, W)
        x = torch.clamp(x, 0.0, 1.0)
        outputs = [x]

        # Progressive refinement stages
        for i in range(depth - 1):
            x = self.refinement_blocks[i](x)                           # (B, 3, H, W)
            x = torch.clamp(x, 0.0, 1.0)
            outputs.append(x)

        return outputs

    def extra_repr(self) -> str:
        return (
            f'out_channel_M={self.out_channel_M}, '
            f'out_channel_N={self.out_channel_N}, '
            f'num_refinement_blocks={self.num_refinement_blocks}'
        )
