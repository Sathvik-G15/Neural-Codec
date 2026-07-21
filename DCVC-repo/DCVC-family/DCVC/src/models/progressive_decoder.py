# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Progressive extension — replaces contextualDecoder_part1 + contextualDecoder_part2
# in DCVC_net.py with a multi-stage decoder that supports variable compute depth.
#
# Architecture (per plan §2.3):
#   - Feature-space refinement (between part1 and part2, on 64-channel features)
#   - Depthwise-separable convolutions for low FLOPs per stage (~5 GFLOPs vs ~4x base)
#   - Target: Tier4/Tier1 FLOPs ratio in 1.3x-2.0x range

import torch
import torch.nn as nn
import torch.nn.functional as F

from .video_net import ResBlock, GDN
from ..layers.layers import subpel_conv3x3


# ---------------------------------------------------------------------------
# Feature-space depthwise-separable refinement block (plan §2.3)
# ---------------------------------------------------------------------------

class FeatureRefinementBlock(nn.Module):
    """Feature-space residual refinement block.

    Operates on part1's 64-channel feature map, inserted between part1 and
    part2. Uses depthwise-separable convolutions to keep per-stage FLOPs low
    at 64-channel width (~5 GFLOPs per block vs the old RGB-space TieredRefinement
    which was ~4x base decoder FLOPs per stage).

    Architecture
    ------------
        x             : (B, N, H, W) where N=64
        dw1           : depthwise 3x3 conv (N -> N, groups=N)
        pw1           : pointwise 1x1 conv (N -> N)
        dw2           : depthwise 3x3 conv (N -> N, groups=N)
        pw2           : pointwise 1x1 conv (N -> N, ZERO-INIT)
        residual      : 0.1 * pw2(dw2(lrelu(pw1(dw1(x)))))
        out           : x + residual

    Key property: with pw2 zeroed at construction the block is the identity
    function. Gradient flow through dw1/pw1/dw2 allows the model to grow
    corrections smoothly during training.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.channels = channels
        self.dw1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels, 1)
        self.dw2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.lrelu = nn.LeakyReLU(0.2)
        nn.init.zeros_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lrelu(self.pw1(self.dw1(x)))
        residual = self.pw2(self.dw2(h))
        return x + 0.1 * residual


# ---------------------------------------------------------------------------
# Progressive Contextual Decoder
# Replaces contextualDecoder_part1 + contextualDecoder_part2 in DCVC_net.py
# ---------------------------------------------------------------------------

class ProgressiveContextualDecoder(nn.Module):
    """Compute-scalable contextual decoder for DCVC.

    This is a **drop-in replacement** for DCVC's two-part fixed decoder:
        contextualDecoder_part1  (y_hat -> upsampled features)
        contextualDecoder_part2  (cat(features, context) -> RGB)

    Architecture (plan §2.2)
    -----------------------
    Part 1  (base upsample - identical to original DCVC part1):
        y_hat  (B, out_channel_M=96, H/16, W/16)
        -> 4x subpel_conv3x3 upsample with IGDN
        -> features  (B, out_channel_N=64, H, W)

    Refinement stages R_1 .. R_k  (feature-space, between part1 and part2):
        features       -> features_1  (via R_1)
        features_1     -> features_2  (via R_2)
        features_2     -> features_3  (via R_3)

    Part 2  (context fusion - identical to original DCVC part2):
        cat(features, context)  (B, 2N, H, W) -> RGB (B, 3, H, W)

    Per plan §2.3: "Tier 1 output is identical to unmodified DCVC decode"

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

        self.part1 = nn.Sequential(
            subpel_conv3x3(out_channel_M, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock(out_channel_N, out_channel_N, 3),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock(out_channel_N, out_channel_N, 3),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
        )

        self.part2 = nn.Sequential(
            nn.Conv2d(out_channel_N * 2, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, 3, 3, stride=1, padding=1),
        )

        self.refinement_blocks = nn.ModuleList([
            FeatureRefinementBlock(out_channel_N)
            for _ in range(num_refinement_blocks)
        ])

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
            depth:   Number of refinement blocks to apply (1..num_refinement_blocks+1).
                     depth=1 -> base decoder only, outputs=[recon_1] (Tier 1)
                     depth=k -> base + first k-1 refinement blocks, outputs=[recon_1, ..., recon_k]
                     None means produce all stages (Tier 1 through Tier 4).

        Returns:
            List of (B, 3, H, W) tensors.
            outputs[0]  = Tier 1 (base, same as original DCVC)
            outputs[-1] = Tier 4 (full quality, all refinements applied)

        Invariant: len(outputs) == depth for all depth in {1,..,num_refinement_blocks+1}
        """
        if depth is None:
            depth = self.num_refinement_blocks + 1
        depth = max(1, min(depth, self.num_refinement_blocks + 1))

        features = self.part1(y_hat)

        refined_features = features
        outputs = []

        for i in range(depth - 1):
            refined_features = self.refinement_blocks[i](refined_features)
            recon = self.part2(torch.cat([refined_features, context], dim=1))
            recon = torch.clamp(recon, 0.0, 1.0)
            outputs.append(recon)

        base_recon = self.part2(torch.cat([features, context], dim=1))
        base_recon = torch.clamp(base_recon, 0.0, 1.0)
        outputs.insert(0, base_recon)

        return outputs

    def forward_with_depth(
        self,
        y_hat: torch.Tensor,
        context: torch.Tensor,
        active_depth: int,
    ) -> list[torch.Tensor]:
        """Forward with explicit active_depth for training (plan §3.1).

        active_depth in {1,2,3,4}:
            active_depth=1 -> base decoder only, outputs=[recon_1] (Tier 1)
            active_depth=2 -> base + R_1, outputs=[recon_1, recon_2] (Tiers 1-2)
            active_depth=3 -> base + R_1 + R_2, outputs=[recon_1, recon_2, recon_3]
            active_depth=4 -> base + R_1 + R_2 + R_3, outputs=[recon_1, ..., recon_4]

        Invariant: len(outputs) == active_depth
        """
        return self.forward(y_hat, context, depth=active_depth)
