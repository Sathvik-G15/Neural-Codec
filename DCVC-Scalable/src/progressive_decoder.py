import torch
import torch.nn as nn
import torch.nn.functional as F


class GDN(nn.Module):
    """Generalized Divisive Normalization (Ballé et al., 2016).

    Uses a full N×N gamma matrix for cross-channel interactions,
    matching the original formulation:

        norm[d] = sqrt( beta[d] + Σ_c  gamma[c, d] * x[c]² )

    In inverse mode (IGDN) the input is multiplied by the normalizer
    instead of divided, implementing the inverse transform used in decoders.
    """

    def __init__(self, channels, inverse=False, beta_min=1e-6, gamma_init=0.1):
        super().__init__()
        self.inverse = inverse
        self.beta_min = beta_min

        self.beta = nn.Parameter(torch.ones(channels))
        # Full cross-channel gamma matrix (C × C)
        self.gamma = nn.Parameter(torch.eye(channels) * gamma_init)

    def forward(self, x):
        beta = (self.beta.abs() + self.beta_min)           # (C,)
        gamma = self.gamma.abs()                            # (C, C)

        x_sq = x.pow(2)                                    # (B, C, H, W)
        # Cross-channel weighted sum: norm[d] = beta[d] + Σ_c gamma[c,d] * x_sq[c]
        norm = torch.einsum('bchw,cd->bdhw', x_sq, gamma)  # (B, C, H, W)
        norm = (beta.view(1, -1, 1, 1) + norm).clamp(min=1e-6).sqrt()

        if self.inverse:
            return x * norm
        else:
            return x / norm


class RefinementBlock(nn.Module):
    """Lightweight residual refinement block.

    Each block takes the current feature representation and adds a small
    scaled correction. This is the compute-scalable component — skipping
    blocks trades quality for speed at decode time.
    """

    def __init__(self, channels, residual_scale=0.1):
        super().__init__()
        self.channels = channels
        self.residual_scale = residual_scale

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = self.conv2(self.lrelu(self.conv1(x)))
        return x + self.residual_scale * residual


class BaseDecoder(nn.Module):
    """Base decoder: y_hat (B, 96, H/16, W/16) → features (B, 64, H, W).

    4-stage PixelShuffle upsampling: each stage is 2× spatial (total 16×).
    Input is H/16 → Output is H (full resolution) in feature space.
    Final output is 64 channels (NOT RGB) for refinement blocks.
    """

    def __init__(self, out_channel_M=96, out_channel_N=64):
        super().__init__()

        # Stage 1: H/16 → H/8
        self.stage1 = nn.Sequential(
            nn.Conv2d(out_channel_M, out_channel_N * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 2: H/8 → H/4
        self.stage2 = nn.Sequential(
            nn.Conv2d(out_channel_N, out_channel_N * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 3: H/4 → H/2
        self.stage3 = nn.Sequential(
            nn.Conv2d(out_channel_N, out_channel_N * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 4: H/2 → H
        self.stage4 = nn.Sequential(
            nn.Conv2d(out_channel_N, out_channel_N * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final feature projection: 64 channels (NOT RGB)
        self.final_conv = nn.Conv2d(out_channel_N, out_channel_N, kernel_size=3, padding=1)

    def forward(self, y_hat):
        x = self.stage1(y_hat)   # (B, 64, H/8, W/8)
        x = self.stage2(x)       # (B, 64, H/4, W/4)
        x = self.stage3(x)       # (B, 64, H/2, W/2)
        x = self.stage4(x)       # (B, 64, H, W)
        x = self.final_conv(x)   # (B, 64, H, W)
        return x


class ProgressiveDecoder(nn.Module):
    """Progressive residual decoder for compute-scalable video decoding.

    Architecture (as per research plan §2):
        y_hat (B, 96, H/16, W/16)
            → BaseDecoder → features (B, 64, H, W)
            → [64→3 conv] → RGB (Tier 1 output)
            → [3→64 conv] → features → RefinementBlock(64) → features → [64→3 conv] → RGB (Tier 2)
            → repeat for Tier 3, 4

    Key property: RefinementBlock operates on 64 channels (same as base decoder output)
    as specified in research plan §2.3. Projection layers bridge RGB↔feature space.

    Depth control (anytime decoding):
        depth=1 : base features → RGB
        depth=2 : base + refinement block 1
        depth=3 : base + refinement blocks 1-2
        depth=4 : base + all 3 refinement blocks
    """

    def __init__(self, out_channel_M=96, out_channel_N=64, num_refinement_blocks=3):
        super().__init__()
        self.out_channel_M = out_channel_M
        self.out_channel_N = out_channel_N
        self.num_refinement_blocks = num_refinement_blocks

        self.base_decoder = BaseDecoder(out_channel_M, out_channel_N)

        # Per-tier RGB projection: 64-ch features → 3-ch RGB
        self.tier_projections = nn.ModuleList([
            nn.Conv2d(out_channel_N, 3, kernel_size=3, padding=1)
            for _ in range(num_refinement_blocks + 1)
        ])

        # Refinement blocks: operate on 64-channel feature space (plan §2.3)
        self.refinement_blocks = nn.ModuleList([
            RefinementBlock(out_channel_N, residual_scale=0.1)
            for _ in range(num_refinement_blocks)
        ])

    def forward(self, latent, depth=None):
        """Forward pass with optional depth control.

        Args:
            latent: Compressed latent (B, 96, H/16, W/16)
            depth:  Number of total output stages to produce (1 to num_refinement_blocks+1).

        Returns:
            List of full-resolution RGB tensors, one per depth level.
        """
        if depth is None:
            depth = self.num_refinement_blocks + 1

        depth = max(1, min(depth, self.num_refinement_blocks + 1))

        # Base decode: y_hat → 64-channel features at full resolution
        features = self.base_decoder(latent)   # (B, 64, H, W)
        outputs = []

        # Tier 1: base features → RGB
        rgb = self.tier_projections[0](features)
        rgb = torch.clamp(rgb, 0.0, 1.0)
        outputs.append(rgb)

        # Progressive refinement in 64-channel feature space
        current_features = features
        for i in range(depth - 1):
            # Residual refinement in feature space (plan §2.3: 64 channels)
            refined_features = self.refinement_blocks[i](current_features)
            current_features = current_features + 0.1 * refined_features
            # Project to RGB for output
            rgb = self.tier_projections[i + 1](current_features)
            rgb = torch.clamp(rgb, 0.0, 1.0)
            outputs.append(rgb)

        return outputs

    def get_flops_per_stage(self, latent_shape):
        """Estimate cumulative FLOPs for each decode stage.

        Args:
            latent_shape: (B, C, H_lat, W_lat) of the latent (H/16, W/16)

        Returns:
            List of cumulative FLOPs: [base, base+R1, ..., base+R1+...+R3]
        """
        B, C, H_lat, W_lat = latent_shape
        H_full = H_lat * 16
        W_full = W_lat * 16

        base_flops = 0
        for m in self.base_decoder.modules():
            if isinstance(m, nn.Conv2d):
                kH, kW = m.kernel_size
                out_h = (H_full // 16) if 'stage1' in str(type(m)) else H_full
                # Use appropriate spatial dimension per stage
                if 'stage1' in [n for n, _ in self.base_decoder.named_modules()]:
                    h, w = H_lat * 2, W_lat * 2   # H/8
                elif 'stage2' in str(type(m)):
                    h, w = H_lat * 4, W_lat * 4   # H/4
                elif 'stage3' in str(type(m)):
                    h, w = H_lat * 8, W_lat * 8   # H/2
                else:
                    h, w = H_full, W_full          # H
                base_flops += m.in_channels * m.out_channels * kH * kW * h * w

        ref_flops = 0
        if self.num_refinement_blocks > 0:
            for m in self.refinement_blocks[0].modules():
                if isinstance(m, nn.Conv2d):
                    kH, kW = m.kernel_size
                    ref_flops += m.in_channels * m.out_channels * kH * kW * H_full * W_full

        tier_flops = [base_flops]
        for _ in range(self.num_refinement_blocks):
            tier_flops.append(tier_flops[-1] + ref_flops)

        return tier_flops

    def extra_repr(self):
        return (f'out_channel_M={self.out_channel_M}, '
                f'out_channel_N={self.out_channel_N}, '
                f'num_refinement_blocks={self.num_refinement_blocks}')


class subpel_conv3x3(nn.Module):
    """Sub-pixel convolution for upsampling."""
    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels * (upscale_factor ** 2),
            kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))