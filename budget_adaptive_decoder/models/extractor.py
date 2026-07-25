"""
BitstreamContentExtractor module for Budget-Constrained Neural Decoder.

FIX (S1): Rewritten to consume the actual tensors DCVC provides rather
than the idealized "latent_quantized + motion_latent + frame_type_id"
described in the original design.

DCVC's forward() (verified in teacher.py) returns:
  - latent:   feature_renorm tensor      [B, C_lat,  H/16, W/16]
              (channel count is codec-config dependent)
  - motion:   quant_mv_upsample_refine   [B, C_mot,  H/4,  W/4]
              (motion-compensated features)
  - context:  motion-compensated context [B, 64,     H/4,  W/4]
              (the tensor actually fed to the original DCVC decoder head)
  - recon_image: [B, 3, H, W]            (DCVC base reconstruction)

Frame-type-id is NOT available from DCVC in this configuration; we drop it.

Inputs to this extractor (matching verified DCVC outputs):
  - context:    [B, 64, H/4, W/4]
  - recon_image: [B, 3, H, W]   (DCVC base reconstruction)
  - prev_frame: [B, 3, H, W]   (previous decoded frame)
  - latent:     [B, C_lat, H/16, W/16] (optional, default kwargs)

Content difficulty proxy:
  residual = |recon_image - prev_frame|
  High residual means the current frame differs significantly from the
  reference -- a strong signal for whether refinement stages will help.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class BitstreamContentExtractor(nn.Module):
    """
    Lightweight feature network operating on DCVC tensors.

    Inputs:
        context:     motion-compensated features [B, 64, H/4, W/4]
        recon_image: DCVC base reconstruction   [B, 3, H, W]
        prev_frame:  previous decoded frame      [B, 3, H, W]
        latent:      optional feature tensor    [B, C_lat, ...]
                     (defaults to None for inference size reduction)

    Output:
        content_features: aggregated feature vector [B, feature_dim]

    Architecture:
        1. context is spatially pooled and projected.
        2. residual = |recon_image - prev_frame| is pooled and projected.
        3. prev_frame statistics are pooled and projected.
        4. Optional latent is pooled and projected.
        5. Concatenate all and pass through MLP fusion.
        6. Output is per-frame (frame-local) feature vector.
    """

    POOL_SIZE = 8  # adaptive avg pool spatial size

    def __init__(
        self,
        context_channels: int = 64,
        latent_channels: int = 96,
        feature_dim: int = 64,
        use_latent: bool = False,
    ):
        """
        Args:
            context_channels: Number of channels in DCVC's `context` tensor (matches
                              student decoder's context_channels). Default 64.
            latent_channels:  Channel count of DCVC's `feature_renorm` tensor
                              (variable -- 96 used as nominal DCVC default).
                              Only used if use_latent=True.
            feature_dim: Output feature dimension. Default 64.
            use_latent:  Whether to include latent features. Default False
                         (smaller, faster, matches most deployment scenarios).
        """
        super().__init__()
        self.context_channels = context_channels
        self.latent_channels = latent_channels
        self.feature_dim = feature_dim
        self.use_latent = use_latent

        self.context_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.POOL_SIZE),
            nn.Flatten(),
            nn.Linear(context_channels * self.POOL_SIZE * self.POOL_SIZE, 32),
            nn.ReLU(inplace=True),
        )

        # Residual between DCVC base reconstruction and prev_frame:
        # high residual => current frame differs strongly from reference
        # => likely benefits from refinement stages.
        self.residual_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.POOL_SIZE),
            nn.Flatten(),
            nn.Linear(3 * self.POOL_SIZE * self.POOL_SIZE, 16),
            nn.ReLU(inplace=True),
        )

        # Statistics of previous frame (texture content proxy)
        self.prev_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.POOL_SIZE // 2),
            nn.Flatten(),
            nn.Linear(3 * (self.POOL_SIZE // 2) * (self.POOL_SIZE // 2), 16),
            nn.ReLU(inplace=True),
        )

        if use_latent:
            self.latent_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(latent_channels, 16),
                nn.ReLU(inplace=True),
            )
            fusion_input_dim = 32 + 16 + 16 + 16
        else:
            fusion_input_dim = 32 + 16 + 16

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim),
        )

    def forward(
        self,
        context: torch.Tensor,
        recon_image: torch.Tensor,
        prev_frame: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract content features from DCVC bitstream representations PLUS
        the previous frame.

        Args:
            context:     DCVC context        [B, 64, H/4, W/4]
            recon_image: DCVC base recon     [B, 3, H, W]
            prev_frame:  previous frame      [B, 3, H, W]
            latent:      optional DCVC latent [B, C_lat, ...] (only if use_latent)

        Returns:
            content_features: [B, feature_dim]
        """
        ctx_feat = self.context_proj(context)

        # Residual as content difficulty proxy
        residual = (recon_image - prev_frame).abs()
        res_feat = self.residual_proj(residual)
        prev_feat = self.prev_proj(prev_frame)

        if self.use_latent and latent is not None:
            lat_feat = self.latent_proj(latent)
            fused = torch.cat([ctx_feat, res_feat, prev_feat, lat_feat], dim=1)
        else:
            fused = torch.cat([ctx_feat, res_feat, prev_feat], dim=1)

        return self.fusion(fused)


def verify_resolution_invariance():
    """
    Verify that the extractor produces the same output shape regardless
    of input resolution. This is a unit test for the design requirement
    that content features are resolution-invariant (post-pool).
    """
    import torch

    extractor = BitstreamContentExtractor()
    extractor.eval()

    resolutions = [
        (256, 448),
        (480, 854),
        (720, 1280),
    ]
    batch_size = 2

    print("=" * 60)
    print("BitstreamContentExtractor Resolution Invariance Test (FIXED)")
    print("=" * 60)
    print("Inputs: context [B, 64, H/4, W/4], recon [B, 3, H, W], "
          "prev [B, 3, H, W] -- matching verified DCVC outputs")
    print()

    for h, w in resolutions:
        context = torch.randn(batch_size, 64, h // 4, w // 4)
        recon_image = torch.rand(batch_size, 3, h, w)
        prev_frame = torch.rand(batch_size, 3, h, w)

        with torch.no_grad():
            features = extractor(context, recon_image, prev_frame)

        print(f"Input resolution: {h}x{w}  (context {h//4}x{w//4})")
        print(f"Output shape: {features.shape}")
        print(f"Expected: [{batch_size}, {extractor.feature_dim}]")
        assert features.shape == (batch_size, extractor.feature_dim), \
            "shape mismatch"
        print()

    print("Resolution invariance verified for all tested resolutions!")
    print("=" * 60)


if __name__ == "__main__":
    verify_resolution_invariance()