import torch
import torch.nn as nn
import torch.nn.functional as F


class DCVCProgressive(nn.Module):
    """DCVC with Progressive Decoder.

    Encodes the current frame optionally conditioned on a reference frame,
    then decodes progressively at variable depth for compute-scalable video decoding.

    When a reference frame is supplied the model acts as a temporal conditional
    codec (P-frame): the reference latent is fused with the current latent
    before decoding.  Without a reference it degrades to an image codec (I-frame).

    Args:
        out_channel_M:          Latent channel dimension.
        out_channel_N:          Decoder internal channel dimension.
        num_refinement_blocks:  Number of progressive refinement stages (default 3).
    """

    def __init__(self, out_channel_M=96, out_channel_N=64, num_refinement_blocks=3):
        super().__init__()
        self.out_channel_M = out_channel_M
        self.out_channel_N = out_channel_N
        self.num_refinement_blocks = num_refinement_blocks

        self._build_encoder()
        self._build_decoder(num_refinement_blocks)
        self._freeze_encoder()  # encoder is FROZEN — only decoder is trained (plan §2.1)

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _build_encoder(self):
        """Current-frame encoder + reference context encoder + fusion layer."""

        class ImageEncoder(nn.Module):
            """Shared encoder architecture (8× spatial downscale)."""
            def __init__(self, out_channels):
                super().__init__()
                self.conv1 = nn.Conv2d(3,   64,  kernel_size=5, stride=2, padding=2)
                self.conv2 = nn.Conv2d(64,  128, kernel_size=3, stride=2, padding=1)
                self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
                self.conv4 = nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = self.conv4(x)           # no activation — latent can be negative
                return x

        # Current frame encoder
        self.img_code = ImageEncoder(self.out_channel_M)

        # Reference (context) encoder — separate weights allow specialisation:
        # img_code learns "what to transmit", ctx_code learns "what to subtract".
        self.ctx_code = ImageEncoder(self.out_channel_M)

        # Context fusion: concat(current_latent, ctx_latent) → M channels
        self.ctx_fusion = nn.Sequential(
            nn.Conv2d(self.out_channel_M * 2, self.out_channel_M, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def _build_decoder(self, num_refinement_blocks):
        """Build the progressive decoder."""
        try:
            from .progressive_decoder import ProgressiveDecoder
        except ImportError:
            from progressive_decoder import ProgressiveDecoder

        self.decoder = ProgressiveDecoder(
            out_channel_M=self.out_channel_M,
            out_channel_N=self.out_channel_N,
            num_refinement_blocks=num_refinement_blocks,
        )

    # ------------------------------------------------------------------
    # Freeze helpers (plan §2.1: encoder is never trained)
    # ------------------------------------------------------------------

    def _freeze_encoder(self):
        """Freeze all encoder parameters — gradients will not propagate through them.

        Called once at construction. This saves VRAM during training because
        PyTorch will not store encoder activations in the computation graph.
        """
        for module in (self.img_code, self.ctx_code, self.ctx_fusion):
            for param in module.parameters():
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, x, ref=None):
        """Encode current frame, optionally conditioned on a reference frame.

        Args:
            x:   Current frame  (B, 3, H, W)
            ref: Reference frame (B, 3, H, W), or None for I-frame mode.

        Returns:
            Latent tensor (B, M, H/8, W/8)
        """
        y = self.img_code(x)
        if ref is not None:
            ctx = self.ctx_code(ref)
            y = self.ctx_fusion(torch.cat([y, ctx], dim=1))
        return y

    def decode(self, latent, depth=None):
        """Decode latent to a full-resolution reconstruction list.

        Args:
            latent: Latent tensor (B, M, H/8, W/8)
            depth:  Number of stages to run (1 to num_refinement_blocks+1).
                    None = run all stages.

        Returns:
            List of (B, 3, H, W) tensors, one per depth level.
        """
        return self.decoder(latent, depth=depth)

    def forward(self, x, ref=None, depth=None):
        """Full forward pass: encode then decode.

        Args:
            x:     Current frame  (B, 3, H, W)
            ref:   Reference frame (B, 3, H, W), or None for I-frame mode.
            depth: Decode depth (1 to num_refinement_blocks+1).

        Returns:
            dict:
                'reconstructions': list of full-resolution (B, 3, H, W) tensors
                'latent':          the (optionally context-fused) latent tensor
        """
        y = self.encode(x, ref=ref)
        reconstructions = self.decode(y, depth=depth)

        return {
            'reconstructions': reconstructions,
            'latent': y,
        }


class DCVCProgressiveLoss(nn.Module):
    """Rate-distortion training loss for DCVC Progressive.

    L = Σ_i  w_i · MSE(recon_i, target)   +   lambda_rd · ||latent||₁

    Deep supervision:
        - All decoder stages receive a gradient via the weighted MSE sum.
        - Earlier stages are weighted higher (they're always executed at any depth).

    Bitrate proxy:
        - L1 norm of the latent encourages sparsity, acting as a proxy for the
          entropy / bitrate of the compressed representation.
        - lambda_rd is the Lagrange multiplier controlling the rate-distortion
          trade-off: higher → smaller latents (lower bitrate, higher distortion).
    """

    def __init__(self, weights=None, lambda_rd=0.01):
        super().__init__()
        self.weights   = weights if weights is not None else [1.0, 0.9, 0.8, 0.7]
        self.lambda_rd = lambda_rd

    def forward(self, outputs, target, latent=None):
        """Compute weighted multi-stage rate-distortion loss.

        Args:
            outputs: List of reconstructions (B, 3, H, W) from the progressive decoder.
            target:  Ground-truth frame (B, 3, H, W) in [0, 1].
            latent:  Latent tensor for the bitrate penalty term (optional).

        Returns:
            Scalar loss tensor.
        """
        distortion = 0.0
        for i, recon in enumerate(outputs):
            w = self.weights[i] if i < len(self.weights) else self.weights[-1]
            distortion = distortion + w * F.mse_loss(recon, target)

        loss = distortion

        if latent is not None and self.lambda_rd > 0:
            # L1 penalty on latent: proxy for entropy / compressed bitrate
            loss = loss + self.lambda_rd * torch.mean(torch.abs(latent))

        return loss


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size=(1, 3, 256, 256)):
    """Print a concise model summary."""
    print(f"\n{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*60}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    if hasattr(model, 'decoder'):
        d = model.decoder
        print(f"\nDecoder structure:")
        print(f"  - Base decoder : {d.out_channel_N} channels, 8× upsample (3× PixelShuffle)")
        print(f"  - Refinement   : {d.num_refinement_blocks} blocks, residual_scale=0.1")
        print(f"  - Output res   : same as input (full resolution)")

    if hasattr(model, 'ctx_code'):
        print(f"\nContext encoder: enabled (P-frame / temporal conditioning)")

    print(f"{'='*60}\n")