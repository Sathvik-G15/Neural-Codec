# DCVC Integration Layer
#
# This module connects DCVC's encoder with the StatelessProgressiveDecoder.
# Due to DCVC's internal relative imports, we use a simplified approach:
# 1. Load DCVC checkpoint weights directly with torch.load
# 2. Re-implement the encoder forward pass using the loaded weights
# 3. Extract y_hat for training the progressive decoder
#
# The progressive decoder is STATELESS — it takes only y_hat as input.
# Temporal context is encoded in y_hat by DCVC's entropy model.

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import DCVC_net from the DCVC-repo checkout
# ---------------------------------------------------------------------------
_DCVC_REPO = Path(__file__).parent.parent.parent / 'DCVC-repo' / 'DCVC-family' / 'DCVC'
if _DCVC_REPO.exists() and str(_DCVC_REPO) not in sys.path:
    sys.path.insert(0, str(_DCVC_REPO))

try:
    from src.models.DCVC_net import DCVC_net
except ImportError as _e:
    raise ImportError(
        f"Cannot import DCVC_net. Expected DCVC-repo at:\n  {_DCVC_REPO}\n"
        f"Original error: {_e}"
    ) from _e



def load_dcvc_state_dict(checkpoint_path, device='cuda'):
    """Load DCVC state dict from checkpoint.

    Handles different checkpoint formats from DCVC-family.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def get_default_checkpoint():
    """Find default DCVC checkpoint path."""
    possible_paths = [
        Path(__file__).parent.parent.parent / 'DCVC-repo' / 'DCVC-family' / 'DCVC' / 'checkpoints' / 'extracted' / 'model_dcvc_quality_3_psnr.pth',
        Path(__file__).parent.parent.parent / 'checkpoints' / 'model_dcvc_quality_3_psnr.pth',
    ]
    for p in possible_paths:
        if p.exists():
            return str(p)
    return None


# =============================================================================
# DCVC Encoder Re-implementation
# =============================================================================
# The following classes re-implement DCVC's encoding path to extract y_hat.
# This avoids the complex import chain of DCVC-family's package structure.
#
# DCVC encoding flow:
#     input_image + referframe → ME → MC → temporal prior
#                            → contextualEncoder → feature
#                            → priorEncoder → z
#                            → compress_ar(feature) → y_hat
#
# We extract y_hat after torch.round(feature) as a quantized latent approximation.
# This matches the forward pass quantization used in DCVC's training.
# =============================================================================


class DCVCEncoderExtractor(nn.Module):
    """Frozen DCVC encoder + entropy model for extracting y_hat.

    This module wraps DCVC's encode path and extracts the quantized latent
    y_hat at the boundary where it enters the entropy coding / decoder.

    IMPORTANT: We intentionally extract y_hat BEFORE the bitstream encoding,
    not after decoding. This is because:
    1. Training needs gradients — bitstream encoding is non-differentiable
    2. Forward quantization (torch.round) gives a good approximation of y_hat
    3. The entropy model is frozen anyway, so we can use forward pass directly

    The key insight: DCVC's auto-regressive entropy model uses y_hat itself
    (quantized from feature) to predict distributions. We replicate this
    by quantizing in the forward pass.
    """

    def __init__(self, checkpoint_path=None, device='cuda'):
        super().__init__()

        # Load DCVC model
        self.dcvc = DCVC_net()

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading DCVC checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'state_dict' in checkpoint:
                self.dcvc.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.dcvc.load_state_dict(checkpoint, strict=False)
        else:
            print("Warning: No DCVC checkpoint loaded. Using random initialization.")
            print(f"Expected checkpoint at: {checkpoint_path}")

        # Move to device and freeze
        self.dcvc = self.dcvc.to(device)
        self._freeze_dcvc()

        # Channel dimension for latent
        self.out_channel_M = self.dcvc.out_channel_M  # 96 for DCVC

        print(f"DCVC encoder loaded. out_channel_M = {self.out_channel_M}")

    def _freeze_dcvc(self):
        """Freeze all DCVC parameters."""
        for param in self.dcvc.parameters():
            param.requires_grad = False
        # Verify no gradients
        for name, param in self.dcvc.named_parameters():
            assert param.requires_grad is False, f"Parameter {name} still has gradients"

    def forward(self, input_image, referframe):
        """Extract y_hat from DCVC encoder.

        Args:
            input_image: (B, 3, H, W) current frame
            referframe: (B, 3, H, W) reference frame

        Returns:
            y_hat: (B, 96, H/16, W/16) quantized latent for progressive decoding
        """
        # This replicates the forward pass quantization from DCVC_net.forward()
        # but only extracts y_hat, not the full reconstruction

        # Motion estimation
        estmv = self.dcvc.opticFlow(input_image, referframe)
        mvfeature = self.dcvc.mvEncoder(estmv)
        z_mv = self.dcvc.mvpriorEncoder(mvfeature)
        compressed_z_mv = torch.round(z_mv)

        # Motion compensation
        quant_mv = torch.round(mvfeature)
        params_mv = self.dcvc.mvpriorDecoder(compressed_z_mv)
        ctx_params_mv = self.dcvc.auto_regressive_mv(quant_mv)
        gaussian_params_mv = self.dcvc.entropy_parameters_mv(
            torch.cat((params_mv, ctx_params_mv), dim=1)
        )
        means_hat_mv, scales_hat_mv = gaussian_params_mv.chunk(2, 1)

        quant_mv_upsample = self.dcvc.mvDecoder_part1(quant_mv)
        quant_mv_upsample_refine = self.dcvc.mv_refine(referframe, quant_mv_upsample)
        context = self.dcvc.motioncompensation(referframe, quant_mv_upsample_refine)

        # Temporal prior
        temporal_prior_params = self.dcvc.temporalPriorEncoder(context)

        # Main encoding path
        feature = self.dcvc.contextualEncoder(
            torch.cat((input_image, context), dim=1)
        )
        z = self.dcvc.priorEncoder(feature)
        compressed_z = torch.round(z)

        # Prior params from hyperprior
        params = self.dcvc.priorDecoder(compressed_z)

        # Quantize feature to get y_hat
        compressed_y_renorm = torch.round(feature)

        # Context and gaussian params for BPP calculation
        ctx_params = self.dcvc.auto_regressive(compressed_y_renorm)
        gaussian_params = self.dcvc.entropy_parameters(
            torch.cat((temporal_prior_params, params, ctx_params), dim=1)
        )
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        # Calculate exact BPP using entropy models
        total_bits_y, _ = self.dcvc.feature_probs_based_sigma(feature, means_hat, scales_hat)
        total_bits_mv, _ = self.dcvc.feature_probs_based_sigma(mvfeature, means_hat_mv, scales_hat_mv)
        total_bits_z, _ = self.dcvc.iclr18_estrate_bits_z(compressed_z)
        total_bits_z_mv, _ = self.dcvc.iclr18_estrate_bits_z_mv(compressed_z_mv)

        B, C, H, W = input_image.size()
        pixel_num = B * H * W
        bpp_y = total_bits_y / pixel_num
        bpp_z = total_bits_z / pixel_num
        bpp_mv_y = total_bits_mv / pixel_num
        bpp_mv_z = total_bits_z_mv / pixel_num
        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z

        return {
            'y_hat': compressed_y_renorm,
            'bpp': bpp
        }


class DCVCProgressiveModel(nn.Module):
    """Full model: frozen DCVC encoder + stateless progressive decoder.

    Architecture:
        input_image + referframe → DCVC encoder → y_hat (B, 96, H/16, W/16)
                                 → StatelessProgressiveDecoder → [Recon_1, ..., Recon_4]

    Training:
        - DCVC encoder/entropy model: FROZEN (no gradients)
        - Progressive decoder: TRAINABLE (gradients flow here)

    The progressive decoder is stateless — it takes only y_hat, not context.
    This is the key difference from DCVC's original decoder which concatenates
    context with upsampled features.

    Args:
        dcvc_checkpoint: Path to DCVC pre-trained checkpoint
        num_refinement_blocks: Number of progressive refinement stages (default 3)
        residual_scale: Residual scaling factor for training stability (default 0.1)
        device: Device to run on
    """
    def __init__(self, dcvc_checkpoint=None, num_refinement_blocks=3,
                 residual_scale=0.1, device='cuda'):
        super().__init__()

        # Frozen DCVC encoder
        self.encoder = DCVCEncoderExtractor(
            checkpoint_path=dcvc_checkpoint,
            device=device
        )

        # Trainable progressive decoder
        from progressive_decoder import ProgressiveDecoder
        self.decoder = ProgressiveDecoder(
            out_channel_M=self.encoder.out_channel_M,
            num_refinement_blocks=num_refinement_blocks,
        )

        print(f"\nModel created:")
        print(f"  - DCVC encoder: FROZEN, out_channel_M={self.encoder.out_channel_M}")
        print(f"  - Progressive decoder: TRAINABLE, {num_refinement_blocks} refinement blocks")

    def forward(self, current, ref=None, depth=None):
        """Forward pass through DCVC encoder + progressive decoder.

        Args:
            current: (B, 3, H, W) current frame
            ref: (B, 3, H, W) reference frame (required for P-frame mode)
            depth: Number of progressive stages (1-4, None=all)

        Returns:
            dict with:
                - 'reconstructions': list of RGB tensors at each depth
                - 'y_hat': the latent tensor from DCVC encoder
        """
        if ref is None:
            raise ValueError("DCVC requires a reference frame for P-frame encoding")

        # Extract y_hat and bpp from frozen DCVC encoder
        enc_out = self.encoder(current, ref)
        y_hat = enc_out['y_hat']
        bpp = enc_out['bpp']

        # Progressive decode (only decoder has gradients)
        reconstructions = self.decoder(y_hat, depth=depth)

        return {
            'reconstructions': reconstructions,
            'y_hat': y_hat,
            'bpp': bpp
        }

    def get_trainable_params(self):
        """Get only trainable parameters (progressive decoder)."""
        return self.decoder.parameters()

    def freeze_encoder(self):
        """Ensure encoder is frozen."""
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder (for fine-tuning scenarios)."""
        for param in self.encoder.parameters():
            param.requires_grad = True


class DCVCCompressor:
    """Standalone DCVC compression utility.

    Provides an easy interface for compressing/decompressing video frames
    using DCVC's encoding + our progressive decoding.

    Example:
        compressor = DCVCCompressor(checkpoint_path='checkpoints/model.pth')
        result = compressor.compress(current_frame, reference_frame, depth=4)
        # result = {reconstructions: [...], y_hat: tensor}
    """

    def __init__(self, checkpoint_path=None, num_refinement_blocks=3,
                 residual_scale=0.1, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Load model
        self.model = DCVCProgressiveModel(
            dcvc_checkpoint=checkpoint_path,
            num_refinement_blocks=num_refinement_blocks,
            residual_scale=residual_scale,
            device=self.device
        )
        self.model.eval()

        # Channel info
        self.out_channel_M = self.model.encoder.out_channel_M

    def compress(self, current, ref, depth=None):
        """Compress a frame pair and return progressive reconstructions.

        Args:
            current: (B, 3, H, W) current frame
            ref: (B, 3, H, W) reference frame
            depth: Progressive decode depth (1-4, None=all)

        Returns:
            dict with reconstructions and y_hat
        """
        with torch.no_grad():
            current = current.to(self.device)
            ref = ref.to(self.device)
            return self.model(current, ref, depth=depth)

    def decompress_from_bitstream(self, bitstream, ref, height, width, depth=None):
        """Decompress from a DCVC bitstream using progressive decoder.

        Args:
            bitstream: bytes from DCVC encoding
            ref: reference frame tensor
            height, width: original frame dimensions
            depth: progressive decode depth

        Returns:
            list of reconstructions at each depth
        """
        raise NotImplementedError(
            "Bitstream decompression requires integrating DCVC's decompress_ar() "
            "and entropy decoding. For training, use compress() instead which "
            "extracts y_hat from the forward pass."
        )


def get_default_checkpoint():
    """Find default DCVC checkpoint path."""
    possible_paths = [
        Path(__file__).parent.parent.parent / 'DCVC-repo' / 'DCVC-family' / 'DCVC' / 'checkpoints' / 'extracted' / 'model_dcvc_quality_3_psnr.pth',
        Path(__file__).parent.parent.parent / 'checkpoints' / 'model_dcvc_quality_3_psnr.pth',
    ]
    for p in possible_paths:
        if p.exists():
            return str(p)
    return None