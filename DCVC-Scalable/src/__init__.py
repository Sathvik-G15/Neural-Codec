"""
DCVC Progressive - Compute-Scalable Neural Video Decoding

This package implements a progressive residual decoder for compute-adaptive
neural video streaming, integrated with Microsoft's DCVC codec.

Main components:
    - stateless_progressive_decoder.py: Research-plan-compliant stateless
      progressive decoder (y_hat input only, no context)
    - dcvc_integration.py: Integration layer with frozen DCVC encoder
    - train_progressive.py: Training script with deep supervision

Legacy (custom encoder, NOT bitstream-compatible):
    - progressive_decoder.py: Original custom progressive decoder
    - dcvc_progressive.py: Full custom encoder + progressive decoder model
    - train.py: Training script with deep supervision
"""

from .progressive_decoder import (
    ProgressiveDecoder,
    RefinementBlock,
    BaseDecoder,
    GDN,
)
from .dcvc_integration import DCVCProgressiveModel, DCVCCompressor, get_default_checkpoint

__version__ = '2.0.0'
__all__ = [
    'ProgressiveDecoder',
    'RefinementBlock',
    'BaseDecoder',
    'GDN',
    'DCVCProgressiveModel',
    'DCVCCompressor',
    'get_default_checkpoint',
]