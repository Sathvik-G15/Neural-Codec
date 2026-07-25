"""
DCVC Wrapper module for Budget-Constrained Neural Decoder.

Uses the ORIGINAL Microsoft DCVC model which has single-stage output.
The checkpoint was trained with the original DCVC architecture.

Path resolution order (first non-empty wins):
    1. explicit constructor arg (pretrained_path or dcvc_src_path)
    2. environment variable DCVC_PRETRAINED_PATH / DCVC_SRC_PATH
    3. hardcoded defaults matching the local DCVC-original layout
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn


class DCVCWrapper(nn.Module):
    """
    Frozen DCVC teacher with Option A interface.

    The original DCVC produces a SINGLE reconstruction (K=1).
    The student decoder has K=4 stages for learned refinement.
    During Phase 1, only stage 1 has a direct teacher target.
    Stages 2-4 learn to improve upon stage 1's output.

    Interface contract (Option A):
        Input:  raw current frame + raw reference frame (both [B,3,H,W])
        Output: dict with reconstruction and intermediate representations
    """

    # Original DCVC is single-stage (1 output)
    DCVC_STAGE_COUNT = 1
    DEFAULT_DCVC_SRC_PATH = (
        Path(__file__).parent.parent.parent
        / "DCVC-original"
        / "DCVC-family"
        / "DCVC"
    )
    DEFAULT_DCVC_CHECKPOINT_PATH = (
        Path(__file__).parent.parent.parent
        / "DCVC-repo"
        / "DCVC-family"
        / "DCVC"
        / "checkpoints"
        / "model_dcvc_quality_3_psnr.pth"
    )

    def __init__(
        self,
        device: Optional[torch.device] = None,
        load_pretrained: bool = True,
        dcvc_src_path: Optional[str] = None,
        pretrained_path: Optional[str] = None,
    ):
        """
        Args:
            device: torch device for the teacher.
            load_pretrained: whether to load the .pth.tar checkpoint.
            dcvc_src_path: directory containing DCVC's `src/` tree
                (the directory whose `src/models/DCVC_net.py` is importable).
                Defaults to local DCVC-original/, or env var DCVC_SRC_PATH.
            pretrained_path: absolute path to model_dcvc_quality_3_psnr.pth
                (or any DCVC-style state dict). Defaults to local
                DCVC-repo/.../checkpoints/, or env var DCVC_PRETRAINED_PATH.
        """
        super().__init__()

        # Resolve src / pretrained paths with priority order:
        #   explicit arg > env var > repo-relative default
        default_src = str(self.DEFAULT_DCVC_SRC_PATH)
        default_ckpt = str(self.DEFAULT_DCVC_CHECKPOINT_PATH)
        self.dcvc_src_path = (
            dcvc_src_path
            or os.environ.get("DCVC_SRC_PATH")
            or default_src
        )
        self.pretrained_path = (
            pretrained_path
            or os.environ.get("DCVC_PRETRAINED_PATH")
            or default_ckpt
        )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dcvc = None
        self._load_dcvc(load_pretrained)

    def _load_dcvc(self, load_pretrained: bool):
        """Load DCVC model from original Microsoft DCVC repo.

        The src directory must be importable on sys.path so that
        `from src.models.DCVC_net import DCVC_net` resolves. This is
        how DCVC's own code expects to be imported.
        """
        try:
            dcvc_repo_src = str(Path(self.dcvc_src_path) / "src")
            if dcvc_repo_src not in sys.path:
                sys.path.insert(0, dcvc_repo_src)
            sys_path_parent = str(Path(self.dcvc_src_path))
            if sys_path_parent not in sys.path:
                sys.path.insert(0, sys_path_parent)

            from src.models.DCVC_net import DCVC_net

            self.dcvc = DCVC_net()

            if load_pretrained:
                checkpoint_path = Path(self.pretrained_path)
                if checkpoint_path.exists():
                    state_dict = torch.load(
                        checkpoint_path,
                        map_location=self.device,
                        weights_only=False
                    )
                    # Match DCVC's load_dict: it handles both raw and
                    # `state_dict=` wrapped checkpoints.
                    if isinstance(state_dict, dict):
                        for k in ("state_dict", "model_state_dict", "model"):
                            if k in state_dict:
                                state_dict = state_dict[k]
                                break
                    self.dcvc.load_dict(state_dict)
                    print(
                        f"Successfully loaded DCVC pretrained weights from "
                        f"{checkpoint_path}"
                    )
                else:
                    print(f"DCVC checkpoint not found at {checkpoint_path}")

            self.dcvc = self.dcvc.to(self.device)
            self._freeze_all_parameters()

        except ImportError as e:
            print(f"Warning: Could not import DCVC from {self.dcvc_src_path}: {e}")
            self.dcvc = None
        except Exception as e:
            print(f"Warning: Could not load DCVC: {e}")
            self.dcvc = None

    def _freeze_all_parameters(self):
        """Freeze all parameters - teacher is never trained."""
        if self.dcvc is not None:
            for param in self.dcvc.parameters():
                param.requires_grad_(False)
            self.dcvc.eval()

    def train(self, mode=True):
        """Override to prevent accidental unfreezing."""
        return self

    @torch.no_grad()
    def get_stage_targets(self, curr_frame: torch.Tensor, ref_frame: torch.Tensor) -> Dict:
        """
        Primary interface for Phase 1 distillation.

        Original DCVC only produces ONE output (final reconstruction).
        For the student's K=4 stages, we provide:
        - stage_recons[0] = base reconstruction (same as final, for stage 1 target)
        - stage_recons[1..3] = same final recon (stages 2-4 learn to improve upon stage 1)

        Args:
            curr_frame: [B, 3, H, W] current frame, pixels in [0,1]
            ref_frame:  [B, 3, H, W] reference frame, pixels in [0,1]

        Returns:
            dict:
                'stage_recons': list of 4 tensors [B,3,H,W]
                                All contain the same DCVC reconstruction
                                (student has no multi-stage teacher)
                'latent':       quantized latent tensor [B, 96, H/16, W/16]
                'motion':       motion representation [B, 128, H/4, W/4]
                'context':      motion-compensated context [B, 64, H/4, W/4]
                'base_recon':   base reconstruction (same as stage_recons[0])
        """
        if self.dcvc is None:
            return self._stub_get_stage_targets(curr_frame, ref_frame)

        self.dcvc.eval()
        result = self.dcvc(ref_frame, curr_frame)

        recon_image = result['recon_image']
        context = result.get('context', recon_image)

        # Original DCVC produces one output
        # For student K=4 stages, all 4 stages target the same teacher output
        stage_recons = [recon_image] * 4

        # Latent is at H/16 resolution (from contextualEncoder output)
        # Motion is upsampled to H/4
        latent = result.get('feature_renorm')
        motion = result.get('quant_mv_upsample_refine', context)

        return {
            'stage_recons': stage_recons,  # All 4 same - teacher is single-stage
            'latent': latent,
            'motion': motion,
            'context': context,
            'base_recon': recon_image
        }

    @torch.no_grad()
    def get_encoded_representations(self, curr_frame: torch.Tensor, ref_frame: torch.Tensor) -> Dict:
        """Returns latent and motion tensors for student decoder."""
        if self.dcvc is None:
            return {'latent': None, 'motion': None}

        self.dcvc.eval()
        result = self.dcvc(ref_frame, curr_frame)

        return {
            'latent': result.get('feature_renorm'),
            'motion': result.get('quant_mv_upsample_refine'),
            'context': result.get('context'),
        }

    def _stub_get_stage_targets(self, curr_frame, ref_frame):
        """Stub when DCVC is not available."""
        B, C, H, W = curr_frame.shape
        recon = ref_frame.clone()

        return {
            'stage_recons': [recon.clone()] * 4,
            'latent': torch.randn(B, 96, H//16, W//16, device=curr_frame.device),
            'motion': torch.randn(B, 128, H//4, W//4, device=curr_frame.device),
            'context': torch.randn(B, 64, H//4, W//4, device=curr_frame.device),
            'base_recon': recon.clone()
        }

    def forward(self, curr_frame: torch.Tensor, ref_frame: torch.Tensor) -> torch.Tensor:
        """Forward pass: return reconstruction."""
        result = self.get_stage_targets(curr_frame, ref_frame)
        return result['stage_recons'][0]


def verify_teacher_is_frozen():
    """Verify that the teacher has all parameters frozen."""
    import torch

    teacher = DCVCWrapper(load_pretrained=True)

    print("=" * 60)
    print("DCVCWrapper Frozen Parameter Verification")
    print("=" * 60)

    if teacher.dcvc is not None:
        frozen_count = sum(1 for p in teacher.dcvc.parameters() if not p.requires_grad)
        trainable_count = sum(1 for p in teacher.dcvc.parameters() if p.requires_grad)
        print(f"DCVC frozen parameters: {frozen_count}")
        print(f"DCVC trainable parameters: {trainable_count}")

        if trainable_count > 0:
            print("WARNING: Teacher has trainable parameters!")
        else:
            print("PASSED: Teacher is fully frozen")

        print("\nRunning forward pass test...")
        device = teacher.device
        B, C, H, W = 1, 3, 256, 448
        curr = torch.randn(B, C, H, W, device=device)
        ref = torch.randn(B, C, H, W, device=device)

        result = teacher.get_stage_targets(curr, ref)
        print(f"get_stage_targets returned:")
        print(f"  stage_recons: {len(result['stage_recons'])} items")
        for i, sr in enumerate(result['stage_recons']):
            print(f"    stage_recons[{i}]: {sr.shape}")
        print(f"  latent: {result['latent'].shape if result['latent'] is not None else None}")
        print(f"  motion: {result['motion'].shape if result['motion'] is not None else None}")
        print(f"  context: {result['context'].shape if result['context'] is not None else None}")
    else:
        print("DCVC not loaded (stub mode)")

    print("\n" + "=" * 60)
    print("Teacher verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    verify_teacher_is_frozen()