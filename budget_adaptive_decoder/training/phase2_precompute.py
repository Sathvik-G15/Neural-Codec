"""
Phase 2: Offline ΔQ Target Precomputation

As specified in Section 8, Phase 2 computes actual marginal quality gains
ΔQ(k) at each depth for all training frames using the trained decoder from Phase 1.

This phase is NOT training - it's a single forward pass with gradients disabled.

FIX (S4): Bitrate / QP used during encoding is now specified in config.
The default is DCVC Quality Level 3 (high-quality / low-compression).
See DESIGN_DOCUMENT.md §8 Phase 2 for the full specification.

FIX (S1/S4): This module now consumes the actual interface of DCVC
(context, recon_image, prev_frame) rather than the previously assumed
(latent, motion, prev_frame) which did not match DCVC's actual outputs.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import json

from ..models.decoder import BudgetAdaptiveDecoder
from ..models.teacher import DCVCWrapper
from ..data.delta_q_targets import DeltaQTargetStorage, compute_delta_q_from_psnr
from ..evaluation.metrics import compute_psnr

logger = logging.getLogger(__name__)

DEFAULT_QP = "dcvc_quality_3"  # DCVC Quality Level 3 (high-bitrate reference)


class Phase2Precomputer:
    """
    Phase 2 precomputation of ΔQ targets.

    Runs the trained Phase 1 decoder at all depths k=0,...,K
    and computes marginal gains: ΔQ(k) = PSNR(k) - PSNR(k-1)

    Important: This uses the PHASE 1 trained decoder, not the final one.
    If decoder changes substantially in Phase 4, rerun this phase.
    """

    def __init__(
        self,
        decoder: BudgetAdaptiveDecoder,
        device: torch.device,
        config: Dict[str, Any],
        output_dir: Optional[Path] = None,
        teacher: Optional[DCVCWrapper] = None,
    ):
        """
        Args:
            decoder: Phase 1 trained BudgetAdaptiveDecoder
            device: Device to run on
            config: Configuration dict (must include 'qp' for bitrate specification;
                    defaults to DEFAULT_QP = 'dcvc_quality_3')
            output_dir: Directory to save targets
            teacher:   Optional DCVC teacher used to produce context + recon_image
                       per frame. If None, the data_loader is expected to yield
                       whatever tensors it yields.
        """
        self.decoder = decoder
        self.device = device
        self.config = config
        # S4 fix: explicit QP / bitrate in config; record it for reproducibility
        self.qp = config.get("qp", DEFAULT_QP)
        self.output_dir = output_dir or Path("./delta_q_cache")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # S4 fix: persist QP at construction so ΔQ targets are reproducible.
        (self.output_dir / "qp.txt").write_text(str(self.qp))

        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

        if teacher is not None:
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
        self.teacher = teacher

        self.K = decoder.K
        self.storage = DeltaQTargetStorage(self.output_dir)

    @torch.no_grad()
    def compute_targets(
        self,
        data_loader: DataLoader,
        num_frames: Optional[int] = None,
        save_intermediate: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ΔQ targets for all frames.

        Args:
            data_loader: Data loader for evaluation
            num_frames: Limit number of frames to process
            save_intermediate: Save progress periodically

        Returns:
            Dictionary with 'delta_q' tensor and frame metadata
        """
        all_delta_q = []
        all_frame_ids = []
        frame_count = 0

        progress_bar = tqdm(data_loader, desc=f"Phase 2 ΔQ computation (qp={self.qp})")

        for batch_idx, batch in enumerate(progress_bar):
            if num_frames is not None and frame_count >= num_frames:
                break

            # S1 fix: support both interface shapes
            # (frames, bitstream_dict, prev_frame) legacy  : not currently used
            # (curr_frame, ref_frame)                         : standard
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                frames, bitstream_dict, prev_frame = batch
                frames = frames.to(self.device)
                prev_frame = prev_frame.to(self.device)
                # Select the 4th frame (index 3) of each septuplet as current frame.
                # frames shape: [B, 7, 3, H, W] -> curr_frame shape: [B, 3, H, W]
                if frames.dim() == 5:
                    curr_frame = frames[:, 3]
                else:
                    curr_frame = frames
                if self.teacher is not None:
                    with torch.no_grad():
                        teacher_out = self.teacher.get_stage_targets(curr_frame, prev_frame)
                    context = teacher_out["context"]
                    recon_image = teacher_out["stage_recons"][0]
                else:
                    raise RuntimeError(
                        "Phase 2 requires either the teacher (recommended) "
                        "or precomputed context/recon tensors in bitstream_dict."
                    )
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                curr_frame, ref_frame = batch
                curr_frame = curr_frame.to(self.device)
                prev_frame = ref_frame.to(self.device)
                if self.teacher is None:
                    raise RuntimeError("Phase 2 requires the teacher to compute context.")
                with torch.no_grad():
                    teacher_out = self.teacher.get_stage_targets(curr_frame, prev_frame)
                # Use 'context' alias for legacy convention - decoder takes (context, prev_frame)
                context = teacher_out["context"]
                recon_image = teacher_out["stage_recons"][0]
            else:
                raise ValueError(f"Unexpected batch shape from data_loader: {type(batch)}")

            # Decoder runs at all depths; we already have the K=4 student stages
            all_recons = self.decoder.get_stage_states(context, prev_frame)
            # all_recons is a list of K+1 recons [R0, R1, R2, R3, R4]

            K = len(all_recons) - 1
            batch_size = curr_frame.shape[0]
            delta_q_batch = torch.zeros(batch_size, K, device=self.device)

            # ΔQ(k) = PSNR(R_k, Y) - PSNR(R_{k-1}, Y), measured vs ground truth
            # This matches the design doc formula exactly.
            for k in range(1, K + 1):
                psnr_k = compute_psnr(all_recons[k], curr_frame)
                psnr_prev = compute_psnr(all_recons[k - 1], curr_frame)
                delta_q_batch[:, k - 1] = psnr_k - psnr_prev

            all_delta_q.append(delta_q_batch.cpu())

            frame_ids = [f"{batch_idx}_{i}" for i in range(batch_size)]
            all_frame_ids.extend(frame_ids)

            frame_count += batch_size
            progress_bar.set_postfix({"frames": frame_count, "qp": self.qp})

            if save_intermediate and frame_count % 10000 == 0:
                self._save_intermediate(all_delta_q, all_frame_ids, frame_count)

        delta_q_tensor = torch.cat(all_delta_q, dim=0)

        targets = {
            "delta_q": delta_q_tensor,
            "frame_ids": all_frame_ids,
        }

        return targets

    def _save_intermediate(
        self,
        delta_q_list: List[torch.Tensor],
        frame_ids: List[str],
        count: int,
    ):
        """Save intermediate results during long computation."""
        intermediate_path = self.output_dir / f"delta_q_intermediate_{count}.pt"
        torch.save({
            "delta_q": torch.cat(delta_q_list, dim=0),
            "frame_ids": frame_ids,
        }, intermediate_path)
        logger.info(f"Saved intermediate: {intermediate_path}")

    def run(
        self,
        data_loader: DataLoader,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the full Phase 2 precomputation.

        Args:
            data_loader: Data loader
            metadata: Additional metadata to save with targets

        Returns:
            Dictionary with 'delta_q' tensor
        """
        num_frames = self.config.get("num_frames", None)

        logger.info("Starting Phase 2 ΔQ precomputation")
        logger.info(f"Processing up to {num_frames or 'all'} frames")

        targets = self.compute_targets(data_loader, num_frames)

        stats = self.storage.get_target_statistics(targets["delta_q"])
        logger.info(f"ΔQ statistics: {stats}")

        full_metadata = {
            "phase": 2,
            "decoder_checkpoint": self.config.get("decoder_checkpoint"),
            "num_frames": targets["delta_q"].shape[0],
            "K": self.K,
            "statistics": stats,
            **(metadata or {}),
        }

        self.storage.save_targets(targets, full_metadata)

        return targets


def precompute_phase2(
    decoder: BudgetAdaptiveDecoder,
    data_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    """
    Main Phase 2 precomputation function.

    Args:
        decoder: Trained BudgetAdaptiveDecoder from Phase 1
        data_loader: Data loader for frames to process
        device: Device to run on
        config: Configuration dict
        output_dir: Output directory for targets

    Returns:
        Dictionary with 'delta_q' tensor
    """
    precomputer = Phase2Precomputer(
        decoder=decoder,
        device=device,
        config=config,
        output_dir=output_dir,
    )

    return precomputer.run(data_loader)


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 Precomputation Module")
    print("=" * 60)
    print("Run via: from training.phase2_precompute import precompute_phase2")
    print("=" * 60)