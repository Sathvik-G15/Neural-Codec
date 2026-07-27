"""
Phase 1: Budget-Adaptive Decoder Distillation

As specified in Section 9 (revised for K=1 teacher), Phase 1 trains the
Budget-Adaptive Decoder with a hybrid supervision strategy:

Stage 1: Teacher-supervised (MSE to DCVC reconstruction)
Stages 2-4: Self-supervised (MSE to ground truth with increasing weights)

The teacher is a frozen DCVC that produces:
    - teacher_recon: DCVC's reconstruction [B, 3, H, W]
    - context: Motion-compensated features [B, 64, H/4, W/4]

The student decoder takes (context, prev_frame) and produces
4 progressive refinements R_1, R_2, R_3, R_4.

CLI Usage (Kaggle)
------------------
    python -m budget_adaptive_decoder.training.phase1_distill \
        --pretrained  /kaggle/input/.../model_dcvc_quality_3_psnr.pth \
        --data_root   /kaggle/input/.../vimeo_septuplet \
        --output_dir  /kaggle/working/checkpoints/phase1 \
        --batch_size 2 --epochs 30 --lr 1e-4
"""

import argparse
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ..models.decoder import BudgetAdaptiveDecoder
from ..models.teacher import DCVCWrapper


def _log(msg, *, err=False):
    print(msg, file=sys.stderr if err else sys.stdout, flush=True)


# Stage weights for ground truth supervision (stages 2, 3, 4)
GT_WEIGHTS = {2: 0.5, 3: 0.75, 4: 1.0}


# Maximum wall-clock training time. The script gracefully stops and saves a
# checkpoint once this budget is exceeded (Kaggle's session limit is 12h).
MAX_TRAINING_SECONDS = 12 * 60 * 60


# Cadence for periodic side-disk checkpoints. We overwrite the SAME file every
# time so that disk usage stays bounded.
CHECKPOINT_CADENCE_BATCHES = 500
SIDE_CKPT_FILENAME = "phase1_latest.pt"


# Cadence for in-loop progress logs (per-batch). One line every N batches to
# respect Kaggle's 20MB log cap.
LOG_CADENCE_BATCHES = 1000


# Middle frame of the septuplet matches the Phase 2 / Phase 4 indexing.
# We pick index 3 of frames [7, 3, H, W] (1-indexed: 4th frame).
CURR_FRAME_INDEX = 3


def _unpack_batch(batch):
    """
    Normalise a batch into (curr_frame, ref_frame) regardless of dataset.

    Supports both:
      - Vimeo90kDataset  →  (frames[B,7,3,H,W], bitstream_dict, prev_frame[B,3,H,W])
      - GenericVideoDataset → (curr_frame[B,3,H,W], ref_frame[B,3,H,W])

    For the 3-tuple form we extract:
        curr_frame = frames[:, CURR_FRAME_INDEX, :, :]  (middle frame, the predicted one)
        ref_frame  = prev_frame                         (frame im1 of the septuplet)
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        frames, _bitstream_dict, prev_frame = batch
        curr_frame = frames[:, CURR_FRAME_INDEX]
        ref_frame = prev_frame
        return curr_frame, ref_frame

    # Fall through: 2-tuple (curr_frame, ref_frame)
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        curr_frame, ref_frame = batch
        return curr_frame, ref_frame

    raise ValueError(
        f"Unsupported batch format from dataloader: "
        f"type={type(batch).__name__}, len={len(batch) if hasattr(batch, '__len__') else 'n/a'}"
    )


class Phase1Trainer:
    """
    Trainer for Phase 1: Hybrid Supervision Distillation.

    Stage 1: supervised by DCVC teacher reconstruction (MSE)
    Stages 2-4: supervised by ground truth (weighted MSE)

    The teacher produces a SINGLE reconstruction (DCVC has K=1).
    Stages 2-4 learn to EXCEED the teacher by refining toward ground truth.
    """

    def __init__(
        self,
        decoder: BudgetAdaptiveDecoder,
        teacher: DCVCWrapper,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        checkpoint_dir: Optional[Path] = None,
        log_every: Optional[int] = None,
        ckpt_every_batches: Optional[int] = None,
        max_training_seconds: Optional[float] = None,
    ):
        """
        Args:
            decoder: BudgetAdaptiveDecoder (student)
            teacher: DCVCWrapper (frozen teacher)
            optimizer: Optimizer for decoder
            device: Device to train on
            config: Training configuration dict
            checkpoint_dir: Directory to save checkpoints
            log_every: Print a training log line every N batches (default 1000).
                       Set to 0 or None to disable mid-epoch logging (will only log
                       per-epoch + validation summary).
            ckpt_every_batches: Save a "latest" checkpoint every N batches,
                       overwriting the previous one (default 500). Set 0 to disable.
            max_training_seconds: Hard wall-clock time budget. Training is stopped
                       (with a final checkpoint) once the budget is exhausted.
                       Default = 12 hours.
        """
        self.decoder = decoder
        self.teacher = teacher
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._verify_teacher_frozen()

        self.K = decoder.K
        self.patience = config.get("patience", 10)
        self.min_delta = config.get("min_delta", 0.001)
        self.best_loss = float("inf")
        self.wait = 0

        # Per-batch logging cadence. Default to 1000 to keep the log under
        # Kaggle's 20MB cap; user-configurable via --log_every or config.
        if log_every is None:
            log_every = int(config.get("log_every", LOG_CADENCE_BATCHES))
        self.log_every = max(int(log_every), 0)
        self._tqdm_disabled = True  # we never use tqdm in this trainer

        # Side checkpoint cadence: every N batches we save and OVERWRITE the
        # same "latest" file to keep disk usage bounded.
        if ckpt_every_batches is None:
            ckpt_every_batches = int(
                config.get("ckpt_every_batches", CHECKPOINT_CADENCE_BATCHES)
            )
        self.ckpt_every_batches = max(int(ckpt_every_batches), 0)
        self.side_ckpt_path = self.checkpoint_dir / SIDE_CKPT_FILENAME

        # Wall-clock budget. Kaggle sessions terminate at ~12h, so default to
        # that ceiling. The trainer stops as soon as the budget is hit.
        if max_training_seconds is None:
            max_training_seconds = float(
                config.get("max_training_seconds", MAX_TRAINING_SECONDS)
            )
        self.max_training_seconds = float(max_training_seconds)
        self.train_start_time = time.time()

        self.loss_history = []
        self.val_loss_history = []

    def _verify_teacher_frozen(self):
        """Verify that the teacher has all parameters frozen."""
        if self.teacher.dcvc is not None:
            trainable_params = sum(1 for p in self.teacher.dcvc.parameters() if p.requires_grad)
            if trainable_params > 0:
                raise ValueError(
                    f"Teacher has {trainable_params} trainable parameters. "
                    "Teacher must be frozen during Phase 1 distillation."
                )
        _log("Teacher verification passed: all parameters frozen")

    def compute_phase1_loss(
        self,
        student_recons: List[torch.Tensor],
        teacher_recon: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the revised Phase 1 loss.

        Stage 1: MSE to teacher reconstruction
        Stages 2-4: Weighted MSE to ground truth

        Args:
            student_recons: List of K reconstructions [R_1, ..., R_K]
            teacher_recon: DCVC reconstruction [B, 3, H, W]
            ground_truth: Ground truth frame [B, 3, H, W]

        Returns:
            total_loss: Combined loss
            loss_components: Dict with individual loss values
        """
        # Stage 1: teacher-supervised
        loss_s1 = F.mse_loss(student_recons[0], teacher_recon)

        # Stages 2-4: ground truth-supervised with increasing weights
        loss_gt = torch.tensor(0.0, device=self.device)
        loss_components = {
            'loss_s1_teacher': loss_s1.item(),
            'loss_s2_gt': 0.0,
            'loss_s3_gt': 0.0,
            'loss_s4_gt': 0.0,
        }

        for k in range(1, self.K):
            stage_gt_loss = F.mse_loss(student_recons[k], ground_truth)
            weight = GT_WEIGHTS.get(k + 1, 1.0)
            loss_gt = loss_gt + weight * stage_gt_loss
            loss_components[f'loss_s{k+1}_gt'] = stage_gt_loss.item()

        # Combined loss
        total_loss = loss_s1 + loss_gt
        loss_components['loss_total'] = total_loss.item()
        loss_components['loss_teacher_total'] = loss_s1.item()
        loss_components['loss_gt_total'] = loss_gt.item()

        return total_loss, loss_components

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Tuple[Dict[str, float], bool]:
        """
        Train for one epoch.

        Side effects:
          * Prints a one-line training log every `self.log_every` batches
            (default 1000) to stay under Kaggle's 20MB log cap.
          * Saves a "latest" checkpoint (overwriting the same file) every
            `self.ckpt_every_batches` batches (default 500).
          * Stops early if the wall-clock budget (`self.max_training_seconds`,
            default 12h) is exhausted. The flag returned in the second tuple
            element signals the caller to break out of the outer epoch loop.

        Args:
            train_loader: DataLoader yielding either:
                            - the 3-tuple (frames, bitstream_dict, prev_frame) from
                              Vimeo90kDataset (frames: [B, 7, 3, H, W] septuplet)
                            - or the 2-tuple (curr_frame, ref_frame) from
                              GenericVideoDataset
                In both cases, curr_frame = frame at chosen septuplet position (we
                use the middle frame index = 3, matching Phase 2 / Phase 4 wiring
                for consistency), ref_frame = frames[0].

            epoch: Current epoch number

        Returns:
            (avg_losses, time_budget_exceeded)
        """
        self.decoder.train()
        epoch_losses = {
            'loss_total': 0.0,
            'loss_s1_teacher': 0.0,
            'loss_s2_gt': 0.0,
            'loss_s3_gt': 0.0,
            'loss_s4_gt': 0.0,
        }
        num_batches = 0
        # Discard per-step totals to avoid Python int accumulation overhead.
        # We only emit a log every `log_every` batches.
        log_every = self.log_every
        ckpt_every = self.ckpt_every_batches
        time_budget_exceeded = False

        for batch_idx, batch in enumerate(train_loader, start=1):
            curr_frame, ref_frame = _unpack_batch(batch)
            curr_frame = curr_frame.to(self.device)
            ref_frame = ref_frame.to(self.device)

            self.optimizer.zero_grad()

            # Teacher: encode curr_frame relative to ref_frame
            # Returns teacher_recon + context for student
            with torch.no_grad():
                teacher_out = self.teacher.get_stage_targets(curr_frame, ref_frame)
                teacher_recon = teacher_out['stage_recons'][0]
                context = teacher_out['context']

            # Student: decode using context + ref_frame
            # CRITICAL: use phase1 path that detaches between Stage 1 and Stage 2
            # so Stage 1 receives only the teacher MSE signal (not propagated GT loss)
            student_recons = self.decoder.decode_all_stages_phase1(context, ref_frame)

            # Compute Phase 1 loss
            loss, loss_components = self.compute_phase1_loss(
                student_recons, teacher_recon, curr_frame
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            for k, v in loss_components.items():
                if k in epoch_losses:
                    epoch_losses[k] += v
            num_batches += 1

            # Periodic per-batch log: keep the running loss in a single short
            # string. We only build/print a log line every `log_every` batches.
            if log_every > 0 and (batch_idx % log_every == 0):
                running_loss = epoch_losses['loss_total'] / batch_idx
                elapsed_s = time.time() - self.train_start_time
                _log(
                    f"[trn] ep={epoch} step={batch_idx}/{len(train_loader)} "
                    f"avg_loss={running_loss:.6f} step_loss={loss.item():.6f} "
                    f"elapsed={elapsed_s/3600:.2f}h"
                )

            # Periodic side checkpoint: overwrite a SINGLE file every
            # `ckpt_every_batches` batches. Keep only this latest snapshot on
            # disk to honour the user's space constraint.
            if ckpt_every > 0 and (batch_idx % ckpt_every == 0):
                self.save_latest_batch_checkpoint(
                    epoch=epoch,
                    global_batch_idx=batch_idx,
                    extra={"running_avg_loss": epoch_losses['loss_total'] / batch_idx},
                )

            # Wall-clock budget check. Bail out as soon as the ceiling is hit;
            # the caller will save a final checkpoint and exit.
            if self.max_training_seconds > 0:
                elapsed = time.time() - self.train_start_time
                if elapsed >= self.max_training_seconds:
                    remaining = max(self.max_training_seconds - elapsed, 0)
                    _log(
                        f"[time-budget] reached {self.max_training_seconds}s "
                        f"(elapsed={elapsed/3600:.2f}h, remaining={remaining:.1f}s); "
                        f"stopping at ep={epoch} step={batch_idx}",
                        err=True,
                    )
                    time_budget_exceeded = True
                    break

        # Average losses
        avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        avg_losses['loss_total'] = avg_losses.pop('loss_total')

        self.loss_history.append(avg_losses)

        return avg_losses, time_budget_exceeded

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Accepts both 2-tuple (curr_frame, ref_frame) from
        GenericVideoDataset and 3-tuple (frames, bitstream_dict, prev_frame)
        from Vimeo90kDataset.

        Logs at most once (one-line summary) to stay within Kaggle's
        20MB log cap.
        """
        self.decoder.eval()
        epoch_losses = {
            'loss_total': 0.0,
            'loss_s1_teacher': 0.0,
            'loss_s2_gt': 0.0,
            'loss_s3_gt': 0.0,
            'loss_s4_gt': 0.0,
        }
        num_batches = 0

        for batch_idx, batch in enumerate(val_loader, start=1):
            curr_frame, ref_frame = _unpack_batch(batch)
            curr_frame = curr_frame.to(self.device)
            ref_frame = ref_frame.to(self.device)

            teacher_out = self.teacher.get_stage_targets(curr_frame, ref_frame)
            teacher_recon = teacher_out['stage_recons'][0]
            context = teacher_out['context']

            student_recons = self.decoder.decode_all_stages_phase1(context, ref_frame)

            loss, loss_components = self.compute_phase1_loss(
                student_recons, teacher_recon, curr_frame
            )

            for k, v in loss_components.items():
                if k in epoch_losses:
                    epoch_losses[k] += v
            num_batches += 1

        avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        avg_losses = {f"val_{k}": v for k, v in avg_losses.items()}

        self.val_loss_history.append(avg_losses)

        # Single-line per-stage summary (cheap to print, doesn't burn log budget).
        _log(
            f"[val] batches={num_batches} "
            f"total={avg_losses['val_loss_total']:.4f} "
            f"stage1={avg_losses['val_loss_s1_teacher']:.4f} "
            f"stage2={avg_losses['val_loss_s2_gt']:.4f} "
            f"stage3={avg_losses['val_loss_s3_gt']:.4f} "
            f"stage4={avg_losses['val_loss_s4_gt']:.4f}"
        )

        return avg_losses

    def should_stop_early(self, val_loss: float) -> bool:
        """Check if early stopping criteria are met."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                _log(f"Early stopping triggered after {self.wait} epochs without improvement")
                return True
        return False

    def save_checkpoint(self, epoch: int, metrics: Dict, is_final: bool = False):
        """Save a training checkpoint."""
        checkpoint_name = f"phase1_epoch{epoch}.pt" if not is_final else "phase1_final.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        torch.save({
            "epoch": epoch,
            "decoder_state_dict": self.decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_loss": self.best_loss,
        }, checkpoint_path)

        _log(f"Checkpoint saved: {checkpoint_path}")

    def save_latest_batch_checkpoint(
        self,
        epoch: int,
        global_batch_idx: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save (and overwrite) the single "latest" side checkpoint.

        A space-constrained fallback that is independent of the regular
        per-epoch checkpoint cadence. Always writes to the same file so disk
        usage stays bounded.
        """
        extra = extra or {}
        torch.save({
            "epoch": epoch,
            "global_batch_idx": global_batch_idx,
            "decoder_state_dict": self.decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "extra": extra,
            "elapsed_seconds": time.time() - self.train_start_time,
        }, self.side_ckpt_path)
        _log(
            f"[side-ckpt] ep={epoch} batch={global_batch_idx} -> "
            f"{self.side_ckpt_path.name}"
        )
        return self.side_ckpt_path

    def load_checkpoint(self, checkpoint_path: Path):
        """Load a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        return checkpoint.get("epoch", 0)


def train_phase1(
    decoder: BudgetAdaptiveDecoder,
    teacher: DCVCWrapper,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict[str, Any],
    checkpoint_dir: Optional[Path] = None,
    num_epochs: Optional[int] = None,
    log_every: Optional[int] = 1000,
    ckpt_every_batches: Optional[int] = 500,
    max_training_seconds: Optional[float] = 12 * 60 * 60,
) -> Tuple[BudgetAdaptiveDecoder, Dict]:
    """
    Run Phase 1 training with Kaggle-friendly low-volume logging.

    Args:
        decoder: BudgetAdaptiveDecoder (student)
        teacher: DCVCWrapper (frozen teacher)
        train_loader: Training data loader yielding (curr_frame, ref_frame)
        val_loader: Validation data loader
        optimizer: Optimizer for decoder
        device: Device to train on
        config: Training configuration dict
        checkpoint_dir: Directory for checkpoints
        num_epochs: Number of epochs (default from config)
        log_every: Per-batch log cadence (default 1000, set 0 to disable)
        ckpt_every_batches: Overwrite the "latest" checkpoint every N batches
            (default 500). Set to 0 to disable in-loop checkpoints.
        max_training_seconds: Hard wall-clock budget (default 12h). Training
            stops and saves a final checkpoint once reached.

    Returns:
        Tuple of (trained_decoder, final_metrics)
    """
    num_epochs = num_epochs or config.get("num_epochs", 100)

    trainer = Phase1Trainer(
        decoder=decoder,
        teacher=teacher,
        optimizer=optimizer,
        device=device,
        config=config,
        checkpoint_dir=checkpoint_dir,
        log_every=log_every,
        ckpt_every_batches=ckpt_every_batches,
        max_training_seconds=max_training_seconds,
    )

    n_train_batches = len(train_loader)
    log_every_str = (
        f"{trainer.log_every} batches" if trainer.log_every > 0 else "off"
    )
    _log(
        f"Starting Phase 1 training for {num_epochs} epochs "
        f"({n_train_batches} train batches; logs every {log_every_str}; "
        f"side-ckpt every {trainer.ckpt_every_batches} batches; "
        f"time budget {trainer.max_training_seconds/3600:.2f}h)"
    )

    time_budget_hit = False
    for epoch in range(1, num_epochs + 1):
        train_metrics, time_budget_hit = trainer.train_epoch(train_loader, epoch)

        if time_budget_hit:
            # Skip validation once we are out of wall-clock budget; just persist.
            val_metrics = {"val_loss_total": float("nan")}
            _log(
                f"[time-budget] exit at ep={epoch}; saving final checkpoint "
                f"without validation",
                err=True,
            )
            trainer.save_checkpoint(
                epoch, {**train_metrics, **val_metrics}, is_final=True
            )
            trainer.save_latest_batch_checkpoint(
                epoch=epoch,
                global_batch_idx=-1,
                extra={"reason": "time-budget-exceeded"},
            )
            break

        val_metrics = trainer.validate(val_loader)

        # Compact single-line per-epoch summary. Replaces the previous two-line
        # dump; keeps per-stage metrics on the same line.
        elapsed_h = (time.time() - trainer.train_start_time) / 3600.0
        _log(
            f"[epoch {epoch}/{num_epochs} done] "
            f"train_total={train_metrics['loss_total']:.6f} "
            f"S1_T={train_metrics['loss_s1_teacher']:.6f} "
            f"S2={train_metrics['loss_s2_gt']:.6f} "
            f"S3={train_metrics['loss_s3_gt']:.6f} "
            f"S4={train_metrics['loss_s4_gt']:.6f} "
            f"| val_total={val_metrics['val_loss_total']:.6f} "
            f"elapsed={elapsed_h:.2f}h"
        )

        if trainer.should_stop_early(val_metrics['val_loss_total']):
            _log(f"Early stopping at epoch {epoch}")
            trainer.save_checkpoint(
                epoch, {**train_metrics, **val_metrics}, is_final=True
            )
            break

        if epoch % config.get("save_every", 10) == 0:
            trainer.save_checkpoint(epoch, {**train_metrics, **val_metrics})

        # Catch the time budget between epochs too in case the last in-loop
        # batch overshot (cheap, single time check).
        if trainer.max_training_seconds > 0 and (
            time.time() - trainer.train_start_time >= trainer.max_training_seconds
        ):
            _log("[time-budget] reached between epochs; stopping", err=True)
            trainer.save_checkpoint(
                epoch, {**train_metrics, **val_metrics}, is_final=True
            )
            break

    if not time_budget_hit:
        trainer.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_final=True)

    return decoder, {
        "final_train_loss": train_metrics['loss_total'],
        "final_val_loss": val_metrics.get('val_loss_total', float("nan")),
        "stopped_reason": (
            "time-budget-exceeded" if time_budget_hit else "completed"
        ),
    }


# ============================================================================
# CLI entry-point (Kaggle-friendly)
# ============================================================================

def _build_kaggle_loaders(args, dataset_cls):
    """
    Build train / val DataLoaders from a Vimeo90k-style dataset class.

    Robust to two layouts (auto-detected by the dataset):
      Kaggle : <root>/sep_{train,test}list.txt + sequences/<SSSSS>/<CCCC>/im{1..7}.png
      Flat   : <root>/{train,val}_list.txt + <folder>/im{1..7}.png
    """
    train_ds = dataset_cls(root=args.data_root, split="train")
    val_ds = dataset_cls(root=args.data_root, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(args.num_workers, 1),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    print(
        f"  Dataset: train={len(train_ds)} clips ({len(train_loader)} batches), "
        f"val={len(val_ds)} clips ({len(val_loader)} batches)"
    )
    return train_loader, val_loader


def _parse_args():
    """CLI argument parser for Phase 1 (Kaggle-friendly)."""
    p = argparse.ArgumentParser(
        description="Train Phase 1 of Budget-Adaptive Decoder on Vimeo90k"
    )
    p.add_argument("--pretrained", required=True,
                   help="Path to DCVC .pth.tar baseline "
                        "(e.g., /kaggle/input/.../model_dcvc_quality_3_psnr.pth)")
    p.add_argument("--data_root", required=True,
                   help="Root containing sep_trainlist.txt + sequences/  "
                        "(e.g. /kaggle/input/.../vimeo_septuplet)")
    p.add_argument("--output_dir", default="/kaggle/working/checkpoints/phase1")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log_every", type=int, default=1000,
                   help="Emit a training log line every N batches (default 1000).")
    p.add_argument("--ckpt_every_batches", type=int, default=500,
                   help="Overwrite the 'latest' side checkpoint every N batches "
                        "(default 500). Set 0 to disable.")
    p.add_argument("--max_hours", type=float, default=12.0,
                   help="Wall-clock training budget in hours (default 12). "
                        "Training stops and saves a final checkpoint when reached.")
    p.add_argument("--dcvc_src", default=None,
                   help="Directory with DCVC's src/ importable tree; "
                        "defaults to env DCVC_SRC_PATH or local default.")
    p.add_argument("--lambda_rd", type=float, default=1.0,
                   help="Reserved weight (currently unused in Phase 1 hybrid loss).")
    p.add_argument("--resume", default=None,
                   help="Path to phase1_*.pt checkpoint to resume from.")
    return p.parse_args()


def main_entry():
    """Phase 1 driver: builds teacher + decoder + loaders, then trains."""
    args = _parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\n[Teacher] loading DCVC from {args.pretrained}")
    teacher = DCVCWrapper(
        device=device,
        load_pretrained=True,
        pretrained_path=args.pretrained,
        dcvc_src_path=args.dcvc_src,
    )
    if teacher.dcvc is None:
        raise RuntimeError(
            "DCVC teacher failed to import. Ensure --dcvc_src points to the "
            "DCVC-family/DCVC directory whose src/ tree is on sys.path."
        )
    print(f"  Teacher params (trainable / total): "
          f"{sum(p.numel() for p in teacher.dcvc.parameters() if p.requires_grad):,} / "
          f"{sum(p.numel() for p in teacher.dcvc.parameters()):,}")

    print("\n[Student] creating BudgetAdaptiveDecoder")
    decoder = BudgetAdaptiveDecoder().to(device)
    n_train = sum(p.numel() for p in decoder.parameters())
    print(f"  Decoder params: {n_train:,}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        print(f"  Resumed decoder weights from {args.resume}")

    print(f"\n[Data] loading Vimeo90k from {args.data_root}")
    from budget_adaptive_decoder.data.vimeo90k import Vimeo90kDataset
    train_loader, val_loader = _build_kaggle_loaders(args, Vimeo90kDataset)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    config = {
        "num_epochs": args.epochs,
        "save_every": args.save_every,
        "patience": args.epochs + 1,
        "min_delta": 0.001,
        "lambda_rd": args.lambda_rd,
    }

    print(f"\n[Train] output_dir={args.output_dir} "
          f"batch_size={args.batch_size} epochs={args.epochs} lr={args.lr}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    decoder, final_metrics = train_phase1(
        decoder=decoder,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=config,
        checkpoint_dir=Path(args.output_dir),
        num_epochs=args.epochs,
        log_every=args.log_every,
        ckpt_every_batches=args.ckpt_every_batches,
        max_training_seconds=args.max_hours * 3600.0,
    )

    print(f"\nDone. Final metrics: {final_metrics}")
    print(f"Checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main_entry()