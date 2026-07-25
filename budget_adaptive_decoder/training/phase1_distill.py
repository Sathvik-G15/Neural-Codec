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
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm

from ..models.decoder import BudgetAdaptiveDecoder
from ..models.teacher import DCVCWrapper


logger = logging.getLogger(__name__)


# Stage weights for ground truth supervision (stages 2, 3, 4)
GT_WEIGHTS = {2: 0.5, 3: 0.75, 4: 1.0}


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
    ):
        """
        Args:
            decoder: BudgetAdaptiveDecoder (student)
            teacher: DCVCWrapper (frozen teacher)
            optimizer: Optimizer for decoder
            device: Device to train on
            config: Training configuration dict
            checkpoint_dir: Directory to save checkpoints
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
        logger.info("Teacher verification passed: all parameters frozen")

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
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader yielding (curr_frame, ref_frame)
            epoch: Current epoch number

        Returns:
            Dictionary of loss metrics
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

        pbar = tqdm(train_loader, desc=f"Phase 1 Epoch {epoch}")

        for curr_frame, ref_frame in pbar:
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

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Average losses
        avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        avg_losses['loss_total'] = avg_losses.pop('loss_total')

        self.loss_history.append(avg_losses)

        return avg_losses

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader yielding (curr_frame, ref_frame)

        Returns:
            Dictionary of validation metrics
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

        for curr_frame, ref_frame in val_loader:
            curr_frame = curr_frame.to(self.device)
            ref_frame = ref_frame.to(self.device)

            # Teacher
            teacher_out = self.teacher.get_stage_targets(curr_frame, ref_frame)
            teacher_recon = teacher_out['stage_recons'][0]
            context = teacher_out['context']

            # Student
            student_recons = self.decoder.decode_all_stages_phase1(context, ref_frame)

            # Loss
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

        return avg_losses

    def should_stop_early(self, val_loss: float) -> bool:
        """Check if early stopping criteria are met."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info(f"Early stopping triggered after {self.wait} epochs without improvement")
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

        logger.info(f"Checkpoint saved: {checkpoint_path}")

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
) -> Tuple[BudgetAdaptiveDecoder, Dict]:
    """
    Run Phase 1 training.

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
    )

    logger.info(f"Starting Phase 1 training for {num_epochs} epochs")

    for epoch in range(1, num_epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader)

        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss = {train_metrics['loss_total']:.6f}, "
            f"Val Loss = {val_metrics['val_loss_total']:.6f}"
        )
        logger.info(
            f"  S1 (teacher): {train_metrics['loss_s1_teacher']:.6f} | "
            f"S2 (GT): {train_metrics['loss_s2_gt']:.6f} | "
            f"S3 (GT): {train_metrics['loss_s3_gt']:.6f} | "
            f"S4 (GT): {train_metrics['loss_s4_gt']:.6f}"
        )

        if trainer.should_stop_early(val_metrics['val_loss_total']):
            logger.info(f"Early stopping at epoch {epoch}")
            break

        if epoch % config.get("save_every", 10) == 0:
            trainer.save_checkpoint(epoch, {**train_metrics, **val_metrics})

    trainer.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_final=True)

    return decoder, {
        "final_train_loss": train_metrics['loss_total'],
        "final_val_loss": val_metrics['val_loss_total'],
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 Training Module")
    print("=" * 60)
    print("Use this module via import from training package")
    print("=" * 60)