"""
Phase 4: Decoder Adaptation

As specified in Section 8, Phase 4 allows the decoder to adapt to the
specific depths selected by the policy while preserving Phase 1 specialization.

Key requirements:
- Frozen policy (no gradient to policy)
- Reconstruction loss only (no regression term in loss)
- Decoder learning rate = 0.05 * base_lr
- Policy Network is frozen - no gradient updates in Phase 4
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm

from ..models.policy import PolicyNetwork
from ..models.extractor import BitstreamContentExtractor
from ..models.decoder import BudgetAdaptiveDecoder
from ..models.scheduler import Scheduler
from ..data.augmentation import ReferenceFrameCorruption
from .budget_sampler import MixtureBudgetSampler


logger = logging.getLogger(__name__)


class Phase4Trainer:
    """
    Trainer for Phase 4: Decoder Adaptation.

    Policy Network is frozen - no gradient flow to policy.
    Scheduling decision is discrete argmax - no gradient through it.

    Loss:
        L_Phase4 = ||Y_hat - Y_star||_2^2

    Only the Budget-Adaptive Decoder receives gradients.

    Decoder learning rate: 0.05 * policy_lr

    Phase 4 uses the TEACHER to generate context from frames_tensor so that
    the decoder can run. The teacher runs in inference mode (no gradients).
    """

    LR_RATIO = 0.05

    def __init__(
        self,
        policy: PolicyNetwork,
        extractor: BitstreamContentExtractor,
        decoder: BudgetAdaptiveDecoder,
        scheduler: Scheduler,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        teacher: Optional[Any] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Args:
            policy: Frozen PolicyNetwork (no gradient)
            extractor: Frozen BitstreamContentExtractor (no gradient)
            decoder: BudgetAdaptiveDecoder to adapt
            scheduler: Scheduler to compute depth
            optimizer: Optimizer for decoder (lr = 0.05 * base_lr)
            device: Device to train on
            config: Training configuration
            teacher: DCVC teacher for context generation (optional, for Phase 4 context)
            checkpoint_dir: Directory to save checkpoints
        """
        self.policy = policy
        self.extractor = extractor
        self.decoder = decoder
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.teacher = teacher
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.K = decoder.K

        self.policy.eval()
        self.extractor.eval()
        for param in self.policy.parameters():
            param.requires_grad = False
        for param in self.extractor.parameters():
            param.requires_grad = False

        self.beta_sampler = MixtureBudgetSampler()

        self.corruption = ReferenceFrameCorruption(enabled=True)

        self.trainable_params = list(self.decoder.parameters())

    def sample_budget(self, batch_size: int) -> torch.Tensor:
        """Sample budget from Beta(2,2) distribution."""
        return self.beta_sampler.sample(batch_size).to(self.device)

    @torch.no_grad()
    def get_policy_predictions(
        self,
        curr_frame: torch.Tensor,
        latent: torch.Tensor,
        prev_frame: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get policy predictions (no gradient flow).

        Args:
            curr_frame: Current frame [B, 3, H, W] - used by teacher for context/recon_image
            latent: Latent tensor [B, C_lat, H, W] - passed to extractor if use_latent
            prev_frame: Previous frame [B, 3, H, W]
            budget: Sampled budget [B]

        Returns:
            Predicted ΔQ values [B, K]
        """
        if self.teacher is not None:
            teacher_out = self.teacher.get_stage_targets(curr_frame, prev_frame)
            context = teacher_out['context']
            recon_image = teacher_out.get('base_recon', teacher_out.get('recon_image'))
        else:
            B, _, H, W = prev_frame.shape
            context = torch.randn(B, 64, H // 4, W // 4, device=prev_frame.device)
            recon_image = torch.zeros_like(prev_frame)

        content_features = self.extractor(
            context, recon_image, prev_frame, latent if self.extractor.use_latent else None
        )

        delta_q_pred = self.policy(content_features, budget)

        return delta_q_pred

    def forward_pass(
        self,
        curr_frame: torch.Tensor,
        latent: torch.Tensor,
        prev_frame: torch.Tensor,
        frame_type_id: torch.Tensor,
        budget: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Run forward pass and compute reconstruction loss.

        Args:
            curr_frame: Current frame [B, 3, H, W] - used to generate context via teacher
            latent: Latent tensor [B, C_lat, H, W] - passed to extractor only if use_latent
            prev_frame: Previous frame [B, 3, H, W]
            frame_type_id: Frame type [B] - kept for API compat but unused
            budget: Sampled budget [B]
            ground_truth: Ground truth frame [B, 3, H, W]

        Returns:
            Tuple of (reconstruction, ground_truth, selected_depth)
        """
        delta_q_pred = self.get_policy_predictions(
            curr_frame, latent, prev_frame, budget
        )

        budget_scalar = budget[0] if budget.dim() > 0 else budget
        k_star = self.scheduler.select_depth(delta_q_pred, budget_scalar.item())

        if self.teacher is not None:
            with torch.no_grad():
                teacher_out = self.teacher.get_stage_targets(curr_frame, prev_frame)
                context = teacher_out['context']
        else:
            context = latent

        # Use the first sample's depth for the batch (multi-sample with different
        # depths requires per-sample decoding, which Phase 4 currently doesn't support).
        # In deployment each frame would be processed independently.
        k_star_val = k_star[0].item() if k_star.dim() > 0 else k_star.item()

        reconstruction = self.decoder.run_to_depth(
            context, prev_frame, k_star_val
        )

        return reconstruction, ground_truth, k_star_val

    def compute_loss(
        self,
        reconstruction: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss (MSE only - no regression term).

        L_Phase4 = ||Y_hat - Y_star||_2^2
        """
        return torch.mean((reconstruction - ground_truth) ** 2)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of loss metrics
        """
        self.decoder.train()

        total_loss = 0.0
        num_batches = 0
        depth_distribution = {k: 0 for k in range(self.K + 1)}

        pbar = tqdm(train_loader, desc=f"Phase 4 Epoch {epoch}")

        for batch_idx, (frames, bitstream_dict, prev_frame) in enumerate(pbar):
            curr_frame = frames[:, 3].to(self.device)
            latent = bitstream_dict["latent"].to(self.device)
            prev_frame = prev_frame.to(self.device)

            ground_truth = frames[:, -1].to(self.device)

            batch_size = latent.shape[0]
            budget = self.sample_budget(batch_size).to(self.device)

            prev_frame_corrupted = self.corruption(
                prev_frame,
                frame_position_in_sequence=0,
            )

            prev_frame_corrupted = self.corruption(
                prev_frame,
                frame_position_in_sequence=0,
            )

            reconstruction, gt, selected_depth = self.forward_pass(
                curr_frame, latent, prev_frame_corrupted, None, budget, ground_truth
            )

            loss = self.compute_loss(reconstruction, gt)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            depth_distribution[selected_depth] = depth_distribution.get(selected_depth, 0) + 1

            pbar.set_postfix({"loss": loss.item(), "depth": selected_depth})

        avg_loss = total_loss / max(num_batches, 1)

        return {
            "loss": avg_loss,
            "depth_distribution": depth_distribution,
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.decoder.eval()

        total_loss = 0.0
        num_batches = 0

        for frames, bitstream_dict, prev_frame in val_loader:
            curr_frame = frames[:, 3].to(self.device)
            latent = bitstream_dict["latent"].to(self.device)
            prev_frame = prev_frame.to(self.device)

            ground_truth = frames[:, -1].to(self.device)

            batch_size = latent.shape[0]
            budget = self.sample_budget(batch_size).to(self.device)

            reconstruction, gt, selected_depth = self.forward_pass(
                curr_frame, latent, prev_frame, None, budget, ground_truth
            )

            loss = self.compute_loss(reconstruction, gt)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        return {"val_loss": avg_loss}

    def save_checkpoint(self, epoch: int, metrics: Dict, is_final: bool = False):
        """Save a training checkpoint."""
        checkpoint_name = f"phase4_epoch{epoch}.pt" if not is_final else "phase4_final.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        torch.save({
            "epoch": epoch,
            "decoder_state_dict": self.decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, checkpoint_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("epoch", 0)


class BetaSamplerPhase4:
    """DEPRECATED: replaced by MixtureBudgetSampler (Issue 3.3 fix). Kept for
    backwards-compatibility with any external imports.
    """

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample from Beta(alpha, beta). Deprecated; use MixtureBudgetSampler."""
        x = torch._standard_gamma(torch.full((batch_size,), self.alpha))
        y = torch._standard_gamma(torch.full((batch_size,), self.beta))

        samples = x / (x + y)

        return samples


def train_phase4(
    policy: PolicyNetwork,
    extractor: BitstreamContentExtractor,
    decoder: BudgetAdaptiveDecoder,
    scheduler: Scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    base_lr: float,
    device: torch.device,
    config: Dict[str, Any],
    teacher: Optional[Any] = None,
    checkpoint_dir: Optional[Path] = None,
    num_epochs: Optional[int] = None,
) -> Tuple[BudgetAdaptiveDecoder, Dict]:
    """
    Main Phase 4 training loop.

    Args:
        policy: Frozen PolicyNetwork
        extractor: Frozen BitstreamContentExtractor
        decoder: BudgetAdaptiveDecoder to adapt
        scheduler: Scheduler for depth selection
        train_loader: Training data loader
        val_loader: Validation data loader
        base_lr: Base learning rate (decoder_lr = 0.05 * base_lr)
        device: Device to train on
        config: Training configuration
        teacher: DCVC teacher for context generation (optional)
        checkpoint_dir: Directory for checkpoints
        num_epochs: Number of epochs (default from config)

    Returns:
        Tuple of (adapted_decoder, final_metrics)
    """
    decoder_lr = Phase4Trainer.LR_RATIO * base_lr
    num_epochs = num_epochs or config.get("num_epochs", 50)

    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=decoder_lr,
        **config.get("optimizer_kwargs", {}),
    )

    trainer = Phase4Trainer(
        policy=policy,
        extractor=extractor,
        decoder=decoder,
        scheduler=scheduler,
        optimizer=optimizer,
        device=device,
        config=config,
        teacher=teacher,
        checkpoint_dir=checkpoint_dir,
    )

    logger.info(f"Starting Phase 4 training for {num_epochs} epochs")
    logger.info(f"Decoder learning rate: {decoder_lr:.6f} (0.05 * {base_lr})")

    for epoch in range(1, num_epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader)

        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss = {train_metrics['loss']:.6f}, "
            f"Val Loss = {val_metrics['val_loss']:.6f}"
        )

        depth_dist = train_metrics.get("depth_distribution", {})
        depth_str = ", ".join([f"k={k}: {v}" for k, v in sorted(depth_dist.items())])
        logger.info(f"  Depth distribution: {depth_str}")

        if epoch % config.get("save_every", 10) == 0:
            trainer.save_checkpoint(epoch, {**train_metrics, **val_metrics})

    trainer.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_final=True)

    return decoder, {
        "final_train_loss": train_metrics["loss"],
        "final_val_loss": val_metrics["val_loss"],
        "decoder_lr": decoder_lr,
        "depth_distribution": train_metrics.get("depth_distribution", {}),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 Training Module")
    print("=" * 60)
    print("Use this module via import from training package")
    print("=" * 60)