"""
Phase 3: Policy Network Training via Regression

As specified in Section 8, Phase 3 trains the Policy Network to predict
ΔQ(k) from content features and budget.

Key requirements:
- Beta(2,2) budget sampling
- Content feature extraction pre-decoding (zero extra decode cost)
- ΔQ regression loss (MSE to precomputed targets)
- Policy Network + BitstreamContentExtractor trainable, decoder frozen
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import numpy as np

from ..models.policy import PolicyNetwork
from ..models.extractor import BitstreamContentExtractor
from ..models.decoder import BudgetAdaptiveDecoder
from ..data.augmentation import ReferenceFrameCorruption
from .budget_sampler import MixtureBudgetSampler


logger = logging.getLogger(__name__)


class Phase3Trainer:
    """
    Trainer for Phase 3: Policy Network Training via Regression.

    Budget B is sampled from a mixture distribution (Issue 3.3):
    50% Beta(2,2), 20% Beta(1,1), 15% Beta(5,2), 15% point masses
    at {0.3, 0.5, 0.9} for deployment-relevant coverage.
    Content features are extracted pre-decoding using the teacher
    to provide context and recon_image.
    Decoder is frozen.

    Loss:
        L_Phase3 = (1/K) * sum_{k=1}^{K} ||ΔQ_hat(k) - ΔQ_star(k)||_2^2
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        extractor: BitstreamContentExtractor,
        decoder: BudgetAdaptiveDecoder,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        teacher: Optional[Any] = None,
        delta_q_targets: Optional[Dict[str, torch.Tensor]] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Args:
            policy: PolicyNetwork to train
            extractor: BitstreamContentExtractor to train
            decoder: Frozen BudgetAdaptiveDecoder
            optimizer: Optimizer for policy and extractor
            device: Device to train on
            config: Training configuration
            teacher: Frozen DCVC teacher for context generation (optional)
            delta_q_targets: Precomputed ΔQ targets from Phase 2
            checkpoint_dir: Directory to save checkpoints
        """
        self.policy = policy
        self.extractor = extractor
        self.decoder = decoder
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.teacher = teacher
        self.delta_q_targets = delta_q_targets or {}
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.K = decoder.K

        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.beta_sampler = MixtureBudgetSampler()

        self.corruption = ReferenceFrameCorruption(enabled=True)

        self.best_loss = float("inf")
        self.patience = config.get("patience", 15)
        self.wait = 0
        self.min_delta = config.get("min_delta", 0.001)

        self.trainable_params = list(self.policy.parameters()) + list(self.extractor.parameters())

    def sample_budget(self, batch_size: int) -> torch.Tensor:
        """Sample budget from mixture distribution (Issue 3.3 fix).

        Mixture: 50% Beta(2,2), 20% Beta(1,1), 15% Beta(5,2),
        15% point masses at {0.3, 0.5, 0.9} for deployment-relevant
        budget coverage.
        """
        return self.beta_sampler.sample(batch_size, self.device)

    def forward_pass(
        self,
        curr_frame: torch.Tensor,
        latent: torch.Tensor,
        prev_frame: torch.Tensor,
        budget: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run forward pass and return predictions and targets.

        Args:
            curr_frame: Current frame [B, 3, H, W] - used by teacher for context
            latent: Latent tensor [B, C_lat, H, W] - passed to extractor if use_latent
            prev_frame: Previous frame [B, 3, H, W]
            budget: Sampled budget [B]

        Returns:
            Tuple of (delta_q_pred, delta_q_target, content_features)
        """
        if self.teacher is not None:
            with torch.no_grad():
                teacher_out = self.teacher.get_stage_targets(curr_frame, prev_frame)
                context = teacher_out['context']
                recon_image = teacher_out.get('base_recon', teacher_out.get('recon_image'))
        else:
            B, _, H, W = prev_frame.shape
            context = torch.zeros(B, 64, H // 4, W // 4, device=prev_frame.device)
            recon_image = torch.zeros_like(prev_frame)

        content_features = self.extractor(
            context, recon_image, prev_frame,
            latent if self.extractor.use_latent else None
        )

        delta_q_pred = self.policy(content_features, budget)

        with torch.no_grad():
            delta_q_target = self._compute_delta_q_targets(
                context, prev_frame
            )

        return delta_q_pred, delta_q_target, content_features

    def _compute_delta_q_targets(
        self,
        context: torch.Tensor,
        prev_frame: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ΔQ targets by running decoder at all depths.

        ΔQ(k) = PSNR(k) - PSNR(k-1) for k=1..K
        Uses get_stage_states() which returns K+1 recons (Q(0), Q(1), ..., Q(K)).
        Returns delta_q of shape [B, K].
        """
        decoder = self.decoder.to(self.device)

        reconstructions = decoder.get_stage_states(context, prev_frame)

        from ..evaluation.metrics import compute_psnr

        K = min(self.K, len(reconstructions) - 1)
        delta_q = torch.zeros(context.shape[0], K, device=self.device)

        for k in range(1, K + 1):
            psnr_prev = compute_psnr(reconstructions[k - 1], reconstructions[-1])
            psnr_curr = compute_psnr(reconstructions[k], reconstructions[-1])
            delta_q[:, k - 1] = (psnr_curr - psnr_prev) / 100.0

        return delta_q

    def compute_loss(
        self,
        delta_q_pred: torch.Tensor,
        delta_q_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MSE regression loss.

        L_Phase3 = (1/K) * sum_{k=1}^{K} ||ΔQ_hat(k) - ΔQ_star(k)||_2^2
        """
        return torch.mean((delta_q_pred - delta_q_target) ** 2)

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
        self.policy.train()
        self.extractor.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Phase 3 Epoch {epoch}")

        for batch_idx, (frames, bitstream_dict, prev_frame) in enumerate(pbar):
            curr_frame = frames[:, 3].to(self.device)
            latent = bitstream_dict["latent"].to(self.device)
            prev_frame = prev_frame.to(self.device)

            batch_size = latent.shape[0]
            budget = self.sample_budget(batch_size)

            prev_frame_corrupted = self.corruption(
                prev_frame,
                frame_position_in_sequence=0,
            )

            delta_q_pred, delta_q_target, content_features = self.forward_pass(
                curr_frame, latent, prev_frame_corrupted, budget
            )

            loss = self.compute_loss(delta_q_pred, delta_q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": loss.item(),
                "ΔQ_pred_mean": delta_q_pred.mean().item(),
                "ΔQ_target_mean": delta_q_target.mean().item(),
            })

        avg_loss = total_loss / max(num_batches, 1)

        return {"loss": avg_loss}

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
        self.policy.eval()
        self.extractor.eval()

        total_loss = 0.0
        num_batches = 0

        for frames, bitstream_dict, prev_frame in val_loader:
            curr_frame = frames[:, 3].to(self.device)
            latent = bitstream_dict["latent"].to(self.device)
            prev_frame = prev_frame.to(self.device)

            batch_size = latent.shape[0]
            budget = self.sample_budget(batch_size)

            delta_q_pred, delta_q_target, _ = self.forward_pass(
                curr_frame, latent, prev_frame, budget
            )

            loss = self.compute_loss(delta_q_pred, delta_q_target)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        return {"val_loss": avg_loss}

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
        checkpoint_name = f"phase3_epoch{epoch}.pt" if not is_final else "phase3_final.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        torch.save({
            "epoch": epoch,
            "policy_state_dict": self.policy.state_dict(),
            "extractor_state_dict": self.extractor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, checkpoint_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.extractor.load_state_dict(checkpoint["extractor_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("epoch", 0)


class BetaSampler:
    """DEPRECATED: replaced by MixtureBudgetSampler (Issue 3.3 fix). Kept for
    backwards-compatibility with any external imports.
    """

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample from Beta(alpha, beta).

        Deprecated: use MixtureBudgetSampler from budget_sampler.py.
        """
        import math

        x = torch._standard_gamma(torch.full((batch_size,), self.alpha))
        y = torch._standard_gamma(torch.full((batch_size,), self.beta))

        samples = x / (x + y)

        return samples


def train_phase3(
    policy: PolicyNetwork,
    extractor: BitstreamContentExtractor,
    decoder: BudgetAdaptiveDecoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict[str, Any],
    teacher: Optional[Any] = None,
    delta_q_targets: Optional[Dict[str, torch.Tensor]] = None,
    checkpoint_dir: Optional[Path] = None,
    num_epochs: Optional[int] = None,
) -> Tuple[PolicyNetwork, BitstreamContentExtractor, Dict]:
    """
    Main Phase 3 training loop.

    Args:
        policy: PolicyNetwork to train
        extractor: BitstreamContentExtractor to train
        decoder: Frozen BudgetAdaptiveDecoder
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer for policy and extractor
        device: Device to train on
        config: Training configuration
        teacher: Frozen DCVC teacher for context generation (optional)
        delta_q_targets: Precomputed ΔQ targets from Phase 2
        checkpoint_dir: Directory for checkpoints
        num_epochs: Number of epochs (default from config)

    Returns:
        Tuple of (trained_policy, trained_extractor, final_metrics)
    """
    num_epochs = num_epochs or config.get("num_epochs", 100)

    trainer = Phase3Trainer(
        policy=policy,
        extractor=extractor,
        decoder=decoder,
        optimizer=optimizer,
        device=device,
        config=config,
        teacher=teacher,
        delta_q_targets=delta_q_targets,
        checkpoint_dir=checkpoint_dir,
    )

    logger.info(f"Starting Phase 3 training for {num_epochs} epochs")

    for epoch in range(1, num_epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader)

        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss = {train_metrics['loss']:.6f}, "
            f"Val Loss = {val_metrics['val_loss']:.6f}"
        )

        if trainer.should_stop_early(val_metrics["val_loss"]):
            logger.info(f"Early stopping at epoch {epoch}")
            break

        if epoch % config.get("save_every", 10) == 0:
            trainer.save_checkpoint(epoch, {**train_metrics, **val_metrics})

    trainer.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_final=True)

    return policy, extractor, {
        "final_train_loss": train_metrics["loss"],
        "final_val_loss": val_metrics["val_loss"],
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3 Training Module")
    print("=" * 60)
    print("Use this module via import from training package")
    print("=" * 60)