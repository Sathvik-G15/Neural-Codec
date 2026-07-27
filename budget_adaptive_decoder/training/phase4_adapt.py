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

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time

from ..models.policy import PolicyNetwork
from ..models.extractor import BitstreamContentExtractor
from ..models.decoder import BudgetAdaptiveDecoder
from ..models.scheduler import Scheduler
from ..data.augmentation import ReferenceFrameCorruption
from .budget_sampler import MixtureBudgetSampler


def _log(msg, *, err=False):
    print(msg, file=sys.stderr if err else sys.stdout, flush=True)


# Maximum wall-clock training time. The script gracefully stops and saves a
# checkpoint once this budget is exceeded (Kaggle's session limit is 12h).
MAX_TRAINING_SECONDS = 12 * 60 * 60


# Side checkpoint cadence. We overwrite the SAME file every time so that disk
# usage stays bounded.
CHECKPOINT_CADENCE_BATCHES = 500
SIDE_CKPT_FILENAME = "phase4_latest.pt"


# In-loop per-batch log cadence (one line every N batches).
LOG_CADENCE_BATCHES = 1000


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
        log_every: Optional[int] = None,
        ckpt_every_batches: Optional[int] = None,
        max_training_seconds: Optional[float] = None,
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
            log_every: Per-batch log cadence (default 1000, set 0 to disable)
            ckpt_every_batches: Overwrite the "latest" side checkpoint every N
                batches (default 500). Set 0 to disable.
            max_training_seconds: Hard wall-clock budget (default 12h).
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

        # In-loop per-batch logging cadence (default 1000). tqdm is removed to
        # avoid per-batch stdout spam and to keep us in control of the cadence.
        if log_every is None:
            log_every = int(config.get("log_every", LOG_CADENCE_BATCHES))
        self.log_every = max(int(log_every), 0)

        # Side checkpoint cadence (overwrite the same file every N batches).
        if ckpt_every_batches is None:
            ckpt_every_batches = int(
                config.get("ckpt_every_batches", CHECKPOINT_CADENCE_BATCHES)
            )
        self.ckpt_every_batches = max(int(ckpt_every_batches), 0)
        self.side_ckpt_path = self.checkpoint_dir / SIDE_CKPT_FILENAME

        # Wall-clock budget. Kaggle sessions terminate at ~12h, so default to
        # that ceiling unless overridden.
        if max_training_seconds is None:
            max_training_seconds = float(
                config.get("max_training_seconds", MAX_TRAINING_SECONDS)
            )
        self.max_training_seconds = float(max_training_seconds)
        self.train_start_time = time.time()

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
    ) -> Tuple[Dict[str, float], bool]:
        """
        Train for one epoch.

        Side effects:
          * Prints a one-line training log every `self.log_every` batches
            (default 1000) to stay under Kaggle's 20MB log cap.
          * Saves a "latest" side checkpoint (overwriting the same file) every
            `self.ckpt_every_batches` batches (default 500).
          * Stops early if the wall-clock budget (`self.max_training_seconds`,
            default 12h) is exhausted. The flag returned in the second tuple
            element signals the caller to break out of the outer epoch loop.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            (metrics_dict, time_budget_exceeded)
        """
        self.decoder.train()

        total_loss = 0.0
        num_batches = 0
        depth_distribution = {k: 0 for k in range(self.K + 1)}
        log_every = self.log_every
        ckpt_every = self.ckpt_every_batches
        time_budget_exceeded = False

        for batch_idx, (frames, bitstream_dict, prev_frame) in enumerate(
            train_loader, start=1
        ):
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

            if log_every > 0 and (batch_idx % log_every == 0):
                running_loss = total_loss / batch_idx
                elapsed_s = time.time() - self.train_start_time
                _log(
                    f"[phase4 trn] ep={epoch} step={batch_idx}/"
                    f"{len(train_loader)} avg_loss={running_loss:.6f} "
                    f"step_loss={loss.item():.6f} depth={selected_depth} "
                    f"elapsed={elapsed_s/3600:.2f}h"
                )

            if ckpt_every > 0 and (batch_idx % ckpt_every == 0):
                self.save_latest_batch_checkpoint(
                    epoch=epoch,
                    global_batch_idx=batch_idx,
                    extra={
                        "running_avg_loss": total_loss / batch_idx,
                        "selected_depth": selected_depth,
                    },
                )

            if self.max_training_seconds > 0:
                elapsed = time.time() - self.train_start_time
                if elapsed >= self.max_training_seconds:
                    remaining = max(self.max_training_seconds - elapsed, 0)
                    _log(
                        f"[phase4 time-budget] reached "
                        f"{self.max_training_seconds}s (elapsed="
                        f"{elapsed/3600:.2f}h, remaining={remaining:.1f}s); "
                        f"stopping at ep={epoch} step={batch_idx}",
                        err=True,
                    )
                    time_budget_exceeded = True
                    break

        avg_loss = total_loss / max(num_batches, 1)

        return {
            "loss": avg_loss,
            "depth_distribution": depth_distribution,
        }, time_budget_exceeded

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

        _log(f"Checkpoint saved: {checkpoint_path}")

    def save_latest_batch_checkpoint(
        self,
        epoch: int,
        global_batch_idx: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save (and overwrite) the single "latest" side checkpoint."""
        extra = extra or {}
        torch.save({
            "epoch": epoch,
            "global_batch_idx": global_batch_idx,
            "decoder_state_dict": self.decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "extra": extra,
            "elapsed_seconds": time.time() - self.train_start_time,
        }, self.side_ckpt_path)
        _log(
            f"[phase4 side-ckpt] ep={epoch} batch={global_batch_idx} -> "
            f"{self.side_ckpt_path.name}"
        )
        return self.side_ckpt_path

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
    log_every: Optional[int] = 1000,
    ckpt_every_batches: Optional[int] = 500,
    max_training_seconds: Optional[float] = 12 * 60 * 60,
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
        log_every: Per-batch log cadence (default 1000, set 0 to disable)
        ckpt_every_batches: Overwrite the "latest" side checkpoint every N
            batches (default 500). Set 0 to disable.
        max_training_seconds: Hard wall-clock budget (default 12h).

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
        log_every=log_every,
        ckpt_every_batches=ckpt_every_batches,
        max_training_seconds=max_training_seconds,
    )

    _log(
        f"Starting Phase 4 training for {num_epochs} epochs "
        f"(decoder lr {decoder_lr:.6f}; logs every {trainer.log_every} "
        f"batches; side-ckpt every {trainer.ckpt_every_batches} batches; "
        f"time budget {trainer.max_training_seconds/3600:.2f}h)"
    )

    time_budget_hit = False
    for epoch in range(1, num_epochs + 1):
        train_metrics, time_budget_hit = trainer.train_epoch(train_loader, epoch)

        if time_budget_hit:
            val_metrics = {"val_loss": float("nan")}
            _log(
                f"[phase4 time-budget] exit at ep={epoch}; saving final "
                f"checkpoint without validation",
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

        elapsed_h = (time.time() - trainer.train_start_time) / 3600.0
        _log(
            f"Epoch {epoch}: "
            f"Train Loss = {train_metrics['loss']:.6f}, "
            f"Val Loss = {val_metrics['val_loss']:.6f} "
            f"elapsed={elapsed_h:.2f}h"
        )

        depth_dist = train_metrics.get("depth_distribution", {})
        depth_str = ", ".join([f"k={k}: {v}" for k, v in sorted(depth_dist.items())])
        _log(f"  Depth distribution: {depth_str}")

        if epoch % config.get("save_every", 10) == 0:
            trainer.save_checkpoint(epoch, {**train_metrics, **val_metrics})

        if trainer.max_training_seconds > 0 and (
            time.time() - trainer.train_start_time >= trainer.max_training_seconds
        ):
            _log("[phase4 time-budget] reached between epochs; stopping", err=True)
            trainer.save_checkpoint(
                epoch, {**train_metrics, **val_metrics}, is_final=True
            )
            break

    if not time_budget_hit:
        trainer.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_final=True)

    return decoder, {
        "final_train_loss": train_metrics["loss"],
        "final_val_loss": val_metrics.get("val_loss", float("nan")),
        "decoder_lr": decoder_lr,
        "depth_distribution": train_metrics.get("depth_distribution", {}),
        "stopped_reason": (
            "time-budget-exceeded" if time_budget_hit else "completed"
        ),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 Training Module")
    print("=" * 60)
    print("Use this module via import from training package")
    print("=" * 60)