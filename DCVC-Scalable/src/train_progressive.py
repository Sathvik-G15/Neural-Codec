# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Training script for DCVC + StatelessProgressiveDecoder integration.
#
# Research plan alignment:
#     - DCVC encoder/entropy model: FROZEN (Section 2.1)
#     - Progressive residual decoder: TRAINABLE (Section 2.2)
#     - Deep supervision with early-stage priority (Section 3.1)
#     - 3-phase training curriculum (Section 3.2)
#     - Weighted multi-stage loss: w=[1.0, 0.9, 0.8, 0.7] (Section 3.1)
#     - Residual scale=0.1 for training stability (Section 2.3)
#
# Training procedure:
#     For each frame pair (ref, current):
#         1. y_hat = DCVC_encoder(current, ref)  [FROZEN - no gradients]
#         2. reconstructions = progressive_decoder(y_hat, depth=k)
#         3. loss = Σ w_i · MSE(recon_i, target) + λ_rd · L1(y_hat)
#         4. backward: only progressive decoder weights updated
#
# =============================================================================

import math
import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dcvc_integration import DCVCProgressiveModel, get_default_checkpoint


# ===========================================================================
# Dataset (reused from train.py — unchanged)
# ===========================================================================

class VideoFrameDataset(torch.utils.data.Dataset):
    """Video frame dataset. Loads consecutive-frame clips from directories."""

    def __init__(self, data_root, sequence_length=2, stride=1,
                 transform=None, allowed_dirs=None):
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.allowed_dirs = (
            set(str(d) for d in allowed_dirs) if allowed_dirs is not None else None
        )
        self.sequences = []
        self._build_sequence_list()

    def _build_sequence_list(self):
        # Vimeo90k has a nested structure: data_root/group_dir/clip_dir/*.png
        # allowed_dirs contains exactly the group_dirs (e.g. sequences/00001)
        if self.allowed_dirs is not None:
            group_dirs = sorted([Path(d) for d in self.allowed_dirs])
        else:
            group_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
            
        seq_dirs = []
        for group_dir in group_dirs:
            for clip_dir in group_dir.iterdir():
                if clip_dir.is_dir():
                    seq_dirs.append(clip_dir)

        for seq_dir in seq_dirs:
            frames = sorted(
                f for f in seq_dir.iterdir()
                if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')
            )
            # Vimeo90k septuplet has 7 frames. We need sequences of length `sequence_length`.
            # For septuplet (7 frames), we need pairs (length 2)
            if len(frames) >= self.sequence_length:
                for start in range(0, len(frames) - self.sequence_length + 1, self.stride):
                    self.sequences.append((seq_dir, frames[start:start + self.sequence_length]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _, frames = self.sequences[idx]
        clip = []
        for frame_path in frames:
            img = Image.open(frame_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            clip.append(img)
        return torch.stack(clip, dim=0)


def build_datasets(data_root, sequence_length=2, train_stride=1, val_stride=2,
                   train_transform=None, val_transform=None, val_ratio=0.2):
    """Build non-overlapping train/val datasets split by video sequence."""
    data_root = Path(data_root)
    seq_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])

    if not seq_dirs:
        raise ValueError(f"No sequence subdirectories found in '{data_root}'")

    split_idx = max(1, int(len(seq_dirs) * (1 - val_ratio)))
    # We take the absolute paths as strings for prefix matching later
    train_dirs = [str(d) for d in seq_dirs[:split_idx]]
    val_dirs = [str(d) for d in seq_dirs[split_idx:]] if split_idx < len(seq_dirs) else [str(seq_dirs[-1])]

    print(f"Allocated {len(train_dirs)} top-level sequence groups for training, {len(val_dirs)} for validation.")

    print(f"Train sequences ({len(train_dirs)} groups)")
    print(f"Val   sequences ({len(val_dirs)} groups)")

    train_dataset = VideoFrameDataset(
        data_root, sequence_length, train_stride, train_transform, allowed_dirs=train_dirs
    )
    val_dataset = VideoFrameDataset(
        data_root, sequence_length, val_stride, val_transform, allowed_dirs=val_dirs
    )
    return train_dataset, val_dataset


def get_transforms(resize_h=256, crop_h=256):
    """Return (train_transform, val_transform)."""
    crop_w = (int(crop_h * 16 / 9) // 8) * 8
    resize_w = (int(resize_h * 16 / 9) // 8) * 8

    train_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        transforms.RandomCrop((crop_h, crop_w)),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((crop_h, crop_w)),
        transforms.ToTensor(),
    ])
    return train_transform, val_transform


# ===========================================================================
# Loss Function (from research plan Section 3.1)
# ===========================================================================

class DCVCProgressiveLoss(nn.Module):
    """Weighted multi-stage rate-distortion loss.

    L = Σ_i  w_i · MSE(recon_i, target)  +  λ_rd · ||y_hat||₁

    Weights from research plan (early-stage priority):
        - Tier 1 (base): 1.0
        - Tier 2 (base+R1): 0.9
        - Tier 3 (base+R1+R2): 0.8
        - Tier 4 (base+R1+R2+R3): 0.7

    The L1 on y_hat acts as a bitrate proxy (encourages sparse latents).
    """
    def __init__(self, weights=None, lambda_rd=0.01):
        super().__init__()
        self.weights = weights if weights is not None else [1.0, 0.9, 0.8, 0.7]
        self.lambda_rd = lambda_rd

    def forward(self, outputs, target, latent=None):
        distortion = 0.0
        for i, recon in enumerate(outputs):
            w = self.weights[i] if i < len(self.weights) else self.weights[-1]
            distortion = distortion + w * F.mse_loss(recon, target)

        loss = distortion

        if latent is not None and self.lambda_rd > 0:
            loss = loss + self.lambda_rd * torch.mean(torch.abs(latent))

        return loss


# ===========================================================================
# Trainer
# ===========================================================================

class DCVCProgressiveTrainer:
    """Training loop for DCVC + StatelessProgressiveDecoder.

    Key properties (from research plan):
        - DCVC encoder: FROZEN throughout training
        - Only progressive decoder weights are updated
        - 3-phase curriculum:
            Phase 1 (epochs 0-199):   fixed depth=4
            Phase 2 (epochs 200-399): random [1,4]
            Phase 3 (epochs 400+):    random [1,4] fine-tune
    """

    def __init__(self, model, train_loader, val_loader, device,
                 log_dir='logs', checkpoint_dir='checkpoints',
                 lr=1e-4, weight_decay=1e-5, gradient_clip=1.0,
                 log_interval=100, val_interval=1000, checkpoint_interval=5000,
                 loss_fn=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.checkpoint_interval = checkpoint_interval

        # Loss
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = DCVCProgressiveLoss()

        # Optimizer — only for trainable params (progressive decoder)
        self.optimizer = optim.Adam(
            self.model.get_trainable_params(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.phase = 1

        # Verify encoder is frozen
        self._verify_frozen()

    def _verify_frozen(self):
        """Verify that only the progressive decoder has trainable parameters."""
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())

        trainable_params = list(self.model.get_trainable_params())
        frozen_params = [p for p in self.model.parameters() if not p.requires_grad]

        print(f"\nFrozen verification:")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Encoder (frozen): {sum(p.numel() for p in encoder_params):,}")
        print(f"  Decoder (trainable): {sum(p.numel() for p in decoder_params):,}")
        print(f"  Encoder has gradients: {any(p.grad is not None for p in encoder_params)}")
        print(f"  Decoder has gradients: {any(p.grad is not None for p in decoder_params)}")

        assert all(not p.requires_grad for p in encoder_params), \
            "Encoder should be frozen but has trainable parameters!"

    def train_step(self, batch, depth):
        """Single training step.

        Args:
            batch: (B, T, 3, H, W) video clip
            depth: Active progressive depth (1-4)

        Returns:
            loss value (float)
        """
        self.model.train()

        # Use frame pairs: ref=frame[0], current=frame[1], target=frame[1]
        ref = batch[:, 0].to(self.device)
        current = batch[:, 1].to(self.device)
        target = batch[:, 1].to(self.device)

        use_amp = self.device.type == 'cuda'
        with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
            result = self.model(current, ref=ref, depth=depth)
            outputs = result['reconstructions']
            y_hat = result['y_hat']

            loss = self.loss_fn(outputs, target, y_hat)

        self.optimizer.zero_grad()
        loss.backward()

        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_params(), self.gradient_clip
            )

        self.optimizer.step()
        return loss.item()

    def validate(self):
        """Validation — runs depth=4 (full quality).

        Returns:
            (avg_loss, avg_psnr): averaged over up to 50 validation batches
        """
        self.model.eval()
        val_loss = 0.0
        val_mse = 0.0
        num_batches = min(len(self.val_loader), 50)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= num_batches:
                    break

                ref = batch[:, 0].to(self.device)
                current = batch[:, 1].to(self.device)
                target = batch[:, 1].to(self.device)

                result = self.model(current, ref=ref, depth=4)
                outputs = result['reconstructions']
                y_hat = result['y_hat']

                loss = self.loss_fn(outputs, target, y_hat)
                val_loss += loss.item()

                # PSNR from highest-quality reconstruction
                final_recon = outputs[-1].clamp(0.0, 1.0)
                val_mse += F.mse_loss(final_recon, target).item()

        avg_loss = val_loss / num_batches
        avg_mse = val_mse / num_batches
        avg_psnr = 10 * math.log10(1.0 / avg_mse) if avg_mse > 0 else float('inf')
        return avg_loss, avg_psnr

    def train_epoch(self, epoch):
        """Train for one epoch with curriculum depth scheduling.

        Curriculum (research plan Section 3.2):
            Phase 1 (epochs   0-199): depth=4 (fixed, establish baseline)
            Phase 2 (epochs 200-399): random [1,2,3,4] (anytime behavior)
            Phase 3 (epochs 400+   ): random [1,2,3,4] (fine-tune)
        """
        if epoch < 200:
            self.phase = 1
            depth = 4
        elif epoch < 400:
            self.phase = 2
            depth = int(np.random.choice([1, 2, 3, 4]))
        else:
            self.phase = 3
            depth = int(np.random.choice([1, 2, 3, 4]))

        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.train_step(batch, depth)
            epoch_loss += loss

            if batch_idx % self.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch} | Batch {batch_idx}/{num_batches} | "
                    f"Loss: {avg_loss:.4f} | Phase: {self.phase} | Depth: {depth}"
                )
                self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                self.writer.add_scalar('train/phase', self.phase, self.global_step)
                self.writer.add_scalar('train/depth', depth, self.global_step)
                self.writer.add_scalar(
                    'train/lr', self.optimizer.param_groups[0]['lr'], self.global_step
                )

            if batch_idx % self.val_interval == 0 and batch_idx > 0:
                val_loss, val_psnr = self.validate()
                self.scheduler.step(val_loss)
                print(f"Validation | Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB")
                self.writer.add_scalar('val/loss', val_loss, self.global_step)
                self.writer.add_scalar('val/psnr', val_psnr, self.global_step)

            if batch_idx % self.checkpoint_interval == 0 and batch_idx > 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            self.global_step += 1

        return epoch_loss / num_batches

    def train(self, num_epochs):
        """Main training loop."""
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device:             {self.device}")
        print(f"Training batches:   {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"Checkpoint dir:     {self.checkpoint_dir}")
        print(f"Log dir:            {self.writer.log_dir}")

        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} complete | Avg Loss: {epoch_loss:.4f}")

            if epoch % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

        self.save_checkpoint("checkpoint_final.pt")
        print("Training complete!")

    def save_checkpoint(self, filename):
        """Save checkpoint — only saves progressive decoder weights."""
        checkpoint = {
            'decoder_state_dict': self.model.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'phase': self.phase,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filepath):
        """Load checkpoint — restores progressive decoder weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.phase = checkpoint.get('phase', 1)
        print(f"Checkpoint loaded: {filepath}")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train DCVC + StatelessProgressiveDecoder'
    )
    parser.add_argument('--data_root', type=str, default='data/uvg',
                        help='Root directory of training data')
    parser.add_argument('--dcvc_checkpoint', type=str, default=None,
                        help='Path to DCVC pre-trained checkpoint')
    parser.add_argument('--epochs', type=int, default=600,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (keep at 1 for 1080p on 8 GB GPU)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lambda_rd', type=float, default=0.01,
                        help='Rate-distortion trade-off weight')
    parser.add_argument('--num_refinement_blocks', type=int, default=3,
                        help='Number of progressive refinement stages')
    parser.add_argument('--residual_scale', type=float, default=0.1,
                        help='Residual scale for refinement blocks')
    parser.add_argument('--log_dir', type=str, default='logs_dcvc_progressive',
                        help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint save directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from a checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Auto-find DCVC checkpoint if not specified
    if args.dcvc_checkpoint is None:
        args.dcvc_checkpoint = get_default_checkpoint()
        if args.dcvc_checkpoint:
            print(f"Auto-found DCVC checkpoint: {args.dcvc_checkpoint}")
        else:
            print("Warning: No DCVC checkpoint found. Using random initialization.")

    # Create model
    print("\nCreating DCVC + Progressive model...")
    model = DCVCProgressiveModel(
        dcvc_checkpoint=args.dcvc_checkpoint,
        num_refinement_blocks=args.num_refinement_blocks,
        residual_scale=args.residual_scale,
        device=device
    )

    # Loss function
    loss_fn = DCVCProgressiveLoss(
        weights=[1.0, 0.9, 0.8, 0.7],
        lambda_rd=args.lambda_rd,
    )

    # Transforms
    train_transform, val_transform = get_transforms(resize_h=256, crop_h=256)

    # Build datasets
    train_dataset, val_dataset = build_datasets(
        data_root=args.data_root,
        sequence_length=2,
        train_stride=1,
        val_stride=2,
        train_transform=train_transform,
        val_transform=val_transform,
        val_ratio=0.2,
    )
    print(f"\nDataset sizes — train: {len(train_dataset)}, val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    # Create trainer
    trainer = DCVCProgressiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        lr=args.lr,
        loss_fn=loss_fn,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()