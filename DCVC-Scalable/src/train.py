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
import glob
import numpy as np


# ===========================================================================
# Datasets
# ===========================================================================

class VideoFrameDataset(torch.utils.data.Dataset):
    """Video frame dataset.

    Loads consecutive-frame clips from a directory of video sequences.
    Each sequence lives in its own subdirectory.

    Directory layout::

        data_root/
            sequence1/
                frame_0000.png
                frame_0001.png
                ...
            sequence2/
                ...

    Args:
        data_root:       Root directory containing per-sequence subdirs.
        sequence_length: Number of consecutive frames per clip.
        stride:          Step between clip start frames (controls overlap).
        transform:       torchvision transform applied to each frame.
        allowed_dirs:    If provided, only sequences inside these directories
                         are included (used for train/val splitting).
    """

    def __init__(self, data_root, sequence_length=8, stride=2,
                 transform=None, allowed_dirs=None):
        self.data_root       = Path(data_root)
        self.sequence_length = sequence_length
        self.stride          = stride
        self.transform       = transform
        # Normalise to strings for fast set look-up
        self.allowed_dirs = (
            set(str(d) for d in allowed_dirs) if allowed_dirs is not None else None
        )

        self.sequences = []
        self._build_sequence_list()

    def _build_sequence_list(self):
        """Populate self.sequences with (seq_dir, [frame_paths]) tuples."""
        seq_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])

        if self.allowed_dirs is not None:
            seq_dirs = [d for d in seq_dirs if str(d) in self.allowed_dirs]

        for seq_dir in seq_dirs:
            frames = sorted(
                f for f in seq_dir.iterdir()
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
            )

            if len(frames) >= self.sequence_length:
                for start in range(
                    0, len(frames) - self.sequence_length + 1, self.stride
                ):
                    self.sequences.append(
                        (seq_dir, frames[start : start + self.sequence_length])
                    )

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

        return torch.stack(clip, dim=0)   # (T, C, H, W)


class UVGDataset(torch.utils.data.Dataset):
    """UVG Dataset — kept for reference / evaluation use."""

    UVG_SEQUENCES = [
        'Beauty', 'Bosphorus', 'FoodMarket', 'HoneyBee',
        'Jockey', 'ReadySetGo', 'ShakeNDrop', 'Sistine',
    ]

    def __init__(self, data_root, sequences=None, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.clips     = []

        if sequences is None:
            sequences = self.UVG_SEQUENCES

        for seq_name in sequences:
            seq_dir = self.data_root / seq_name
            if seq_dir.exists():
                frames = sorted(
                    f for f in seq_dir.iterdir()
                    if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
                )
                if len(frames) >= 8:
                    for i in range(0, min(len(frames) - 8, 32), 2):
                        self.clips.append(frames[i : i + 8])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = []
        for frame_path in self.clips[idx]:
            img = Image.open(frame_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            clip.append(img)
        return torch.stack(clip, dim=0)


def build_datasets(
    data_root,
    sequence_length=8,
    train_stride=4,
    val_stride=8,
    train_transform=None,
    val_transform=None,
    val_ratio=0.2,
):
    """Build train/val datasets with non-overlapping video sequences.

    Splits by sequence directory so train and val **never share the same video**.

    Args:
        data_root:        Root directory containing per-sequence subdirs.
        sequence_length:  Frames per clip.
        train_stride:     Step between clip starts for training (more overlap = more data).
        val_stride:       Step between clip starts for validation.
        train_transform:  Transform applied to training frames.
        val_transform:    Transform applied to validation frames.
        val_ratio:        Fraction of sequences reserved for validation (default 20 %).

    Returns:
        (train_dataset, val_dataset)
    """
    data_root = Path(data_root)
    seq_dirs  = sorted([d for d in data_root.iterdir() if d.is_dir()])

    if not seq_dirs:
        raise ValueError(f"No sequence subdirectories found in '{data_root}'.")

    split_idx  = max(1, int(len(seq_dirs) * (1 - val_ratio)))
    train_dirs = seq_dirs[:split_idx]
    # Edge case: if only one sequence exists, val reuses the last sequence
    val_dirs   = seq_dirs[split_idx:] if split_idx < len(seq_dirs) else seq_dirs[-1:]

    print(f"Train sequences ({len(train_dirs)}): "
          f"{[d.name for d in train_dirs]}")
    print(f"Val   sequences ({len(val_dirs)}  ): "
          f"{[d.name for d in val_dirs]}")

    train_dataset = VideoFrameDataset(
        data_root, sequence_length, train_stride, train_transform,
        allowed_dirs=train_dirs,
    )
    val_dataset = VideoFrameDataset(
        data_root, sequence_length, val_stride, val_transform,
        allowed_dirs=val_dirs,
    )
    return train_dataset, val_dataset


# ===========================================================================
# Transforms
# ===========================================================================

def get_transforms(resize_h=256, crop_h=256):
    """Return (train_transform, val_transform, target_size=None).

    The decoder now fully inverts the encoder (3× PixelShuffle = 8× upsample,
    matching the encoder's 8× downscale), so no target downsampling is needed.
    target_size is returned as None for backward compatibility.

    Width is rounded to the nearest multiple of 8 so that stride-2 convolutions
    divide evenly through the encoder.
    """
    crop_w   = (int(crop_h   * 16 / 9) // 8) * 8
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

    # target_size=None: model outputs full resolution, no interpolation needed
    return train_transform, val_transform, None


# ===========================================================================
# Metrics
# ===========================================================================

def psnr_from_mse(mse: float) -> float:
    """Convert MSE → PSNR (dB), assuming pixel values in [0, 1]."""
    if mse <= 0.0:
        return float('inf')
    return 10.0 * math.log10(1.0 / mse)


# ===========================================================================
# Trainer
# ===========================================================================

class Trainer:
    """Training loop for DCVC Progressive."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        log_dir='logs',
        checkpoint_dir='checkpoints',
        lr=1e-4,
        weight_decay=1e-5,
        gradient_clip=1.0,
        log_interval=100,
        val_interval=1000,
        checkpoint_interval=5000,
        loss_fn=None,
    ):
        self.model       = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device

        self.gradient_clip       = gradient_clip
        self.log_interval        = log_interval
        self.val_interval        = val_interval
        self.checkpoint_interval = checkpoint_interval

        # Default: DCVCProgressiveLoss imported lazily to avoid circular imports
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            from dcvc_progressive import DCVCProgressiveLoss
            self.loss_fn = DCVCProgressiveLoss()

        # Only optimise the progressive decoder — encoder is FROZEN (plan Section 2.1)
        decoder_params = (
            model.decoder.parameters()
            if hasattr(model, 'decoder')
            else model.parameters()
        )
        self.optimizer = optim.Adam(
            decoder_params, lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.writer          = SummaryWriter(log_dir)
        self.checkpoint_dir  = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.phase       = 1

    # ------------------------------------------------------------------
    # Core train / validate steps
    # ------------------------------------------------------------------

    def train_step(self, batch, depth):
        """Single training step using a frame pair (reference + current).

        Uses clip[:, 0] as the *reference* frame (temporal context) and
        clip[:, 1] as the *current* frame to compress and reconstruct.
        This trains the model as a proper conditional video codec rather
        than an image auto-encoder.

        Args:
            batch: Tensor of shape (B, T, C, H, W), T >= 2.
            depth: Number of progressive decoder stages to run.

        Returns:
            Scalar loss value (float).
        """
        self.model.train()

        clip    = batch.to(self.device)   # (B, T, C, H, W)
        ref     = clip[:, 0]              # reference frame  (B, C, H, W)
        current = clip[:, 1]              # current frame to compress
        target  = clip[:, 1]             # reconstruct current at full resolution

        use_amp = self.device.type == 'cuda'
        with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
            result  = self.model(current, ref=ref, depth=depth)
            outputs = result['reconstructions']   # list of (B, 3, H, W) at full res
            latent  = result['latent']

            loss = self.loss_fn(outputs, target, latent)

        self.optimizer.zero_grad()
        loss.backward()

        if self.gradient_clip > 0:
            # Clip only decoder parameters (encoder is frozen — no gradients there)
            decoder_params = (
                self.model.decoder.parameters()
                if hasattr(self.model, 'decoder')
                else self.model.parameters()
            )
            torch.nn.utils.clip_grad_norm_(decoder_params, self.gradient_clip)

        self.optimizer.step()
        return loss.item()

    def validate(self):
        """Validation loop.

        Returns:
            (avg_loss, avg_psnr_dB): averaged over up to 50 validation batches.
        """
        self.model.eval()
        val_loss = 0.0
        val_mse  = 0.0
        num_batches = min(len(self.val_loader), 50)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= num_batches:
                    break

                clip    = batch.to(self.device)
                ref     = clip[:, 0]
                current = clip[:, 1]
                target  = clip[:, 1]

                result  = self.model(current, ref=ref, depth=4)
                outputs = result['reconstructions']
                latent  = result['latent']

                loss = self.loss_fn(outputs, target, latent)
                val_loss += loss.item()

                # PSNR from the final (highest-quality) reconstruction
                final_recon = outputs[-1].clamp(0.0, 1.0)
                val_mse    += F.mse_loss(final_recon, target).item()

        avg_loss = val_loss / num_batches
        avg_mse  = val_mse  / num_batches
        avg_psnr = psnr_from_mse(avg_mse)
        return avg_loss, avg_psnr

    # ------------------------------------------------------------------
    # Epoch / training loop
    # ------------------------------------------------------------------

    def train_epoch(self, epoch):
        """Train for one epoch with curriculum depth scheduling.

        Phase 1 (epochs   0-199): always depth=4 to stabilise all decoder stages.
        Phase 2 (epochs 200-399): random depth — teaches anytime behaviour.
        Phase 3 (epochs 400+   ): random depth — continued fine-tuning.
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

        epoch_loss  = 0.0
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
                self.writer.add_scalar('train/loss',  avg_loss,  self.global_step)
                self.writer.add_scalar('train/phase', self.phase, self.global_step)
                self.writer.add_scalar('train/depth', depth,      self.global_step)
                self.writer.add_scalar(
                    'train/lr',
                    self.optimizer.param_groups[0]['lr'],
                    self.global_step,
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

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step':          self.global_step,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        print(f"Checkpoint loaded: {filepath}")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Train DCVC Progressive')
    parser.add_argument('--data_root',      type=str,   default='data/uvg',
                        help='Root directory of training data')
    parser.add_argument('--epochs',         type=int,   default=600,
                        help='Number of training epochs')
    parser.add_argument('--batch_size',     type=int,   default=1,
                        help='Batch size (keep at 1 for 1080p on 8 GB GPU)')
    parser.add_argument('--lr',             type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lambda_rd',      type=float, default=0.01,
                        help='Rate-distortion trade-off weight '
                             '(higher → smaller latent / lower bitrate)')
    parser.add_argument('--log_dir',        type=str,   default='logs',
                        help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str,   default='checkpoints',
                        help='Checkpoint save directory')
    parser.add_argument('--resume',         type=str,   default=None,
                        help='Resume training from a checkpoint file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from dcvc_progressive import DCVCProgressive, DCVCProgressiveLoss, count_parameters

    model = DCVCProgressive(
        out_channel_M=96,
        out_channel_N=64,
        num_refinement_blocks=3,
    )
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Rate-distortion loss with configurable lambda
    loss_fn = DCVCProgressiveLoss(
        weights=[1.0, 0.9, 0.8, 0.7],
        lambda_rd=args.lambda_rd,
    )

    # Transforms — target_size is None (model now outputs full resolution)
    train_transform, val_transform, _ = get_transforms(resize_h=256, crop_h=256)

    # Build non-overlapping train/val datasets (split by video sequence)
    train_dataset, val_dataset = build_datasets(
        data_root=args.data_root,
        sequence_length=8,
        train_stride=4,
        val_stride=8,
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

    trainer = Trainer(
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