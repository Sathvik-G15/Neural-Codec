"""
Dry-run Phase 1 training with synthetic data.

Tests the training pipeline end-to-end without requiring Vimeo90k.
"""

import sys
sys.path.insert(0, 'E:\\Sathvik\\programming\\Research\\Neural-Codec\\budget_adaptive_decoder')

import torch
import tempfile
import os
import numpy as np
from PIL import Image

from models.decoder import BudgetAdaptiveDecoder
from models.teacher import DCVCWrapper
from training.phase1_distill import Phase1Trainer
from data.generic_video_dataset import GenericVideoDataset, make_phase1_loader


def create_synthetic_video_dataset(tmpdir, n_frames=10, height=256, width=448):
    """Create fake video frames for testing."""
    for i in range(n_frames):
        # Create random RGB image
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(os.path.join(tmpdir, f"frame_{i:03d}.png"))
    print(f"Created {n_frames} synthetic frames in {tmpdir}")
    return tmpdir


def run_dry_run():
    print("=" * 60)
    print("Phase 1 Dry-Run with Synthetic Data")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create synthetic dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        create_synthetic_video_dataset(tmpdir, n_frames=20)

        # Create data loader
        train_loader = make_phase1_loader(
            tmpdir, batch_size=4, num_workers=0, shuffle=True
        )
        val_loader = make_phase1_loader(
            tmpdir, batch_size=4, num_workers=0, shuffle=False
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Create models
        print("\nLoading teacher (DCVC)...")
        teacher = DCVCWrapper(load_pretrained=True).to(device)
        print(f"Teacher loaded. DCVC is None: {teacher.dcvc is None}")

        print("\nCreating decoder...")
        decoder = BudgetAdaptiveDecoder().to(device)
        print(f"Decoder created. Parameters: {sum(p.numel() for p in decoder.parameters()):,}")

        # Optimizer
        optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

        # Config
        config = {
            "patience": 5,
            "min_delta": 0.001,
            "save_every": 100,
            "num_epochs": 3,
        }

        # Create trainer
        trainer = Phase1Trainer(
            decoder=decoder,
            teacher=teacher,
            optimizer=optimizer,
            device=device,
            config=config,
            checkpoint_dir=Path("./phase1_dryrun_checkpoints"),
        )

        # Run 3 epochs
        print("\n" + "=" * 60)
        print("Training for 3 epochs (dry-run)")
        print("=" * 60)

        for epoch in range(1, 4):
            train_metrics = trainer.train_epoch(train_loader, epoch)
            val_metrics = trainer.validate(val_loader)

            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss_total']:.6f}")
            print(f"    S1 (teacher): {train_metrics['loss_s1_teacher']:.6f}")
            print(f"    S2 (GT): {train_metrics['loss_s2_gt']:.6f}")
            print(f"    S3 (GT): {train_metrics['loss_s3_gt']:.6f}")
            print(f"    S4 (GT): {train_metrics['loss_s4_gt']:.6f}")
            print(f"  Val Loss: {val_metrics['val_loss_total']:.6f}")

            # Check for NaN
            if torch.isnan(torch.tensor(train_metrics['loss_total'])):
                print("  WARNING: NaN detected in training loss!")
                break

        print("\n" + "=" * 60)
        print("Dry-run completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Get real video data (Vimeo90k or other)")
        print("2. Run Experiment 1 on real data")
        print("3. If passed: run full Phase 1 training")


if __name__ == "__main__":
    from pathlib import Path
    try:
        run_dry_run()
    except Exception as e:
        print(f"\nError during dry-run: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)