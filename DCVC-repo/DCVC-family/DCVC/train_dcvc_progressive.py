"""
train_dcvc_progressive.py
=========================
Fine-tune DCVC's progressive contextual decoder while keeping everything
else (encoder, optical flow, entropy model) **frozen**.

The bitstream is unchanged — all four compute tiers decode the same bits.

Usage
-----
    python train_dcvc_progressive.py \\
        --pretrained  checkpoints/dcvc_baseline.pth.tar \\
        --data_root   ../../DCVC-Scalable/data/uvg \\
        --lambda_rd   0.0483 \\
        --epochs      300 \\
        --batch_size  2 \\
        --log_dir     runs/progressive \\
        --ckpt_dir    checkpoints/progressive

Training strategy
-----------------
Phase 0  (epochs 0-49)   : Freeze ALL, train decoder ONLY, depth=4 fixed,
                           MSE loss only (no rate term) — decoder alignment.
Phase 1  (epochs 50-299) : Freeze encoder/entropy/motion, train decoder,
                           random depth each batch, full RD loss.
"""

import gc
import os
import sys
import argparse
import random
import psutil
from pathlib import Path
import ctypes

try:
    libc = ctypes.CDLL("libc.so.6")
except Exception:
    libc = None

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from PIL import Image
import numpy as np

from src.models.DCVC_net import DCVC_net


# =============================================================================
# Dataset
# =============================================================================

class Vimeo90kDataset(Dataset):
    """Yields consecutive frame pairs (ref, current) from Vimeo-90k septuplets."""

    def __init__(self, root: str, list_file: str, patch_size: int = 256):
        base_dir = Path(root)
        # Handle cases where the dataset is extracted flat (like in Kaggle) 
        # or nested in 'vimeo_septuplet' (like the official zip)
        if (base_dir / 'vimeo_septuplet').exists():
            base_dir = base_dir / 'vimeo_septuplet'

        self.root = base_dir / 'sequences'
        self.patch_size = patch_size
        self.sequences = []
        
        list_path = base_dir / list_file
        if list_path.exists():
            with open(list_path, 'r') as f:
                self.sequences = [line.strip() for line in f if line.strip()]
        else:
            print(f"Warning: {list_path} not found. Ensure Vimeo-90k is downloaded.")

        print(f"  Dataset ({list_file}): {len(self.sequences)} sequences.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.root / self.sequences[idx]
        
        # Randomly select a pair of consecutive frames from the 7 available
        frame_idx = random.randint(1, 6)
        ref_path = seq_path / f"im{frame_idx}.png"
        cur_path = seq_path / f"im{frame_idx + 1}.png"
        
        ref = self._load(ref_path)
        cur = self._load(cur_path)
        
        # Random crop
        _, H, W = ref.shape
        ps = self.patch_size
        if H > ps and W > ps:
            y0 = random.randint(0, H - ps)
            x0 = random.randint(0, W - ps)
            ref = ref[:, y0:y0+ps, x0:x0+ps]
            cur = cur[:, y0:y0+ps, x0:x0+ps]
            
        self._counter = getattr(self, "_counter", 0) + 1
        
        if libc is not None and self._counter % 200 == 0:
            libc.malloc_trim(0)
            
        return ref, cur

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        try:
            import torchvision.io as io
            # read_image returns uint8 tensor of shape [C, H, W]
            # Bypasses PIL and Numpy entirely, avoiding glibc fragmentation leaks!
            tensor = io.read_image(str(path))
            return tensor.float() / 255.0
        except Exception:
            # Fallback for missing files during download
            return torch.zeros(3, 256, 256)


# =============================================================================
# Loss
# =============================================================================

class ProgressiveLoss(nn.Module):
    """RD loss with deep supervision across all 4 decode tiers.

    L = bpp  +  lambda_rd * sum_i(w_i * MSE(recon_i, target))

    Base weights [1.0, 0.9, 0.8, 0.7] prioritise early tiers so the base 
    tier (Tier 1) provides strong initial quality. These weights are 
    normalized to sum to 1.0 to ensure the overall gradient magnitude 
    matches the original DCVC scaling for lambda_rd.
    """

    def __init__(self, lambda_rd: float = 0.0483, use_rate: bool = True):
        super().__init__()
        self.lambda_rd = lambda_rd
        self.use_rate  = use_rate
        self.mse = nn.MSELoss()
        
        base_weights = [1.0, 0.9, 0.8, 0.7]
        norm = sum(base_weights)
        self.TIER_WEIGHTS = [w / norm for w in base_weights]

    def forward(self, recon_images, target, bpp):

        distortion = 0.0

        for i, recon in enumerate(recon_images):
            w = self.TIER_WEIGHTS[i] if i < len(self.TIER_WEIGHTS) else self.TIER_WEIGHTS[-1]
            distortion = distortion + w * self.mse(recon, target)

        rate = bpp if self.use_rate else 0.0
        loss = rate + self.lambda_rd * distortion

        mse_best = self.mse(recon_images[-1], target).item()
        psnr = 10 * np.log10(1.0 / (mse_best + 1e-8))

        return {
            'loss':       loss,
            'rate':       rate.detach() if isinstance(rate, torch.Tensor) else rate,
            'distortion': distortion.detach(),
            'psnr':       psnr,
        }



# =============================================================================
# Freeze / unfreeze helpers
# =============================================================================

def _freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)


def freeze_encoder_and_entropy(model: DCVC_net):
    """Freeze everything except the progressive decoder."""
    _freeze(model)                              # freeze all
    _unfreeze(model.progressive_decoder)        # unfreeze only our module
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}  "
          f"({100*trainable/total:.1f}%)")


# =============================================================================
# Trainer
# =============================================================================

class Trainer:

    def __init__(self, args):
        self.args   = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        # ── Model ──────────────────────────────────────────────────────────
        self.model = DCVC_net().to(self.device)
        self.start_epoch = 0
        
        # Load Microsoft base weights first
        self._load_pretrained(args.pretrained)
        freeze_encoder_and_entropy(self.model)

        # Wrap in DataParallel to use both T4 GPUs on Kaggle
        if torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel!")
            self.parallel_model = nn.DataParallel(self.model)
        else:
            self.parallel_model = self.model

        # ── Loss ───────────────────────────────────────────────────────────
        self.loss_fn_warmup = ProgressiveLoss(lambda_rd=args.lambda_rd, use_rate=False)
        self.loss_fn_train  = ProgressiveLoss(lambda_rd=args.lambda_rd, use_rate=True)

        # ── Optimiser (only progressive decoder params) ────────────────────
        self.optimizer = optim.Adam(
            self.model.progressive_decoder.parameters(),
            lr=args.lr,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-6,
        )
        self.scaler = torch.cuda.amp.GradScaler()

        # ── Resume State ───────────────────────────────────────────────────
        if args.resume:
            self._load_resume(args.resume)

        # ── Data ───────────────────────────────────────────────────────────
        train_ds = Vimeo90kDataset(args.data_root, list_file='sep_trainlist.txt', patch_size=args.patch_size)
        val_ds   = Vimeo90kDataset(args.data_root, list_file='sep_testlist.txt',  patch_size=args.patch_size)

        # DataLoader: Use 2 workers with spawn+file_system to bypass Kaggle memory
        # fragmentation and /dev/shm exhaustion. Keep pin_memory=False for safety.
        self.train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True
        )
        # Limit validation to 100 random pairs per epoch to speed it up
        self.val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            pin_memory=False,
            persistent_workers=True
        )

        # ── Logging ────────────────────────────────────────────────────────
        self.writer   = SummaryWriter(args.log_dir)
        self.ckpt_dir = Path(args.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_psnr = 0.0

    # ------------------------------------------------------------------

    def _load_pretrained(self, ckpt_path: str):
        if not ckpt_path or not Path(ckpt_path).exists():
            print("  No pretrained checkpoint — training from scratch (not recommended).")
            return
        print(f"  Loading pretrained weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        # Handle 'state_dict' / 'model_state_dict' wrappers
        if isinstance(state, dict):
            for key in ('state_dict', 'model_state_dict', 'model'):
                if key in state:
                    state = state[key]
                    break
        self.model.load_dict(state)
        print("  Pretrained weights loaded.")

    # ------------------------------------------------------------------
    
    def _load_resume(self, resume_path: str):
        if not Path(resume_path).exists():
            print(f"  Resume checkpoint {resume_path} not found.")
            return
        print(f"  Resuming from checkpoint {resume_path}")
        state = torch.load(resume_path, map_location=self.device, weights_only=True)
        self.start_epoch = state['epoch'] + 1
        self.best_val_psnr = state.get('val_psnr', 0.0)
        self.model.progressive_decoder.load_state_dict(state['progressive_decoder_state'])
        
        if 'optimizer_state' in state:
            self.optimizer.load_state_dict(state['optimizer_state'])
        if 'scheduler_state' in state:
            self.scheduler.load_state_dict(state['scheduler_state'])
        if 'scaler_state' in state:
            self.scaler.load_state_dict(state['scaler_state'])
            
        print(f"  Resumed at epoch {self.start_epoch}. Best PSNR: {self.best_val_psnr:.2f} dB")

    # ------------------------------------------------------------------

    def _is_warmup(self, epoch: int) -> bool:
        return epoch < self.args.warmup_epochs

    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        # Keep frozen modules in eval mode so BatchNorm stats don't drift
        self.model.opticFlow.eval()
        self.model.contextualEncoder.eval()
        self.model.temporalPriorEncoder.eval()

        loss_fn = self.loss_fn_warmup if self._is_warmup(epoch) else self.loss_fn_train
        total_loss = total_psnr = 0.0
        n = 0

        total_batches = len(self.train_loader)

        for ref, cur in self.train_loader:
            ref = ref.to(self.device)
            cur = cur.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass through DataParallel model
            # PyTorch automatically disables gradient tracking for the frozen 
            # encoder/entropy parts and tracks gradients ONLY for the decoder.
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = self.parallel_model(ref, cur)
                recon_images = out['recon_images']
                bpp_val = out['bpp']
                
                # DataParallel might return a list or stacked tensor
                if isinstance(bpp_val, torch.Tensor):
                    bpp_val = bpp_val.mean()
                elif isinstance(bpp_val, list):
                    bpp_val = sum(bpp_val) / len(bpp_val)

                metrics = loss_fn(recon_images, cur, bpp_val)

            self.scaler.scale(metrics['loss']).backward()

            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.progressive_decoder.parameters(), max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += metrics['loss'].item()
            total_psnr += metrics['psnr']
            n += 1
            
            # Explicitly delete tensors to prevent any reference accumulation
            del out
            del recon_images
            del bpp_val
            del metrics
            del ref, cur

            # Force standard print to Kaggle log every 1000 batches (bypassing Jupyter buffers)
            if n % 1000 == 0:
                # Actively release Python-side garbage and defragment the CUDA
                # allocator pool before checking memory stats.
                gc.collect()
                torch.cuda.empty_cache()

                process   = psutil.Process(os.getpid())
                ram_gb    = process.memory_info().rss / 1024**3
                gpu_alloc = torch.cuda.memory_allocated(0) / 1024**3
                gpu_rsrv  = torch.cuda.memory_reserved(0)  / 1024**3  # allocator pool
                print(f"  Epoch {epoch} [Train] Batch {n}/{total_batches} | "
                      f"loss: {total_loss/n:.4f} | psnr: {total_psnr/n:.2f} | "
                      f"RAM: {ram_gb:.2f}GB | "
                      f"GPU alloc: {gpu_alloc:.2f}GB / reserved: {gpu_rsrv:.2f}GB",
                      flush=True)

        return {'loss': total_loss / n, 'psnr': total_psnr / n}

    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        self.model.eval()
        total_psnr = total_bpp = 0.0
        n = 0

        for ref, cur in self.val_loader:
            ref = ref.to(self.device)
            cur = cur.to(self.device)
            out = self.parallel_model(ref, cur)
            recon_images = out['recon_images']

            mse  = nn.functional.mse_loss(recon_images[-1], cur).item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-8))
            total_psnr += psnr
            
            bpp_val = out['bpp']
            if isinstance(bpp_val, torch.Tensor):
                bpp_val = bpp_val.mean().item()
            elif isinstance(bpp_val, list):
                bpp_val = sum(v.mean().item() if isinstance(v, torch.Tensor) else v for v in bpp_val) / len(bpp_val)
            total_bpp += bpp_val

            # Free GPU tensors immediately — without this, each iteration holds two
            # batches' worth of reconstruction tensors alive simultaneously.
            del out, recon_images, ref, cur

            n += 1
            if n >= 100:  # Early stop validation to save time (Vimeo90k test set is large)
                break

        return {'psnr': total_psnr / n, 'bpp': total_bpp / n}

    # ------------------------------------------------------------------

    def run(self):
        args = self.args
        print(f"\nTraining for {args.epochs} epochs "
              f"({args.warmup_epochs} warmup, then full RD loss)\n")

        for epoch in range(self.start_epoch, args.epochs):
            phase = 'warmup' if self._is_warmup(epoch) else 'train'
            train_metrics = self.train_one_epoch(epoch)
            val_metrics   = self.validate(epoch)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']

            print(
                f"Epoch {epoch:4d}/{args.epochs}  [{phase:6s}]  "
                f"loss={train_metrics['loss']:.4f}  "
                f"train_psnr={train_metrics['psnr']:.2f}  "
                f"val_psnr={val_metrics['psnr']:.2f}  "
                f"val_bpp={val_metrics['bpp']:.4f}  "
                f"lr={lr:.2e}"
            )

            # TensorBoard
            self.writer.add_scalar('train/loss',  train_metrics['loss'], epoch)
            self.writer.add_scalar('train/psnr',  train_metrics['psnr'], epoch)
            self.writer.add_scalar('val/psnr',    val_metrics['psnr'],   epoch)
            self.writer.add_scalar('val/bpp',     val_metrics['bpp'],    epoch)
            self.writer.add_scalar('lr',          lr,                     epoch)

            # Checkpoint (best + periodic)
            if val_metrics['psnr'] > self.best_val_psnr:
                self.best_val_psnr = val_metrics['psnr']
                self._save(epoch, tag='best')

            if (epoch + 1) % 50 == 0:
                self._save(epoch, tag=f'epoch{epoch+1:04d}')

        self.writer.close()
        print(f"\nDone. Best val PSNR: {self.best_val_psnr:.2f} dB")

    def _save(self, epoch: int, tag: str):
        path = self.ckpt_dir / f'progressive_{tag}.pth.tar'
        temp_path = self.ckpt_dir / f'progressive_{tag}_temp.pth.tar'
        
        torch.save({
            'epoch':             epoch,
            'val_psnr':          self.best_val_psnr,
            # Only save the trainable decoder + optimiser state.
            # Saving the full frozen model via model.state_dict() causes a large
            # CPU-RAM spike at every checkpoint (torch.save builds an in-memory
            # copy before writing to disk). The full model can be reconstructed
            # at resume time by loading the pretrained weights first, then
            # overlaying progressive_decoder_state.
            'progressive_decoder_state': self.model.progressive_decoder.state_dict(),
            'optimizer_state':   self.optimizer.state_dict(),
            'scheduler_state':   self.scheduler.state_dict(),
            'scaler_state':      self.scaler.state_dict(),
        }, temp_path)
        
        # Atomic rename prevents corrupted checkpoints if Kaggle kills the job during save
        os.replace(temp_path, path)
        print(f"  Saved checkpoint: {path}")


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Fine-tune DCVC progressive decoder (encoder frozen)'
    )
    p.add_argument('--pretrained',     type=str, default='',
                   help='Path to pretrained DCVC checkpoint (.pth.tar)')
    p.add_argument('--resume',         type=str, default='',
                   help='Path to progressive_best.pth.tar to resume training after a timeout')
    p.add_argument('--data_root',      type=str, default='../../../DCVC-Scalable/data',
                   help='Directory containing the vimeo_septuplet/ folder')
    p.add_argument('--lambda_rd',      type=float, default=0.0483,
                   help='RD tradeoff weight. DCVC uses: 0.0483 | 0.0250 | 0.0130 | 0.0067')
    p.add_argument('--epochs',         type=int, default=300)
    p.add_argument('--warmup_epochs',  type=int, default=50,
                   help='Epochs of MSE-only training before enabling rate loss')
    p.add_argument('--batch_size',     type=int, default=2)
    p.add_argument('--patch_size',     type=int, default=256)
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--log_dir',        type=str, default='runs/progressive')
    p.add_argument('--ckpt_dir',       type=str, default='checkpoints/progressive')
    return p.parse_args()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")

    torch.backends.cudnn.benchmark = True
    
    args = parse_args()
    trainer = Trainer(args)
    trainer.run()
