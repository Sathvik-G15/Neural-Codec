"""Train Progressive Decoder (Phase A/B/C curriculum, descending-weight deep supervision).

Strategy
--------
Phase A (0..4999)       : fixed depth=4 (model alignment)
Phase B (5000..34999)   : UNIFORM random depth [1..4]
Phase C (35000..59999)  : UNIFORM random depth [1..4]

Both late phases use uniform sampling to train all tiers equally.
Endpoint-biased sampling under-trains the mid tiers.

Loss
----
L = lambda_rd * sum_i w_i * MSE(recon_i, target)
weights = (1.0, 0.9, 0.8, 0.7)  (descending stabilisation, no rate term)

Tier 1 output equals the original DCVC reconstruction by construction
(TieredRefinement has zero-init proj_out at construction time).

Lambda escalation rule
----------------------
Start at lambda_rd = 1e-3. After ~5k..10k steps, if mean tier gap < 0.2 dB,
restart with --lambda_rd 2e-3. Do not exceed 3e-3.
Escalation is MANUAL (not auto-applied inside the script).

Usage (Kaggle)
--------------
    !python dcvc_progressive/scripts/train_progressive.py \
        --pretrained /kaggle/input/dcvc-baseline/model_dcvc_quality_3_psnr.pth.tar \
        --data_root  /kaggle/input/vimeo90k-septuplet \
        --total_steps 60000 \
        --batch_size 4 \
        --lambda_rd 1e-3 \
        --log_dir    /kaggle/working/runs/progressive \
        --ckpt_dir   /kaggle/working/checkpoints/progressive
"""

import argparse
import gc
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.DCVC_net import DCVC_net

# =============================================================================
# Loss
# =============================================================================

class DCVCProgressiveLoss(nn.Module):
    """Deep-supervision loss across all available tiers of recon_images.

    L = lambda_rd * sum_i w_i * MSE(recon_i, target)

    Weights default to (1.0, 0.9, 0.8, 0.7) -- descending. Earlier tiers
    receive more gradient; this stabilises Tier 1 quality.

    The weight list is sliced to len(recon_list) so the same loss handles
    variable depth returned by DCVC_net.forward(...).
    """

    def __init__(
        self,
        weights: tuple = (1.0, 0.9, 0.8, 0.7),
        lambda_rd: float = 1e-3,
    ):
        super().__init__()
        self.weights = list(weights)
        self.lambda_rd = lambda_rd

    def forward(
        self,
        recon_list,
        target: torch.Tensor,
    ) -> torch.Tensor:
        w_eff = self.weights[:len(recon_list)]
        distortion = sum(
            w * F.mse_loss(r, target)
            for w, r in zip(w_eff, recon_list)
        )
        return self.lambda_rd * distortion


# =============================================================================
# Dataset (Vimeo-90k Septuplet)
# =============================================================================

class Vimeo90kSeptupletDataset(Dataset):
    """Yields consecutive frame pairs (ref, current) from Vimeo-90k septuplets."""

    def __init__(self, root: str, list_file: str, patch_size: int = 256):
        base_dir = Path(root)
        if (base_dir / "vimeo_septuplet").exists():
            base_dir = base_dir / "vimeo_septuplet"

        self.root = base_dir / "sequences"
        self.patch_size = patch_size
        self.sequences = []

        list_path = base_dir / list_file
        if list_path.exists():
            with open(list_path, "r") as f:
                self.sequences = [line.strip() for line in f if line.strip()]
            print(f"  [Vimeo90k:{list_file}] {len(self.sequences)} sequences.")
        else:
            print(f"  [Vimeo90k:WARN] {list_path} not found. "
                  f"Ensure vimeo_septuplet/{list_file} is on the data path.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.root / self.sequences[idx]
        frame_idx = random.randint(1, 6)
        ref = self._load(seq_path / f"im{frame_idx}.png")
        cur = self._load(seq_path / f"im{frame_idx + 1}.png")

        _, H, W = ref.shape
        ps = self.patch_size
        if H > ps and W > ps:
            y0 = random.randint(0, H - ps)
            x0 = random.randint(0, W - ps)
            ref = ref[:, y0:y0 + ps, x0:x0 + ps]
            cur = cur[:, y0:y0 + ps, x0:x0 + ps]
        return ref, cur

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        try:
            with Image.open(str(path)) as img:
                img = img.convert("RGB")
                arr = np.array(img, copy=True)
            return torch.from_numpy(arr).permute(2, 0, 1)
        except Exception:
            return torch.zeros(3, 256, 256)


# =============================================================================
# Trainer
# =============================================================================

class ProgressiveTrainer:
    """Decoder-only trainer with deep supervision across all tiers."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # ----- Model -----
        self.model = DCVC_net().to(self.device)
        self._load_pretrained(args.pretrained)
        self._freeze_encoder_and_entropy()
        self._apply_residual_damping(args.residual_damping)

        # ----- Loss / Optimiser -----
        self.loss_fn = DCVCProgressiveLoss(
            weights=(1.0, 0.9, 0.8, 0.7),
            lambda_rd=args.lambda_rd,
        )
        self.decoder_params = [
            p for p in self.model.progressive_decoder.parameters()
            if p.requires_grad
        ]
        self.optimizer = optim.Adam(self.decoder_params, lr=args.lr)

        # ----- Datasets / Loaders -----
        train_ds = Vimeo90kSeptupletDataset(
            args.data_root, "sep_trainlist.txt", args.patch_size
        )
        val_ds = Vimeo90kSeptupletDataset(
            args.data_root, "sep_testlist.txt", args.patch_size
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            pin_memory=False,
            persistent_workers=True,
        )

        # ----- Logging / Checkpoints -----
        self.writer = SummaryWriter(args.log_dir)
        self.ckpt_dir = Path(args.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        print(f"Log dir:    {args.log_dir}")
        print(f"Ckpt dir:   {args.ckpt_dir}")

        # ----- Resume State -----
        if args.resume:
            self._load_resume(args.resume)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_pretrained(self, ckpt_path: str):
        if not ckpt_path or not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"pretrained checkpoint not found at: {ckpt_path!r}"
            )
        print(f"  Loading pretrained DCVC weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict):
            for k in ("state_dict", "model_state_dict", "model"):
                if k in state:
                    state = state[k]
                    break
        self.model.load_dict(state)
        print("  Pretrained weights loaded (DCVC_net.load_dict mapping contextualDecoder -> progressive_decoder.part).")

    def _freeze_encoder_and_entropy(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.progressive_decoder.parameters():
            p.requires_grad_(True)
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_tota = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable params: {n_train:,} / {n_tota:,}  "
              f"({100 * n_train / n_tota:.2f}%)")

    def _apply_residual_damping(self, damping: float):
        if damping != 1.0:
            for block in self.model.progressive_decoder.refinement_blocks:
                block.damping = damping
            print(f"  Applied residual_damping = {damping} to all refinement blocks.")

    def _load_resume(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"--resume path not found: {path!r}")
        print(f"  Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.progressive_decoder.load_state_dict(
            ckpt["progressive_decoder_state"], strict=False
        )
        self.step = ckpt.get("global_step", 0)
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"  Resumed at step {self.step} "
              f"(phase {ckpt.get('phase', '?')}).")

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    @staticmethod
    def sample_depth(step: int) -> int:
        """Curriculum depth sampler (uniform in B and C)."""
        if step < 5000:
            return 4
        return int(torch.randint(0, 4, (1,)).item()) + 1

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def train_step(self, ref, cur) -> float:
        depth = self.sample_depth(self.step)
        self.model._current_depth = depth
        out = self.model(ref, cur)
        self.model._current_depth = None

        recon_images = out["recon_images"]
        loss = self.loss_fn(recon_images, cur)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.decoder_params, max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        tier_sums = None
        n = 0

        for ref, cur in self.val_loader:
            ref = ref.to(self.device).float().div_(255.0)
            cur = cur.to(self.device).float().div_(255.0)

            self.model._current_depth = None
            out = self.model(ref, cur)
            self.model._current_depth = None
            recon_images = out["recon_images"]

            if tier_sums is None:
                tier_sums = [0.0] * len(recon_images)

            for i, r in enumerate(recon_images):
                mse = F.mse_loss(r, cur).item()
                tier_sums[i] += 10.0 * np.log10(1.0 / (mse + 1e-8))
            n += 1
            if n >= self.args.val_max_pairs:
                break

        self.model.train()
        if tier_sums is None:
            return {}
        return {f"psnr_t{i+1}_mean": s / max(n, 1) for i, s in enumerate(tier_sums)}

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_ckpt(self, tag: str):
        path = self.ckpt_dir / f"progressive_{tag}.pth.tar"
        tmp = self.ckpt_dir / f"_tmp_progressive_{tag}.pth.tar"
        torch.save(
            {
                "progressive_decoder_state": self.model.progressive_decoder.state_dict(),
                "optimizer_state":          self.optimizer.state_dict(),
                "epoch": -1,
                "global_step": self.step,
                "phase": (
                    "A" if self.step < 5_000
                    else "B" if self.step < 35_000
                    else "C"
                ),
                "config": vars(self.args),
            },
            tmp,
        )
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        print(f"\nTraining for {self.args.total_steps} steps.")
        print(f"  Phase A: 0..4999        (depth=4)")
        print(f"  Phase B: 5000..34999   (uniform [1..4])")
        print(f"  Phase C: 35000..59999   (uniform [1..4])\n")

        self.model.train()
        ref_cur = iter(self.train_loader)
        log_loss = 0.0
        log_n = 0
        t_last = None
        import time as _time

        while self.step < self.args.total_steps:
            try:
                ref, cur = next(ref_cur)
            except StopIteration:
                ref_cur = iter(self.train_loader)
                ref, cur = next(ref_cur)

            ref = ref.to(self.device, non_blocking=True).float().div_(255.0)
            cur = cur.to(self.device, non_blocking=True).float().div_(255.0)

            loss = self.train_step(ref, cur)
            self.step += 1
            log_loss += loss
            log_n += 1

            if self.step % 200 == 0:
                avg = log_loss / log_n
                log_loss = 0.0
                log_n = 0
                depth_now = self.sample_depth(self.step - 1)
                if t_last is None:
                    print(
                        f"  step {self.step}/{self.args.total_steps}  "
                        f"loss={avg:.6f}  depth={depth_now}",
                        flush=True,
                    )
                    t_last = self.step
                else:
                    print(
                        f"  step {self.step}/{self.args.total_steps}  "
                        f"loss={avg:.6f}  depth={depth_now}",
                        flush=True,
                    )
                self.writer.add_scalar("train/loss", avg, self.step)

            if self.step % self.args.val_every == 0:
                m = self.validate()
                vals = " ".join(
                    f"T{i}={m.get(f'psnr_t{i}_mean', float('nan')):.3f}"
                    for i in range(1, 5)
                )
                if len(m) == 4:
                    t1, t2, t3, t4 = [m[f"psnr_t{i}_mean"] for i in range(1, 5)]
                    gap = t4 - t1
                    order = (
                        "OK"
                        if (t1 < t2 < t3 < t4)
                        else "INVERSION_DETECTED"
                    )
                else:
                    gap = float("nan")
                    order = "INCOMPLETE"
                phase = (
                    "A" if self.step < 5000
                    else "B" if self.step < 35000
                    else "C"
                )
                print(
                    f"[VAL step {self.step} phase={phase}] {vals} gap={gap:.3f} {order}",
                    flush=True,
                )
                for k, v in m.items():
                    self.writer.add_scalar(f"val/{k}", v, self.step)

            if self.step % self.args.save_every == 0:
                self.save_ckpt(f"step{self.step}")

        self.save_ckpt("final")
        self.writer.close()
        print(f"\nDone after {self.step} steps. Final ckpt at {self.ckpt_dir}.")



# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train progressive decoder (uniform depth sampling, "
                    "descending-weight deep supervision)"
    )
    p.add_argument("--pretrained", type=str, required=True,
                   help="Path to Microsoft DCVC baseline .pth.tar")
    p.add_argument("--data_root", type=str, required=True,
                   help="Path containing vimeo_septuplet/sequences/ and "
                        "{sep_trainlist.txt, sep_testlist.txt}")
    p.add_argument("--lambda_rd", type=float, default=1e-3,
                   help="Distortion weight. Start at 1e-3; raise to 2e-3 if "
                        "gap < 0.2 dB after 5k-10k steps. Hard cap 3e-3.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--total_steps", type=int, default=60_000,
                   help="Total optimisation steps. 60k ~ 3.7 epochs on Vimeo-90k.")
    p.add_argument("--val_every", type=int, default=2_000)
    p.add_argument("--val_max_pairs", type=int, default=100)
    p.add_argument("--save_every", type=int, default=2_000)
    p.add_argument("--log_dir", type=str, default="runs/progressive")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/progressive")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--residual_damping", type=float, default=1.0,
                   help="TieredRefinement residual scale. Default 1.0 "
                        "(no damping). Set to 0.8 if training becomes unstable.")
    p.add_argument("--resume", type=str, default="",
                   help="Path to progressive_step*.pth.tar to resume from. "
                        "Loads progressive_decoder_state and optimizer_state.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determinism
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    trainer = ProgressiveTrainer(args)
    trainer.run()

