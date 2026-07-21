"""Train Progressive Decoder (Phase 0/1/2/3 curriculum, descending-weight deep supervision).

Phase 0 (0..10K):   Fixed depth=4 - CEILING CHECK: does Tier 4 approach stock DCVC PSNR?
Phase 1 (0..60K):   Fixed depth=4 - Establish full-depth baseline quality
Phase 2 (0..60K):   Random depth [1..4] - Force early stages to learn standalone representations
Phase 3 (0..60K):   Random depth [1..4] - Fine-tune with full training set

Per plan §3.2: phase boundaries are CHECKPOINTS TO RESUME FROM, not restart points.
Early-advance and mid-phase-resume are both explicitly permitted.

Go/no-go decision (Phase 0 -> Phase 1):
    Compare Tier 4 PSNR against stock DCVC PSNR per UVG sequence.
    Advance if mean within ±1σ; flag if >1 sequence outside ±2σ.
    σ is pinned to M0.5's measurement, not training-run σ.

Loss
----
L = lambda_rd * sum_i w_i * MSE(recon_i, target)
weights = (1.0, 0.9, 0.8, 0.7)  (descending, per plan §3.1)

Invariant: len(outputs) == active_depth (weight slicing in loss handles this).

Usage (Kaggle)
--------------
    python train_progressive.py \
        --pretrained /kaggle/input/dcvc-baseline/model_dcvc_quality_3_psnr.pth.tar \
        --data_root  /kaggle/input/vimeo90k-septuplet \
        --total_steps 190000 \
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

MAX_STEPS_PER_PHASE = {
    0: 10_000,
    1: 60_000,
    2: 60_000,
    3: 60_000,
}
CUMULATIVE_STEPS = {
    0: MAX_STEPS_PER_PHASE[0],
    1: MAX_STEPS_PER_PHASE[0] + MAX_STEPS_PER_PHASE[1],
    2: MAX_STEPS_PER_PHASE[0] + MAX_STEPS_PER_PHASE[1] + MAX_STEPS_PER_PHASE[2],
    3: sum(MAX_STEPS_PER_PHASE.values()),
}
DEPTH_BY_PHASE = {
    0: [4],
    1: [4],
    2: [1, 2, 3, 4],
    3: [1, 2, 3, 4],
}


class DCVCProgressiveLoss(nn.Module):
    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0), lambda_rd=0.05):
        super().__init__()
        self.weights = list(weights)
        self.lambda_rd = lambda_rd

    def forward(self, recon_list, target):
        w_eff = self.weights[:len(recon_list)]
        distortion = sum(w * F.mse_loss(r, target) for w, r in zip(w_eff, recon_list))
        return self.lambda_rd * distortion


class Vimeo90kSeptupletDataset(Dataset):
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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.root / self.sequences[idx]
        frame_idx = random.randint(1, 6)
        ref = self._load(seq_path / f"im{frame_idx}.png")
        cur = self._load(seq_path / f"im{frame_idx + 1}.png")
        _, H, W = ref.shape
        if H > self.patch_size and W > self.patch_size:
            y0 = random.randint(0, H - self.patch_size)
            x0 = random.randint(0, W - self.patch_size)
            ref = ref[:, y0:y0 + self.patch_size, x0:x0 + self.patch_size]
            cur = cur[:, y0:y0 + self.patch_size, x0:x0 + self.patch_size]
        return ref, cur

    @staticmethod
    def _load(path: Path):
        try:
            with Image.open(str(path)) as img:
                img = img.convert("RGB")
                arr = np.array(img, copy=True)
            return torch.from_numpy(arr).permute(2, 0, 1)
        except Exception:
            return torch.zeros(3, 256, 256)


class ProgressiveTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.model = DCVC_net().to(self.device)
        self._load_pretrained(args.pretrained)
        self._freeze_encoder()

        self.loss_fn = DCVCProgressiveLoss(
            weights=(1.0, 0.9, 0.8, 0.7),
            lambda_rd=args.lambda_rd,
        )
        self.decoder_params = [
            p for p in self.model.progressive_decoder.parameters()
            if p.requires_grad
        ]
        self.optimizer = optim.Adam(self.decoder_params, lr=args.lr)

        train_ds = Vimeo90kSeptupletDataset(args.data_root, "sep_trainlist.txt", args.patch_size)
        val_ds = Vimeo90kSeptupletDataset(args.data_root, "sep_testlist.txt", args.patch_size)

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
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            persistent_workers=True,
        )

        self.writer = SummaryWriter(args.log_dir)
        self.ckpt_dir = Path(args.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        print(f"Log dir: {args.log_dir}")
        print(f"Ckpt dir: {self.ckpt_dir}")

        if args.resume:
            self._load_resume(args.resume)

    def _load_pretrained(self, ckpt_path: str):
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
        print(f"  Loading pretrained DCVC weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict):
            for k in ("state_dict", "model_state_dict", "model"):
                if k in state:
                    state = state[k]
                    break
        self.model.load_dict(state)
        print("  Pretrained weights loaded.")

    def _freeze_encoder(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.progressive_decoder.parameters():
            p.requires_grad_(True)
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable params: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)")

    def _load_resume(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"Resume path not found: {path}")
        print(f"  Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.progressive_decoder.load_state_dict(ckpt["progressive_decoder_state"], strict=False)
        self.step = ckpt.get("global_step", 0)
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"  Resumed at step {self.step} (phase {ckpt.get('phase', '?')}).")

    def get_phase(self, global_step: int) -> int:
        if global_step < CUMULATIVE_STEPS[0]:
            return 0
        elif global_step < CUMULATIVE_STEPS[1]:
            return 1
        elif global_step < CUMULATIVE_STEPS[2]:
            return 2
        else:
            return 3

    def get_phase_step(self, global_step: int) -> int:
        phase = self.get_phase(global_step)
        if phase == 0:
            return global_step
        elif phase == 1:
            return global_step - CUMULATIVE_STEPS[0]
        elif phase == 2:
            return global_step - CUMULATIVE_STEPS[1]
        else:
            return global_step - CUMULATIVE_STEPS[2]

    def sample_depth(self, global_step: int) -> int:
        phase = self.get_phase(global_step)
        depths = DEPTH_BY_PHASE[phase]
        return int(torch.randint(0, len(depths), (1,)).item()) + depths[0]

    def phase_summary(self, global_step: int) -> str:
        phase = self.get_phase(global_step)
        phase_step = self.get_phase_step(global_step)
        return f"phase={phase} (step_in_phase={phase_step}/{MAX_STEPS_PER_PHASE[phase]})"

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

    @torch.no_grad()
    def validate(self) -> dict:
        saved_training = self.model.training
        saved_depth = self.model._current_depth

        self.model.train()
        self.model._current_depth = None

        tier_sums = [0.0] * 4
        n = 0

        for ref, cur in self.val_loader:
            ref = ref.to(self.device).float().div_(255.0)
            cur = cur.to(self.device).float().div_(255.0)

            out = self.model(ref, cur)
            recon_images = out["recon_images"]

            for i, r in enumerate(recon_images):
                mse = F.mse_loss(r, cur).item()
                tier_sums[i] += 10.0 * np.log10(1.0 / (mse + 1e-8))
            n += 1
            if n >= self.args.val_max_pairs:
                break

        self.model._current_depth = saved_depth
        self.model.train(saved_training)

        if tier_sums is None:
            return {}
        return {f"psnr_t{i+1}_mean": s / max(n, 1) for i, s in enumerate(tier_sums)}

    def save_ckpt(self, tag: str):
        path = self.ckpt_dir / f"progressive_{tag}.pth.tar"
        tmp = self.ckpt_dir / f"_tmp_progressive_{tag}.pth.tar"
        phase = self.get_phase(self.step)
        torch.save(
            {
                "progressive_decoder_state": self.model.progressive_decoder.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": -1,
                "global_step": self.step,
                "phase": phase,
                "phase_step": self.get_phase_step(self.step),
                "config": vars(self.args),
            },
            tmp,
        )
        os.replace(tmp, path)
        print(f"  [CKPT] step={self.step} {self.phase_summary(self.step)} -> {path.name}")
        return path

    def run(self):
        total_phases = len(MAX_STEPS_PER_PHASE)
        max_steps = CUMULATIVE_STEPS[total_phases - 1]
        print(f"\nTraining for up to {max_steps} steps ({total_phases} phases).")
        print(f"  Phase 0: 0..{MAX_STEPS_PER_PHASE[0]-1}         (depth=4, ceiling check)")
        print(f"  Phase 1: {CUMULATIVE_STEPS[0]}..{CUMULATIVE_STEPS[1]-1}      (depth=4, establish baseline)")
        print(f"  Phase 2: {CUMULATIVE_STEPS[1]}..{CUMULATIVE_STEPS[2]-1}   (random [1..4], learn early tiers)")
        print(f"  Phase 3: {CUMULATIVE_STEPS[2]}..{CUMULATIVE_STEPS[3]-1}   (random [1..4], fine-tune)")
        print(f"\n  Early advance: enabled (plateau detection in Phases 0-1)")
        print(f"  Checkpoint-based resume: phase/phase_step tracked in ckpt\n")

        self.model.train()
        ref_cur = iter(self.train_loader)
        log_loss = 0.0
        log_n = 0
        val_count = 0

        while self.step < max_steps:
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

            phase = self.get_phase(self.step - 1)

            if self.step % 200 == 0:
                avg = log_loss / log_n
                log_loss = 0.0
                log_n = 0
                depth_now = self.sample_depth(self.step - 1)
                print(
                    f"  step {self.step}/{max_steps}  "
                    f"loss={avg:.6f} ({avg:.3e})  depth={depth_now}  [{self.phase_summary(self.step)}]",
                    flush=True,
                )
                self.writer.add_scalar("train/loss", avg, self.step)
                self.writer.add_scalar("train/depth", depth_now, self.step)
                self.writer.add_scalar("train/phase", phase, self.step)

            if self.step % 1000 == 0:
                try:
                    m = self.validate()
                    val_count += 1
                    vals = " ".join(
                        f"T{i}={m.get(f'psnr_t{i}_mean', float('nan')):.3f}"
                        for i in range(1, 5)
                    )
                    if len(m) == 4:
                        t1, t2, t3, t4 = [m[f"psnr_t{i}_mean"] for i in range(1, 5)]
                        gap = t4 - t1
                        order = "OK" if (t1 <= t2 <= t3 <= t4) else "INVERSION_DETECTED"
                    else:
                        gap = float("nan")
                        order = "INCOMPLETE"
                    print(
                        f"[VAL@step{self.step}] {vals} gap={gap:.3f} {order}  [{self.phase_summary(self.step)}]",
                        flush=True,
                    )
                    for k, v in m.items():
                        self.writer.add_scalar(f"val/{k}", v, self.step)

                    if phase == 0 and self.get_phase_step(self.step) >= MAX_STEPS_PER_PHASE[0]:
                        print(f"  [PHASE_0_COMPLETE] Ceiling check done at step {self.step}")
                        break

                except Exception as exc:
                    print(f"[VAL@step{self.step}] skipped: {exc!r}", flush=True)

            if self.step % self.args.save_every == 0:
                self.save_ckpt(f"step{self.step}")

        self.save_ckpt("final")
        self.writer.close()
        print(f"\nDone after {self.step} steps. Final ckpt at {self.ckpt_dir}.")
        print(f"Ran {val_count} validation cycles.")


def parse_args():
    p = argparse.ArgumentParser(description="Train progressive decoder (Phase 0-3 curriculum)")
    p.add_argument("--pretrained", type=str, required=True,
                   help="Path to Microsoft DCVC baseline .pth.tar")
    p.add_argument("--data_root", type=str, required=True,
                   help="Path containing vimeo_septuplet/sequences/ and {sep_trainlist.txt, sep_testlist.txt}")
    p.add_argument("--lambda_rd", type=float, default=0.05,
                   help="Distortion weight. Start at 1e-3; raise to 2e-3 if gap < 0.2 dB after 5k-10k steps.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--total_steps", type=int, default=190000,
                   help="Total optimisation steps across Phases 0-3 (190k default).")
    p.add_argument("--val_max_pairs", type=int, default=100)
    p.add_argument("--save_every", type=int, default=2_000)
    p.add_argument("--log_dir", type=str, default="runs/progressive")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/progressive")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--residual_damping", type=float, default=1.0,
                   help="FeatureRefinementBlock residual scale. Default 1.0.")
    p.add_argument("--resume", type=str, default="",
                   help="Path to progressive_step*.pth.tar to resume from.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    trainer = ProgressiveTrainer(args)
    trainer.run()