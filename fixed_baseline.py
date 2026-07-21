"""M5: Fixed-Decoder Baseline Comparison.

Trains four independent fixed-size decoders (one per tier compute budget)
and compares against the single progressive model. This tests the claim in
§10 ("why not just train separate models?") empirically, not just rhetorically.

Per plan §4.3:
  - Width-scale part2 only; hold part1 fixed across all four baselines
  - Match FLOPs within ~5% tolerance of progressive model tiers
  - Compare PSNR at matched FLOPs budgets

Usage:
    python fixed_baseline.py \
        --pretrained <dcvc_baseline.pth> \
        --data_root <vimeo90k_path> \
        --output_dir fixed_baseline_results

Output:
    - fixed_baseline_results/comparison.json: PSNR comparison per tier
    - fixed_baseline_results/baseline_t{i}.pth.tar: trained baseline checkpoints
"""

import argparse
import json
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

DCVC_DIR = Path(__file__).parent / "DCVC-repo" / "DCVC-family" / "DCVC"
sys.path.insert(0, str(DCVC_DIR))
from src.models.DCVC_net import DCVC_net
from src.models.video_net import ResBlock


class DCVCProgressiveLoss(nn.Module):
    def __init__(self, weights=(1.0, 0.9, 0.8, 0.7), lambda_rd=1e-3):
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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.root / self.sequences[idx]
        frame_idx = random.randint(1, 6)
        ref = self._load(seq_path / f"im{frame_idx}.png")
        cur = self._load(seq_path / f"im{frame_idx + 1}.png")
        _, H, W = ref.shape
        if H > self.patch_size and W > self.patch_size:
            y0, x0 = random.randint(0, H - self.patch_size), random.randint(0, W - self.patch_size)
            ref, cur = ref[:, y0:y0 + self.patch_size, x0:x0 + self.patch_size], \
                       cur[:, y0:y0 + self.patch_size, x0:x0 + self.patch_size]
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


class FixedTierDecoder(nn.Module):
    """Fixed-scope decoder for baseline comparison.

    Architecture: identical to progressive_decoder's part1 + part2,
    but with a configurable part2 channel width to trade quality for FLOPs.
    """

    def __init__(self, out_channel_M=96, out_channel_N=64, part2_width=None):
        super().__init__()
        from src.layers.layers import subpel_conv3x3
        from src.models.video_net import GDN

        if part2_width is None:
            part2_width = out_channel_N

        self.part1 = nn.Sequential(
            subpel_conv3x3(out_channel_M, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock(out_channel_N, out_channel_N, 3),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock(out_channel_N, out_channel_N, 3),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
        )

        self.proj = nn.Conv2d(out_channel_N * 2, part2_width, 1) if part2_width != out_channel_N else None
        self.part2 = nn.Sequential(
            nn.Conv2d(part2_width, part2_width, 3, stride=1, padding=1),
            ResBlock(part2_width, part2_width, 3),
            ResBlock(part2_width, part2_width, 3),
            nn.Conv2d(part2_width, 3, 3, stride=1, padding=1),
        )

    def forward(self, y_hat, context):
        features = self.part1(y_hat)
        x = torch.cat([features, context], dim=1)
        if self.proj is not None:
            x = self.proj(x)
        recon = self.part2(x)
        return recon.clamp(0.0, 1.0)


class BaselineTrainer:
    def __init__(self, args, tier: int, part2_width: int):
        self.args = args
        self.tier = tier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DCVC_net().to(self.device)
        self._load_pretrained(args.pretrained)
        self._freeze_encoder()

        self.baseline_decoder = FixedTierDecoder(
            out_channel_M=96, out_channel_N=64, part2_width=part2_width
        ).to(self.device)

        self.loss_fn = DCVCProgressiveLoss(lambda_rd=args.lambda_rd)
        params = list(self.baseline_decoder.parameters())
        self.optimizer = optim.Adam(params, lr=args.lr)

        train_ds = Vimeo90kSeptupletDataset(args.data_root, "sep_trainlist.txt", args.patch_size)
        val_ds = Vimeo90kSeptupletDataset(args.data_root, "sep_testlist.txt", args.patch_size)
        self.train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

        self.step = 0
        self.writer = SummaryWriter(args.log_dir / f"tier{tier}")
        self.output_dir = Path(args.output_dir) / f"tier{tier}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_pretrained(self, ckpt_path):
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict):
            state = state.get("state_dict", state)
        self.model.load_dict(state)

    def _freeze_encoder(self):
        for p in self.model.parameters():
            p.requires_grad_(False)

    def train_step(self, ref, cur):
        with torch.no_grad():
            y_hat = self.model.contextualEncoder(torch.cat([ref, cur], dim=1))
            y_hat = torch.round(y_hat)
            mv = self.model.opticFlow(ref, cur)
            context = self.model.motioncompensation(ref, mv)

        recon = self.baseline_decoder(y_hat, context)
        loss = self.loss_fn([recon], cur)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def validate(self):
        self.baseline_decoder.eval()
        psnrs = []
        for ref, cur in self.val_loader:
            ref = ref.to(self.device).float().div_(255.0)
            cur = cur.to(self.device).float().div_(255.0)
            y_hat = self.model.contextualEncoder(torch.cat([ref, cur], dim=1))
            y_hat = torch.round(y_hat)
            mv = self.model.opticFlow(ref, cur)
            context = self.model.motioncompensation(ref, mv)
            recon = self.baseline_decoder(y_hat, context)
            mse = F.mse_loss(recon, cur).item()
            psnrs.append(10.0 * np.log10(1.0 / (mse + 1e-8)))
            if len(psnrs) >= self.args.val_max_pairs:
                break
        self.baseline_decoder.train()
        return np.mean(psnrs) if psnrs else 0.0

    def save_ckpt(self, tag):
        path = self.output_dir / f"baseline_t{self.tier}_{tag}.pth.tar"
        torch.save({
            "baseline_decoder_state": self.baseline_decoder.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_step": self.step,
            "tier": self.tier,
            "config": vars(self.args),
        }, path)
        return path

    def run(self):
        print(f"\nTraining baseline Tier {self.tier} for {self.args.total_steps} steps...")
        self.model.eval()
        self.baseline_decoder.train()
        ref_cur = iter(self.train_loader)
        log_loss = 0.0

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

            if self.step % 200 == 0:
                print(f"  Tier {self.tier} step {self.step}/{self.args.total_steps} loss={log_loss / 200:.6f}")
                log_loss = 0.0

            if self.step % 1000 == 0:
                val_psnr = self.validate()
                print(f"  [VAL] Tier {self.tier} step {self.step}: PSNR = {val_psnr:.3f}")
                self.writer.add_scalar("val/psnr", val_psnr, self.step)

            if self.step % self.args.save_every == 0:
                self.save_ckpt(f"step{self.step}")

        final_path = self.save_ckpt("final")
        print(f"  Tier {self.tier} done. Final ckpt: {final_path}")
        return self.validate()


def main():
    parser = argparse.ArgumentParser(description="M5: Fixed-Decoder Baseline Comparison")
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="fixed_baseline_results")
    parser.add_argument("--lambda_rd", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--total_steps", type=int, default=60_000)
    parser.add_argument("--val_max_pairs", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--part2_widths", type=str, default="64,56,48,32",
                        help="Comma-separated part2 channel widths for Tiers 1-4")
    args = parser.parse_args()

    widths = [int(w) for w in args.part2_widths.split(",")]
    if len(widths) != 4:
        print("[ERROR] Must specify exactly 4 part2_widths for Tiers 1-4")
        return

    args.log_dir = Path(args.output_dir) / "logs"

    print("=" * 60)
    print("M5: Fixed-Decoder Baseline Comparison")
    print("=" * 60)
    print(f"\nPart2 channel widths per tier: T1={widths[0]}, T2={widths[1]}, T3={widths[2]}, T4={widths[3]}")
    print("Per plan §4.3: width-scale part2 only, hold part1 fixed")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = {}
    for tier, width in enumerate(widths, start=1):
        print(f"\n--- Training Tier {tier} (part2_width={width}) ---")
        trainer = BaselineTrainer(args, tier=tier, part2_width=width)
        val_psnr = trainer.run()
        results[f"Tier {tier}"] = {"part2_width": width, "val_psnr": val_psnr}

    print("\n" + "=" * 60)
    print("Baseline Comparison Results")
    print("=" * 60)
    for tier, data in results.items():
        print(f"  {tier} (width={data['part2_width']}): PSNR = {data['val_psnr']:.3f} dB")

    comparison_path = Path(args.output_dir) / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {comparison_path}")
    print("\nNote: Compare these PSNR values against the progressive model's PSNR at matching FLOPs.")
    print("If progressive model matches within ~0.5 dB while requiring one model vs four, it validates the approach.")


if __name__ == "__main__":
    main()