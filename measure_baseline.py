"""M0.5: Stock DCVC baseline PSNR/σ measurement on UVG sequences.

This script runs the unmodified DCVC decoder on all UVG sequences and records
per-sequence PSNR mean and standard deviation. This establishes the baseline
variance (σ) that defines "meaningful improvement" in §6 of the plan.

Usage:
    python measure_baseline.py

Outputs:
    - Per-sequence PSNR mean and std (printed + saved to baseline_results.json)
    - σ values needed for M2 go/no-go decision rule (plan §3.2)
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DCVC_DIR = Path(__file__).parent / "DCVC-repo" / "DCVC-family" / "DCVC"
sys.path.insert(0, str(DCVC_DIR))

from src.models.DCVC_net import DCVC_net

UVG_DATA_DIR = Path(__file__).parent / "DCVC-Scalable" / "data" / "uvg"
CHECKPOINT_DIR = DCVC_DIR / "checkpoints"

UVG_SEQUENCES = [
    "Beauty",
    "Bosphorus",
    "Boxer",
    "Honeycomb",
    "Jockey",
    "ReadySetGo",
    "ShakeNDry",
    "Speech",
]


def _load_image(path: Path) -> torch.Tensor:
    with Image.open(str(path)) as img:
        img = img.convert("RGB")
        arr = np.array(img, copy=True)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(torch.float32).mul_(1.0 / 255.0)


def _pad_to_multiple(x: torch.Tensor, m: int = 64):
    _, _, H, W = x.shape
    pad_h = (m - H % m) % m
    pad_w = (m - W % m) % m
    if pad_h or pad_w:
        import torch.nn.functional as F
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (H, W)


def _unpad(x: torch.Tensor, orig_HW: tuple) -> torch.Tensor:
    H, W = orig_HW
    return x[:, :, :H, :W]


def measure_sequence(model: DCVC_net, seq_dir: Path, max_pairs: int = 100):
    """Measure PSNR for all consecutive frame pairs in a sequence.

    Returns:
        List of PSNR values (one per frame pair)
    """
    pngs = sorted(seq_dir.glob("*.png"))
    if len(pngs) < 2:
        print(f"  [WARN] {seq_dir.name}: only {len(pngs)} frames found, skipping")
        return []

    psnrs = []
    model.eval()

    for i in range(min(len(pngs) - 1, max_pairs)):
        ref = _load_image(pngs[i])
        cur = _load_image(pngs[i + 1])

        ref, orig_hw = _pad_to_multiple(ref)
        cur, _ = _pad_to_multiple(cur)

        ref = ref.cuda()
        cur = cur.cuda()

        with torch.no_grad():
            out = model(ref, cur)
        recon = out["recon_image"]
        recon = recon.cuda()

        recon_crop = _unpad(recon, orig_hw)
        cur_crop = _unpad(cur, orig_hw)

        mse = torch.nn.functional.mse_loss(recon_crop, cur_crop).item()
        psnr = 10.0 * np.log10(1.0 / (mse + 1e-8))
        psnrs.append(psnr)

    return psnrs


def main():
    print("=" * 60)
    print("M0.5: Stock DCVC Baseline PSNR/σ Measurement")
    print("=" * 60)

    print("\nLoading DCVC model...")
    model = DCVC_net().cuda()
    ckpt_path = CHECKPOINT_DIR / "model_dcvc_quality_3_psnr.pth"
    if not ckpt_path.exists():
        extracted = CHECKPOINT_DIR / "extracted" / "model_dcvc_quality_3_psnr.pth"
        if extracted.exists():
            ckpt_path = extracted
    state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model_state_dict", state))
    model.load_dict(state)
    print(f"  Loaded: {ckpt_path}")

    results = {}
    all_psnrs = []

    print(f"\nMeasuring {len(UVG_SEQUENCES)} UVG sequences...")
    for seq_name in UVG_SEQUENCES:
        seq_dir = UVG_DATA_DIR / seq_name
        if not seq_dir.exists():
            print(f"  [SKIP] {seq_name}: directory not found at {seq_dir}")
            continue

        psnrs = measure_sequence(model, seq_dir, max_pairs=100)
        if not psnrs:
            continue

        mean_psnr = np.mean(psnrs)
        std_psnr = np.std(psnrs)
        results[seq_name] = {"mean": mean_psnr, "std": std_psnr, "n_frames": len(psnrs)}
        all_psnrs.extend(psnrs)

        print(f"  {seq_name:20s}: PSNR = {mean_psnr:.3f} ± {std_psnr:.3f} dB  (n={len(psnrs)})")

    if not results:
        print("\n[ERROR] No sequences found. Check UVG_DATA_DIR path.")
        return

    overall_mean = np.mean(all_psnrs)
    overall_std = np.std(all_psnrs)

    print("-" * 60)
    print(f"Overall mean PSNR: {overall_mean:.3f} dB")
    print(f"Overall σ (pooled std): {overall_std:.3f} dB")

    output_path = Path(__file__).parent / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "per_sequence": {k: {"mean": float(v["mean"]), "std": float(v["std"]), "n": v["n_frames"]}
                                 for k, v in results.items()},
                "overall_mean": float(overall_mean),
                "overall_std": float(overall_std),
                "sigma_reference": "M0.5: per-sequence std from unmodified model_dcvc_quality_3_psnr.pth on UVG",
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")
    print("\nσ reference for §6 'meaningful improvement' definition:")
    print(f"  Use per-sequence σ from this file, NOT training-run σ")
    print(f"  Go/no-go rule: Tier 4 mean within ±1σ of stock DCVC = advance")


if __name__ == "__main__":
    main()