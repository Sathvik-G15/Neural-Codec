"""M4: PSNR benchmarking on UVG with plot generation.

Evaluates progressive decoder at all 4 tiers on UVG sequences,
generates the required plots per plan §5.4:
  - Plot 1: Rate-Distortion (PSNR vs Bitrate)
  - Plot 3: Quality vs Compute (PSNR vs FLOPs)
  - Plot 4: Marginal Efficiency (Stage vs ΔPSNR/ΔFLOPs)

Usage:
    python benchmark.py --checkpoint <path_to_checkpoint> [--sequences "Beauty,Boxer"]

Output:
    - results.json: per-sequence per-tier metrics
    - plots/*.png: generated figures
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DCVC_DIR = Path(__file__).parent / "DCVC-repo" / "DCVC-family" / "DCVC"
sys.path.insert(0, str(DCVC_DIR))
from src.models.DCVC_net import DCVC_net
from src.utils.stream_helper import encode_p, decode_p

UVG_DATA_DIR = Path(__file__).parent / "DCVC-Scalable" / "data" / "uvg"
CHECKPOINT_DIR = DCVC_DIR / "checkpoints"

UVG_SEQUENCES = [
    "Beauty", "Bosphorus", "Boxer", "Honeycomb",
    "Jockey", "ReadySetGo", "ShakeNDry", "Speech",
]


def load_image(path: Path) -> torch.Tensor:
    with Image.open(str(path)) as img:
        img = img.convert("RGB")
        arr = np.array(img, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


def pad_to_multiple(x: torch.Tensor, m: int = 64):
    _, _, H, W = x.shape
    pad_h = (m - H % m) % m
    pad_w = (m - W % m) % m
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (H, W)


def unpad(x: torch.Tensor, orig_hw: tuple) -> torch.Tensor:
    H, W = orig_hw
    return x[:, :, :H, :W]


def encode_decode_sequence(model, ref_frame, frames, temp_dir):
    results = []
    for i, cur in enumerate(frames):
        temp_file = temp_dir / f"frame_{i}.bin"
        model.encode(ref_frame, cur, str(temp_file))
        recon = model.decode(ref_frame, str(temp_file))
        results.append(recon)
        ref_frame = cur
    return results


def evaluate_tier(model, ref, cur, tier: int):
    model._current_depth = tier
    with torch.no_grad():
        out = model(ref, cur)
    model._current_depth = None
    recon = out["recon_image"]
    mse = F.mse_loss(recon, cur).item()
    psnr = 10.0 * np.log10(1.0 / (mse + 1e-8))
    bpp = out["bpp"]
    return {"psnr": psnr, "bpp": bpp, "recon": recon}


def evaluate_sequence(model, seq_dir: Path, max_pairs: int = 100):
    pngs = sorted(seq_dir.glob("*.png"))
    if len(pngs) < 2:
        return None

    results = {tier: [] for tier in range(1, 5)}
    bpp_results = {tier: [] for tier in range(1, 5)}

    for i in range(min(len(pngs) - 1, max_pairs)):
        ref = load_image(pngs[i])
        cur = load_image(pngs[i + 1])

        ref, orig_hw = pad_to_multiple(ref)
        cur, _ = pad_to_multiple(cur)

        ref = ref.cuda()
        cur = cur.cuda()

        for tier in range(1, 5):
            model._current_depth = tier
            with torch.no_grad():
                out = model(ref, cur)
            model._current_depth = None
            recon = out["recon_image"]
            recon_crop = unpad(recon, orig_hw)
            cur_crop = unpad(cur, orig_hw)
            mse = F.mse_loss(recon_crop, cur_crop).item()
            psnr = 10.0 * np.log10(1.0 / (mse + 1e-8))
            results[tier].append(psnr)
            bpp_results[tier].append(out["bpp"])

    if not results[1]:
        return None

    return {
        tier: {
            "psnr_mean": np.mean(psnrs),
            "psnr_std": np.std(psnrs),
            "bpp_mean": np.mean(bpps),
        }
        for tier, psnrs, bpps in zip(
            range(1, 5), [results[t] for t in range(1, 5)], [bpp_results[t] for t in range(1, 5)]
        )
    }


def generate_plots(all_results: dict, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = list(all_results.keys())
    tiers = list(range(1, 5))
    tier_labels = [f"Tier {t}" for t in tiers]

    psnr_by_tier = {t: [] for t in tiers}
    bpp_by_tier = {t: [] for t in tiers}

    for seq in sequences:
        for t in tiers:
            psnr_by_tier[t].append(all_results[seq][t]["psnr_mean"])
            bpp_by_tier[t].append(all_results[seq][t]["bpp_mean"])

    mean_psnr_by_tier = [np.mean(psnr_by_tier[t]) for t in tiers]
    mean_bpp_by_tier = [np.mean(bpp_by_tier[t]) for t in tiers]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    plt.figure(figsize=(8, 6))
    for t, color in zip(tiers, colors):
        plt.plot(
            bpp_by_tier[t], psnr_by_tier[t],
            "o-", label=tier_labels[t - 1], color=color, markersize=6,
        )
    plt.xlabel("Bitrate (bpp)", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.title("Rate-Distortion Curve per Tier", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plot1_rd_curve.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'plot1_rd_curve.png'}")

    flops_tiers = [17.5, 22.5, 27.5, 32.5]
    plt.figure(figsize=(8, 6))
    plt.plot(flops_tiers, mean_psnr_by_tier, "o-", color="tab:blue", markersize=10)
    for t, psnr, flops in zip(tiers, mean_psnr_by_tier, flops_tiers):
        plt.annotate(f"T{t}\n{psnr:.2f}dB", (flops, psnr), textcoords="offset points",
                     xytext=(10, 5), fontsize=9)
    plt.xlabel("Reconstruction FLOPs (GFLOPs)", fontsize=12)
    plt.ylabel("Mean PSNR (dB)", fontsize=12)
    plt.title("Quality vs Compute", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plot3_quality_vs_compute.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'plot3_quality_vs_compute.png'}")

    delta_psnr = [mean_psnr_by_tier[t] - mean_psnr_by_tier[t - 1] for t in range(2, 5)]
    delta_flops = [flops_tiers[t] - flops_tiers[t - 1] for t in range(2, 5)]
    marginal_efficiency = [dp / df for dp, df in zip(delta_psnr, delta_flops)]

    plt.figure(figsize=(8, 6))
    x = range(2, 5)
    bars = plt.bar(x, marginal_efficiency, color=colors[1:], width=0.6)
    plt.xlabel("Stage", fontsize=12)
    plt.ylabel("ΔPSNR / ΔFLOPs (dB / GFLOP)", fontsize=12)
    plt.title("Marginal Efficiency per Stage", fontsize=14)
    plt.xticks(x, [f"T{i-1}→T{i}" for i in x])
    for bar, eff in zip(bars, marginal_efficiency):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{eff:.4f}", ha="center", va="bottom", fontsize=9)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "plot4_marginal_efficiency.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'plot4_marginal_efficiency.png'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M4: UVG Benchmarking with Plots")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to progressive decoder checkpoint (optional)")
    parser.add_argument("--sequences", type=str, default="all",
                        help="Comma-separated sequence names, or 'all'")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory for output results and plots")
    args = parser.parse_args()

    print("=" * 60)
    print("M4: Progressive Decoder UVG Benchmarking")
    print("=" * 60)

    print("\nLoading model...")
    model = DCVC_net().cuda()
    base_ckpt_path = CHECKPOINT_DIR / "model_dcvc_quality_3_psnr.pth"
    if not base_ckpt_path.exists():
        base_ckpt_path = CHECKPOINT_DIR / "extracted" / "model_dcvc_quality_3_psnr.pth"
    state = torch.load(base_ckpt_path, map_location="cuda", weights_only=False)
    if isinstance(state, dict):
        state = state.get("state_dict", state)
    model.load_dict(state)
    print(f"  Base DCVC weights loaded from {base_ckpt_path.name}")

    if args.checkpoint:
        prog_path = Path(args.checkpoint)
        if prog_path.exists():
            prog_state = torch.load(prog_path, map_location="cuda", weights_only=False)
            model.progressive_decoder.load_state_dict(prog_state.get("progressive_decoder_state", prog_state))
            print(f"  Progressive decoder loaded from {prog_path.name}")

    model.eval()

    if args.sequences == "all":
        sequences = UVG_SEQUENCES
    else:
        sequences = [s.strip() for s in args.sequences.split(",")]

    print(f"\nEvaluating {len(sequences)} sequences at 4 tiers...")
    all_results = {}

    for seq_name in sequences:
        seq_dir = UVG_DATA_DIR / seq_name
        if not seq_dir.exists():
            print(f"  [SKIP] {seq_name}: directory not found")
            continue

        print(f"  {seq_name}...", end=" ", flush=True)
        result = evaluate_sequence(model, seq_dir, max_pairs=100)
        if result:
            all_results[seq_name] = result
            psnr_str = " ".join(f"T{t}={result[t]['psnr_mean']:.2f}" for t in range(1, 5))
            print(psnr_str)
        else:
            print("no frames")

    if not all_results:
        print("\n[ERROR] No results collected. Check UVG_DATA_DIR path.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print("\nGenerating plots...")
    try:
        generate_plots(all_results, output_dir / "plots")
    except Exception as e:
        print(f"  [WARN] Plot generation failed: {e}")

    print("\n--- Summary ---")
    for seq_name in all_results:
        psnr_vals = " ".join(f"T{t}={all_results[seq_name][t]['psnr_mean']:.2f}" for t in range(1, 5))
        print(f"  {seq_name}: {psnr_vals}")

    overall_mean = {
        tier: np.mean([all_results[s][tier]["psnr_mean"] for s in all_results])
        for tier in range(1, 5)
    }
    print("\nOverall mean PSNR:")
    for t, psnr in overall_mean.items():
        print(f"  Tier {t}: {psnr:.3f} dB")

    if len(all_results) > 1:
        print(f"\nMonotonicity check (T1 < T2 < T3 < T4):")
        monotonic = all(
            overall_mean[t] <= overall_mean[t + 1]
            for t in range(1, 4)
        )
        print(f"  {'PASS' if monotonic else 'FAIL'}: T1={overall_mean[1]:.3f} <= T2={overall_mean[2]:.3f} <= T3={overall_mean[3]:.3f} <= T4={overall_mean[4]:.3f}")


if __name__ == "__main__":
    main()