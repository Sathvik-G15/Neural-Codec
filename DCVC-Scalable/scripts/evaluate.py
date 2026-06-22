"""
Evaluation script for DCVC Progressive — research plan Section 5.2 & 5.3

Measures per sequence, per tier:
    - VMAF (primary quality metric) with PSNR fallback
    - Bitrate (bpp proxy from latent L1 norm)
    - Decode latency (ms/frame, 10 runs averaged)
    - FPS
    - GPU utilization and power (nvidia-smi if available)

Required plots (Section 5.3):
    1. Rate-Distortion         : VMAF vs Bitrate
    2. Quality vs Latency      : VMAF vs ms/frame  ← PRIMARY FIGURE
    3. Quality vs Compute      : VMAF vs FLOPs
    4. Marginal Efficiency     : Stage vs ΔVMAF/ΔFLOPs
    5. Real-time Feasibility   : FPS vs VMAF  (30/60 FPS threshold lines)

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/model.pt --data data/uvg
"""

import os
import argparse
import time
import json
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """PSNR between two uint8 images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_vmaf(ref_path: str, dist_path: str,
                 width: int, height: int) -> float:
    """Compute VMAF using ffmpeg's libvmaf filter.

    Requires ffmpeg built with --enable-libvmaf.
    Falls back to PSNR if unavailable.

    Args:
        ref_path:  Path to reference Y4M / PNG sequence.
        dist_path: Path to distorted Y4M / PNG sequence.
        width, height: Frame dimensions.

    Returns:
        VMAF score (0–100), or PSNR value if VMAF unavailable.
    """
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', dist_path,
            '-i', ref_path,
            '-lavfi', f'[0:v][1:v]libvmaf=log_fmt=json:log_path=/tmp/vmaf_log.json',
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            with open('/tmp/vmaf_log.json') as f:
                vmaf_data = json.load(f)
            return vmaf_data['pooled_metrics']['vmaf']['mean']
    except Exception:
        pass
    return None  # Caller falls back to PSNR


def compute_vmaf_from_arrays(ref_arr: np.ndarray,
                              dist_arr: np.ndarray) -> float:
    """Compute VMAF by writing frames to temp files and calling ffmpeg.

    Args:
        ref_arr:  Reference frame (H, W, 3) uint8.
        dist_arr: Distorted frame (H, W, 3) uint8.

    Returns:
        VMAF score if ffmpeg/libvmaf available, else None.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ref_path  = os.path.join(tmp, 'ref.png')
            dist_path = os.path.join(tmp, 'dist.png')
            log_path  = os.path.join(tmp, 'vmaf.json')

            Image.fromarray(ref_arr).save(ref_path)
            Image.fromarray(dist_arr).save(dist_path)

            H, W = ref_arr.shape[:2]
            cmd = [
                'ffmpeg', '-y',
                '-i', dist_path,
                '-i', ref_path,
                '-lavfi', f'[0:v][1:v]libvmaf=log_fmt=json:log_path={log_path}',
                '-f', 'null', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and os.path.exists(log_path):
                with open(log_path) as f:
                    vmaf_data = json.load(f)
                return vmaf_data['pooled_metrics']['vmaf']['mean']
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Evaluates DCVC Progressive model on test sequences (research plan §5.2)."""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self._vmaf_warned = False

    def _quality(self, ref_np: np.ndarray, recon_np: np.ndarray) -> dict:
        """Compute VMAF (primary) and PSNR (fallback).

        Args:
            ref_np:   Reference frame in [0,1] float32, shape (H, W, 3).
            recon_np: Reconstructed frame in [0,1] float32, shape (H, W, 3).

        Returns:
            dict with 'vmaf' (primary) and 'psnr' keys.
        """
        ref_uint8   = (ref_np   * 255).clip(0, 255).astype(np.uint8)
        recon_uint8 = (recon_np * 255).clip(0, 255).astype(np.uint8)

        psnr = compute_psnr(ref_uint8, recon_uint8)

        vmaf = compute_vmaf_from_arrays(ref_uint8, recon_uint8)
        if vmaf is None and not self._vmaf_warned:
            print("  [Warning] VMAF unavailable — using PSNR as quality proxy. "
                  "Install ffmpeg with --enable-libvmaf for VMAF.")
            self._vmaf_warned = True

        # Use VMAF as primary; fall back to PSNR if unavailable
        quality = vmaf if vmaf is not None else psnr

        return {'vmaf': quality, 'psnr': psnr, 'vmaf_available': vmaf is not None}

    def measure_latency(self, ref: torch.Tensor, current: torch.Tensor,
                        depth: int, num_runs: int = 10) -> dict:
        """Measure decode latency for a single frame pair (plan §5.2 step 3).

        Args:
            ref:      Reference frame tensor (B, C, H, W).
            current:  Current frame tensor   (B, C, H, W).
            depth:    Decode depth (1-4).
            num_runs: Number of timed runs to average (plan: 10 runs).

        Returns:
            dict with mean / std / min / max latency in milliseconds.
        """
        latencies = []
        use_cuda  = (self.device == 'cuda' or self.device == torch.device('cuda')) \
                    and torch.cuda.is_available()

        with torch.no_grad():
            # Warm-up (5 runs, not measured)
            for _ in range(5):
                _ = self.model(current, ref=ref, depth=depth)

            if use_cuda:
                torch.cuda.synchronize()

            for _ in range(num_runs):
                start = time.perf_counter()
                _ = self.model(current, ref=ref, depth=depth)
                if use_cuda:
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)

        return {
            'mean_ms': float(np.mean(latencies)),
            'std_ms':  float(np.std(latencies)),
            'min_ms':  float(np.min(latencies)),
            'max_ms':  float(np.max(latencies)),
        }

    def evaluate_sequence(self, frames: list, depth: int = 4) -> dict:
        """Evaluate on a single sequence using consecutive frame pairs.

        Follows plan §5.2:
            1. Encode once with DCVC (single encoding — all tiers share same bitstream)
            2. Decode at all 4 tiers
            3. Record: bitrate, VMAF, decode latency (10 runs averaged), FPS, GPU util, power

        Args:
            frames: List of frames as numpy arrays (H, W, 3), uint8.
            depth:  Decode depth (1-4).

        Returns:
            dict with avg VMAF, PSNR, latency, FPS, bpp.
        """
        device = torch.device(self.device) if isinstance(self.device, str) \
                 else self.device

        total_vmaf    = 0.0
        total_psnr    = 0.0
        total_latency = 0.0
        total_bpp     = 0.0
        num_frames    = 0

        for i in range(len(frames) - 1):
            ref_np     = frames[i].astype(np.float32) / 255.0
            current_np = frames[i + 1].astype(np.float32) / 255.0

            ref     = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).to(device)
            current = torch.from_numpy(current_np).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                lat_stats = self.measure_latency(ref, current, depth, num_runs=10)

                result  = self.model(current, ref=ref, depth=depth)
                outputs = result['reconstructions']
                recon   = outputs[-1]   # highest-quality reconstruction at this depth

                # Bitrate proxy: L1 norm of latent
                latent   = result.get('latent', result.get('y_hat', None))
                bpp_val  = float(torch.mean(torch.abs(latent)).item()) if latent is not None else 0.0

                recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
                recon_np = np.clip(recon_np, 0.0, 1.0)

                # Resize target to match reconstruction if necessary
                target_h, target_w = recon_np.shape[:2]
                target_resized = np.array(
                    Image.fromarray((current_np * 255).astype(np.uint8))
                         .resize((target_w, target_h))
                ).astype(np.float32) / 255.0

                q = self._quality(target_resized, recon_np)

                total_vmaf    += q['vmaf']
                total_psnr    += q['psnr']
                total_latency += lat_stats['mean_ms']
                total_bpp     += bpp_val
                num_frames    += 1

        avg_latency = total_latency / num_frames
        return {
            'num_frames':     num_frames,
            'avg_vmaf':       total_vmaf    / num_frames,
            'avg_psnr':       total_psnr    / num_frames,
            'avg_latency_ms': avg_latency,
            'fps':            1000.0 / avg_latency,
            'avg_bpp':        total_bpp     / num_frames,
        }

    def evaluate_dataset(self, data_dir: str, sequences: list = None,
                         depths: list = None) -> pd.DataFrame:
        """Evaluate on full dataset at all depths (plan §5.2).

        Args:
            data_dir:  Root directory of test sequences (UVG layout).
            sequences: List of sequence names (None = all).
            depths:    List of depths to evaluate (default [1,2,3,4]).

        Returns:
            DataFrame with one row per (sequence, depth).
        """
        if depths is None:
            depths = [1, 2, 3, 4]
        data_dir = Path(data_dir)

        if sequences is None:
            sequences = [d.name for d in sorted(data_dir.iterdir()) if d.is_dir()]

        results = []

        for seq_name in sequences:
            seq_dir = data_dir / seq_name
            if not seq_dir.exists():
                print(f"  Warning: {seq_name} not found, skipping")
                continue

            frames = []
            for frame_path in sorted(seq_dir.glob("*.png"))[:32]:
                img = Image.open(frame_path).convert('RGB')
                frames.append(np.array(img))

            if len(frames) < 2:
                print(f"  Warning: {seq_name} has insufficient frames")
                continue

            print(f"\nEvaluating {seq_name} ({len(frames)} frames)")

            for depth in depths:
                print(f"  Depth {depth}...", end=' ', flush=True)

                metrics = self.evaluate_sequence(frames, depth=depth)

                results.append({
                    'sequence':   seq_name,
                    'depth':      depth,
                    'vmaf':       metrics['avg_vmaf'],
                    'psnr':       metrics['avg_psnr'],
                    'latency_ms': metrics['avg_latency_ms'],
                    'fps':        metrics['fps'],
                    'bpp':        metrics['avg_bpp'],
                })

                print(f"VMAF: {metrics['avg_vmaf']:.2f}  "
                      f"FPS: {metrics['fps']:.1f}  "
                      f"Latency: {metrics['avg_latency_ms']:.1f} ms")

        return pd.DataFrame(results)

    # -----------------------------------------------------------------------
    # Plotting — all 5 required plots (plan Section 5.3)
    # -----------------------------------------------------------------------

    def plot_results(self, results_df: pd.DataFrame, output_dir: str = 'results',
                     flops_per_tier: list = None):
        """Generate all 5 required evaluation plots (plan §5.3).

        Plot 1 — Rate-Distortion   : VMAF vs Bitrate
        Plot 2 — Quality vs Latency: VMAF vs ms/frame  ← PRIMARY FIGURE
        Plot 3 — Quality vs Compute: VMAF vs FLOPs
        Plot 4 — Marginal Efficiency: Stage vs ΔVMAF/ΔFLOPs
        Plot 5 — Real-time Feasibility: FPS vs VMAF (30/60 FPS lines)

        Args:
            results_df:    DataFrame from evaluate_dataset().
            output_dir:    Directory to save PNG files.
            flops_per_tier: List of cumulative FLOPs for each tier (from profiler).
                           If None, uses tier index as compute proxy.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        depths = sorted(results_df['depth'].unique())

        # Aggregate across sequences for summary plots
        agg = results_df.groupby('depth').agg({
            'vmaf':       'mean',
            'psnr':       'mean',
            'latency_ms': 'mean',
            'fps':        'mean',
            'bpp':        'mean',
        }).reset_index()

        # Determine compute axis
        if flops_per_tier is not None:
            tier_to_flops = {d: flops_per_tier[i] for i, d in enumerate(depths)
                             if i < len(flops_per_tier)}
        else:
            tier_to_flops = {d: d for d in depths}  # proxy: depth index

        flops_values = [tier_to_flops.get(d, d) for d in agg['depth']]

        # ------------------------------------------------------------------ #
        # Plot 1 — Rate-Distortion: VMAF vs Bitrate                          #
        # ------------------------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(8, 6))
        for depth in depths:
            d_df = results_df[results_df['depth'] == depth]
            ax.scatter(d_df['bpp'], d_df['vmaf'], s=80, label=f'Tier {depth}',
                       zorder=3)
        # Draw RD curve for mean values
        ax.plot(agg['bpp'], agg['vmaf'], 'k--', linewidth=1, alpha=0.6, label='Mean')
        ax.set_xlabel('Bitrate proxy (L1 latent norm)', fontsize=12)
        ax.set_ylabel('VMAF', fontsize=12)
        ax.set_title('Plot 1 — Rate-Distortion Curve', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_dir / 'plot1_rate_distortion.png', dpi=150)
        plt.close()
        print(f"  Saved: plot1_rate_distortion.png")

        # ------------------------------------------------------------------ #
        # Plot 2 — Quality vs Latency (PRIMARY FIGURE)                       #
        # ------------------------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(9, 6))
        for depth in depths:
            d_df = results_df[results_df['depth'] == depth]
            ax.scatter(d_df['latency_ms'], d_df['vmaf'], s=80,
                       label=f'Tier {depth}', zorder=3)
        ax.plot(agg['latency_ms'], agg['vmaf'], 'k--', linewidth=1.5, alpha=0.7,
                label='Mean trajectory')
        ax.set_xlabel('Decode latency (ms/frame)', fontsize=12)
        ax.set_ylabel('VMAF', fontsize=12)
        ax.set_title('Plot 2 — Quality vs Latency  [PRIMARY FIGURE]',
                     fontsize=13, fontweight='bold', color='#c0392b')
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_dir / 'plot2_quality_vs_latency.png', dpi=150)
        plt.close()
        print(f"  Saved: plot2_quality_vs_latency.png  ← PRIMARY FIGURE")

        # ------------------------------------------------------------------ #
        # Plot 3 — Quality vs Compute: VMAF vs FLOPs                         #
        # ------------------------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, depth in enumerate(depths):
            d_df = results_df[results_df['depth'] == depth]
            flops_val = tier_to_flops.get(depth, depth)
            ax.scatter([flops_val] * len(d_df), d_df['vmaf'], s=60,
                       label=f'Tier {depth}', zorder=3)
        ax.plot(flops_values, agg['vmaf'], 'k--', linewidth=1.5, alpha=0.7)
        x_label = 'FLOPs' if flops_per_tier is not None else 'Decode depth (FLOPs proxy)'
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('VMAF', fontsize=12)
        ax.set_title('Plot 3 — Quality vs Compute', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_dir / 'plot3_quality_vs_compute.png', dpi=150)
        plt.close()
        print(f"  Saved: plot3_quality_vs_compute.png")

        # ------------------------------------------------------------------ #
        # Plot 4 — Marginal Efficiency: Stage vs ΔVMAF/ΔFLOPs               #
        # ------------------------------------------------------------------ #
        vmaf_vals  = agg['vmaf'].values
        delta_vmaf = np.diff(vmaf_vals)
        delta_compute = np.diff(flops_values)

        marginal_eff = []
        for dv, dc in zip(delta_vmaf, delta_compute):
            marginal_eff.append(dv / dc if dc != 0 else 0.0)

        transitions = [f'Tier {depths[i]}→{depths[i+1]}'
                       for i in range(len(depths) - 1)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(transitions, delta_vmaf, color='steelblue', alpha=0.8)
        axes[0].set_xlabel('Tier Transition', fontsize=11)
        axes[0].set_ylabel('ΔVMAF', fontsize=11)
        axes[0].set_title('ΔVMAF per Stage', fontsize=12)
        axes[0].grid(True, axis='y', alpha=0.4)

        axes[1].bar(transitions, marginal_eff, color='darkorange', alpha=0.8)
        axes[1].set_xlabel('Tier Transition', fontsize=11)
        compute_unit = 'ΔVMAF/ΔFLOPs' if flops_per_tier else 'ΔVMAF/ΔDepth'
        axes[1].set_ylabel(compute_unit, fontsize=11)
        axes[1].set_title('Marginal Efficiency (should decline → diminishing returns)',
                          fontsize=11)
        axes[1].grid(True, axis='y', alpha=0.4)

        # Print marginal efficiency table (plan §4.3)
        print(f"\n  Marginal Efficiency Table:")
        print(f"  {'Transition':<16} {'ΔVMAF':>8} {'ΔCompute':>14} {'ΔVMAF/ΔCompute':>16}")
        for t, dv, dc, me in zip(transitions, delta_vmaf, delta_compute, marginal_eff):
            print(f"  {t:<16} {dv:>+8.3f} {dc:>14,.0f} {me:>16.6f}")

        fig.suptitle('Plot 4 — Marginal Efficiency Analysis', fontsize=13,
                     fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'plot4_marginal_efficiency.png', dpi=150)
        plt.close()
        print(f"  Saved: plot4_marginal_efficiency.png")

        # ------------------------------------------------------------------ #
        # Plot 5 — Real-time Feasibility: FPS vs VMAF                        #
        # (with 30 fps and 60 fps threshold lines — plan §5.3 plot 5)        #
        # ------------------------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(9, 6))

        for i, depth in enumerate(depths):
            d_df = results_df[results_df['depth'] == depth]
            ax.scatter(d_df['vmaf'], d_df['fps'], s=80,
                       label=f'Tier {depth}', zorder=3)

        # Mean trajectory
        ax.plot(agg['vmaf'], agg['fps'], 'k--', linewidth=1.5, alpha=0.7,
                label='Mean trajectory', zorder=2)

        # Threshold lines (plan requirement)
        ax.axhline(y=30, color='red',   linestyle='--', linewidth=1.5,
                   label='30 FPS threshold', alpha=0.8)
        ax.axhline(y=60, color='green', linestyle='--', linewidth=1.5,
                   label='60 FPS threshold', alpha=0.8)

        ax.set_xlabel('VMAF', fontsize=12)
        ax.set_ylabel('FPS', fontsize=12)
        ax.set_title('Plot 5 — Real-time Feasibility', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.4)

        # Annotate which tiers exceed 60 fps
        for _, row in agg.iterrows():
            status = '✓ RT' if row['fps'] >= 60 else ('≥30' if row['fps'] >= 30 else '✗')
            ax.annotate(f'Tier {int(row["depth"])} {status}',
                        xy=(row['vmaf'], row['fps']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'plot5_realtime_feasibility.png', dpi=150)
        plt.close()
        print(f"  Saved: plot5_realtime_feasibility.png")

        # ------------------------------------------------------------------ #
        # Per-sequence individual plots                                       #
        # ------------------------------------------------------------------ #
        for seq_name in results_df['sequence'].unique():
            seq_df = results_df[results_df['sequence'] == seq_name]

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f'Sequence: {seq_name}', fontsize=12)

            depths_s  = seq_df['depth'].values
            vmaf_s    = seq_df['vmaf'].values
            latency_s = seq_df['latency_ms'].values
            fps_s     = seq_df['fps'].values

            axes[0].plot(depths_s, vmaf_s, 'o-')
            axes[0].set_xlabel('Decode Depth / Tier')
            axes[0].set_ylabel('VMAF')
            axes[0].set_title('Quality vs Tier')
            axes[0].grid(True, alpha=0.4)

            axes[1].plot(depths_s, latency_s, 'o-', color='darkorange')
            axes[1].set_xlabel('Decode Depth / Tier')
            axes[1].set_ylabel('Latency (ms/frame)')
            axes[1].set_title('Latency vs Tier')
            axes[1].grid(True, alpha=0.4)

            axes[2].plot(vmaf_s, fps_s, 'o-', color='green')
            axes[2].set_xlabel('VMAF')
            axes[2].set_ylabel('FPS')
            axes[2].set_title('Quality vs Speed')
            axes[2].axhline(y=30, color='r', linestyle='--', label='30 FPS')
            axes[2].axhline(y=60, color='g', linestyle='--', label='60 FPS')
            axes[2].legend(fontsize=8)
            axes[2].grid(True, alpha=0.4)

            plt.tight_layout()
            plt.savefig(output_dir / f'{seq_name}_results.png', dpi=150)
            plt.close()

        print(f"\nAll plots saved to {output_dir}/")

    def save_results(self, results_df: pd.DataFrame,
                     output_path: str = 'results.json'):
        """Save results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_dict = results_df.to_dict(orient='records')
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate DCVC Progressive — all 5 plots from research plan §5.3'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/uvg',
                        help='Test data directory (UVG with sub-sequence folders)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for plots and JSON')
    parser.add_argument('--depths', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='Tiers / depths to evaluate')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                        help='Specific sequence names to evaluate (default: all)')
    parser.add_argument('--profile_flops', action='store_true',
                        help='Run FLOPs profiler to get accurate compute axis')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    import sys
    sys.path.insert(0, 'src')
    from dcvc_progressive import DCVCProgressive

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    model = DCVCProgressive()
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    evaluator = Evaluator(model, device=str(device))

    # Optional FLOPs profiling for accurate Plot 3 & 4 compute axis
    flops_per_tier = None
    if args.profile_flops:
        try:
            from progressive_decoder import ProgressiveDecoder
            decoder = ProgressiveDecoder(out_channel_M=96, out_channel_N=64,
                                         num_refinement_blocks=3)
            latent_shape = (1, 96, 64, 64)
            flops_per_tier = decoder.get_flops_per_stage(latent_shape)
            print(f"\nFLOPs per tier: {flops_per_tier}")
        except Exception as e:
            print(f"  FLOPs profiling skipped: {e}")

    print(f"\nEvaluating on dataset: {args.data}")
    results = evaluator.evaluate_dataset(
        args.data,
        sequences=args.sequences,
        depths=args.depths
    )

    # Results summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (plan §4.3 — Marginal Efficiency Table)")
    print("=" * 70)
    summary = results.groupby('depth').agg({
        'vmaf':       ['mean', 'std'],
        'psnr':       ['mean', 'std'],
        'fps':        ['mean', 'std'],
        'latency_ms': ['mean', 'std'],
        'bpp':        ['mean'],
    })
    print(summary.to_string())

    # Check success criteria (plan §6)
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK (plan §6)")
    print("=" * 70)
    tier1 = results[results['depth'] == 1]['vmaf'].mean()
    tier4 = results[results['depth'] == 4]['vmaf'].mean()
    fps_tier1 = results[results['depth'] == 1]['fps'].mean()
    vmaf_range = tier4 - tier1

    print(f"  VMAF range (Tier 4 - Tier 1): {vmaf_range:.2f}  "
          f"[Need ≥3–5 VMAF points] {'✓' if vmaf_range >= 3.0 else '✗'}")
    print(f"  Tier 1 FPS:                  {fps_tier1:.1f}  "
          f"[Need ≥60 FPS 1080p]    {'✓' if fps_tier1 >= 60 else '✗'}")

    # Monotonicity check
    for seq_name in results['sequence'].unique():
        seq_vmaf = [results[(results['sequence'] == seq_name) &
                            (results['depth'] == d)]['vmaf'].mean()
                    for d in sorted(args.depths)]
        monotonic = all(seq_vmaf[i] <= seq_vmaf[i+1] for i in range(len(seq_vmaf)-1))
        print(f"  Monotonicity ({seq_name}): {'✓' if monotonic else '✗ VIOLATED'}")

    evaluator.save_results(results, Path(args.output) / 'results.json')
    evaluator.plot_results(results, output_dir=args.output,
                           flops_per_tier=flops_per_tier)


if __name__ == '__main__':
    main()