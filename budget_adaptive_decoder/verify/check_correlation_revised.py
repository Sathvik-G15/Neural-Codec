"""
Experiment 1: Check ΔQ-Content Correlation (Revised for K=1 Teacher, FIXED for N2).

Since DCVC has K=1 (single-stage output), the correlation check measures
whether content features predict the QUALITY GAIN from running additional
refinement stages (vs ground truth, not vs teacher).

ΔQ(k) = PSNR(k) - PSNR(k-1) measured against GROUND TRUTH

This tells us whether the policy can predict which frames benefit from
additional compute before any decoding runs.

FIX (N2): Content features used here MUST match the inputs the deployed
BitstreamContentExtractor (after S1 fix) will receive at inference:

  - context_energy:    pooled energy of DCVC `context` tensor
                        (proxy for motion-compensated feature magnitude)
  - recon_residual:     |recon_image - prev_frame|, mean magnitude
                        (S1 fix: replaces unspecified `motion_magnitude`)
  - prev_stats:         mean/std/energy of prev_frame pixels
  - context_spatial_std: spatial variability of `context`

Using synthetic stand-ins (e.g. "latent_energy" without a latent tensor)
is not allowed because the deployed extractor cannot access them.

Run AFTER Phase 1 training (untrained decoder = not informative).
Run on REAL data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import json
from tqdm import tqdm

from models.decoder import BudgetAdaptiveDecoder
from models.teacher import DCVCWrapper
from models.extractor import BitstreamContentExtractor
from data.generic_video_dataset import GenericVideoDataset


CORRELATION_THRESHOLD = 0.3
MIN_SAMPLES = 500


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between prediction and target."""
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def run_experiment1(
    teacher: DCVCWrapper,
    decoder: BudgetAdaptiveDecoder,
    dataset,
    n_samples: int = MIN_SAMPLES,
    device: torch.device = None,
    output_file: Path = None,
) -> Dict:
    """
    Run Experiment 1 on real video data.

    Measures whether content features correlate with ΔQ(k) where
    ΔQ(k) = PSNR at depth k - PSNR at depth k-1 (measured vs ground truth).

    Since DCVC has K=1:
    - Stage 1 output vs ground truth = base quality
    - Stage 2,3,4 outputs vs ground truth = refinement quality
    - ΔQ(k) measures how much stage k improves over stage k-1

    Args:
        teacher: DCVCWrapper (frozen)
        decoder: BudgetAdaptiveDecoder (can be untrained or trained)
        dataset: Video dataset yielding (curr_frame, ref_frame)
        n_samples: Number of samples to evaluate
        device: Device to run on
        output_file: Optional file to save results

    Returns:
        Dict with correlation results and PASS/FAIL status
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = teacher.to(device)
    decoder = decoder.to(device)
    teacher.dcvc.eval() if teacher.dcvc else None
    decoder.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    delta_q_records = []
    content_records = []
    psnr_absolute = []

    print("=" * 65)
    print("Experiment 1: ΔQ-Content Correlation Check")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Samples: {n_samples}")
    # The decoder's trained state is determined externally (the main()
    # function loads a checkpoint if --checkpoint is provided). The
    # framework otherwise cannot introspect from inside run_experiment1.
    decoder_is_trained = bool(getattr(decoder, '_trained', False))
    if decoder_is_trained:
        decoder_state = "TRAINED"
    else:
        decoder_state = "UNTRAINED (expected low correlation)"
    print(f"Decoder: {decoder_state}")
    print()

    feature_names = [
        'context_energy',
        'recon_residual',
        'prev_energy',
        'context_spatial_std',
    ]

    with torch.no_grad():
        for i, (curr_frame, ref_frame) in enumerate(tqdm(loader, desc="Running")):
            if i >= n_samples:
                break

            curr_frame = curr_frame.to(device)
            ref_frame = ref_frame.to(device)

            # Teacher: encode
            teacher_out = teacher.get_stage_targets(curr_frame, ref_frame)
            context = teacher_out['context']  # [B, 64, H/4, W/4]
            recon_image = teacher_out['stage_recons'][0]  # [B, 3, H, W] DCVC base recon

            # Content features (post S1 fix): ONLY these are available to the
            # deployed BitstreamContentExtractor at inference.
            context_energy    = context.abs().mean().item()
            recon_residual    = (recon_image - ref_frame).abs().mean().item()
            prev_energy       = ref_frame.abs().mean().item()
            context_spatial_std = context.std(dim=(-2, -1)).mean().item()

            content_records.append([
                context_energy,
                recon_residual,
                prev_energy,
                context_spatial_std,
            ])

            # Decoder: run at all depths
            # get_stage_states returns [R0, R1, R2, R3, R4]
            all_recons = decoder.get_stage_states(context, ref_frame)

            # Measure PSNR at each depth vs GROUND TRUTH (curr_frame)
            q_vals = []
            for recon in all_recons:
                psnr = compute_psnr(recon, curr_frame)
                q_vals.append(psnr)

            psnr_absolute.append(q_vals)

            # ΔQ(k) = PSNR(k) - PSNR(k-1)
            delta_q = [q_vals[k] - q_vals[k-1] for k in range(1, len(q_vals))]
            delta_q_records.append(delta_q)

            if i % 100 == 0:
                print(f"  {i}/{n_samples} | "
                      f"Q0:{q_vals[0]:.1f} Q1:{q_vals[1]:.1f} "
                      f"Q2:{q_vals[2]:.1f} Q3:{q_vals[3]:.1f} "
                      f"Q4:{q_vals[4]:.1f}")

    delta_q_arr = np.array(delta_q_records)
    content_arr = np.array(content_records)
    psnr_arr = np.array(psnr_absolute)

    # ΔQ Distribution
    print("\n" + "=" * 65)
    print("ΔQ Distribution (PSNR gain per stage vs ground truth)")
    print("=" * 65)
    print(f"{'Stage':8} {'Mean':>7} {'Std':>7} {'Min':>7} "
          f"{'Max':>7} {'%>0':>7} {'%>0.1dB':>9}")
    print("-" * 65)

    for k in range(decoder.K + 1):  # +1 for base (k=0)
        if k == 0:
            dq = delta_q_arr[:, 0]  # Q1 - Q0 vs GT
        else:
            dq = delta_q_arr[:, k - 1]

    for k in range(1, decoder.K + 1):
        dq = delta_q_arr[:, k - 1]
        print(f"Stage {k}  "
              f"{dq.mean():>7.3f} "
              f"{dq.std():>7.3f} "
              f"{dq.min():>7.3f} "
              f"{dq.max():>7.3f} "
              f"{(dq>0).mean():>7.1%} "
              f"{(dq>0.1).mean():>9.1%}")

    # Key checks
    print("\nKey checks:")
    mean_dq_s1 = delta_q_arr[:, 0].mean()
    if mean_dq_s1 < 0:
        print(f"  PROBLEM: Stage 1 mean ΔQ = {mean_dq_s1:.3f} dB (NEGATIVE)")
        print(f"    Stage 1 is REDUCING quality vs base reconstruction.")
    elif mean_dq_s1 < 0.05:
        print(f"  EXPECTED: Stage 1 mean ΔQ = {mean_dq_s1:.3f} dB (very small, untrained decoder)")
        print(f"    Decoder needs training before meaningful ΔQ emerges.")
    else:
        print(f"  OK: Stage 1 mean ΔQ = {mean_dq_s1:.3f} dB")

    # Correlation Matrix
    print("\n" + "=" * 65)
    print("Pearson Correlation: Content Feature vs ΔQ(stage k)")
    print("Threshold: |r| > 0.3 for at least one pair")
    print("=" * 65)

    header = f"{'Feature':<20}" + "".join(
        f"{'Stage'+str(k):>10}" for k in range(1, decoder.K + 1)
    )
    print(header)
    print("-" * 65)

    passed = False
    best_r = 0.0
    best_pair = None
    corr_matrix = np.zeros((len(feature_names), decoder.K))

    for fi, feat_name in enumerate(feature_names):
        feat_vals = content_arr[:, fi]
        row = f"{feat_name:<20}"

        for k in range(1, decoder.K + 1):
            r, p = scipy.stats.pearsonr(feat_vals, delta_q_arr[:, k - 1])
            corr_matrix[fi, k - 1] = r
            marker = "*" if abs(r) > CORRELATION_THRESHOLD else " "
            row += f"  {r:+.3f}{marker}"

            if abs(r) > CORRELATION_THRESHOLD:
                passed = True
            if abs(r) > abs(best_r):
                best_r = r
                best_pair = (feat_name, k)

        print(row)

    print("-" * 65)
    print("* indicates |r| > 0.3 threshold")

    # Decision
    print("\n" + "=" * 65)
    if passed:
        print(f"PASSED — Best |r| = {abs(best_r):.3f} ({best_pair[0]} vs Stage {best_pair[1]})")
        print("Proceed to Phase 1 training (if not done) or Phase 2.")
    else:
        print(f"FAILED — Maximum |r| = {abs(best_r):.3f} < {CORRELATION_THRESHOLD} threshold")
        all_dq_std = delta_q_arr.std(axis=0)
        if all_dq_std.max() < 0.05:
            print("\nNOTE: ΔQ variance is very low (expected for UNTRAINED decoder).")
            print("Action: Run Phase 1 training first, then re-run Experiment 1.")
        else:
            print("\nNOTE: ΔQ has variance but no content correlation.")
            print("Action: Reconsider content feature design.")
    print("=" * 65)

    result = {
        "passed": passed,
        "best_correlation": float(best_r),
        "best_pair": best_pair,
        "correlation_matrix": corr_matrix.tolist(),
        "feature_names": feature_names,
        "delta_q_mean": [float(delta_q_arr[:, k].mean()) for k in range(decoder.K)],
        "delta_q_std": [float(delta_q_arr[:, k].std()) for k in range(decoder.K)],
        "psnr_absolute_mean": [float(psnr_arr[:, k].mean()) for k in range(decoder.K + 1)],
        "num_samples": len(delta_q_records),
    }

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 1: ΔQ-Content Correlation")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to video frames directory")
    parser.add_argument("--n_samples", type=int, default=MIN_SAMPLES,
                        help=f"Number of samples (default {MIN_SAMPLES})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    parser.add_argument("--height", type=int, default=256,
                        help="Frame height (default 256)")
    parser.add_argument("--width", type=int, default=448,
                        help="Frame width (default 448)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to decoder checkpoint (defaults to untrained)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    teacher = DCVCWrapper(load_pretrained=True).to(device)
    decoder = BudgetAdaptiveDecoder().to(device)
    if args.checkpoint:
        print(f"Loading decoder checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "decoder_state_dict" in ckpt:
            decoder.load_state_dict(ckpt["decoder_state_dict"])
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            decoder.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            decoder.load_state_dict(ckpt["state_dict"])
        else:
            decoder.load_state_dict(ckpt)
        decoder._trained = True
        print("Decoder loaded from checkpoint.")
    print("Decoder ready:", "TRAINED" if args.checkpoint else "UNTRAINED")
    print("Models loaded.")

    dataset = GenericVideoDataset(
        root=args.data_root,
        height=args.height,
        width=args.width,
    )

    output_file = Path(args.output) if args.output else None

    result = run_experiment1(
        teacher=teacher,
        decoder=decoder,
        dataset=dataset,
        n_samples=args.n_samples,
        device=device,
        output_file=output_file,
    )

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()