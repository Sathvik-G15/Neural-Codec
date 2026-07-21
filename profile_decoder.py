"""M1: Profile decoder FLOPs to confirm 1.3x-2.0x Tier4/Tier1 ratio.

This script profiles the progressive decoder's reconstruction FLOPs per tier
to verify the feature-space depthwise-separable design (§2.3) hits the
target ratio before any training begins.

Usage:
    python profile_decoder.py

Outputs:
    - Per-tier FLOPs and the Tier4/Tier1 ratio
    - Comparison against the plan's 1.86x estimate
    - Confirmation/rejection of the design decision before training investment
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

DCVC_DIR = Path(__file__).parent / "DCVC-repo" / "DCVC-family" / "DCVC"
sys.path.insert(0, str(DCVC_DIR))

from src.models.progressive_decoder import ProgressiveContextualDecoder, FeatureRefinementBlock
from src.models.DCVC_net import DCVC_net


def count_flops_linear(C_in, C_out, H, W, groups=1):
    return C_in * C_out * H * W / groups


def count_conv2d_flops(layer: nn.Conv2d, H: int, W: int) -> float:
    C_in = layer.in_channels
    C_out = layer.out_channels
    kH, kW = layer.kernel_size
    groups = layer.groups
    if isinstance(kH, int):
        flops = C_in * C_out * kH * kW * H * W / groups
    else:
        flops = C_in * C_out * kH[0] * kW[1] * H * W / groups
    return flops


def profile_refinement_block(block: FeatureRefinementBlock, H: int, W: int) -> float:
    flops = 0.0
    flops += count_conv2d_flops(block.dw1, H, W)
    flops += count_conv2d_flops(block.pw1, H, W)
    flops += count_conv2d_flops(block.dw2, H, W)
    flops += count_conv2d_flops(block.pw2, H, W)
    return flops


def profile_part1(part1: nn.Sequential, y_hat_H: int, y_hat_W: int) -> float:
    H, W = y_hat_H * 4, y_hat_W * 4
    flops = 0.0
    for layer in part1:
        if isinstance(layer, nn.Conv2d):
            flops += count_conv2d_flops(layer, H, W)
        elif isinstance(layer, nn.Sequential):
            for sublayer in layer:
                if isinstance(sublayer, nn.Conv2d):
                    flops += count_conv2d_flops(sublayer, H, W)
        H = H * 2
        W = W * 2
    return flops


def profile_part2(part2: nn.Sequential, H: int, W: int) -> float:
    flops = 0.0
    h, w = H, W
    for layer in part2:
        if isinstance(layer, nn.Conv2d):
            flops += count_conv2d_flops(layer, h, w)
        elif isinstance(layer, nn.Sequential):
            for sublayer in layer:
                if isinstance(sublayer, nn.Conv2d):
                    flops += count_conv2d_flops(sublayer, h, w)
    return flops


def profile_decoder(decoder: ProgressiveContextualDecoder, resolution: tuple = (1080, 1920)):
    """Profile FLOPs for each tier of the progressive decoder.

    Returns:
        Dict with per-tier FLOPs and the Tier4/Tier1 ratio.
    """
    H, W = resolution
    y_hat_H, y_hat_W = H // 16, W // 16
    context_H, context_W = H, W

    part1_flops = 0.0
    h, w = y_hat_H, y_hat_W
    for layer in decoder.part1:
        if isinstance(layer, nn.Conv2d):
            part1_flops += count_conv2d_flops(layer, h, w)
        elif isinstance(layer, nn.Sequential):
            for sl in layer:
                if isinstance(sl, nn.Conv2d):
                    part1_flops += count_conv2d_flops(sl, h, w)
        if hasattr(layer, 'stride') and layer.stride == (2, 2):
            h, w = h * 2, w * 2

    part2_flops = 0.0
    for layer in decoder.part2:
        if isinstance(layer, nn.Conv2d):
            part2_flops += count_conv2d_flops(layer, context_H, context_W)
        elif isinstance(layer, nn.Sequential):
            for sl in layer:
                if isinstance(sl, nn.Conv2d):
                    part2_flops += count_conv2d_flops(sl, context_H, context_W)

    base_recon_flops = part1_flops + part2_flops

    refinement_flops_per_block = profile_refinement_block(
        decoder.refinement_blocks[0], context_H, context_W
    )

    tier_flops = {
        "Tier 1 (base)": base_recon_flops,
        "Tier 2 (+R_1)": base_recon_flops + refinement_flops_per_block,
        "Tier 3 (+R_1,R_2)": base_recon_flops + 2 * refinement_flops_per_block,
        "Tier 4 (+R_1,R_2,R_3)": base_recon_flops + 3 * refinement_flops_per_block,
    }

    return {
        "part1_flops": part1_flops,
        "part2_flops": part2_flops,
        "base_recon_flops": base_recon_flops,
        "refinement_flops_per_block": refinement_flops_per_block,
        "tier_flops": tier_flops,
        "ratio_T4_to_T1": (base_recon_flops + 3 * refinement_flops_per_block) / base_recon_flops,
    }


def profile_with_thop(decoder: ProgressiveContextualDecoder, resolution: tuple = (1080, 1920)):
    """Profile using thop (if available) for more accurate CUDA FLOPs.

    Falls back to analytical counting if thop is not installed.
    """
    try:
        from thop import profile
        model = decoder
        y_hat = torch.randn(1, 96, resolution[0] // 16, resolution[1] // 16).cuda()
        context = torch.randn(1, 64, resolution[0], resolution[1]).cuda()
        model = model.cuda()
        model.eval()
        with torch.no_grad():
            flops, params = profile(model, inputs=(y_hat, context, 4), verbose=False)
        return flops
    except ImportError:
        print("  thop not installed. Using analytical FLOPs estimate.")
        return None


def main():
    print("=" * 60)
    print("M1: Progressive Decoder FLOPs Profiling")
    print("=" * 60)

    print("\n--- Analytical FLOPs Estimate (1080p, 1920x1080) ---")

    decoder = ProgressiveContextualDecoder(
        out_channel_M=96,
        out_channel_N=64,
        num_refinement_blocks=3,
    )
    decoder = decoder.cuda()
    decoder.eval()

    results = profile_decoder(decoder, resolution=(1080, 1920))

    print(f"\nPart 1 (feature extraction):     {results['part1_flops'] / 1e9:.2f} GFLOPs")
    print(f"Part 2 (context fusion -> RGB): {results['part2_flops'] / 1e9:.2f} GFLOPs")
    print(f"Base reconstruction (Part 1+2): {results['base_recon_flops'] / 1e9:.2f} GFLOPs")
    print(f"Refinement block (depthwise-sep): {results['refinement_flops_per_block'] / 1e9:.2f} GFLOPs")
    print()
    print("Per-tier reconstruction FLOPs:")
    for tier, flops in results["tier_flops"].items():
        print(f"  {tier}: {flops / 1e9:.2f} GFLOPs")

    ratio = results["ratio_T4_to_T1"]
    print(f"\nTier4/Tier1 FLOPs ratio: {ratio:.3f}x")
    print(f"Target range: 1.3x - 2.0x")

    if 1.3 <= ratio <= 2.0:
        print("\n[PASS] Ratio is within target range!")
    else:
        print(f"\n[FAIL] Ratio {ratio:.3f}x is outside target [1.3x, 2.0x]")
        print("  Per plan §2.3: fall back to internal channel bottleneck design")

    plan_estimate = 1.86
    ratio_diff = abs(ratio - plan_estimate) / plan_estimate * 100
    print(f"\n--- Comparison with plan §2.3 estimate ---")
    print(f"Plan estimated: {plan_estimate:.2f}x")
    print(f"Actual ratio:   {ratio:.2f}x")
    print(f"Difference:    {ratio_diff:.1f}%")
    if ratio_diff <= 10:
        print("[PASS] Within ±10% of plan estimate")
    elif ratio_diff <= 20:
        print("[WARN] Between 10-20% of plan estimate - investigate discrepancy")
    else:
        print("[FAIL] More than 20% off - review assumptions")

    print("\n--- Sensitivity to Base Decoder Cost (plan §2.3 note) ---")
    base_costs = [15e9, 17.5e9, 20e9]
    for base_cost in base_costs:
        tier4 = base_cost + 3 * results["refinement_flops_per_block"]
        sens_ratio = tier4 / base_cost
        print(f"  If base = {base_cost/1e9:.1f} GFLOPs -> Tier4 = {tier4/1e9:.2f} GFLOPs, ratio = {sens_ratio:.2f}x")

    print("\n--- Actual CUDA Profiling (thop) ---")
    thop_flops = profile_with_thop(decoder, resolution=(1080, 1920))
    if thop_flops:
        print(f"thop reports: {thop_flops / 1e9:.2f} GFLOPs (Tier 4)")


if __name__ == "__main__":
    main()