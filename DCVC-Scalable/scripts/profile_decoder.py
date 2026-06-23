"""
FLOPs profiling utility for DCVC Progressive decoder

Run this script to compute exact FLOPs breakdown per layer and per tier.

Usage:
    python scripts/profile_decoder.py
"""

import torch
import torch.nn as nn

import sys
sys.path.insert(0, 'src')

from progressive_decoder import ProgressiveContextualDecoder, RefinementBlock, GDN


def count_flops_conv(m, H, W):
    """Count multiply-accumulate FLOPs for a single Conv2d layer.

    FLOPs = in_channels * out_channels * kernel_h * kernel_w * out_h * out_w
    """
    if isinstance(m, nn.Conv2d):
        out_h = (H + 2 * m.padding[0] - m.kernel_size[0]) // m.stride[0] + 1
        out_w = (W + 2 * m.padding[1] - m.kernel_size[1]) // m.stride[1] + 1
        return m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] * out_h * out_w
    return 0


def profile_decoder_layers(decoder, spatial_dim=(64, 64)):
    """Profile each layer in the decoder.

    Args:
        decoder:      ProgressiveContextualDecoder instance
        spatial_dim:  (H, W) of the *latent* tensor fed into the decoder.

    Returns:
        dict mapping layer name -> FLOPs.

    Notes on spatial dimensions:
        - Base decoder layers (conv1-4, GDN) operate at latent spatial dim.
        - The upsample block has 3× PixelShuffle(2) stages, giving 8× total upscale.
        - Refinement layers (proj, block, out) all operate at the full OUTPUT dim
          = latent_dim × 8  (encoder 8× down → decoder 8× up → full resolution).
    """
    lat_H, lat_W = spatial_dim
    out_H, out_W = lat_H * 8, lat_W * 8   # 3× PixelShuffle(2) = 8× spatial upsample
    results = {}

    # ------------------------------------------------------------------ #
    #  BASE DECODER  (operates at latent spatial resolution)              #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print(f"BASE DECODER LAYER PROFILE  (latent: {lat_H}×{lat_W})")
    print("=" * 70)

    total_base_flops = 0
    for name, module in decoder.part1.named_modules():
        if len(list(module.children())) == 0:
            flops = count_flops_conv(module, lat_H, lat_W)
            if flops > 0:
                results[f"part1.{name}"] = flops
                total_base_flops += flops
                print(f"  part1.{name:34s}: {flops:>12,} FLOPs")
                
    for name, module in decoder.part2.named_modules():
        if len(list(module.children())) == 0:
            flops = count_flops_conv(module, out_H, out_W)
            if flops > 0:
                results[f"part2.{name}"] = flops
                total_base_flops += flops
                print(f"  part2.{name:34s}: {flops:>12,} FLOPs")

    print(f"\n  TOTAL BASE DECODER: {total_base_flops:,} FLOPs")

    # ------------------------------------------------------------------ #
    #  REFINEMENT BLOCKS  (operate at decoder OUTPUT resolution)          #
    #                                                                      #
    #  Each stage has THREE sub-layers:                                    #
    #    proj  : Conv2d(3,  N, 1×1) — project RGB → feature space        #
    #    block : Conv2d(N,  N, 3×3) × 2 — residual refinement            #
    #    out   : Conv2d(N,  3, 1×1) — project feature space → RGB        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print(f"REFINEMENT BLOCK PROFILE  (full output res: {out_H}×{out_W})")
    print("=" * 70)

    for i in range(decoder.num_refinement_blocks):
        block_flops = 0

        # proj layer: Conv2d(3 → N, 1×1)
        proj_conv  = decoder.refinement_projs[i][0]   # first element is the Conv2d
        proj_flops = count_flops_conv(proj_conv, out_H, out_W)
        block_flops += proj_flops

        # refinement block: two Conv2d(N → N, 3×3)
        ref_block_flops = 0
        for _name, module in decoder.refinement_blocks[i].named_modules():
            if len(list(module.children())) == 0:
                ref_block_flops += count_flops_conv(module, out_H, out_W)
        block_flops += ref_block_flops

        # out layer: Conv2d(N -> 3, 1x1)
        out_conv   = decoder.refinement_outs[i]
        out_flops  = count_flops_conv(out_conv, out_H, out_W)
        block_flops += out_flops

        results[f"refinement_{i}"] = block_flops
        print(f"  Refinement Block {i}:")
        print(f"    proj  (3->N, 1x1)    : {proj_flops:>12,} FLOPs")
        print(f"    block (N->N, 3x3 x2) : {ref_block_flops:>12,} FLOPs")
        print(f"    out   (N->3, 1x1)    : {out_flops:>12,} FLOPs")
        print(f"    SUBTOTAL            : {block_flops:>12,} FLOPs")

    return results


def compute_tier_flops(decoder, layer_results, spatial_dim=(64, 64)):
    """Compute cumulative FLOPs per compute tier.

    Args:
        decoder:       ProgressiveContextualDecoder instance
        layer_results: Output from profile_decoder_layers
        spatial_dim:   (H, W) of latent (used only for display)

    Returns:
        dict {tier_index: cumulative_flops}
    """
    base_flops = sum(v for k, v in layer_results.items() if k.startswith('part'))

    # Tier 1 = base only; each subsequent tier adds one refinement block
    tiers = {1: base_flops}
    for i in range(decoder.num_refinement_blocks):
        ref_flops = layer_results.get(f'refinement_{i}', 0)
        tiers[i + 2] = tiers[i + 1] + ref_flops

    max_tier  = max(tiers)
    max_flops = tiers[max_tier]

    print("\n" + "=" * 70)
    print("COMPUTE TIER FLOPs BREAKDOWN")
    print("=" * 70)

    for tier, flops in tiers.items():
        pct = 100 * flops / max_flops
        print(f"  Tier {tier}: {flops:>14,} FLOPs  ({pct:5.1f}% of Tier {max_tier})")

    print(f"\n  FLOPs Variation (Tier {max_tier} / Tier 1): {max_flops / tiers[1]:.2f}x")

    return tiers


def profile_with_thop(decoder, input_shape=(1, 96, 64, 64)):
    """Profile using the thop library for PyTorch-accurate FLOPs.

    Args:
        decoder:      ProgressiveContextualDecoder instance
        input_shape:  Shape of latent input tensor
    """
    from thop import profile as thop_profile

    print("\n" + "=" * 70)
    print("THOP PROFILING (PyTorch-accurate)")
    print("=" * 70)

    decoder.eval()
    x = torch.randn(*input_shape)

    for depth in [1, 2, 3, 4]:
        flops, params = thop_profile(decoder, inputs=(x, depth), verbose=False)
        print(f"  Depth {depth}: {flops:>12,} FLOPs,  {params:,} params")


def main():
    print("=" * 70)
    print("DCVC PROGRESSIVE DECODER PROFILING")
    print("=" * 70)

    decoder = ProgressiveContextualDecoder(
        out_channel_M=96,
        out_channel_N=64,
        num_refinement_blocks=3
    )
    decoder.eval()

    print(f"\nDecoder configuration:")
    print(f"  - out_channel_M:         {decoder.out_channel_M}")
    print(f"  - out_channel_N:         {decoder.out_channel_N}")
    print(f"  - num_refinement_blocks: {decoder.num_refinement_blocks}")
    print(f"  - residual_scale:        0.1")

    # spatial_dim = latent spatial resolution.
    # For a 256×448 input with encoder stride 8: latent is 32×56.
    # Use 64×64 as a round canonical value for the paper table.
    spatial_dim = (64, 64)

    layer_results = profile_decoder_layers(decoder, spatial_dim)
    tier_flops    = compute_tier_flops(decoder, layer_results, spatial_dim)

    try:
        profile_with_thop(decoder, input_shape=(1, 96, *spatial_dim))
    except Exception as e:
        print(f"\n(thop profiling skipped: {e})")

    # ------------------------------------------------------------------ #
    #  PAPER TABLE                                                         #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("FLOPs TABLE FOR PAPER")
    print("=" * 70)
    print(f"  Latent spatial: {spatial_dim[0]}×{spatial_dim[1]}  |  "
          f"Output spatial: {spatial_dim[0]*8}×{spatial_dim[1]*8}  (8× upsample)\n")

    base_total = sum(v for k, v in layer_results.items() if k.startswith('part'))
    ref_ea     = layer_results.get('refinement_0', 0)
    max_flops  = tier_flops[max(tier_flops)]

    col_w = 24
    print(f"  {'Module':<{col_w}} | {'FLOPs':>14} | {'% of Base':>10}")
    print(f"  {'-'*col_w}-+-{'-'*14}-+-{'-'*10}")
    for name, flops in layer_results.items():
        if name.startswith('part'):
            short = name
            print(f"  {short:<{col_w}} | {flops:>14,} | {100*flops/base_total:>8.1f}%")
    print(f"  {'-'*col_w}-+-{'-'*14}-+-{'-'*10}")
    print(f"  {'Base Decoder (total)':<{col_w}} | {base_total:>14,} |   100.0%")
    print(f"  {'Refinement block (each)':<{col_w}} | {ref_ea:>14,} | {100*ref_ea/base_total:>8.1f}%")
    print(f"  {'-'*col_w}-+-{'-'*14}-+-{'-'*10}")
    for tier, flops in tier_flops.items():
        label = f"Tier {tier}  (depth={tier})"
        print(f"  {label:<{col_w}} | {flops:>14,} | {100*flops/max_flops:>8.1f}%")
    print()


if __name__ == '__main__':
    main()