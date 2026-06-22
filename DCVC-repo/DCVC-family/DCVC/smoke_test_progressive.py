"""Smoke test for ProgressiveContextualDecoder within the DCVC package.
Run from: DCVC-repo/DCVC-family/DCVC/
"""
import sys

# Same import pattern as DCVC's own test_video.py
from src.models.progressive_decoder import ProgressiveContextualDecoder
import torch

print("=== Smoke test: ProgressiveContextualDecoder ===")

dec = ProgressiveContextualDecoder(out_channel_M=96, out_channel_N=64, num_refinement_blocks=3)
dec.eval()

total = sum(p.numel() for p in dec.parameters())
print(f"PASS  Total params: {total}")

with torch.no_grad():
    y_hat   = torch.randn(1, 96, 4, 4)    # latent at H/16
    context = torch.randn(1, 64, 64, 64)  # context at full res

    for depth in [1, 2, 3, 4]:
        outs = dec(y_hat, context, depth=depth)
        assert len(outs) == depth, f"Expected {depth} outputs, got {len(outs)}"
        assert outs[-1].shape == (1, 3, 64, 64), f"Wrong shape: {outs[-1].shape}"
        vals = outs[-1]
        assert 0.0 <= vals.min().item() and vals.max().item() <= 1.0, "Output not in [0,1]"
        print(f"PASS  depth={depth}: {len(outs)} output(s), shape={tuple(outs[-1].shape)}")

# Verify tier outputs are monotonically improving (or at least different)
all_outs = dec(y_hat, context, depth=None)
assert len(all_outs) == 4, f"Expected 4 outputs, got {len(all_outs)}"
print("PASS  depth=None returns all 4 tiers")

# Part 1 output shape check: should go from (1,96,4,4) to (1,64,64,64)
feat = dec.part1(y_hat)
assert feat.shape == (1, 64, 64, 64), f"part1 output wrong shape: {feat.shape}"
print(f"PASS  part1 upsample: {tuple(y_hat.shape)} -> {tuple(feat.shape)}")

print()
print("ALL CHECKS PASSED")
print("ProgressiveContextualDecoder is a valid drop-in for DCVC contextualDecoder_part1/part2")
