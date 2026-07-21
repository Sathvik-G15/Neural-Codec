# Test Results - Modified Plan.md Implementation

**Date**: July 2026
**Status**: Implementation complete

---

## Summary

| # | Test | Status |
|---|------|--------|
| 1 | FeatureRefinementBlock | PASS |
| 2 | Zero-init verification | PASS |
| 3 | Progressive decoder depth invariant | PASS |
| 4 | DCVC model loading | PASS |
| 5 | Loss function + weight slicing | PASS |
| 6 | Inference/training mode split | PASS |
| 7 | FLOPs ratio (analytical) | PASS - 1.359x |
| 8 | Training curriculum (Phase 0-3) | PASS |

---

## M1: FLOPs Profiling

| Component | FLOPs (GFLOPs) |
|-----------|----------------|
| Part 1 (feature extraction) | 5.33 |
| Part 2 (context fusion) | 156.47 |
| Base Total (Tier 1) | 161.80 |
| Refinement block | 19.38 |
| Tier 4 | 219.93 |
| **Tier4/Tier1 Ratio** | **1.359x** (PASSES 1.3-2.0x target) |

---

## Architecture

- FeatureRefinementBlock: depthwise-separable (64ch, ~19 GFLOPs)
- Inference mode: part2 runs ONCE (self.training check)
- Training mode: deep supervision across all tiers

---

## Training Curriculum

| Phase | Steps | Depth | Purpose |
|-------|-------|-------|---------|
| 0 | 0-10K | 4 (fixed) | Ceiling check |
| 1 | 10K-70K | 4 (fixed) | Establish baseline |
| 2 | 70K-130K | random [1,4] | Learn early tiers |
| 3 | 130K-190K | random [1,4] | Fine-tune |

---

## Next Steps

1. Run Phase 0 (10K steps) on Kaggle
2. Compare Tier 4 PSNR against M0.5 baseline σ
3. If within ±1σ, advance to Phase 1