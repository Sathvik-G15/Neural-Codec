# Test Results - Modified Plan.md Implementation

**Date**: July 2026
**Status**: Implementation complete, FLOPs ratio below target

---

## Summary

| Test | Status | Notes |
|------|--------|-------|
| Import Test | PASS | All modules import successfully |
| FeatureRefinementBlock Forward | PASS | Output shape matches input, zero-init verified |
| ProgressiveContextualDecoder Forward | PASS | len(outputs) == depth invariant holds |
| Zero-Init Verification | PASS | All refinement block pw2 weights ~0 |
| **FLOPs Ratio (256x256)** | **FAIL** | 1.095x vs target 1.3x-2.0x |
| FLOPs Ratio (1080p extrapolated) | FAIL | 1.095x vs target 1.3x-2.0x |

---

## M1: FLOPs Profiling Results

### Measured at 256x256 (thop)

| Component | FLOPs (MFLOPs) |
|-----------|---------------|
| part1 (feature extraction) | 4,737.47 |
| part2 (contextual fusion) | 14,608.76 |
| **Base Total (Tier 1)** | **19,346.23** |
| Refinement block (64ch depthwise-sep) | 612.37 |

### Extrapolated to 1080p (1920x1080)

| Component | FLOPs (GFLOPs) |
|-----------|----------------|
| Base Total (Tier 1) | 612.13 |
| Refinement block | 19.38 |
| **Tier 4** | **670.25** |
| **Tier4/Tier1 Ratio** | **1.095x** |

### Target Range: 1.3x - 2.0x

**FINDING: Ratio 1.095x is BELOW the 1.3x-2.0x target range.**

---

## Analysis

### Why is the ratio low?

The base decoder (part1 + part2) consumes ~612 GFLOPs at 1080p while each refinement block consumes only ~19 GFLOPs. This means:
- Refinement blocks are only ~3% of base cost
- Adding 3 refinement blocks increases total by only ~9.5%

### Plan Estimate vs Actual

| Metric | Plan Estimate (§2.3) | Measured |
|--------|---------------------|----------|
| Base decoder | ~15-20 GFLOPs | 612 GFLOPs |
| Refinement block | ~5 GFLOPs | 19.4 GFLOPs |
| Tier4/Tier1 ratio | ~1.86x | 1.095x |

The plan's base decoder estimate appears to have been significantly underestimated. The actual base decoder is ~30-40x heavier than the plan estimated.

### Sensitivity (per plan §2.3 note)

Even at the plan's optimistic base estimate (15 GFLOPs), with 3 × 5 GFLOPs refinement blocks:
- Tier4 = 15 + 15 = 30 GFLOPs
- Ratio = 30/15 = 2.0x (at edge of target)

But with actual measured base (612 GFLOPs) and measured refine (19.4 GFLOPs):
- Tier4 = 612 + 3×19.4 = 670 GFLOPs
- Ratio = 670/612 = 1.095x

**The ratio is below target regardless of base decoder cost assumptions.**

---

## Recommendation

Per plan M1: "profiled ratio agrees with this 1.86x estimate within ±10%, else revisit design"

The profiled ratio (1.095x) does NOT agree with the estimate (1.86x). Per the plan's own criteria, this requires revisiting the design before training.

### Options:

1. **Increase refinement block complexity** (within feature-space constraint):
   - Add more depthwise-pointwise pairs
   - Use internal channel expansion (e.g., 64→128→64)
   - This increases per-block FLOPs while maintaining feature-space operation

2. **Reduce base decoder complexity**:
   - Not feasible per plan: "hold part1 fixed"

3. **Proceed with caveat**:
   - Document that ratio is lower than target
   - The systems claim (real-time adaptation) may still hold if Tier 1 is fast enough
   - But the "modest increment" claim in §1 would need revision

---

## Next Steps (per plan)

1. **Resolve FLOPs ratio issue** before M2/M3 training
2. Re-profile after any architectural changes
3. If ratio cannot be brought into 1.3x-2.0x range, revisit §2.3 architecture decision

---

## Test Commands Used

```bash
# Basic import and forward test
python -c "
import torch
import sys
sys.path.insert(0, 'DCVC-repo/DCVC-family/DCVC')
from src.models.progressive_decoder import ProgressiveContextualDecoder, FeatureRefinementBlock
# ... test code ...
"

# FLOPs profiling with thop
python -c "
from thop import profile
# ... profiling code ...
"
```

---

## Verified Behaviors

| Behavior | Verified |
|----------|----------|
| FeatureRefinementBlock produces identity at init | Yes (pw2 zero-initialized) |
| len(outputs) == depth invariant | Yes (tested for depth 1-4) |
| Zero-init on all 3 refinement blocks | Yes |
| Model loads DCVC pretrained weights | Yes (via load_dict) |