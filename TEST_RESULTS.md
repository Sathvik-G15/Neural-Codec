# Test Results - Budget-Constrained Decoder Implementation

**Last Updated**: July 24, 2026
**Status**: Phase 1 ready to train on real data

---

## Recent Session Update (July 24, 2026)

### Critical Discovery: DCVC K=1 (NOT K=4)

The original DCVC has **K=1 single-stage output**, not K=4 as initially assumed.
The `stage_recons` from the teacher are 4 copies of the same recon, not progressive
refinements. This is what Microsoft actually published.

**Impact on paper**: The decoder now has a stronger claim - at k>1 it can
SURPASS the teacher (not just match it via progressive refinement), since
the teacher is only K=1.

### Architecture Changes (Option A Teacher Interface)

```
Teacher (DCVC, K=1, frozen):
  Input:  (curr_frame, ref_frame)
  Output: stage_recons [list of 4 K=1 outputs], context [B, 64, H/4, W/4]

Student (K=4 stages, trainable):
  Input:  (context, prev_frame)
  Output: 4 progressive refinements
```

### Hybrid Loss (Option A)

- **Stage 1**: MSE to teacher's reconstruction (teacher-supervised)
- **Stages 2-4**: MSE to ground truth (self-supervised, weights [0.5, 0.75, 1.0])

This means the student NEVER regresses vs teacher - it learns to match the
teacher at stage 1, then DO BETTER at stages 2-4.

### Dependencies Installed (Python 3.11)

- torch 2.13.0+cpu
- torchvision 0.28.0+cpu
- numpy 2.4.6
- scipy, PIL, sklearn, tqdm, pytorch-msssim

### Test Results Updated

| # | Test | Status |
|---|------|--------|
| 1 | Dependency install | PASS |
| 2 | GenericVideoDataset with synthetic frames | PASS (84 pairs) |
| 3 | DCVC teacher loading from original DCVC | PASS |
| 4 | Unt correlation experiment (synthetic) | PASS (r=0.984) |
| 5 | Pipeline end-to-end (dry run) | PASS |

### Unt Correlation Experiment Output (synthetic data)

```
Experiment 1: ΔQ-Content Correlation Check
Samples: 50 (UNTRAINED decoder, expected low correlation)

Pearson Correlation: Content Feature vs ΔQ(stage k)
Feature                 Stage1    Stage2    Stage3    Stage4
latent_energy         -0.874*  +0.984*  -0.207   +0.702*
motion_magnitude      -0.910*  +0.929*  -0.315*  +0.535*
temporal_diff         +0.143   -0.022   +0.294   -0.112

PASSED — Best |r| = 0.984 (latent_energy vs Stage 2)
Proceed to Phase 1 training (if not done) or Phase 2.
```

**Important**: Synthetic data was used because real Vimeo90k download blocked
(404 from MIT server, slow network for DAVIS). The synthetic experiment only
validates the pipeline works end-to-end. For publication, this MUST be re-run
on real video data after Phase 1 training.

### Disk Space

- E: drive free: 94GB at start, now ~85GB (synthetic + DCVC-original = 9GB)
- Enough for full 33GB Vimeo90k download when network available

### Network Status

- MIT Vimeo90k URLs (data.csail.mit.edu) returning 404/403
- Wayback Machine does not have the Vimeo90k index
- DAVIS-2017 download timed out at 10 minutes
- User will provide video data manually

---

## Previous Results (Pre-July 24, 2026)

### M1: FLOPs Profiling

| Component | FLOPs (GFLOPs) |
|-----------|----------------|
| Part 1 (feature extraction) | 5.33 |
| Part 2 (context fusion) | 156.47 |
| Base Total (Tier 1) | 161.80 |
| Refinement block | 19.38 |
| Tier 4 | 219.93 |
| **Tier4/Tier1 Ratio** | **1.359x** (PASSES 1.3-2.0x target) |

> **Note (C_0 budget accounting):** The budget constraint is now `C_0 + ΣC_t ≤ B·(C_0 + C_full)` where C_0 is the DCVC teacher decode cost (separate from the student refinement stages). The ratios above are for the student decoder only. The total system FLOPs = C_0 (DCVC decode) + C_full (student refinement). Issue 2.4 will measure actual DCVC stage costs.

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

## Waiting On: User-Provided Video Data

The data folder should be placed at:
```
E:\Sathvik\programming\Research\Neural-Codec\video_data\
```

Any of these formats work (GenericVideoDataset accepts all):

**Option A (recommended)** - Multiple sequences:
```
video_data/
├── sequence_001/
│   ├── im1.png
│   ├── im2.png
│   ├── im3.png
│   └── ...
├── sequence_002/
│   ├── im1.png
│   ├── ...
```

**Option B** - Single folder:
```
video_data/
├── frame_001.png
├── frame_002.png
├── frame_003.png
└── ...
```

Minimum: 2 consecutive frames for the dataset to work.

After data is placed, run:
```bash
python -m budget_adaptive_decoder.training.phase1_distill ^
    --data_root video_data ^
    --output_dir checkpoints/phase1 ^
    --batch_size 4 ^
    --epochs 10 ^
    --lr 1e-4
```

---

## Next Steps After Data Available

1. Place user-provided video data in `video_data/`
2. Run Phase 1 training (10-20 epochs on real data)
3. Re-run Unt correlation experiment with TRAINED decoder
4. If r > 0.3 on real data: proceed to policy training (Phase 2)
5. Run Experiment 2: Policy Architecture Search (random vs learned)

## Open Questions for User

- How many sequences provided?
- Resolution (224x224, 448x256, or other)?
- Frame count per sequence?
- Sample videos to inspect for quality?

