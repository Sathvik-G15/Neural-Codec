# Verification Scripts

This directory contains pre-implementation verification scripts for the
Budget-Constrained Neural Decoder project.

## Running Verification Checks

### 1. Stage Cost Verification (`check_stage_costs.py`)

Measures FLOPs per stage and verifies budget alignment.

```bash
cd budget_adaptive_decoder
python verify/check_stage_costs.py
```

**What it checks:**
- Each stage cost in GFLOPs
- Ratios to C_full
- Cumulative stage boundaries
- Whether design document budgets {0.2, 0.3, 0.44, 0.5, 0.67, 0.8, 0.89, 0.95, 1.0} align with stage boundaries

**Expected output:**
- Analytical stage costs based on design document ratios (4:2:2:1)
- Budget alignment table
- Feasibility check (are at least 2 depths feasible at each budget?)

---

### 2. ΔQ-Content Correlation Check (`check_correlation.py`)

**CRITICAL: Run this BEFORE Phase 3 training.**

Validates the core assumption that marginal quality gains ΔQ(k) correlate
with observable bitstream features.

```bash
cd budget_adaptive_decoder
python verify/check_correlation.py
```

**Required before running:**
1. Complete Phase 1 (decoder distillation)
2. Complete Phase 2 (ΔQ precomputation) OR use synthetic data for testing

**What it checks:**
- Loads DCVC teacher and trained decoder
- Runs on Vimeo90k subset (1000 frames by default)
- Computes ΔQ at each depth
- Computes Pearson correlation between content features and ΔQ
- Prints correlation table with PASS/FAIL indicators

**Success criterion:**
- At least one content feature must have |r| > 0.3 correlation with at least one ΔQ(k)

**CRITICAL OUTPUT:**
```
╔══════════════════════════════════════════════════════════╗
║        STOP — Do not proceed to training                 ║
╚══════════════════════════════════════════════════════════╝
```

If you see this message, **do not proceed to Phase 3 training**.
The content features do not provide sufficient signal for the policy
to distinguish frames that benefit from additional stages.
The core research claim must be reconsidered.

---

## Workflow Order

1. **First run**: `check_stage_costs.py`
   - Verifies stage cost assumptions
   - Update `configs/phase*.yaml` with actual costs if different from 4:2:2:1

2. **After Phase 1 + Phase 2**: `check_correlation.py`
   - CRITICAL validation before Phase 3
   - If PASS: proceed to Phase 3 training
   - If FAIL: STOP and reconsider approach

3. **After Phase 3**: Run experiments (see `evaluation/` directory)
   - Experiment 1: ΔQ-Content Correlation (repeat)
   - Experiment 2: k* vs Content at Fixed Budget
   - Experiment 3: Quality-Compute Pareto Curve

---

## Options

### check_correlation.py options

```bash
python verify/check_correlation.py --help

Options:
  --samples N     Number of samples to evaluate (default: 1000)
  --data_root DIR Dataset root directory
  --output FILE   Output JSON file for results
```

### check_stage_costs.py

No options required. Uses design document ratios by default.

---

## Interpreting Results

### Correlation Table Format

```
Feature          ΔQ(k=1)    ΔQ(k=2)    ΔQ(k=3)    ΔQ(k=4)
------------------------------------------------------------
latent_energy      0.5232*    0.4121*    0.3012     0.1523
motion_magnitude   0.3121*    0.2891     0.1823     0.0932
temporal_diff      0.1823     0.1521     0.1232     0.0821
frame_type         0.0523     0.0412     0.0312     0.0212
```

- `*` indicates p < 0.05 (statistically significant)
- Look for |r| > 0.3 at any feature-stage combination
- Higher correlations earlier (k=1, k=2) are expected as motion
  complexity affects early stages most

### Stage Cost Table Format

```
Stage 1: 4.0 units (44.44% of total)
Stage 2: 2.0 units (22.22% of total)
Stage 3: 2.0 units (22.22% of total)
Stage 4: 1.0 units (11.11% of total)
Full (C_full): 9.0 units

Cumulative stage boundaries:
  k=1: cumulative = 4, budget threshold = 0.4444
  k=2: cumulative = 6, budget threshold = 0.6667
  k=3: cumulative = 8, budget threshold = 0.8889
  k=4: cumulative = 9, budget threshold = 1.0000
```

---

## Troubleshooting

### "fvcore not installed" warning

Install fvcore for accurate FLOP measurements:
```bash
pip install fvcore
```

The script will use analytical estimates if fvcore is not available.

### "compressai not installed" warning

Install compressai for DCVC loading:
```bash
pip install compressai
```

The script will use stub implementations if compressai is not available.

### Out of memory

Reduce `--samples` to 500 or lower:
```bash
python verify/check_correlation.py --samples 500
```