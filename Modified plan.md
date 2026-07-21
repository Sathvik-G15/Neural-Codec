# Compute-Scalable Neural Video Decoding for Real-Time Adaptive Streaming

**Version**: 2.4
**Date**: July 2026
**Status**: Advisor-Ready (revised after code audit + peer review of v2.0 + confirmed training environment + peer review of v2.2 + peer review of v2.3)

---

## 0d. Changelog from v2.3

| Issue | Change |
|---|---|
| §2.3's 1.86× estimate didn't acknowledge base-decoder cost uncertainty (15–20 GFLOPs range) | Added explicit sensitivity note: ratio stays in-band (1.75×–1.86×) across the plausible base-cost range |
| §3.1 pseudocode didn't state the weights-slicing invariant, risking a misread that all 4 weights are required even at active_depth < 4 | Added explicit invariant line, matches `train_progressive.py:88`'s existing slicing behavior |
| §7 Ablation 2's "same number of steps as Phase 1+2+3" was unpinnable now that the curriculum is checkpoint-based (§3.2) | Anchored to a concrete step count with a plateau-based alternative, explicitly stated |
| §7 Ablation 3 (stretch) had no go/no-go criterion | Added a relative quota-margin criterion instead of an absolute week count, since exact remaining Kaggle budget isn't known in advance |
| §3.2/M2's "approaches stock DCVC PSNR" go/no-go wasn't pinned to a concrete decision rule | Added explicit rule: within ±1σ (all 8 sequences) → advance; ≥1 sequence outside ±2σ → architecture-capacity finding, revisit §2.3 |
| No venue-target statement anywhere in the document | Added to §1: systems venues (MMSys/ICIP/national flagship), with M6 as the deciding factor for workshop- vs. systems-tier |

---

## 0c. Changelog from v2.2

| Issue | Change |
|---|---|
| §2.3 committed to feature-space/depthwise-separable design with no anchoring FLOPs estimate | Added back-of-envelope estimate (~1.86×) with derivation; M1 deliverable now explicitly means "profiled number agrees with this estimate within ±10%, else revisit" |
| §3.2 phase boundaries (60K/120K/180K) read as fixed commitments, but Kaggle session caps make mid-phase disconnects likely | Explicit statement added: boundaries are checkpoints to resume from, not restart points; early-advance and mid-phase-resume both explicitly permitted |
| §4.3 "FLOP-match via architecture scaling" didn't specify width vs. depth scaling | Committed to width-scaling of `part2` only, `part1` held fixed |
| §5.3 didn't address PixelShuffle/subpixel-conv in `part1`, which is on the Tier-1 mobile-export critical path regardless of refinement-block design | Added explicit note; Tier 1 export risk now stated, not implied |
| §6 "σ" was ambiguous between benchmark σ (M0.5) and any training-run σ | Pinned explicitly to the M0.5 measurement on the specific unmodified checkpoint |
| §5.3 and §8 disagreed on whether M6 (mobile export + phone benchmarking) is required or stretch | Resolved: M6 is spike-gated — a 1-day feasibility spike decides required-vs-stretch status, rather than either section asserting it unconditionally |

---

## 0b. Changelog from v2.1

| Issue | Change |
|---|---|
| §3.2 assumed training happened on the 6GB RTX 4050, framed as a VRAM-scarcity problem | **Corrected**: training happens on Kaggle's T4 (16GB) — VRAM is not the binding constraint. The real constraint is Kaggle session runtime caps and weekly GPU quota, requiring resumable checkpointing. RTX 4050 is now correctly scoped as the benchmarking-only device (§5.2, §5.3), not the training device. |

---

## 0a. Changelog from v2.0

| Issue (flagged in v2.0 review) | Change |
|---|---|
| Refinement block channel-space was undecided (RGB vs. feature space) — this is a structural decision, not a thickness tweak | **Decided**: feature space, between `part1`/`part2`, depthwise-separable blocks (Option B). Rationale and fallback stated in §2.3. Confirmed/revised by profiling, not assumed. |
| §6 didn't say which FLOPs ratio (reconstruction vs. end-to-end) gates success | **Decided**: reconstruction-only ratio gates; end-to-end ratio reported, not gating. |
| "Meaningful" PSNR improvement was undefined | Defined via ±σ against stock DCVC PSNR variance on UVG, measured cheaply pre-training (new M0.5) |
| `forward_with_depth` pseudocode indexing didn't match `progressive_decoder.py`'s `range(depth - 1)` | Fixed in §3.1 |
| Ablation 2 (fixed vs. random depth) truncation baseline was broken by zero-init proj_out — would trivially degenerate to base-decoder output | Protocol tightened in §7 |
| M1 blocked "any training" but didn't include the architecture decision itself | M1 now explicitly includes committing to the channel-space interpretation before profiling is treated as complete |
| Device budget was unconfirmed ("lab GPU" assumed) | Confirmed: single RTX 4050 6GB laptop GPU + Realme Narzo 5G phone. VRAM constraint now addressed in training plan (§3.2); phone export pipeline risk addressed in §5.3 |

---

## 0. Changelog from v1.1

| Issue | Change |
|---|---|
| Refinement block was ~4× base FLOPs per stage (code), not "lightweight" (plan) → Tier4/Tier1 ≈ 13×, not 1.5× | Refinement block redesigned to hit target ratio; profiling step added **before** any training |
| Novelty was thin (early-exit applied to a new domain) | Reframed explicitly as a **systems paper**; added real-device heterogeneous benchmarking as primary evidence |
| No comparison against "4 separate fixed-size decoders" baseline | Added as a required experiment, not just a rebuttal argument |
| Missing complexity-scalable coding literature | Added to Related Work; required reading before Architecture section is finalized |
| Weight-strategy rationale conflicted between plan and code docstring | Resolved and stated once, unambiguously |
| "Epochs" vs "steps" ambiguous in training curriculum | Standardized to **steps** throughout |
| VMAF pipeline absent, but treated as required | PSNR is the Milestone-1 metric; VMAF is added only once a checkpoint exists |
| 10-week fixed calendar | Replaced with a **milestone sequence** (dependency-ordered, not date-ordered) |
| 3 ablations planned, likely unaffordable | Reduced to 2; third listed as stretch |

---

## 1. Problem Statement

Modern neural video codecs (e.g., DCVC) assume **fixed decode compute**. A mobile device and a high-end GPU receive the same bitstream and perform identical decode computation, regardless of available resources.

**Research Question**: Can we design a neural video decoder whose compute cost scales dynamically at inference time — from a single trained model — to enable real-time adaptation across heterogeneous devices?

**Framing decision (locked)**: This is a **systems contribution**, not a compression contribution. We do not claim to improve rate-distortion performance over DCVC. We claim to add a capability DCVC does not have: a single bitstream, decoded by a single model, that can trade decode compute for reconstruction quality at runtime. The primary evidence for this claim is measured latency/FPS/power across real, heterogeneous hardware — not PSNR/VMAF gains.

**Relationship to prior art (must be resolved in Week 1, before implementation is finalized)**: Complexity-scalable video coding is a decades-old research area (scalable transform coding, complexity-scalable wavelet coders, layered/SHVC-style scalability). Anytime-inference networks (MSDNet, Skip-Net, and related early-exit architectures) are also established. This paper's contribution is not either idea in isolation — it is applying compute-scalable inference to a *neural* video decoder while preserving *bitstream compatibility* with an unmodified entropy model, and validating it as a systems mechanism across real devices. This distinction must be stated explicitly and defended against both literatures, not just the neural-codec literature.

**Target venue range**: systems-oriented venues (e.g., MMSys, ICIP, national flagship conferences), not a compression-focused top-tier venue — consistent with §11's explicit non-claims (no RD improvement, no novel entropy coding). The heterogeneous-device evidence (M6/§5.3) is the deciding factor between a workshop-tier result (RTX-4050-only, quality-vs-compute story) and a stronger systems-tier result (validated real-time adaptation across genuinely different hardware classes).

---

## 2. Architecture

### 2.1 Design Principle

- Keep **DCVC encoder and entropy model unchanged** (bitstream-compatible)
- Replace the fixed decoder with a **progressive residual decoder**
- All refinement stages share the same compressed latent representation
- No changes to the bitstream format
- **Target: Tier4/Tier1 FLOPs ratio in the range 1.3×–2.0×.** This is a hard design constraint, not a measurement to report after the fact — the refinement block is sized to hit it.

**Clarification carried over from v1.1**: The bitstream is entropy-decoded **fully** before refinement begins. Compute scaling applies only to reconstruction layers, not entropy decoding. All tiers produce identical bitstream interpretations.

**New clarification**: Report total decode FLOPs (entropy + reconstruction), not just reconstruction FLOPs, alongside the reconstruction-only ratio. If entropy decoding is a large fraction of total decode cost, the *effective* end-to-end compute variation will be smaller than 1.5×, and this must be disclosed rather than discovered by a reviewer.

### 2.2 Structure

```
Compressed Bitstream
        │
        ▼
┌───────────────────┐
│   DCVC Entropy    │  (unchanged — fully decoded before refinement)
│     Decoder       │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Base Decoder    │  Target: ~55–65% of total reconstruction FLOPs
│ (majority of DCVC │  (exact % confirmed by profiling, see 2.4)
│  contextual dec.) │
└─────────┬─────────┘
          │
          ▼
       Recon_1  ────────────────────► (Tier 1 output — identical to unmodified DCVC decode)
          │
          ▼
┌───────────────────┐
│  Refinement R_1   │  Target: ~12–15% of total reconstruction FLOPs
└─────────┬─────────┘
          │
          ▼
       Recon_2  ────────────────────► (Tier 2 output)
          │
          ▼
┌───────────────────┐
│  Refinement R_2   │  Target: ~12–15% of total reconstruction FLOPs
└─────────┬─────────┘
          │
          ▼
       Recon_3  ────────────────────► (Tier 3 output)
          │
          ▼
┌───────────────────┐
│  Refinement R_3   │  Target: ~12–15% of total reconstruction FLOPs
└─────────┬─────────┘
          │
          ▼
       Recon_4  ────────────────────► (Tier 4 output)
```

### 2.3 Refinement Block Architecture — DECIDED (was underspecified in v2.0)

**Problem with prior implementation**: the existing code (`progressive_decoder.py`) uses `proj_in(3→64) + 3×ResBlock(64) + proj_out(64→3)` per refinement stage, at roughly 4× base decoder FLOPs *per stage*. With 3 stages, Tier4 ≈ 13× Tier1. This fails the 1.5× target by nearly an order of magnitude.

**v2.0 review correctly flagged that "thin the block" is not a complete decision** — DCVC's decoder is 64-channel after `part1` and 3-channel after `part2`, and the two interpretations (refine in RGB space after `part2`, vs. refine in feature space between `part1`/`part2`) produce structurally different pipelines with different FLOPs, not just different block sizes.

**Decision: refine in feature space (between `part1` and `part2`), using depthwise-separable blocks.**

```python
class RefinementBlock(nn.Module):
    """Operates on part1's 64-channel feature map, inserted before part2.
    Depthwise-separable to keep per-stage FLOPs low at this channel width."""
    def __init__(self, channels=64):
        super().__init__()
        self.dw1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels, 1)
        self.dw2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.lrelu(self.pw1(self.dw1(x)))
        residual = self.pw2(self.dw2(h))
        return x + 0.1 * residual
```

**Rationale**:
- Refining in RGB space (after `part2`) forces every stage through a `proj_in(3→C)` / `proj_out(C→3)` round-trip that has nothing to do with actual refinement — this round-trip is a real contributor to why the existing code overshot its FLOPs budget. Feature-space refinement avoids it entirely.
- Feature-space refinement is also the more architecturally honest framing: the model is refining the *decode process*, not post-processing a rendered image.
- At 64 channels, a naive (non-separable) 2-conv block costs meaningfully more than the same block at 3 channels — depthwise-separable convolutions are what make feature-space refinement affordable within the 1.3×–2.0× target.
- Residual scale stays fixed at 0.1 for training stability, unchanged from v1.1/v2.0.

**This is a recommended default, not an unverified assumption presented as final.** M1 (§8) requires profiling this exact block at DCVC's real 64-channel width before training begins. If profiling shows it still overshoots the target band, the fallback is to reduce channel width further via a bottleneck (e.g., project to 32 channels internally, expand back to 64) rather than reverting to RGB-space refinement — reverting would reopen the round-trip cost problem this decision was made to avoid.

**Back-of-envelope FLOPs estimate (unmeasured — anchors M1's profiling, does not replace it):**

At 64 channels, 1080p feature map (1920×1080):
- `dw1` (3×3 depthwise): H×W × 64 × 9 ≈ 1.19 GFLOPs
- `pw1` (1×1 pointwise): H×W × 64 × 64 ≈ 1.33 GFLOPs
- `dw2`: ≈ 1.19 GFLOPs
- `pw2`: ≈ 1.33 GFLOPs
- **Per refinement block ≈ 5.0 GFLOPs**

If the base decoder (`part1` + `part2` reconstruction) is on the order of ~15–20 GFLOPs at 1080p (prior rough profiling, unconfirmed), then three refinement blocks at ~5 GFLOPs each add ~15 GFLOPs:
- Tier 1 ≈ 17.5 GFLOPs (base only)
- Tier 4 ≈ 32.5 GFLOPs (base + 3 refinement blocks)
- **Estimated ratio ≈ 1.86×** — inside the 1.3×–2.0× target band.

**M1's deliverable is not just "profile and confirm ≥1.3× and ≤2.0×" — it is "profiled ratio agrees with this 1.86× estimate within ±10%, or we stop and revisit the design."** A profiled number that lands inside the target band but far from 1.86× (e.g., 1.35× or 1.98×) means the estimate's assumptions were wrong somewhere, and that discrepancy should be understood before training, not shrugged off because the number happened to clear the bar.

**Sensitivity to base-decoder cost uncertainty**: the ~15–20 GFLOPs figure for `part1`+`part2` is itself a rough estimate, not a profiled number. Even at the upper end of that range (20 GFLOPs base), Tier 4 ≈ 20 + 15 = 35 GFLOPs, giving a ratio ≈ 1.75× — still inside the 1.3×–2.0× target band. At the lower end (15 GFLOPs), the ratio is ≈ 2.0×, at the edge of the band. This is stated explicitly so a reviewer who computes a different base-decoder cost doesn't conclude the estimate is wrong — the estimate is a range (≈1.75×–2.0×), not a single point, until M1 profiles the real number.

### 2.4 Tier Definitions

| Tier | Active Stages | Target cumulative FLOPs (of Tier 4) |
|---|---|---|
| 1 (Minimal) | Base decoder only | ~55–65% |
| 2 (Low) | + R_1 | ~70–80% |
| 3 (Medium) | + R_1, R_2 | ~85–90% |
| 4 (Full) | + R_1, R_2, R_3 | 100% |

Exact percentages are fixed only after profiling (Milestone 0). If profiling shows Option A cannot hit the 1.3×–2.0× target at DCVC's real channel width, escalate to Option B before touching training code.

---

## 3. Training Strategy

### 3.1 Deep Supervision with Early-Stage Priority

For each training batch, randomly sample an active refinement depth and compute loss on all intermediate outputs.

**Weight strategy (rationale stated once, unambiguously)**: earlier stages are executed in every forward pass regardless of sampled depth, so they are weighted higher to ensure Tier 1 is a strong standalone baseline, not merely a truncated version of a network optimized for Tier 4. This is the single rationale — the code's docstring ("descending stabilization, no rate term") and the plan's original rationale should be treated as the same statement, not two competing ones; if a future contributor edits the training script, this section is the source of truth.

```python
def forward_with_depth(x, active_depth):
    # active_depth in {1,2,3,4}; depth=1 -> base decoder only, no refinement.
    # range(active_depth - 1) matches progressive_decoder.py:184 — NOT range(active_depth).
    recon = base_decoder(x)
    outputs = [recon]

    for i in range(active_depth - 1):
        recon = recon + 0.1 * refinement_blocks[i](recon)
        outputs.append(recon)

    return outputs

weights = [1.0, 0.9, 0.8, 0.7]  # earlier stages weighted higher; always executed

for batch in dataloader:
    active_depth = random.choice([1, 2, 3, 4])
    outputs = forward_with_depth(bitstream_decoded, active_depth)

    w_eff = weights[:len(outputs)]  # invariant: len(outputs) == active_depth for all active_depth in {1,2,3,4}
    loss = 0.0
    for i, recon in enumerate(outputs):
        loss += w_eff[i] * rd_loss(recon, target)

    loss.backward()
```

**Invariant (must hold, and must be implemented, not just true in principle)**: `len(outputs) == active_depth` for every `active_depth ∈ {1,2,3,4}`. The weight list `[1.0, 0.9, 0.8, 0.7]` must be **sliced to `len(outputs)`**, not indexed unconditionally — at `active_depth=1`, only `weights[0]` applies, and using the full 4-element list without slicing would either crash or silently misweight the loss. `train_progressive.py:88`'s `DCVCProgressiveLoss` already does this correctly (`w_eff = self.weights[:len(recon_list)]`); this note exists so the plan's pseudocode can't be misread as requiring all four weights regardless of depth.

### 3.2 Training Curriculum (in **steps**, not epochs)

| Phase | Steps | Active Depth | Purpose |
|---|---|---|---|
| 0 (new) | ~5–10K | Depth = 4, fixed | **Ceiling check**: does full-depth model even approach DCVC-level PSNR on the frozen encoder/entropy latents? Gate before investing in Phases 1–3. |
| 1 | ~0–60K | Depth = 4, fixed | Establish full-depth baseline quality |
| 2 | ~60K–120K | Random [1,4] | Force early stages to learn useful standalone representations |
| 3 | ~120K–180K | Random [1,4] | Fine-tune with full training set |

**Phase 0 is the most important addition in this revision.** It answers, in a small fraction of the total compute budget, whether the architecture is viable at all before random-depth training (the most time-consuming and hyperparameter-sensitive phase) is attempted. If Phase 0 shows Tier 4 falling meaningfully short of stock DCVC PSNR, that is an architecture-capacity problem, not a training-schedule problem — stop and revisit Section 2.3 rather than proceeding to Phase 1.

**Hardware split (confirmed): training on Kaggle T4 (16GB), benchmarking on local RTX 4050 (6GB).** These are two different machines serving two different purposes, and the plan should not conflate them:

- **Training happens on Kaggle's T4.** 16GB VRAM is comfortable for this model size — batch size and crop resolution are not the binding constraint here. The binding constraint is **session limits**: Kaggle sessions have a runtime cap (historically ~9–12h) and a weekly GPU quota, and sessions can disconnect without warning. This means:
  - Checkpointing must be frequent and resumable — save every N steps, not just at phase boundaries, and verify `train_progressive.py` can resume mid-phase without restarting a phase from step 0.
  - Sequence the curriculum (§3.2 table) to fit inside session-length chunks where possible, so a disconnect loses minutes, not hours.
  - Track weekly quota usage against the milestone sequence (§8) — if Phase 0 + Phase 1 alone consume most of a week's quota, that changes how M2/M3 are paced, and should be flagged early rather than discovered mid-run.
- **Benchmarking (§5.2, §5.3) happens on the RTX 4050**, because that's the device a real user would decode on — a Kaggle instance is not a deployment target and its latency numbers would not support the paper's real-time claims. A checkpoint trained on the T4 is simply loaded and evaluated on the 4050; this is standard practice and needs no special handling beyond confirming the checkpoint loads correctly on different hardware/CUDA versions.
- Mixed precision (AMP) is still worth using during training for speed (T4 has good FP16 throughput), but it is now a performance optimization rather than a memory-survival requirement.

**Curriculum boundaries are checkpoints, not commitments.** The 60K/120K/180K figures in the table above are *suggested* step counts, not fixed milestones the schedule must hit exactly. Given Kaggle's session-length risk, the curriculum is explicitly **checkpoint-based, not step-based**:
- If Phase 1 shows convergence (loss/PSNR plateau) before 60K steps, advance to Phase 2 early rather than burning remaining budgeted steps.
- If a Kaggle session dies mid-phase (e.g., at step 82K, partway through Phase 2), resume Phase 2 from the last checkpoint at step 82K — do not roll back to the 60K phase boundary and restart Phase 2 from scratch.
- This must be reflected in `train_progressive.py`'s checkpoint/resume logic: the script needs to track which phase it's in and how many steps into that phase, not just a global step count, so a resume lands in the correct sampling regime (fixed depth-4 vs. random depth) rather than silently drifting into the wrong phase.

**Phase 0 / M2 go/no-go decision rule (pinned, not left to judgment when the numbers arrive)**: compare Phase 0's Tier 4 PSNR against stock DCVC PSNR, per UVG sequence, using M0.5's σ as the reference.
- **Advance to Phase 1** if Tier 4's mean PSNR falls within **±1σ** of stock DCVC across all 8 UVG sequences.
- **Architecture-capacity finding, revisit §2.3** if **1 or more sequences** fall outside **±2σ** — this is not a training-schedule problem to fix by running more steps; it means the base decoder or refinement capacity needs to change before Phase 1 is attempted.
- Results between ±1σ and ±2σ: proceed to Phase 1 but flag explicitly in project notes as a borderline case worth re-checking once Phase 1 has more training signal, rather than silently treated as a clean pass.

### 3.3 Loss Function

```
L_total = λ · D(recon, target) + R(bitstream)
```

No compute-cost penalty term. Compute scaling emerges from random-depth sampling forcing early-stage representations to be independently strong, not from an explicit FLOPs regularizer.

---

## 4. Evaluation Metrics

### 4.1 Core Metrics (Milestone 1 — PSNR only)

| Metric | Description |
|---|---|
| PSNR | Peak signal-to-noise ratio, per tier |
| Bitrate (bps) | Bits per second |
| Decode latency (ms/frame) | Time per frame, per tier |
| FPS | Frames decoded per second, per tier |

**VMAF is explicitly deferred.** No VMAF tooling exists yet in the codebase, and installing/validating it before a trained checkpoint exists is premature effort. VMAF is added in Milestone 3 (Section 8), once Phase 0/1 confirm the architecture is viable.

**Baseline variance (new, feeds §6's "meaningful" definition)**: before any training, run stock unmodified DCVC decode on all UVG sequences and record per-sequence PSNR mean and standard deviation (σ). This requires no training — only the existing checkpoint — and is cheap enough to run in the first day of work (folded into M0.5, §8). Without this number, "Tier 4 within noise of stock DCVC" in §6 has no concrete threshold to check against.

### 4.2 Compute Scaling Metrics

| Metric | Formula |
|---|---|
| FLOPs variation (reconstruction-only) | FLOPs_Tier4 / FLOPs_Tier1 |
| FLOPs variation (total decode, incl. entropy) | (FLOPs_Tier4 + FLOPs_entropy) / (FLOPs_Tier1 + FLOPs_entropy) |
| Marginal PSNR/FLOPs | (PSNR_i − PSNR_{i−1}) / (FLOPs_i − FLOPs_{i−1}) |
| Real-time feasibility | True if FPS ≥ video framerate, per device |

### 4.3 Required Baseline Comparison (new, was previously only an argument, not an experiment)

Train **four independent fixed-size decoders**, one per tier's compute budget, with no shared parameters. Compare against the single progressive model at each tier. This directly tests the claim in Section 10 ("why not just train separate models?") empirically instead of only rhetorically. If the progressive model is meaningfully worse than independently-sized models at any tier, that is a real limitation and must be reported as such.

**FLOP-matching protocol — committed, not left open**: width-scale `part2` only; hold `part1` fixed across all four baselines. Concretely, train four decoders with `part2` channel widths (e.g., 64/56/48/32, exact values set once M1 profiling gives real per-width FLOPs) chosen so each baseline's total FLOPs matches the corresponding progressive-model tier's FLOPs budget within a small tolerance (~5%). This is chosen over depth-scaling because it isolates the same variable the progressive model varies (capacity in the refinement/reconstruction path), keeping the comparison interpretable as "single adaptive model vs. four purpose-built models of matched capacity" rather than conflating capacity and depth as separate confounds.

---

## 5. Benchmarking Protocol

### 5.1 Test Dataset

- Primary: UVG dataset (8 sequences, 1080p, various motion levels)
- Secondary: HEVC Class B (if time permits)

### 5.2 Per-Sequence Measurement Protocol

For each sequence, at each tier:
1. Encode once with the DCVC encoder (single encoding — all tiers share the same bitstream)
2. Decode at all 4 tiers
3. Record: bitrate, PSNR, decode latency (10 runs, report **mean and p95**, not mean alone — real-time claims live or die on tail latency), FPS, GPU/device utilization, power draw

### 5.3 Heterogeneous Device Benchmarking (new — primary systems evidence)

The research question explicitly claims "real-time adaptation across heterogeneous devices." This must be measured on more than one device class, or the claim is untested. **Confirmed hardware for this study**:

1. **RTX 4050 (6GB, laptop)** — the "desktop-class" data point. Note this is itself a modest GPU, not a high-end lab card; frame results accordingly rather than implying a stronger reference platform than actually used.
2. **Realme Narzo 5G phone** — the low-power data point.

Even two real data points converts "enables adaptation across heterogeneous devices" from an assertion into a demonstrated result, and remains the single highest-leverage addition available for raising this paper's venue ceiling. But the phone data point carries real engineering risk that must be planned for, not assumed away:

- **Export pipeline required**: PyTorch → ONNX → a mobile runtime (TFLite, NCNN, or ONNX Runtime Mobile). This is new engineering work, not a checkbox — budget explicit time for it (see M6, §8) and expect at least one round of fixing unsupported ops (depthwise-separable convs from §2.3 are generally mobile-runtime-friendly, which is a secondary reason to prefer that design).
- **GPU/NPU delegate support on a budget-tier SoC is not guaranteed.** If hardware acceleration isn't available or reliable, benchmark CPU-only inference and report it as such — do not silently assume GPU delegate execution without confirming it.
- **PixelShuffle/subpixel-conv in `part1` is on the Tier-1 export critical path, independent of the refinement-block design.** The depthwise-separable refinement blocks (§2.3) being mobile-runtime-friendly says nothing about `part1`'s subpixel upsampling — some mobile runtimes (notably older TFLite versions) have had known issues exporting or accelerating PixelShuffle. Since Tier 1 (base decoder only, no refinement) must run on the phone for the heterogeneous-device plot to include all four tiers, this op needs to be verified exportable in the M6 spike (below), not assumed safe because the *refinement* blocks are known to be portable.
- **The phone may not hit real-time (30fps) at any tier, including Tier 1.** This is an acceptable, reportable outcome — "a budget device cannot sustain real-time decode even at minimum compute" is itself a systems finding that supports the paper's motivation (heterogeneous devices genuinely need this kind of adaptivity). It is not a failure condition; do not gate §6's success criteria on the phone hitting real-time. The desktop-class RTX 4050 remains the real-time floor requirement (§6); the phone's role is to characterize the low end, whatever that turns out to be.

### 5.4 Required Plots

| # | Plot | Axes | Interpretation |
|---|---|---|---|
| 1 | Rate-Distortion | PSNR vs Bitrate | Standard RD curve per tier |
| 2 | Quality vs Latency | PSNR vs ms/frame, per device | **Primary figure** — directly maps to the real-time systems claim |
| 3 | Quality vs Compute | PSNR vs FLOPs | Compute scaling effectiveness |
| 4 | Marginal Efficiency | Stage vs ΔPSNR/ΔFLOPs | Diminishing returns |
| 5 | Real-time Feasibility | FPS vs PSNR, per device | With 30fps/60fps threshold lines, one series per device class |

---

## 6. Success Criteria

| Criterion | Threshold | Why |
|---|---|---|
| FLOPs variation — **gating metric** | 1.3×–2.0× between Tier 1 and Tier 4, **reconstruction FLOPs only** | Decoder compute actually scales in a controllable, "modest increment" way — this is the paper's central claim |
| FLOPs variation — reported, not gating | End-to-end (entropy + reconstruction) ratio | Discloses that the *effective* device-level compute swing is smaller than the reconstruction-only number; must be reported so a reviewer doesn't have to compute it themselves |
| PSNR at Tier 4 vs. stock DCVC | Within ±σ of stock DCVC = clean win. **>0.3 dB below** stock DCVC = a finding to report, not a silent shortfall. **σ is pinned to M0.5's measurement only**: per-sequence PSNR standard deviation from the *unmodified* `model_dcvc_quality_3_psnr.pth` checkpoint decoding UVG, computed once before any training. σ from any training run, or from the progressive model itself, is a different, unrelated number and must not be substituted here. | Defines "meaningful" concretely instead of leaving it to interpretation; Tier 4 is not expected to *beat* DCVC given the frozen encoder/entropy model |
| Monotonicity | No tier produces worse PSNR than the previous tier | Each stage adds value |
| Real-time floor | Tier 1 achieves ≥60 FPS 1080p on the **RTX 4050** | Practical for the lowest compute budget, on the actual hardware available — not gated on the phone (§5.3) |
| Marginal efficiency | Declining ΔPSNR/ΔFLOPs across tiers | Natural diminishing returns |
| Single model | All tiers served by one model file | Deployment/storage advantage vs. multiple checkpoints — validated against the fixed-decoder baseline in §4.3, not asserted |
| Heterogeneous characterization | Results reported (not necessarily "passing") on both RTX 4050 and Realme Narzo 5G | Supports the research question as stated; the phone may legitimately fail real-time and that is still a valid, reportable result |

If all criteria are met: proceed to a streaming-policy / content-adaptive tiering extension.
If marginal: publish with limitation acknowledgment, systems framing intact.
If the FLOPs ratio cannot be brought under ~2× without unacceptable PSNR loss: this is an architecture-capacity finding, not a failure — report it as a design-tradeoff result.

---

## 7. Ablations (reduced from 3 to 2; third listed as stretch)

**Ablation 1 — Residual Scaling Factor**: train with scale = 1.0 instead of 0.1. Show training instability or early-tier degradation, demonstrating the 0.1 scale is a meaningful design choice, not an arbitrary constant.

**Ablation 2 — Fixed vs. Random Depth Training**: show that random-depth sampling is necessary for multi-tier performance.

**Protocol must be precise, or this ablation is vacuous.** The refinement blocks use zero-initialized output projections (§2.3-adjacent property, carried from the original code), meaning a model trained *only* at fixed depth 4 and then naively truncated at inference (skip blocks 2 and 3) will not have been forced to make blocks 1–2 individually meaningful — depending on how training unfolds, truncation could degenerate toward simply re-exposing the base decoder's output, which would make random-depth training "win" trivially rather than for the reason we want to demonstrate.

Correct protocol: (1) train a fixed-depth-4 model for a **concrete, pinned step count**: use whichever comes first — the main run's actual total steps once Phases 1–3 complete (checkpoint-based, per §3.2, so this number isn't known until then), **or a hard cap of 180,000 steps** (the original v1.1 sum of 60K+60K+60K), applying the same loss/PSNR plateau definition used to decide early-advance in §3.2. Do not leave this open-ended — pin the number once the main run's actual total is known, and use 180K as the planning default until then. (2) At inference, evaluate it at depth 2 by skipping refinement blocks 2 and 3 (not by retraining); (3) compare its depth-2 PSNR against the random-depth-trained model's depth-2 PSNR, same UVG sequences. The comparison is meaningful only if both models are evaluated at the same depth using the same held-out data — state this explicitly in the paper's ablation section so a reviewer doesn't have to infer it.

**Stretch Ablation 3 — Weight Strategy Comparison** (early-priority vs. late-priority vs. uniform).

**Go/no-go criterion (relative, not a fixed week count — exact remaining Kaggle quota isn't known in advance)**: run Ablation 3 only if, after Ablations 1–2 complete, the remaining Kaggle weekly quota is sufficient to run one additional full training curriculum (Phases 1–3, or their eventual pinned step-count equivalent) without delaying M8/M9/M10. If quota margin is unclear at that point, default to **not** running it and reporting it as future work — an unrun stretch ablation costs nothing; a half-finished one that gets cut mid-run costs a wasted week and an awkward gap in the results section.

---

## 8. Milestone Sequence (dependency-ordered, not calendar-ordered)

| Milestone | Deliverable | Blocks |
|---|---|---|
| M0 | Repo cleanup: unify `DCVC-Scalable/` and `DCVC-repo/DCVC-family/DCVC/`, fix `eval_uvg.py` paths, remove/mark inert legacy path | Everything downstream |
| M0.5 | Stock DCVC baseline: run unmodified decoder on all UVG sequences, record per-sequence PSNR mean/σ (no training required) | Defines "meaningful" threshold in §6 |
| M1 | **Commit** to feature-space/depthwise-separable refinement block (§2.3) as the implementation target; profile DCVC's actual 64-channel decoder at the injection point; confirm the committed design hits 1.3×–2.0× at real channel width; implement `profile_decoder.py`. If it doesn't hit target, fall back to internal channel bottleneck (§2.3), not a redesign from scratch. | Any training |
| M2 | Phase 0 ceiling check (fixed depth-4, ~5–10K steps, on Kaggle T4, checkpointed for resumability): confirm Tier 4 approaches stock DCVC PSNR (within M0.5's σ) on frozen latents | Phases 1–3 |
| M3 | Full curriculum (Phases 1–3) on Kaggle T4, with disconnect-safe checkpointing; produce first checkpoint, then transfer to RTX 4050 for evaluation | Benchmarking |
| M4 | PSNR-only benchmarking on UVG, RTX 4050, all 4 tiers; generate Plots 1, 3, 4 | Baseline comparison, device benchmarking |
| M5 | Fixed-decoder baseline comparison (§4.3) | Paper's rebuttal-to-obvious-question section |
| M5.5 (new) | **Mobile export feasibility spike (~1 day)**: export a trivial model (or just `part1` alone) to ONNX → mobile runtime, confirm it runs *at all* on the Realme Narzo 5G, and specifically confirm PixelShuffle/subpixel-conv (§5.3) exports and executes. This spike's outcome decides M6's status below. | Determines whether M6 is required or downgraded to stretch |
| M6 | **Status: required if M5.5 succeeds cleanly; downgrade to stretch (report RTX 4050-only heterogeneity data as a limitation) if M5.5 reveals export blockers that would cost more than ~1 additional week to resolve.** If proceeding: full export pipeline validated on Realme Narzo 5G; heterogeneous device benchmarking (§5.3) on RTX 4050 + phone; generate Plots 2, 5 | Systems claim validation |
| M7 | VMAF pipeline added; re-run Plot 1/2 with VMAF as secondary metric | Final evaluation completeness |
| M8 | Ablations 1–2, using precise truncation protocol (§7) (+3 if time permits) | Paper completeness |
| M9 | Complexity-scalable coding literature review finalized into Related Work | Paper submission readiness |
| M10 | Draft, figures, supplemental | Submission |

---

## 9. Paper Structure

**Working Title**: *Compute-Scalable Neural Video Decoding: A Progressive Refinement Approach for Heterogeneous Device Streaming*

**Abstract**
- Problem: neural video codecs lack compute scalability
- Approach: progressive residual decoder, controllable runtime depth, single model, bitstream-compatible
- Results: [FLOPs ratio] compute variation with [PSNR range] quality range; validated on [N] device classes
- Implication: enables compute-aware streaming without re-encoding or multiple deployed models

**Introduction**
- Neural codecs assume fixed decode compute
- Heterogeneous devices need adaptive decoding
- Existing codecs expose bitrate scalability but not compute scalability
- We propose a progressive residual decoder validated on real heterogeneous hardware

**Related Work**
- DCVC and neural video codecs
- **Complexity-scalable classical video coding** (new — required, not optional)
- Anytime/early-exit inference (MSDNet, Skip-Net)
- Adaptive/compute-aware video streaming

**Architecture** — includes exact FLOP breakdown table, profiled not estimated

**Training** — deep supervision, Phase 0 ceiling check as a methodological contribution (shows the field how to validate architecture capacity before expensive random-depth training)

**Experiments**
- Setup, dataset, hardware (multiple device classes)
- RD curves, Quality-vs-Latency (primary figure, per device)
- Compute scaling results, marginal efficiency
- Real-time feasibility, per device class
- Comparison against fixed-decoder baseline
- Ablations
- VMAF as secondary validation

**Conclusion**
- Summary
- Limitations: single encoder, fixed entropy model, budget-based (not quality-based) exit
- Future work: content-adaptive or quality-based tier selection, spatial (region-level) adaptivity

---

## 10. Anticipated Reviewer Questions

**Q: "Why not just train separate models per tier?"**
A: Empirically tested in §4.3, not just argued. If the progressive model matches independently-sized models within [X] PSNR while requiring one deployed model instead of four, that's the result. If it doesn't, that's reported as a limitation.

**Q: "This looks like early-exit networks (MSDNet) applied to video decoding — what's new?"**
A: The mechanism is adjacent to known anytime-inference techniques; the contribution is (1) bitstream-compatible integration with an unmodified neural codec's entropy model, and (2) validated real-time adaptation across measured heterogeneous hardware — not just simulated FLOPs. We do not claim novelty in the exit mechanism itself.

**Q: "How is this different from complexity-scalable classical coding (SHVC, scalable transform coding)?"**
A: [To be written after M9 literature review — do not leave this as a placeholder in the actual paper.]

**Q: "Your compute variation is modest — why not scale further?"**
A: Aggressive scaling requires a much smaller base decoder, risking base-tier quality. We prioritize a monotonic, real-time-capable Tier 1 over a larger nominal ratio.

---

## 11. What This Paper Does Not Claim

- Novel entropy coding
- Better rate-distortion performance than state-of-the-art codecs
- Quality-based or content-adaptive learned exit mechanism (explicitly future work, not claimed here)
- Joint bandwidth–compute optimization
- Novelty in the anytime-inference mechanism itself

---

## 12. Key References

1. Li et al., "DCVC: Deep Contextual Video Compression," NeurIPS 2021
2. Li et al., "DCVC-RT: Real-Time Neural Video Coding," CVPR 2025
3. Huang et al., "Multi-Scale Dense Networks for Resource Efficient Image Classification" (MSDNet)
4. Wang et al., "Skip-Net: Learning to Skip State Updates in RNNs"
5. **[Required, pending M9]** Complexity-scalable / SHVC-style classical scalable video coding — minimum 2–3 representative papers, to be added before Related Work is finalized. Do not submit without this section populated.

---

## Open Items Before Next Advisor Meeting

1. ~~Confirm compute budget/environment~~ — **Resolved**: training on Kaggle T4, benchmarking on RTX 4050 + Realme Narzo 5G.
2. Confirm Kaggle weekly GPU quota in practice (varies) and whether it's sufficient to complete Phases 0–3 (§3.2) without excessive week-to-week stalling — if not, this affects M2/M3 pacing and should be flagged to the advisor as a scheduling risk, not silently absorbed.
3. ~~Confirm the mobile export pipeline is technically feasible before committing~~ — **Resolved structurally**: M5.5 is now a scheduled spike whose outcome gates M6's required-vs-stretch status (§8). Still to be *executed*: the spike itself hasn't run yet, so M6's actual status is unknown until M5.5 completes.
4. Confirm advisor's PSNR expectation is understood as: Tier 1 = DCVC baseline by construction (no PSNR gain expected there); any Tier 4 gain over DCVC, if it appears, is a secondary bonus, not the paper's central claim.