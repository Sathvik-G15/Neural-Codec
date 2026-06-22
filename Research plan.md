Incorporating all fixes. Here is the corrected version:
# Compute-Scalable Neural Video Decoding for Real-Time Adaptive Streaming

**Version**: 1.1  
**Date**: June 2026  
**Status**: Advisor-Ready

---

## 1. Problem Statement

Modern neural video codecs (e.g., DCVC) assume **fixed decode compute**. This creates a fundamental mismatch: a mobile device and a high-end GPU both receive the same bitstream and perform identical decode computation — regardless of available resources.

**Research Question**: Can we design a neural video decoder whose compute cost scales dynamically to enable real-time adaptation across heterogeneous devices?

**Key Insight**: Existing neural codecs expose bitrate scalability but **not compute scalability**. This paper addresses that gap.

---

## 2. Architecture

### 2.1 Design Principle

- Keep **DCVC encoder and entropy model unchanged** (bitstream-compatible)
- Replace the fixed decoder with a **progressive residual decoder**
- All refinement stages share the same compressed latent representation
- No changes to the bitstream format

**Important clarification**: The bitstream is entropy-decoded **fully** before refinement begins. Compute scaling applies only to reconstruction layers, not entropy decoding. This ensures all tiers produce identical bitstream interpretations.

### 2.2 Structure

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
│   Base Decoder    │  XX% of reconstruction FLOPs
│ (majority of DCVC │  (exact % measured in Week 1)
│  contextual dec.) │
└─────────┬─────────┘
          │
          ▼
       Recon1  ────────────────────► (Tier 1 output)
          │
          ▼
┌───────────────────┐
│  Refinement R1   │  XX% of reconstruction FLOPs
│ (lightweight      │  (exact % measured in Week 1)
│ residual CNN)     │
└─────────┬─────────┘
          │
          ▼
       Recon_2  ────────────────────► (Tier 2 output)
          │
          ▼
┌───────────────────┐
│  Refinement R_2   │  XX% of reconstruction FLOPs
└─────────┬─────────┘
          │
          ▼
       Recon3  ────────────────────► (Tier 3 output)
          │
          ▼
┌───────────────────┐
│  Refinement R3   │  XX% of reconstruction FLOPs
└─────────┬─────────┘
          │
          ▼
       Recon_4  ────────────────────► (Tier 4 output)

### 2.3 Refinement Block Architecture

Each refinement block `R_i` is a small residual module:

```python
class RefinementBlock(nn.Module):
    def __init__(self, channels):
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        residual = self.conv2(self.lrelu(self.conv1(x)))
        return x + 0.1 * residual  # Small residual scale for training stability
- 2 conv layers per block
- 3×3 kernels
- Same channel count as base decoder
- Additive residual with scaling factor 0.1 to prevent training divergence
2.4 Tier Definitions (Exact FLOPs Required in Week 1)
Tier	Active Stages
1 (Minimal)	Base decoder only
2 (Low)	+ R_1
3 (Medium)	+ R_1, R_2
4 (Full)	+ R_1, R_2, R_3
Note: Exact percentages will be measured during Week 1 implementation. Tier thresholds will be set to produce approximately equal FLOPs increments (e.g., 25% / 50% / 75% / 100%).
3. Training Strategy
3.1 Deep Supervision with Early-Stage Priority
For each training batch, randomly sample active refinement depth and compute loss on all intermediate outputs.
Weight strategy: Earlier stages receive higher weights because they are always executed. Final output still receives meaningful weight to ensure full-tier quality.
def forward_with_depth(x, active_depth):
    recon = base_decoder(x)
    outputs = [recon]
    
    for i in range(active_depth):
        recon = recon + 0.1 * refinement_blocks[i](recon)
        outputs.append(recon)
    
    return outputs

# Training with early-stage priority weights:
weights = [1.0, 0.9, 0.8, 0.7]  # Earlier stages weighted higher

for batch in dataloader:
    active_depth = random.choice([1, 2, 3, 4])
    outputs = forward_with_depth(bitstream_decoded, active_depth)
    
    loss = 0.0
    for i, recon in enumerate(outputs):
        loss += weights[i] * rd_loss(recon, target)
    
    loss.backward()
Rationale: Higher early weights force base decoder + early refinement to carry more reconstruction burden. This ensures Tier 1 is already a strong baseline. Final-tier performance remains competitive because output 4 still receives weight 0.7.
3.2 Training Curriculum
Phase	Epochs	Active Depth	Purpose
1	0–200K	4 fixed	Train full depth, establish baseline
2	200K–400K	Random [1,4]	Force early stages to learn useful representations
3	400K–600K	Random [1,4]	Fine-tune with full training set
3.3 Loss Function
L_total = λ * D(recon, target) + R(bitstream)
No βC penalty. Compute scaling emerges from random depth sampling forcing early-stage representations to be strong.
4. Evaluation Metrics
4.1 Core Metrics
Metric	Description
VMAF	Video Multimethod Assessment Fusion
Bitrate (bps)	Bits per second
Decode latency (ms/frame)	Time per frame
FPS	Frames decoded per second
GPU power (W)	Power draw during decode
GPU utilization (%)	SM occupancy during decode
4.2 Compute Scaling Metrics
Metric	Formula
FLOPs variation	FLOPs_Tier4 / FLOPs_Tier1
Marginal VMAF/FLOPs	(VMAF_i - VMAF_{i-1}) / (FLOPs_i - FLOPs_{i-1})
Real-time feasibility	True if FPS >= video framerate
4.3 Marginal Efficiency Table
After benchmarking, construct this table:
Transition	ΔVMAF	ΔFLOPs
Tier 1 → 2	+x.x	+y%
Tier 2 → 3	+x.x	+y%
Tier 3 → 4	+x.x	+y%
This informs the streaming policy on when to stop decoding.
5. Benchmarking Protocol
5.1 Test Dataset
- Primary: UVG dataset (8 sequences, 1080p, various motion levels)
- Secondary: HEVC Class B (if time permits)
5.2 Per-Sequence Measurement Protocol
For each sequence at each tier:
1. Encode once with DCVC encoder (single encoding — all tiers share same bitstream)
2. Decode at all 4 tiers
3. Record: bitrate, VMAF, decode latency (10 runs averaged), FPS, GPU util, power
5.3 Required Plots
#	Plot	Axis	Interpretation
1	Rate-Distortion	VMAF vs Bitrate	Standard RD curve per tier
2	Quality vs Latency	VMAF vs ms/frame	Core systems result — directly maps to real-time constraint
3	Quality vs Compute	VMAF vs FLOPs	Compute scaling effectiveness
4	Marginal Efficiency	Stage vs ΔVMAF/ΔFLOPs	Diminishing returns visualization
5	Real-time Feasibility	FPS vs VMAF	With 30fps and 60fps threshold lines
Plot 2 is the primary figure for the systems community. It shows the latency budget directly.
6. Success Criteria
Criterion	Threshold	Why
FLOPs variation	≥ 1.5× between Tier 1 and Tier 4	Decoder compute actually scales
VMAF range	≥ 3–5 VMAF points Tier 1 → 4	Quality improves meaningfully with compute
Monotonicity	No tier produces worse VMAF than previous	Each stage adds value
Real-time floor	Tier 1 achieves ≥60 FPS 1080p on mid GPU	Practical for low-power devices
Marginal efficiency	Declining ΔVMAF/ΔFLOPs across tiers	Natural diminishing returns (validates approach)
Single model	All tiers served by one model file	Storage advantage vs. multiple checkpoints
If all criteria met: Proceed to streaming policy design  
If marginal: Publish with limitation acknowledgment  
If failed: Pivot to simpler approach or different architecture
7. Ablations
Ablation 1: Residual Scaling Factor
Train identical model with scale=1.0 instead of 0.1.
Show that:
- Training becomes unstable, OR
- Early-tier performance degrades significantly
This demonstrates that the 0.1 residual scaling is a meaningful design choice.
Ablation 2: Weight Strategy Comparison
Compare:
- Early-stage priority: 1.0, 0.9, 0.8, 0.7
- Late-stage priority: 0.7, 0.8, 0.9, 1.0
- Uniform: 1.0, 1.0, 1.0, 1.0
Show impact on Tier 1 and Tier 4 quality.
Ablation 3: Fixed vs Random Depth Training
Show that random depth sampling is necessary for multi-tier performance.
8. Timeline (10 Weeks)
Week 1–2: Setup & Baseline
- Fork DCVC repository
- Inspect existing decoder architecture (layer counts, channel dimensions)
- Compute exact per-layer FLOPs — present table in paper
- Implement base decoder replacement
- Verify training pipeline runs with original DCVC decoder
- Validate bitstream compatibility
Week 3–4: Single Refinement Block
- Implement one refinement block (R_1)
- Train with fixed 2-stage depth
- Measure Tier 1 vs Tier 2 quality and compute
- Verify FLOP scaling follows prediction
Week 5–6: Full 4-Stage Progressive Decoder
- Add remaining refinement blocks (R2, R3)
- Implement random-depth training sampler
- Train with weighted multi-stage loss
- Verify deep supervision works
Week 7–8: Benchmarking
- Run full evaluation on UVG dataset
- Measure all 4 tiers per sequence
- Compute marginal efficiency table
- Generate all 5 required plots
- Verify success criteria
Week 9–10: Writing & Integration
- Draft paper
- Add streaming simulation (simple tier-selection based on device capability)
- Compare against DCVC baseline
- Prepare supplemental materials
9. Paper Structure
Title (Example)
Progressive Residual Decoding for Compute-Adaptive Neural Video Streaming
Abstract
- Problem: neural video codecs lack compute scalability
- Approach: progressive residual decoder with controllable depth
- Results: 1.5× compute variation with 3–5 VMAF quality range from single model
- Implication: enables compute-aware streaming without re-encoding
Introduction
- Neural codecs assume fixed decode compute (1 paragraph)
- Heterogeneous devices need adaptive decoding (1 paragraph)
- Key contrast: existing codecs expose bitrate scalability but not compute scalability
- We propose progressive residual decoder (1 paragraph)
- Experiments show measurable quality–compute tradeoff (1 paragraph)
Related Work
- DCVC and neural video codecs
- Adaptive computation in neural networks (ACT, Skip-Net, MSDNet)
- Compute-aware video streaming (brief)
Architecture
- DCVC base (unchanged — encoder, entropy, bitstream)
- Progressive residual decoder design
- Refinement block details
- Exact FLOP breakdown table (per-layer, per-tier)
Training
- Deep supervision with early-stage priority weights
- Random depth sampling curriculum
- Training details
Experiments
- Setup (dataset, hardware, measurement protocol)
- Rate-distortion curves
- Quality vs Latency plot (primary figure)
- Compute scaling results (FLOPs, latency, FPS)
- Marginal efficiency analysis
- Real-time feasibility
- Ablations (residual scale, weight strategy, fixed vs random depth)
- Comparison with DCVC baseline
Conclusion
- Summary of contributions
- Limitations (single encoder, fixed entropy model, budget-based exit)
- Future work (quality-based exit, dynamic compute allocation)
10. Answer to Expected Reviewer Attacks
Q: "Why not just train separate models for each compute tier?"
A: 
1. Storage: One model vs. four reduces deployment overhead for edge devices
2. Streaming flexibility: A single encoded bitstream serves multiple device tiers without re-encoding
3. Joint optimization: Deep supervision shares training signal across tiers, improving early-stage representations
Q: "Your compute variation (1.5×) is modest. Why not design for larger scaling?"
A: We prioritize training stability and quality. Aggressive compute scaling (e.g., 4×) would require a much smaller base decoder, risking base-tier quality degradation. The 1.5× range is sufficient for practical real-time adaptation and maintains quality.
Q: "This is just deep supervision. What's novel?"
A: The novelty is applying progressive refinement to the specific problem of decoder compute scaling in neural video codecs. Deep supervision is a training technique; our contribution is the architecture that enables runtime compute adaptation from a single model.
11. What This Paper Does NOT Claim
- Novel entropy coding
- Better RD performance than state-of-art codecs
- Quality-based learned exit mechanism
- Joint bandwidth–compute optimization
12. Key References
1. Li et al., "DCVC: Deep Contextual Video Compression," NeurIPS 2021
2. Li et al., "DCVC-RT: Real-Time Neural Video Coding," CVPR 2025
3. Wang et al., "Skip-Net: Learning to Skip State Updates in RNNs"
4. Huang et al., "Multi-Scale Dense Networks for Resource Efficient Image Classification"
Exact FLOPs percentages will be determined by profiling the DCVC decoder during Week 1. All tier thresholds will be set to approximately equal increments pending those measurements.

---

**Changes incorporated from feedback:**

| Issue | Fix Applied |
|-------|-------------|
| 4× FLOPs unrealistic | Revised to **≥ 1.5×** with explanation |
| Bitstream decode ambiguity | Added clarification: entropy decoded fully before refinement |
| Weight strategy inverted | Changed to **early-stage priority** [1.0, 0.9, 0.8, 0.7] with rationale |
| Fuzzy FLOPs (~70-80%) | Changed to "exact % measured in Week 1" |
| Missing latency plot | Added **Quality vs Latency** as primary systems figure |
| Missing ablation | Added residual scaling ablation |

This is advisor-ready. Copy and take to your meeting.