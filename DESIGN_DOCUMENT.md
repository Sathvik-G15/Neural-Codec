# Budget-Constrained Neural Decoder Scheduling for Learned Video Compression

## A Research and System Design Specification

---

# Part I — Research Foundation

## 1. Motivation

Modern video decoding spans heterogeneous client devices with widely varying compute budgets. The same encoded bitstream may be decoded on a battery-constrained mobile phone, a mid-range laptop, a desktop workstation, or an edge appliance — each tier providing a different envelope of available FLOPs, memory, and per-frame latency for the same reconstruction task. Learned video codecs have favorably reshaped rate-distortion performance, but they impose a structural assumption inherited from their classical counterparts: the decoder must execute the full reconstruction pipeline in order to reconstruct the frame. The full pipeline is run on every device, on every frame, whether the device can afford it or whether the content demands it. The result is a misalignment between deployment reality and the compute footprint of the decoder itself.

The mismatch is twofold. Constrained devices are forced to execute reconstruction work whose cost was never tailored to their available envelope; capable devices, conversely, do not differentiate their compute expenditure across content. A static talking-head frame and a fast-motion sports frame are processed with the same decoder compute, even though the content difficulty, the residual signal, and the resulting reconstruction sensitivity differ substantially. Adaptive mechanisms in today's learned codecs operate almost exclusively on the encoder side — through learned bitrate selection, frame-level quality-mode switching, or quantization adaptation — while the decoder itself behaves as a fixed computational object. The decoder does not know the device it runs on. It does not know the content it is reconstructing. It executes its full reconstruction graph regardless.

This problem becomes increasingly important as learned video codecs transition from research prototypes to deployment across increasingly heterogeneous client hardware. It suggests a reconsideration of the decoder's design. Rather than treating the decoder as a fixed pipeline that all devices must run clumsily, the decoder should adapt its compute allocation to the dual constraints imposed by deployment: the available compute budget on the host device, and the intrinsic difficulty of the frame currently being reconstructed. The central question this work asks is thus:

> *Given a runtime compute budget and the current decoder state, how should a neural video decoder allocate computation across its refinement stages to maximize reconstruction quality?*

> **Contribution framing.** This work makes two separable claims. First, the *budget-adaptive quality frontier*: the trained decoder produces K+1 operating points on the (compute, quality) curve, any of which can be selected at runtime via a budget parameter. This claim is always valid regardless of content variation. Second, the *content-adaptive scheduling*: the policy network learns to select different depths for different frames, providing additional quality gain on top of the budget-adaptive frontier. The content-adaptive claim is gated on empirical validation: if feature-stage correlation |r| < 0.3 across the training set, the system is classified as "budget-adaptive only" — a weaker but still publishable result. See §6.4 for the full two-layer framing and the conditions under which the content-adaptive component is expected to activate.

The motivation is not anchored to any specific application. It applies equally to video-on-demand streaming, live conferencing, cloud-to-edge adaptive decoding, and edge-class inference. What unites them is the underlying phenomenon — heterogeneous devices, varying per-frame content — and the same architectural question arises across all of them.

---

## 2. Research Gap

The literature has produced a rich set of complementary approaches that each touch portions of this question without jointly addressing it.

**Adaptive inference.** Early-exit networks, slimmable networks, and conditional-computation methods demonstrate per-input modulation of compute in classification and recognition tasks. They establish that neural networks need not consume the same FLOPs for every input. They do not, however, address the specific structure of learned video decoding, where refinement operates across a sequential decoding graph and where the bitstream is itself already compressed under a coder.

**Efficient learned video codecs.** Compact-decoder designs, slim architectures, mixed-precision decoders, reduced-channel-count codecs, and knowledge-distilled codecs target the fixed-compute quality-compression frontier. These methods accept a single compute footprint and compress its quality-efficiency frontier. They do not adaptively allocate that footprint across hosts or frames.

**Content-adaptive coding.** Adaptive quantization, learned bitrate ladders, and content-conditional frame-level mode selection modulate the bitstream in response to content. The decoder, however, remains fixed. The adaptation runs on the encoder/orchestration side, not on the decoder execution side.

**Dynamic neural networks.** Conditional computation, mixture-of-experts, and input-dependent routing demonstrate per-input graph selection for deep networks. They generally assume a backbone whose compute at full execution is significantly larger than the budgeted subset, and they target default discriminative tasks — not the sequential, stateful reconstruction structure of video decoding with its latent, context, and motion couplings.

**Constraint-aware inference.** Budget-constrained neural network execution has been studied under energy, latency, and FLOPs budgets. These formulations typically target inference pipelines with homogeneous operations, single-network policies, and discriminative tasks — and rarely concern themselves with the bitstream-decoder coupling that defines video coding.

To the best of our knowledge, existing approaches have not been published that apply this specific instantiation: budget-constrained depth selection over a **sequential refinement decoder** for *learned video compression*, where the decoder graph is stateful across stages and reconstruction is conditioned on motion and context produced earlier in the codec. Structurally similar formulations exist in adjacent domains — early-exit networks, slimmable networks, RL-based depth selection, mixture-of-experts routing — which collectively establish the conceptual toolkit. The gap is therefore not an absence of formulation but an absence of application of these existing formulations to the specific decoder structure examined here. The novel contribution is the application: the decoder is scheduled as an optimizer subject to a budget, conditioned on content, instantiated for sequential refinement decoding of bitstream-conditioned video reconstruction. This honest framing aligns the contribution with what the evaluation supports and avoids overstating novelty in a domain with extensive prior work on related formulations.

---

## 3. Problem Formulation

We model the video codec's decoder as a sequential Budget-Adaptive Decoder. Given the latent, motion, and context already produced by the codec, the Budget-Adaptive Decoder reconstructs the frame through a sequence of refinement stages. Let:

- $K$ denote the number of refinement stages in the decoder. For the remainder of this work, we set $K = 4$. The formulation is stated generally over $K$; this work instantiates $K = 4$ to enable exhaustive scheduling and clean ablation studies.
- $c_t$ denote pre-decoded content features extracted from the bitstream BEFORE any decoder stage runs (by the BitstreamContentExtractor). The policy network operates on $c_t$ — not on the decoder's internal state — to avoid circular dependency between scheduling and execution.
- $C_{\max}$ denote the maximum computation currently available to the decoder at runtime (FLOPs).
- $C_t$ denote the analytical FLOP count of executing refinement stage $t$. Stage costs satisfy $C_1 \ge C_2 \ge \cdots \ge C_K$; the earliest stage dominates compute, with later stages becoming progressively cheaper.
- $C_\text{full} = \sum_{t=1}^{K} C_t$ denote the full decoder compute (refinement stages only; base reconstruction cost $C_0$ is accounted separately).
- $B \in (0, 1]$ denote the runtime budget ratio, defined as the maximum compute currently available divided by the total decoder compute: $B = C_{\max} / (C_0 + C_\text{full})$. $B = 1$ corresponds to executing all stages; smaller $B$ corresponds to a more constrained device.

The Budget-Adaptive Decoder executes a *prefix* of refinement stages. Specifically, if $k \in \{0, 1, \ldots, K\}$ stages are executed, they are always stages $\{1, 2, \ldots, k\}$ — stage $t$ depends on the output of stage $t-1$, so skipping an intermediate stage is not feasible. The compute incurred is $C_0 + \sum_{t=1}^{k} C_t$.

Let $Q(k; s_1)$ denote the reconstruction quality realized by executing the first $k$ refinement stages sequentially, starting from initial decoder state $s_1$. The exact computation of $Q(k; s_1)$ is unavailable at inference. The *Policy Network* $f_\theta$, a lightweight module, produces stage utility estimates given pre-decoded content features $c_t$ and runtime budget $B$:
$$U_t = f_\theta(c_t, B),$$
one scalar per refinement stage. These utilities are surrogate estimates used by the scheduler to approximate each stage's marginal contribution — not an assumption of independence or additivity.

The *Scheduler* $\Omega$ takes the utility estimates $\{U_1, \ldots, U_K\}$ together with the stage costs $\{C_1, \ldots, C_K\}$, base reconstruction cost $C_0$, and budget $B$, and produces the execution depth:
$$k^\star = \Omega(\{U_t\}, \{C_t\}, C_0, B).$$
The scheduler approximates the solution of the constrained optimization problem:
$$
\begin{aligned}
k^\star \;=\; \arg\max_{k \in \{0, \ldots, K\}} \quad & Q(k; s_1) \\
\text{subject to} \quad & C_0 + \sum_{t=1}^{k} C_t \;\le\; B \cdot (C_0 + C_\text{full}),
\end{aligned}
$$
where the objective — the true reconstruction quality — is approximated during inference via the surrogate utilities produced by $f_\theta$. The execution policy itself is the composite $\pi = (f_\theta, \Omega)$.

The Budget-Adaptive Decoder takes $k^\star$ and produces the final reconstruction by executing, in order, stages $1$ through $k^\star$ starting from base reconstruction state $s_1$. Information flow at the architectural level is:
$$
s_1,\, B \;\xrightarrow{f_\theta}\; \{U_t\} \;\xrightarrow{\Omega}\; k^\star \;\xrightarrow{\text{Budget-Adaptive Decoder}}\; \widehat{Y}.
$$

The objective of the remainder of this work is therefore not to redesign neural video decoding itself, but to determine how an existing decoder should allocate its computation under a runtime budget.

### Design Assumptions

1. Motion/context $M$ is computed once before refinement and remains constant across all refinement stages.
2. The runtime compute budget $B$ is known before decoding begins.
3. Refinement stages execute sequentially; each stage operates on the output of the previous stage.
4. Stage costs $C_t$ are analytically determinable and do not vary with content.

---

# Part II — System Design

## 4. Design Objectives

The formulation in Part I defines *what* the system must accomplish: maximize reconstruction quality subject to a runtime compute budget. This section defines *how* the system meets those requirements — the properties each component must satisfy, and why each architectural choice follows from one or more objectives.

**O1 — Runtime budget awareness.** The system must be explicitly informed of the runtime compute budget $B$ before decoding. This budget is external — it is provided by the deployment environment — and the system must condition its execution decisions on it. This is distinct from approaches that select compute depth without an explicit budget input.

**O2 — Encoder-side compatibility.** The system reuses the established DCVC encoding pipeline (motion estimation, context encoding, entropy coding) without modification, and is bitstream-compatible with DCVC's encoded representations. The decoder is a new progressive multi-stage architecture that consumes DCVC's `context` tensor and the previously decoded frame to produce refinements of the DCVC base reconstruction. This preserves deployment-side encoder compatibility (existing DCVC encoders deploy unchanged) while enabling the budget-adaptive scheduling framework on the decoder side. In this work, the codec is instantiated as DCVC; the formulation generalizes to any codec whose output is a tensor plus a previously decoded frame that can be refined progressively.

> **Revision note (S3 fix).** Previous draft claimed "minimal modification of the codec," but the decoder itself is written from scratch. The encoder-side compatibility claim is more precise and is exactly what is preserved — no existing DCVC encoder is modified. The decoder redesign is the architectural contribution of this work.

**O3 — Lightweight runtime overhead.** Decisions about which stages to execute must be made with negligible computational cost relative to the decoder itself. If scheduling overhead is comparable to decoding overhead, the adaptation provides no practical benefit. The Policy Network is therefore designed to be minimal — on the order of tens of thousands of parameters — and the Scheduler performs only simple arithmetic comparisons.

**O4 — Monotonic quality improvement with increasing compute.** If $B_1 > B_2$, the system executing under budget $B_1$ should produce reconstruction quality at least as good as that under $B_2$. This property ensures that the scheduling framework is monotonic with respect to available compute — a natural requirement for any budget-adaptive system — and follows from the sequential structure of the Budget-Adaptive Decoder.

**O5 — Exact scheduling under a small search space.** The constrained optimization problem over depth values $k \in \{0, \ldots, K\}$ admits $K + 1$ feasible solutions. For $K = 4$, the search space contains only 5 depths, enabling exact evaluation rather than approximation. This eliminates any concern about scheduler suboptimality and makes quality differences directly attributable to the Policy Network's utility estimates.

**O6 — Compatibility with teacher-guided training.** The Budget-Adaptive Decoder must be trainable via distillation from a teacher codec, and the Policy Network must be trainable via budget-conditioned supervision, without architectural modifications. Both training paradigms must coexist within the same framework.

These six objectives collectively determine the structure of every component in Part II. A component that violates any objective is rejected; a component that satisfies all six objectives is retained.

---

## 5. Overall System Architecture

The system consists of five principal components, arranged in two stages:

```
    INFERENCE-TIME EXECUTION (per frame):

    BEFORE DECODING (pre-decode, zero extra compute):
    ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
    │  Budget B         │────▶│ BitstreamContent │     │  Policy Network  │
    │  (from device)    │     │ Extractor        │────▶│  f_θ             │
    └──────────────────┘     │ (zero decode cost)│     │  predicts ΔQ(k) │
                              └──────────────────┘     └────────┬─────────┘
                                                                  │
                                                                  ▼
                                             ┌──────────────────────────┐
                                             │  Scheduler Ω             │
│  k* = argmax_{k: C_0+Σ_{t=1}^k C_t≤B·(C_0+C_full)}
                                              │           Σ_{t=1}^{k} ΔQ(t)
                                             └────────────┬─────────────┘
                                                          │ k*
                                                          ▼
    DECODING (only after scheduling decision):
    ┌──────────────────────────────────────────────────┐
    │           Budget-Adaptive Decoder                  │
    │   Base reconstruction + stages 1...k*             │
    └──────────────────────────────────────────────────┘
                                                          │
                                                          ▼
                                                     Ŷ (reconstruction)
```

**Key invariant:** Content features are extracted from the bitstream BEFORE any decoder stage runs. The scheduling decision is made before execution begins. No circular dependency exists between content features and decoder state.

---

## 6. Component Design

### 6.1 DCVC as the Teacher Codec

The teacher codec provides: (i) a base reconstruction target for the Budget-Adaptive Decoder's first stage, and (ii) the final reconstruction target for the full system during Phase 3–4 (policy training).

We instantiate the teacher as DCVC, a state-of-the-art learned video codec that decomposes video compression into motion estimation, context encoding, latent encoding, and sequential refinement decoding.

**Stage count verification (COMPLETED):** DCVC's original decoder produces a **single-stage output** (K=1). Verification performed on the original Microsoft DCVC implementation:
- DCVC forward pass returns exactly one reconstruction (`recon_image`)
- The `context` tensor [B, 64, H/4, W/4] is the motion-compensated feature map
- No intermediate stage targets are available from the teacher

**Implication for student architecture:** The Budget-Adaptive Decoder has K=4 refinement stages but the teacher only provides ONE target. This fundamentally changes the Phase 1 supervision strategy:

| Student Stage | Target | Supervision Signal |
|---------------|--------|-------------------|
| Stage 1 | DCVC `recon_image` | Teacher-supervised (MSE to teacher recon) |
| Stages 2, 3, 4 | Ground truth frame | Self-supervised (MSE to GT) |

**Rationale:** Stages 2–4 learn to EXCEED the teacher by refining toward ground truth. This makes the decoder genuinely useful: k=1 matches DCVC at 44% of full compute budget; k>1 exceeds DCVC at additional compute cost. The paper claim becomes: "Stage 1 matches DCVC quality; Stages 2–4 surpass DCVC."

**Inputs:** Previous decoded frame, current bitstream (entropy-decoded latent and motion representations).  
**Outputs:** Final decoded frame, motion-compensated context features [B, 64, H/4, W/4].  
**Role in training:** Frozen. Produces reconstruction and context signals without being modified.  
**Role at inference:** None. The teacher is used exclusively during training.

---

### 6.2 BitstreamContentExtractor

The content extractor is a lightweight feature network that operates on the tensors DCVC exposes — after DCVC has produced its base reconstruction but before any Budget-Adaptive Decoder refinement stages run. All inputs are available at zero additional decode cost (no extra forward calls into the codec). **Important clarification:** "Pre-decoding" here means before the Budget-Adaptive Decoder's refinement stages (which cost $C_1, \ldots, C_K$). DCVC itself has already executed (at cost $C_0$) to produce `context` and `recon_image`. The $C_0$ cost is fixed and independent of the refinement budget $B$; it is included in the total budget constraint as $C_0 + \sum_{t=1}^{k} C_t \le B \cdot (C_0 + C_\text{full})$ but is not part of the scheduling decision's candidate depths.

> **Revision note (S1 fix).** Earlier draft specified `latent_quantized` + `motion_latent` + `frame_type_id`, but DCVC's verified forward (see `models/teacher.py`) does not expose a separate `motion_latent` tensor in inference mode, and the `frame_type_id` is not signalled in the bitstream output we receive. The actual inputs are the ones listed below. The implementation is updated accordingly (`models/extractor.py`).

**Inputs (available pre-decoding, matching DCVC's verified outputs):**
- `context`:     motion-compensated features $[B, 64, H/4, W/4]$ — the tensor actually fed into DCVC's original decoder head
- `recon_image`: DCVC base reconstruction $[B, 3, H, W]$ — the per-pixel output DCVC produces prior to any refinement
- `prev_frame`:  previous decoded frame $[B, 3, H, W]$ — needed for the motion-compensation path
- `latent` (optional): feature tensor from DCVC's contextual encoder, $[B, C_{lat}, \ldots]$; off by default to keep extractor lightweight

**Architecture:** Each input is reduced via adaptive average pooling to a fixed-size spatial feature map, flattened to a vector, and projected to a lower dimension: `context` (64×8×8 → 32-dim), `residual = |recon_image − prev_frame|` (3×8×8 → 16-dim), `prev_frame` (3×4×4 → 16-dim), and optional `latent` (C_lat×1×1 → 16-dim). These are concatenated and passed through an MLP fusion network producing a `feature_dim=64` content feature vector. No frame-type embedding is used (frame_type_id is not available from DCVC's bitstream output).

**Cost:** ~0.1% of base stage FLOPs. This is a design choice: the feature extractor must be cheap enough that running it pre-decoding does not dominate the budget it is trying to allocate.

**Why this works:** All four inputs are available in the bitstream without executing any decoder stages. The content signal is computed before, not after, partial decoding — eliminating the circular dependency present in earlier designs.

---

### 6.3 Policy Network $f_\theta$

The Policy Network predicts the marginal quality gain $\Delta Q(k)$ that each refinement stage $k$ would contribute, given pre-decoded content features and runtime budget $B$.

**Inputs:**
- `content_features`: aggregated bitstream-derived feature vector from BitstreamContentExtractor
- $B$: the runtime budget ratio, provided by the deployment environment

**Output:** A vector $[\Delta Q(1), \ldots, \Delta Q(K)]$ where $\Delta Q(k) = Q(k) - Q(k-1)$ is the predicted PSNR gain of stage $k$. The output is unconstrained (no softmax or sigmoid). Negative predictions are valid: they indicate the policy predicts a stage will decrease quality for a given frame, which the scheduler's cumulative maximization will avoid by selecting a smaller $k^\star$. Empirically, after Phase 2 regression training, $\Delta Q$ predictions should cluster near the observed positive distribution, but negative values are allowed for the minority of frames where a stage may reduce PSNR.

**Architecture:** Two hidden layers of ~64 units each, with a budget encoder (1 → 16 → 16) that conditions the prediction on runtime budget. Total parameters ~10k–20k. The output is a $K$-dimensional vector of unbounded real values — directly interpreted as marginal PSNR gains in dB.

**Why ΔQ regression instead of utilities:** The policy output is directly interpretable: each $\Delta Q(k)$ is the predicted marginal quality gain of stage $k$. The scheduler selects the feasible depth maximizing $\sum_{t=1}^{k} \Delta Q(t)$. The regression formulation avoids the non-differentiability problem entirely; training is pure supervised regression with no discrete gradient approximations.

**Role in training:** Trained via budget-conditioned regression to precomputed $\Delta Q$ targets (Phase 3), with joint fine-tuning in Phase 4.

---

### 6.4 Scheduler $\Omega$

The Scheduler turns predicted marginal quality gains into an execution depth. It selects the feasible depth that maximizes total predicted quality gain — the sum of marginal gains up to that depth.

**Inputs:** Predicted marginal gains $\{\Delta Q(1), \ldots, \Delta Q(K)\}$, stage costs $\{C_1, \ldots, C_K\}$, base cost $C_0$, budget $B$.

**Decision rule:** Select $k^\star$ as the feasible depth maximizing total predicted gain:
$$k^\star = \arg\max_{k \in \{0, \ldots, K\} : C_0 + \sum_{t=1}^{k} C_t \le B \cdot (C_0 + C_\text{full})} \sum_{t=1}^{k} \Delta Q(t).$$

The objective $\sum_{t=1}^{k} \Delta Q(t)$ is the predicted reconstruction quality at depth $k$ relative to the base reconstruction, because:
$$\widehat{Q}(k) = Q(0) + \sum_{t=1}^{k} \Delta Q(t),$$
and $Q(0)$ is constant across all candidates. The scheduler therefore selects the depth predicted to yield the highest quality within the budget constraint.

**Algorithm:** For $K = 4$, evaluate all $K + 1 = 5$ feasible depths, compute cumulative $\Delta Q$ sums for each, and select the argmax. This is $O(K)$ with no free parameters — the scheduler is fully determined by the policy's $\Delta Q$ predictions and the budget.

> **Acknowledgment on content-awareness vs. budget-awareness (C2 fix).** When the trained decoder's per-stage outputs all have $\Delta Q(k) > 0$ uniformly (which the Phase 1 ground-truth supervision of Stages 2-4 incentivizes), the cumulative sum $\sum_{t=1}^{k} \Delta Q(t)$ is strictly increasing in $k$, and the scheduler selects the maximum feasible depth at every budget. In this regime the policy degenerates to a budget-only rule, equivalent to the greedy baseline (A3 in §0).

> **Honest framing in the paper.** The contribution of this work is therefore presented in two layers:

> 1. **Budget-adaptive quality frontier (always validated):** The trained Budget-Adaptive Decoder produces a one-shot architecture where any depth $k$ is reachable. The $K+1$ discrete operating points (compute, PSNR) for depths $k \in \{0, \ldots, K\}$ constitute the design contribution regardless of whether the policy adds per-frame adaptation on top. The curve appears as a staircase because $k^\star$ can only change at discrete budget boundaries; between boundaries the system executes the same depth. This is correct discrete-depth behavior, not a continuous Pareto-optimal frontier.

> 2. **Content-adaptive scheduling (valid when $\Delta Q$ is heterogeneous):** Content-adaptive benefits emerge in two cases:
>
> - **Higher compression QPs** (lower bitrates), where some frames are ill-served by the DCVC base reconstruction and gain disproportionately from later stages while others are nearly perfect at the base — producing large variance in $\Delta Q$ across frames. The current paper reports only on DCVC Quality Level 3 (high-bitrate) and acknowledges this regime may exhibit low ΔQ variance.
> - **Content-class mixture** (e.g., mixed animation + natural video), where static backgrounds show small stage gains while dynamic content shows large stage gains. The smoothing (K=4) together with per-frame residual and motion proxies produces a content-adaptive benefit.
>
> When both conditions fail (uniform content, single QP, near-saturated base reconstruction), the system still produces $K+1$ valid operating points; it just does not exercise the *content-adaptive* path inside each budget level. The paper reports both:
>
> - the $K+1$ operating points on the quality-compute frontier (always valid), and
> - the depth-vs-content distribution per budget level (valid only when features vary across frames).
>
> The required correlation check (Experiment 1) is the gate: if $|r| < 0.3$ for all feature-stage pairs across the entire training set, the system is reported as "budget-adaptive only" — a weaker but still publishable claim.

> **Why design not redesigned.** We have not redesigned Phase 1 to produce heterogeneously-significant $\Delta Q$ values (Option B in the design review) because that would compromise the cleaner **"decoder always exceeds teacher"** story we have now. The hybrid supervision (Stage 1 ↔ teacher, Stages 2-4 ↔ ground truth) is the cleanest way to write "decoder is at least as good as teacher, and strictly better at deeper depths." Heterogeneous ΔQ is achieved through the multi-QP / multi-content future-work extension rather than via training-time regularization.

**Non-monotonic stage selection note:** The cumulative maximization scheduler can select fewer stages at a higher budget if an intermediate stage has predicted negative ΔQ. For example, if ΔQ(3) < 0, the scheduler may skip k=3 even when budget permits it, because including k=3 reduces the cumulative sum. This means $k^\star(B)$ is not guaranteed to be monotonic in $B$. This is correct behavior when ΔQ predictions are accurate, but it means the stage-count vs. budget curve may have flat sections followed by jumps.

**On O4 quality monotonicity (conditional):** Design Objective O4 ("more budget → better quality") is a quality claim, not a stage-count claim. O4 holds **conditionally** on the Policy Network's prediction accuracy:
- If all $\Delta Q(k) > 0$ (positive predictions): quality strictly increases with budget; scheduler selects max feasible depth.
- If some $\Delta Q(k) < 0$ (negative predictions): scheduler may skip that stage; quality at the higher budget equals quality at the lower budget's depth (not worse), because the skipped stage's prediction was incorrect and skipping it preserves the previous-best quality.
- The worst case for O4: the policy predicts ΔQ(k) > 0 for a stage that actually decreases quality, causing the scheduler to select it at a higher budget. This would violate O4. O4 is therefore not architecturally guaranteed — it is guaranteed only if the policy's positive predictions are accurate.
- Experiment 1b and Experiment 3 verify O4 empirically: if PSNR decreases for >5% of consecutive budget pairs, the decoder training must be revisited.

The asymmetry matters: the scheduler never selects a *worse* depth at higher budget unless the policy's prediction was incorrect. O4 is thus a statement about the joint system (decoder + policy), not about the decoder alone.

**Why not threshold-based:** The threshold rule (run stage $k$ iff $\Delta Q(k) > \text{threshold}(B)$) has an unspecified free parameter `threshold_base` that fundamentally determines system behavior. If `threshold_base` is too small, the system degenerates to greedy scheduling; if too large, it almost always selects $k^\star = 0$. The cumulative maximization rule eliminates this parameter entirely while directly optimizing the stated objective.

**Why not REINFORCE or learned scheduler:** The sequential prefix structure makes the optimization over $k$ exact and trivial. A learned or approximate scheduler would introduce training complexity and variance without any accuracy gain in this small search space ($K = 4$).

---

### 6.5 Budget-Adaptive Decoder

The Budget-Adaptive Decoder is the execution engine of the system. It consists of a base reconstruction stage (cost $C_0$) followed by $K = 4$ sequential refinement stages, each with decreasing computational cost $C_1 \ge C_2 \ge C_3 \ge C_4$.

**Stage cost ratios (approximate, to be verified by analytical FLOP accounting):** Based on the computational structure of each stage (feature dimensions, convolution counts), the approximate cost ratios are:
$$C_1 : C_2 : C_3 : C_4 \approx 4 : 2 : 2 : 1$$
meaning stage 1 consumes approximately 44% of the total refinement budget (excluding $C_0$), and stage 4 consumes approximately 11%. $C_0$ is the cost of the base DCVC reconstruction and is independent of the refinement budget. These ratios must be verified by analytical FLOP accounting once the decoder architecture is fully specified and before Experiment 2 is run. If the actual ratios differ substantially from these estimates, the budget values used in Experiment 2 (B ∈ {0.3, 0.5, 0.7}) must be adjusted to ensure at least 2 depth values are feasible for the majority of frames at each tested budget level.

**Verification requirement:** Before running Experiment 2, verify that for each $B \in \{0.3, 0.5, 0.7\}$, more than 50% of frames have at least 2 feasible depth values. If this fails for a given $B$, either adjust the budget values or document which budget levels are excluded from the content-awareness analysis in Experiment 2.

> **Issue 2.4 — Pending FLOPs measurement:** The $C_1:C_2:C_3:C_4 = 4:2:2:1$ ratios are analytical estimates based on `BLOCKS_PER_STAGE = [8, 4, 4, 2]`. Actual FLOPs must be measured using `fvcore.nn.FlopCountAnalysis` on the trained decoder. Additionally, $C_0$ (DCVC teacher decode cost) must be measured — it is NOT included in the student decoder's stage costs and is independent of the refinement budget. The full system cost is $C_0 + C_\text{full}$. When actual measurements are available, recompute the budget boundary table using the formula $B_k = (C_0 + \sum_{t=1}^{k} C_t) / (C_0 + C_\text{full})$. The current budget values in Experiment 3 ($\{0.44, 0.67, 0.89, 1.0\}$ for $K=4$) are based on the old $C_0=0$ model and must be updated once real measurements are available (Issue 2.4).

**Base Reconstruction:** Produces the initial decoder state from the DCVC teacher's `context` tensor [B, 64, H/4, W/4] and previous frame. This stage is always executed regardless of budget. Input resolution: context at H/4 upsampled to H; output: reconstruction at H×W + feature map at H×W.

**Refinement Stages $1, \ldots, K$:** Sequential stages, each operating on the feature tensor produced by the previous stage. Each refinement stage updates the decoder feature representation and reconstruction via a residual connection. Only stages $1, \ldots, k^\star$ selected by $\Omega$ are executed.

**Monotonicity note:** The refinement stages use a residual architecture where output $= \text{input} + \delta$, but $\delta$ is unconstrained and can be negative. The softplus gate $\text{softplus}(\alpha)$ ensures the stage applies a non-zero correction, not that the correction improves PSNR. A negative $\delta$ can reduce quality. Monotonicity is therefore not architecturally guaranteed — it must be verified empirically.

**Empirical verification protocol (Section 13, Experiment 1b):** After Phase 1 training, measure PSNR at each depth for all validation frames. For each frame compute the depth $k$ that maximizes PSNR. Report what fraction of frames have $k > 0$ and whether PSNR at $k+1 \ge$ PSNR at $k$ for more than 99% of frame-depth pairs. If this holds, the system is empirically monotonic; if not, the decoder training must be adjusted before Phase 3.

| Component | Computational Complexity |
|-----------|--------------------------|
| Decoder (full) | $O(C_\text{base} + \sum_{t=1}^{K} C_t) = O(C_\text{full})$ |

**Outputs:** The final reconstruction $\widehat{Y} = R_{k^\star+1}$ where $k^\star$ is the depth selected by $\Omega$.

**Role during training:** In Phase 1, the Budget-Adaptive Decoder is trained via **hybrid distillation**: Student Stage 1 ↔ DCVC teacher reconstruction (MSE); Student Stages 2-4 ↔ ground truth frame (weighted MSE with weights [0.5, 0.75, 1.0]). DCVC has K=1 (single output), so a stage-aligned mapping is impossible; Stage 1 cleanly matches teacher quality while Stages 2-4 exceed it by always targeting ground truth. In Phase 2, decoder parameters are frozen and ΔQ precomputation runs. In Phase 3, both decoder and Policy Network are fine-tuned jointly.

**Design rationale:** Keeping the decoder architecture fixed — and making only the *execution depth* adaptive — satisfies O2 (minimal codec modification) and O6 (distillation-compatible). The sequential structure and empirical quality monotonicity verification satisfy O4: executing additional stages empirically does not decrease quality for the vast majority of frames. The decreasing stage costs satisfy O5: the search space $\{0, \ldots, K\}$ is small enough for exact evaluation.

---

## 7. Operational Flow

The complete system execution pipeline, end-to-end, is as follows:

**At inference time:**
1. A runtime budget $B$ is provided by the deployment environment (e.g., mapped from device class: phone → 0.3, desktop → 0.9).
2. Content features are extracted from the DCVC bitstream — entropy-decoded `context`, `recon_image`, `prev_frame` — BEFORE the Budget-Adaptive Decoder's refinement stages run. This pre-decoding step happens AFTER DCVC has produced its base reconstruction; it does NOT re-run DCVC. Cost is negligible (~0.1% of base stage FLOPs).
3. The Policy Network evaluates $\Delta Q(k) = f_\theta(\text{content\_features}, B)$ for each $k \in \{1, \ldots, K\}$.
4. The Scheduler selects depth $k^\star$ by maximizing cumulative predicted quality gain subject to the budget constraint: $k^\star = \arg\max_{k: C_0 + \sum_{t=1}^{k} C_t \le B \cdot (C_0 + C_\text{full})} \sum_{t=1}^{k} \Delta Q(t)$.
5. The Budget-Adaptive Decoder executes base reconstruction (cost $C_0$) + refinement stages $1, \ldots, k^\star$ (cost $\sum_{t=1}^{k^\star} C_t$) and returns the final reconstruction.
6. The total FLOPs consumed are $C_0 + \sum_{t=1}^{k^\star} C_t$, which never exceeds $B \cdot (C_0 + C_\text{full})$.

**At training time (Phase 1 — distillation):**
1. The DCVC teacher produces context [B, 64, H/4, W/4] and single reconstruction R_T.
2. The Budget-Adaptive Decoder produces reconstructions R_1, R_2, R_3, R_4.
3. Stage 1 loss: MSE(R_1, R_T) — teacher-supervised.
4. Stages 2–4 loss: weighted MSE(R_k, Y) for ground truth Y — self-supervised.
5. The Policy Network is frozen; only the decoder learns.

**At training time (Phase 2 — ΔQ precomputation):** No forward pass with training. The trained Budget-Adaptive Decoder (from Phase 1) is run at all depths $k = 0, \ldots, K$ on the training set to compute $\Delta Q$ targets. Results are stored to disk for Phase 3.

**At training time (Phase 3 — policy regression):**
1. Budget $B$ is sampled from a mixture distribution: 50% $\text{Beta}(2, 2)$, 20% $\text{Beta}(1, 1)$ (uniform), 15% $\text{Beta}(5, 2)$ (favoring higher budgets), 15% at discrete deployment values $\{0.3, 0.5, 0.9\}$. This ensures adequate training signal at deployment-relevant budgets while maintaining exploration.
2. Content features are extracted from the stored bitstream (pre-decoding; no decoder stage runs).
3. The Policy Network evaluates $\Delta \widehat{Q}(k) = f_\theta(\text{content\_features}, B)$ for $k = 1, \ldots, K$.
4. The decoder is frozen; the Policy Network and BitstreamContentExtractor receive gradients via the $\Delta Q$ regression loss.

**At training time (Phase 4 — decoder adaptation):**
1. Budget $B$ is sampled from a mixture distribution: 50% $\text{Beta}(2, 2)$, 20% $\text{Beta}(1, 1)$ (uniform), 15% $\text{Beta}(5, 2)$ (favoring higher budgets), 15% at discrete deployment values $\{0.3, 0.5, 0.9\}$. This ensures adequate training signal at deployment-relevant budgets while maintaining exploration.
2. Content features are extracted and $\Delta \widehat{Q}$ is predicted using the **frozen** Phase 3 Policy Network (no gradient flow to policy).
3. The Scheduler selects $k^\star$ by maximizing cumulative predicted gain subject to budget.
4. The Budget-Adaptive Decoder executes stages $1, \ldots, k^\star$ and produces $\widehat{Y}$.
5. Gradients flow only to the Budget-Adaptive Decoder at $0.05 \times \eta_\text{base}$. The Policy Network receives no gradient updates in Phase 4.

---

# Part III — Learning Framework

## 8. Training Methodology

The training program consists of four sequential phases. The phase order is strict: Phase 1 must complete before Phase 2 runs, because Phase 2's ΔQ targets are computed using the trained decoder. Each phase freezes components from earlier phases, preventing gradient interference and enabling staged specialization.

### Phase 1 — Budget-Adaptive Decoder Distillation

**Objective:** Train the Budget-Adaptive Decoder so that Stage 1 matches DCVC quality, and Stages 2–4 exceed DCVC by refining toward ground truth.

In this phase, the Policy Network is detached (no gradients flow to it). The Budget-Adaptive Decoder learns via a hybrid supervision strategy that reflects the single-stage teacher limitation.

**Teacher supervision (Stage 1 only):** The DCVC teacher produces a single reconstruction `R_T`. The Budget-Adaptive Decoder's first stage output `R_1` is supervised to match `R_T`:
$$\mathcal{L}_\text{teacher}^{(1)} = \|R_1 - R_T\|_2^2.$$

**Ground truth supervision (Stages 2–4):** The decoder's Stages 2, 3, and 4 learn to exceed the teacher by refining toward the ground truth frame `Y`:
$$\mathcal{L}_\text{gt}^{(k)} = w_k \cdot \|R_k - Y\|_2^2, \quad k \in \{2, 3, 4\}$$
where weights $w_k = [0.5, 0.75, 1.0]$ increasing with stage depth ensure later stages receive stronger gradient signal toward the ground truth.

**Total Phase 1 loss:**
$$\mathcal{L}_\text{Phase 1} = \mathcal{L}_\text{teacher}^{(1)} + \sum_{k=2}^{K} \mathcal{L}_\text{gt}^{(k)}.$$

**Critical implementation requirement (C1 fix):** Stage 1 must receive ONLY the teacher MSE gradient signal. Stages 2-4 must receive ONLY the ground-truth MSE gradient signal. To enforce this, `decoder.decode_all_stages_phase1(context, prev_frame)` calls Stage 1 in the standard way (gradient flows into its weights from `loss_s1`), then **calls `.detach()`** on Stage 1's output before passing it as input to Stage 2. This isolates the two supervision signals at the Stage 1 ↔ Stage 2 boundary.

$$\mathcal{L}_\text{Phase 1}^\text{corrected} = \|R_1 - R_T\|_2^2 \;+\; \sum_{k=2}^{K} w_k \,\cdot\, \|\,R_k(\text{detach}(R_1)) - Y\|_2^2$$

Without `.detach()`, Stages 2-4 propagate their GT-MSE gradient backward through Stage 1, creating two conflicting signals at Stage 1: "match teacher" (from `loss_s1`) and "produce features that help Stage 2-4 reach ground truth" (from propagated GT loss). Stage 1 then converges to a *compromise* between the two, and the paper claim "Stage 1 matches DCVC cleanly" does not hold.

This is implemented in `models/decoder.py:decode_all_stages_phase1()` and used by `training/phase1_distill.py`.

**Why this design is stronger than teacher-matching only:** If all stages matched the teacher, the decoder could at best equal DCVC quality. By supervising Stages 2–4 with ground truth, the decoder learns to surpass the teacher at higher compute budgets. The paper claim becomes: "Stage 1 matches DCVC at 44% of full compute; Stages 2–4 exceed DCVC at 67%, 89%, and 100% of compute respectively."

**Stage weights ablation (N1):** The default weights $[w_2, w_3, w_4] = [0.5, 0.75, 1.0]$ are a heuristic, not ablated. Earlier-iteration weights (smaller $w_2$) let Stage 1 settle the teacher signal first; later-iteration weights (larger $w_4$) are honest about the final ground-truth cost being the most important target.

  | Ablation A14 | Setting | Rationale |
  |--------------|---------|-----------|
  | `equal` | $w_k = 1$ for $k \in \{2,3,4\}$ | Treats all stages equally |
  | `linear` | $w_k = (k-1)/3$ | Smooth ramp |
  | `geometric` | $w_k = 0.5^{4-k}$ | Halves each step |
  | `default` | $w_k = [0.5, 0.75, 1.0]$ | Chosen empirically |

  All four are reported in the appendix; the reported headline numbers in §13 use the `default` weights if not otherwise stated.

**Frozen components:** DCVC teacher (always frozen), Policy Network (detached).  
**Trainable components:** Budget-Adaptive Decoder.  
**Training signal:** Stage 1: teacher MSE; Stages 2–4: weighted ground truth MSE.

**Output:** A trained Budget-Adaptive Decoder capable of producing high-quality reconstructions at each depth. This decoder is used as the basis for all subsequent phases.

---

### Phase 2 — Offline ΔQ Target Precomputation

**Purpose:** Compute actual marginal quality gains $\Delta Q(k)$ at each depth for all training frames using the trained decoder from Phase 1. These become regression targets for the Policy Network.

**Critical ordering:** This phase uses the Phase 1 decoder. Phase 2 must not run before Phase 1 is complete. Using the DCVC teacher directly would produce targets that do not correspond to the student's actual per-stage gain distribution.

**Resolution consideration:** Training uses Vimeo90k at 448×256 resolution. ΔQ targets are therefore computed at 448×256 and reflect the decoder's gain distribution at that resolution. At test time, the policy is evaluated on 1080p sequences. PSNR is already resolution-invariant by definition for the same per-pixel MSE. The spatial correlation structure of codec errors means per-stage gain distributions may still differ across resolutions, but no analytical correction is applied. Resolution-dependent bias is evaluated empirically after training (see Section 14 limitation and post-training evaluation protocol).

**Procedure:** For each training frame, the trained Budget-Adaptive Decoder is executed at all depths $k = 0, 1, \ldots, K$ (with gradients disabled), and PSNR is measured at each depth. The marginal gain is $\Delta Q(k) = \text{PSNR}(k) - \text{PSNR}(k-1)$. Results are stored to disk without analytical normalization.

**Bitrate specification (S4 fix):** The ΔQ distribution is entirely determined by the quantization parameter (QP) used during DCVC encoding. The reference DCVC checkpoint loaded is `model_dcvc_quality_3_psnr.pth`, which corresponds to DCVC's **Quality Level 3** (high-quality, low-compression regime). The actual $\lambda$ value used is derived from the DCVC codebase's preset (`lambda_scale=1` at this quality level). Phase 2 in this work uses this single setting:

  - **Phase 2 (default):** Single QP/$\lambda$ corresponding to DCVC Quality Level 3.
    Captures the high-bitrate regime where ΔQ gains tend to be modest.
    Suitable for the planning paper focused on Schema A.

  - **Multi-bitrate sweep (optional, future work):** Sweep DCVC's full
    64-QP range and concatenate ΔQ targets across bitrates. The Policy
    Network is then conditioned on $B$ alone — the bitrate is implicit in
    the encoded-bitstream behavior — but the policy learns to predict
    ΔQ *relative to* whatever QP produced the bitstream. This requires
    storing per-(frame, QP) targets.

This step avoids the discrete gradient problem entirely: the Policy Network is later trained via pure supervised regression to precomputed targets, with no differentiation through the scheduling decision.

> **Warning (S4 fix).** Earlier draft did not specify the QP used for ΔQ precomputation. Without this specification, the ΔQ distribution is undefined and any downstream Phase 3 policy is uncalibrated. The default $\lambda$ / QP for this paper is fixed at DCVC Quality Level 3 (high-bitrate) and this is documented in §15 metadata and Appendix C.

**Staleness note:** Phase 4 fine-tunes the decoder, which may shift $\Delta Q(k)$ values. Targets computed in Phase 2 are therefore approximate for Phase 4.

**Iterative loop with convergence criterion:** If Phase 4 causes substantial decoder drift (measured as L1 difference in $\Delta Q$ predictions exceeding 5% between pre- and post-Phase 4 versions on a held-out validation set of 100 frames), refresh Phase 2's target file by re-running it with the Phase 4 decoder, then retrain Phase 3 policy, then redo Phase 4. The iterative loop **converges** when:
1. the pre/post Phase 4 $\Delta Q$ prediction difference is below 5%, OR
2. two successive refresh iterations produce $\Delta Q$ prediction differences both below 1% (diminishing returns).

If convergence criterion is not met within 3 refresh iterations, the Phase 1 distillation is repeated from scratch — the system reports Phase 4 wideness as insufficient and reverts to the original Phase 1 decoder. This iterative procedure is documented in §12 and feeds into the "training curriculum" promise: a runtime-callable, deterministic training pipeline that terminates.

**Compute overhead of iteration:** Each refresh iteration costs approximately the same as the original Phase 2 ($\sim$300 A100-hours with stratified sampling). The expected number of refresh iterations is 0-1 in practice (Phase 4 with $\eta_\text{decoder} = 0.05 \cdot \eta_\text{policy}$ produces small decoder drift), so total expected iterative compute remains at $\sim$300 A100-hours. In the worst case (3 iterations + Phase 1 retraining), total compute is bounded at $\sim$1500 A100-hours.

**Compute efficiency:** Phase 2 is the most compute-intensive phase of the curriculum. The 1000 A100-hour estimate for the full unstratified sweep (641k frames at K+1=5 depth passes) is a **first-order approximation** derived from typical DCVC decoder FLOPs (~5×10^10 per frame on 448×256 inputs). Per-frame FLOPs × frame count × depth passes ÷ A100 throughput ≈ hundreds to low-thousands of hours depending on measurement of decoder FLOPs (Issue 2.4 — pending). The exact figure should be re-measured on the actual decoder architecture once Issue 2.4 is resolved; budget planning should include a 2× safety margin. To reduce this cost, Phase 2 uses stratified sampling: 30% of sequences are sampled (stratified by motion proxy — mean frame-to-frame absolute difference — ensuring full coverage of the motion distribution), and within each sequence only P-frames (frames 2–7 of each septuplet) are processed. I-frames are excluded because they have no motion compensation and their $\Delta Q$ distribution differs systematically from P-frames. This reduces Phase 2 to approximately 55,000–80,000 frames, cutting compute by ~70% (bringing expected stratified compute to ~300 A100-hours, with 2× safety margin: target ~600 hours). The exact stratified compute depends on the post-Issue-2.4 decoder measurement.

---

### Phase 3 — Policy Network Training via Regression

**Objective:** Train the Policy Network to predict $\Delta Q(k)$ from content features and budget.

In this phase, the Budget-Adaptive Decoder is frozen. The Policy Network learns via supervised regression to the $\Delta Q$ targets computed in Phase 2.

**Budget sampling:** At each training step, a budget $B$ is sampled from $\text{Beta}(2, 2)$. Note: $\text{Beta}(2, 2)$ places ~33% of its mass in $[0.4, 0.6]$ and relatively little mass near typical deployment budgets (phone → 0.3, desktop → 0.9). If deployment is concentrated at discrete device class budgets, consider a mixture distribution (e.g., $\text{Beta}(2, 2)$ mixed with point masses at the three device class budgets) to ensure adequate training signal at the most relevant budget levels.

**Forward pass:**
1. Budget $B$ is sampled.
2. Content features are extracted from the stored bitstream (pre-decoding, no decoder execution).
3. The Policy Network evaluates $\Delta \widehat{Q}(k) = f_\theta(\text{content\_features}, B)$ for $k = 1, \ldots, K$.
4. The decoder is frozen; the Policy Network and BitstreamContentExtractor receive gradients via the $\Delta Q$ regression loss only. No reconstruction loss is applied in Phase 3.

**Loss:** Mean squared error between predicted and target marginal gains:
$$\mathcal{L}_\text{Phase 3} = \frac{1}{K} \sum_{k=1}^{K} \|\Delta \widehat{Q}(k) - \Delta Q^\star(k)\|_2^2,$$
where $\Delta Q^\star(k)$ is the precomputed target from Phase 2.

**Why no gradient through scheduling:** The Phase 3 loss is purely a regression loss on $\Delta Q$ predictions. We never backprop through the discrete scheduling decision.

**Frozen components:** DCVC teacher, Budget-Adaptive Decoder.  
**Trainable components:** Policy Network, BitstreamContentExtractor.  
**Training signal:** Budget-conditioned MSE regression to precomputed $\Delta Q$ targets.

---

### Phase 4 — Decoder Adaptation

**Objective:** Allow the decoder to adapt to the specific depths selected by the policy, while preserving Phase 1 specialization. The Policy Network is frozen — it does not receive gradients in Phase 4.

**Decoder learning rate ratio:** The decoder's learning rate is $0.05 \times \eta_\text{policy}$. This ratio was selected by validation-set monitoring: the decoder should adapt to policy-selected depths without forgetting the teacher-aligned representations learned in Phase 1. If the ratio is too large (e.g., 0.5), decoder quality at untrained depths degrades. If too small (e.g., 0.001), no meaningful adaptation occurs. The value 0.05 is validated empirically by monitoring per-stage distillation loss on a held-out set during Phase 4. Ablation A13 sweeps the ratio over $\{0.01, 0.05, 0.1, 0.5\}$ to verify 0.05 is near-optimal.

**Gradient flow:** The Policy Network is frozen in Phase 4. The scheduling decision $k^\star = \arg\max_{k: C_0 + \sum_{t=1}^{k} C_t \le B \cdot (C_0 + C_\text{full})} \sum_{t=1}^{k} \Delta \widehat{Q}(t)$ is a discrete argmax — gradients from the reconstruction loss cannot flow through it to the policy. The regression term from Phase 3 is also absent: it would provide gradients to the Policy Network, which is explicitly frozen. Phase 4 trains only the Budget-Adaptive Decoder.

**On the Stage 1→Stage 2 gradient detach (C1 fix) in Phase 4:** The gradient detach inserted between Stage 1 and Stage 2 in `decode_all_stages_phase1()` (see §6.5) was designed specifically for Phase 1's hybrid loss structure: Stage 1 is supervised by the teacher (MSE to teacher recon) while Stages 2-4 are supervised by ground truth (MSE to GT). Without the detach, Stage 2-4's GT-loss gradient would backpropagate into Stage 1 and corrupt the teacher-aligned representation. Phase 4 does NOT use the hybrid loss — it computes a single MSE between the decoder's reconstruction at depth $k^\star$ and ground truth, with no teacher signal and no stage-specific loss splitting. Therefore, Phase 4 calls `run_to_depth()` (which does NOT have the gradient detach) rather than `decode_all_stages_phase1()`. The C1 detach is Phase 1-specific and does not apply to Phase 4's training path.

**Budget sampling:** Same as Phase 3 — $\text{Beta}(2, 2)$.

**Loss:**
$$\mathcal{L}_\text{Phase 4} = \| \widehat{Y} - Y^\star \|_2^2$$

The loss is purely reconstruction MSE — no regression term. All gradients flow to the Budget-Adaptive Decoder only.

**Frozen components:** DCVC teacher, Policy Network, BitstreamContentExtractor.  
**Trainable components:** Budget-Adaptive Decoder ($0.05 \cdot \eta_\text{policy}$).

**Training signal:** Reconstruction quality at the policy-selected depth $k^\star$ (from Phase 3 policy).

### Dataset Augmentation

Three augmentation strategies address known Vimeo90k limitations. The augmentation source must be explicitly defined and verified non-overlapping with test datasets before Phase 2 runs.

**Reference frame corruption (Phase 3 and 4 only — NOT Phase 1):** Vimeo90k septuplets always start from a clean previous frame, which does not reflect deployment conditions where decoded frames accumulate temporal drift. During Phase 3 and Phase 4 only, the previous frame is corrupted before use:
$$\text{prev\_frame\_corrupted} = \text{prev\_frame} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
where $\sigma = \min(0.005 \times \text{frame\_position\_in\_sequence}, 0.02)$. The cap at $\sigma_\text{max} = 0.02$ (approximately 34 dB PSNR degradation) keeps corruption within realistic decoded frame quality range. The slope 0.005 is calibrated by measuring PSNR degradation in actual DCVC-decoded frames at a representative bitrate: decode multiple Vimeo90k sequences, measure per-frame PSNR relative to ground truth, and fit a linear slope to the degradation curve.

**Approximation limitation:** This corruption uses spatially uncorrelated Gaussian noise calibrated to match the average PSNR of DCVC-decoded reference frames. It is an approximation to real temporal error propagation: actual DCVC decoding errors are spatially correlated (blocking artifacts, ringing near edges, motion-compensated drift), whereas Gaussian noise spreads error uniformly. The approximation provides robustness training against imperfect references but does not fully simulate structured codec drift. A more accurate simulation would use actual decoded frames from a cascaded codec pass, which is computationally feasible but was not implemented. This limitation does not affect the validity of quality comparisons but means the policy's robustness to real temporal error patterns is only approximate.

Reference frame corruption is NOT applied during Phase 1. Phase 1 trains the decoder to match DCVC teacher outputs generated from clean references. Applying corruption during Phase 1 creates a systematic mismatch between student input and teacher target, elevating distillation loss without improving decoder quality.

**Principled justification for asymmetric treatment:** The asymmetric corruption scope (Phases 3-4 only, not Phase 1) follows the principle that augmentation should match the deployment distribution of the *supervision target*, not the input. Specifically:
- Phase 1 supervision target = DCVC teacher's reconstruction from clean reference. The teacher itself is trained on clean references, so its output is conditioned on a clean reference distribution. Augmenting Phase 1 inputs without augmenting the teacher target would teach the student to be robust to a mismatch the teacher was never validated against.
- Phase 3-4 supervision target = ground truth frame (Phase 4) or $\Delta Q$ regression target (Phase 3), both of which are independent of the previous frame. These targets are not affected by corruption of the previous frame, so corruption can be added without creating input-target mismatch.
This is a general principle: augmentation is applied to a phase only if the augmentation preserves the supervision target's validity under the augmentation. Corrupting the previous frame preserves Phase 3/4 validity but invalidates Phase 1 distillation; therefore, corruption is applied in Phases 3-4 only.

**Content distribution augmentation (Phase 2 and 3):** Vimeo90k underrepresents high-motion content. Phase 2 and Phase 3 use a stratified training set: 70% Vimeo90k sequences (sampled randomly) plus 30% from the BVI-DVC training set (Bristol Video Inference for Deep Video Compression), which was constructed for learned video codec training and has verified zero overlap with all test datasets (UVG, HEVC Class B/C/D, MCL-JCV). Specific BVI-DVC sequences are selected by motion magnitude: sequences with frame-to-frame pixel difference in the top 30th percentile are included. All augmentation sequences are encoded with DCVC at the same quantization parameters used for Vimeo90k training to ensure consistent ΔQ scale across training data.

**Augmentation dataset specification (required before Phase 2):** The specific BVI-DVC sequences used must be listed in a dataset appendix. Sequences are selected by motion magnitude filtering: compute frame-to-frame absolute difference for all BVI-DVC sequences, rank by mean difference, and include the top 30%. This produces a reproducible, non-overlapping set that covers the high-motion content range encountered at test time. Johnny and FourPeople (JCT-VC Class E test sequences) are excluded from any augmentation source to prevent train-test overlap with evaluation benchmarks.

**No resolution normalization:** As noted in Section 8 Phase 2, PSNR is defined as $10 \log_{10}(\text{MAX}^2 / \text{MSE})$ where MSE is already per-pixel averaged — it is resolution-invariant by definition for spatially independent errors. The spatial correlation structure of codec errors does not permit a simple analytical scaling correction. No analytical resolution normalization is applied to ΔQ targets. Resolution-dependent bias is evaluated empirically after training: if mean absolute prediction error on 1080p test sequences exceeds 0.5 dB above the error on Vimeo90k, Phase 2 targets should be recomputed on a high-resolution training set.

---

## 9. Training Curriculum

The four training phases are executed strictly in order. Phase 2 must not run before Phase 1 is complete — Phase 2's targets are computed using the trained decoder from Phase 1.

| Phase | Steps | Frozen Components | Trainable Components | Budget Sampling |
|-------|-------|-------------------|----------------------|-----------------|
| 1 — Decoder distillation | Until decoder convergence | Teacher, Policy | Budget-Adaptive Decoder | N/A (all stages) |
| 2 — ΔQ precomputation | Once (offline, uses Phase 1 decoder) | All | None | N/A |
| 3 — Policy regression | Until policy convergence | Teacher, Decoder | Policy, ContentExtractor | $\text{Beta}(2, 2)$ |
| 4 — Decoder adaptation | Fixed epoch count | Teacher, Policy | Budget-Adaptive Decoder only | $\text{Beta}(2, 2)$ |

**Phase 1 termination:** Monitor both teacher-supervised loss (Stage 1 vs DCVC recon) and self-supervised loss (Stages 2–4 vs ground truth). Stop when both losses converge or validation loss plateaus.

**Phase 2 note:** This phase is a single forward pass over the training set with a trained decoder; no training occurs. Output is a file of $\Delta Q$ targets for each frame.

**Phase 3 termination:** Monitor $\Delta Q$ regression loss on a validation set. Stop when validation quality plateaus.

**Phase 4 termination:** Fixed number of epochs (determined empirically; typically fewer than Phase 1 or 3 due to adapter-style small learning rate).

---

## 10. Inference Pipeline

At inference time, the system operates entirely without the teacher and without gradient computation. The execution path is:

**Input:** DCVC-encoded bitstream (which, when decoded, produces `context`, `recon_image`, and optionally `latent`), previous decoded frame, runtime budget $B$.

**Step 1 — Content feature extraction (pre-decoding):** The BitstreamContentExtractor processes `context`, `recon_image`, the previous frame (and optionally `latent`) to produce content features. This step runs BEFORE any decoder refinement stage and costs negligible FLOPs.

> **Revision note (N3 fix).** Earlier draft described inputs as "entropy-decoded latent, motion, previous frame, and frame type" — corrected to match what DCVC actually outputs (see S1 fix in §6.2). The `frame_type_id` was removed because DCVC does not signal types in the bitstream we receive.

**Step 2 — Policy prediction:** The Policy Network evaluates $\Delta \widehat{Q}(k) = f_\theta(\text{content\_features}, B)$ for $k = 1, \ldots, K$.

**Step 3 — Scheduling:** The Scheduler takes $\{\Delta \widehat{Q}(k)\}$, $\{C_t\}$, and $B$, and computes $k^\star = \arg\max_{k: C(k) \le B \cdot C_\text{full}} \sum_{t=1}^{k} \Delta \widehat{Q}(t)$. This selects the feasible depth predicted to yield the highest total quality gain, with no free parameters. See §11/§10.5 for the **content-aware vs budget-aware** acknowledgment when all $\Delta Q > 0$.

**Step 4 — Execution:** The Budget-Adaptive Decoder executes base reconstruction + stages $1, \ldots, k^\star$ in sequence and returns the final reconstruction.

**Runtime complexity:** Content extraction + Policy Network + Scheduler together add negligible overhead compared to the decoder itself. For $K = 4$, the scheduler performs at most 4 comparisons; the Policy Network is a small MLP consuming ~10k–20k parameters.

**No circular dependency:** Content features are extracted from DCVC's already-decoded tensors (`context`, `recon_image`) and the previous frame, all of which are available before any decoder refinement stage runs. The decoder's intermediate refinement state is never used as input to the scheduling decision.

---

# Part IV — Evaluation

## 11. Experimental Methodology

### Research Questions

The evaluation is organized around three primary research questions:

**RQ1 — Quality under budget:** Does the proposed framework achieve higher reconstruction quality than static-tier baselines when operating under the same compute budget?

**RQ2 — Scheduler effectiveness:** Is the learned Policy Network better at budget-constrained stage selection than heuristic baselines (greedy, uniform, oracle)?

**RQ3 — Generalization:** Does the framework generalize across content types, motion regimes, and device budgets not seen during training?

### Datasets

**Training dataset:** Vimeo90k (septuplet) is used for training the Budget-Adaptive Decoder and Policy Network, following the standard learned video compression training protocol.

**Test datasets:** Evaluation is performed on multiple datasets representing diverse content:

| Dataset | Content Characteristics | Purpose |
|---------|------------------------|---------|
| UVG | High-quality 1080p sequences; varied motion | Primary quality evaluation |
| HEVC Class B | Basketball, BQTerrace, Cactus, Kimono, ParkScene | Standardized comparison |
| HEVC Class C/D | Medium resolution; diverse motion | Resolution generalization |
| MCL-JCV | High-motion broadcast sequences | Stress-test on complex content |

**Train/test split:** Vimeo90k for training only; UVG, HEVC, and MCL-JCV for evaluation. No overlap between training and test content.

### Metrics

Three complementary metrics capture the system's behavior:

**Quality metrics:**
- **PSNR (dB):** Mean squared error in decibel form. Primary quality metric.
- **MS-SSIM:** Multi-scale structural similarity. Complements PSNR by measuring perceptual quality.
- **LPIPS:** Learned perceptual image patch similarity. Measures perceptual similarity to ground truth.

**Compute metrics:**
- **FLOPs ratio:** $\frac{C(k)}{C_\text{full}}$ — fraction of full decoder compute used.
- **Actual budget adherence:** Verified post-hoc that $C(k) \le B \cdot C_\text{full}$ for all evaluations.

**Efficiency metrics:**
- **Quality per FLOP:** PSNR / FLOPs — measures return on compute investment.
- **Quality-compute frontier:** The set of (FLOPs, quality) pairs achieved by the $K+1$ discrete operating points.

**Metric alignment caveat (Issue 4.1):** The Policy Network is trained to predict $\Delta Q(k)$ in PSNR units (Phase 2 computes "PSNR gain at each depth"; Phase 3 regression targets are PSNR deltas). The scheduler therefore maximizes predicted PSNR, not the metric being displayed. When MS-SSIM or LPIPS curves are plotted for evaluation, the scheduler is operating on predictions calibrated to PSNR — meaning MS-SSIM/LPIPS gaps to oracle are partially explained by **metric misalignment in training**, not architectural limitations. Specifically: a system optimizing for PSNR-optimal scheduling may perform significantly below the MS-SSIM oracle or LPIPS oracle even when the architecture could in principle reach them, because the policy is not tuned for those metrics. The primary metric for all headline claims is therefore PSNR. For MS-SSIM/LPIPS evaluation, the following are reported:
- The proposed system at PSNR-optimal scheduling (primary MS-SSIM/LPIPS result).
- Oracle-MS-SSIM and Oracle-LPIPS (upper bounds; informs the metric-specific performance ceiling).
- An *optional* extension A14: a parallel policy trained against MS-SSIM deltas (or a multi-metric training scheme). A14 is out of scope for the main paper but is documented in §14 as future work.
This caveat is disclosed in the experimental methodology to prevent misinterpretation of non-PSNR results.

### Baselines

The framework is compared against five categories of baselines:

**Fixed-tier baselines (same architecture, no scheduling):**
- **T1:** Always execute only the first refinement stage.
- **T2:** Always execute stages 1–2.
- **T3:** Always execute stages 1–3.
- **T4:** Always execute all stages (full decoder).

**Heuristic baselines (same Policy Network architecture, different training):**
- **Random:** Uniform random depth selection at each frame.
- **Uniform utility:** Policy Network with uniform utility output (ablated).
- **Greedy:** Greedy stage selection (select stages in fixed order until budget exhausted).

**Oracle baselines (not achievable in practice, for upper-bound analysis):**
- **Oracle-per-frame:** For each frame, select the depth that maximizes the metric of interest. Computed separately for each metric: Oracle-PSNR (selects depth maximizing PSNR), Oracle-MS-SSIM (selects depth maximizing MS-SSIM), Oracle-LPIPS (selects depth minimizing LPIPS). PSNR, MS-SSIM, and LPIPS can disagree on the optimal depth for a given frame, so oracle curves are metric-specific.
- **Oracle-budget:** For each budget level $B$, the best quality achievable at exactly that budget (computed by trying all feasible depths and selecting the best for the given metric).

**Prior work baselines:**
- **DCVC (full):** Original DCVC with all computation enabled (our teacher).
- **SlimVC / Mobicodec variants:** Published efficient video codecs at comparable complexity.

**Fixed-compute baselines:**
- **DCVC-half:** DCVC decoder with reduced channel width to approximately match T2 compute.
- **DCVC-quarter:** DCVC decoder with further reduction to match T1 compute.

### Hardware

Evaluation is performed across three device classes to demonstrate budget-awareness in deployment:

| Device Class | Representative Hardware | Budget Range |
|--------------|-------------------------|--------------|
| Mobile | ARM Cortex-A-series, Snapdragon | $B \in [0.2, 0.4]$ |
| Laptop | Intel Iris Xe, mid-range GPU | $B \in [0.4, 0.7]$ |
| Desktop | High-end GPU, workstation | $B \in [0.7, 1.0]$ |

FLOP measurements are performed on each platform to verify actual compute consumption matches the declared budget.

### Statistical Analysis

All reported results are averaged over at least 3 independent runs with different random seeds. Significance testing uses a **length-weighted paired t-test at the sequence level** — for each sequence, the metric values across frames are averaged to produce one independent data point per run, weighted by the sequence's frame count in the t-statistic. Using per-frame values as the unit of analysis violates temporal independence and artificially inflates statistical significance; conversely, unweighted sequence-level averaging treats a 600-frame HEVC Class B sequence and a 100-frame short clip as equally informative, biasing against the system on long sequences. Weight $w_s = N_s$ (frame count of sequence $s$) and compute the weighted t-statistic with effective degrees of freedom equal to the number of sequences minus 1. Confidence intervals are reported at the 95% level using the weighted standard error. The weighting is implemented in `evaluation/statistical_utils.py` and called from each experiment.

---

## 12. Ablation Strategy

Each architectural and training design choice is validated individually. The ablation plan covers 11 components:

| Ablation | Phases Run | Expected Outcome |
|----------|------------|------------------|
| **A1 — No Policy Network (random)** | All phases, random depth at inference | Establishes baseline; RQ2 answer without learned policy |
| **A2 — No Policy Network (zero ΔQ)** | All phases, policy outputs all-zeros | Tests whether learned ΔQ vs. fixed-zero prediction matters |
| **A3 — Fixed depth scheduler** | All phases, greedy by cost only (ignores ΔQ) | Tests whether learned ΔQ improves over budget-only greedy |
| **A4 — Uniform budget sampling** | All phases, Beta(2,2) replaced with Uniform(0,1) | Tests whether biased sampling matters |
| **A5 — Phase 1 only** | Phase 1 only, policy never trained | Tests decoder quality without scheduling |
| **A6 — DCVC as decoder baseline** | DCVC replaces student decoder (no refinement stages); scheduler selects $k=0$ always | Tests whether the multi-stage student decoder adds quality over the DCVC teacher baseline; expected: substantial drop since student cannot exceed teacher quality at any depth |
| **A7 — Phases 1+2+3, no Phase 4** | Phases 1, 2, 3 only (skips Phase 4) | Tests Phase 4 decoder adaptation contribution; expected: A7 ≈ A4 (small Phase 4 benefit) |
| **A8 — Stage cost equality** | All phases, $C_1 = C_2 = C_3 = C_4$ | Equal costs change the feasible depth sets at each budget; if quality is similar, the cost ordering is unimportant for quality but affects compute allocation; staircase curve shifts right/left |
| **A9 — Fewer stages ($K=2$)** | All phases, 2 refinement stages | Coarser quality-compute tradeoff with only 3 operating points; may close less of the oracle gap at moderate budgets; validates whether 4 stages are necessary |
| **A10 — More stages ($K=5$)** | All phases, 5 refinement stages | Finer quality-compute tradeoff with 6 operating points; later stages likely add diminishing returns; increased Phase 2 compute cost |
| **A11 — No teacher distillation** | All phases, decoder trained from scratch | Expected large quality drop; decoder trained from scratch on Vimeo90k may not match DCVC teacher quality; validates teacher guidance is essential |
| **A12 — Lambda sweep** | N/A (ablation removed) | Removed: Phase 4 no longer contains a regression regularization term (Policy Network is frozen), making the lambda parameter unnecessary. |
| **A13 — Decoder LR ratio** | All phases, sweep ratio over {0.01, 0.05, 0.1, 0.5} | Validates 0.05 is near-optimal for Phase 4; too large (0.5) degrades decoder at untrained depths; too small (0.01) produces no meaningful adaptation |

**Expected pattern:** The largest quality drops should occur for A1 (random), A2 (zero ΔQ), and A5 (no scheduling). Smaller drops for A3 (fixed-depth vs learned ΔQ) and A7 (small Phase 4 benefit). This confirms that both the learned policy and the sequential training curriculum are contributing.

---

## 13. Required Content-Awareness Validation Experiments

Three experiments must be run to establish that the system is genuinely content-aware. These are not optional evaluations — they are the fundamental validation of the system's core claim. If any experiment fails, the approach must be reconsidered before proceeding.

**Unified success-criterion principle (Issue 4.4):** All three experiments share a single underlying hypothesis: **content variation in the test distribution must produce a measurable, learnable signal that maps to non-uniform depth selection**. The three experiments probe this hypothesis at three layers:
- **Experiment 1** ($\Delta Q$ vs. content in *training data*): tests whether the underlying signal exists.
- **Experiment 2** ($k^\star$ vs. content in *test data* at fixed $B$): tests whether the policy has learned to use the signal (training→test generalization).
- **Experiment 3** (K+1 operating points vs. static tiers): tests whether the content-adaptive scheduling benefits overall efficiency.

The thresholds ($|r| > 0.3$, $r > 0.2$, lie-on-or-above-tier-curve) are calibrated to the same variance-explained rule of thumb: $|r| = 0.3$ corresponds to $r^2 \approx 0.09$ (at least 9% variance explained), and $r = 0.2$ corresponds to $r^2 \approx 0.04$ (at least 4% variance explained). The Experiment 2 threshold is intentionally weaker than Experiment 1's threshold because Experiment 2 measures a learned-mapping outcome (which can be lossy due to training imperfections) while Experiment 1 measures raw correlation in the ground-truth $\Delta Q$ (which is independent of the policy's mapping accuracy).

**Distribution-shift remediation (train→test mismatch):** Experiment 1 is run on the training distribution (Vimeo90k + BVI-DVC), while Experiment 2 is run on the test distribution (UVG, HEVC, MCL-JCV). Distribution shift is expected and can cause Experiment 1 to pass while Experiment 2 fails. Remediation paths:
- If Experiment 1 passes but Experiment 2 fails (no correlation between $k^\star$ and content in test data), the policy has failed to generalize. **Remediation**: Re-run Phase 3 with a smaller policy capacity (Ablation A14 — capacity sweep) to reduce overfitting to training distribution; or add a domain-confusion regularizer to Phase 3.
- If Experiment 1 fails (no correlation in training), the design is fundamentally impossible. **Remediation**: Re-run Phase 1 with content-difficulty-weighted sampling (upweight frames with high-motion magnitude, downweight static frames); or report as "budget-adaptive only".
- If Experiment 2 fails but Experiment 1 passes at $B$ budget, but **passes at a different $B$**, the policy learned but its learned preference is budget-dependent. **Remediation**: Re-examine the Budget sampling distribution in Phase 3 (Issue 3.3 — already fixed to use a mixture distribution).

These remediation paths are documented as part of the training curriculum and are explicit procedures, not vague "try again" suggestions.

### Experiment 1 — ΔQ–Content Correlation (Run After Phase 2, Before Phase 3)

**Purpose:** Validate the core assumption that marginal quality gains $\Delta Q(k)$ correlate with observable bitstream features. If this correlation is weak, content-aware scheduling is fundamentally impossible regardless of architecture.

**Method:** For all training frames, compute $\Delta Q(k)$ at each depth (Phase 2 precomputation). For each frame, also record observable content features: latent energy, motion magnitude, temporal difference, frame type. Compute Pearson correlation between each content feature and each $\Delta Q(k)$ across frames.

**Success criterion:** At least one content feature has $|r| > 0.3$ correlation with at least one $\Delta Q(k)$. This threshold is task-specific: $|r| = 0.3$ means content features explain at least 9% of the variance in per-stage marginal gains ($r^2 \ge 0.09$). If $|r| < 0.3$ for all feature-stage pairs, the content features provide insufficient signal for the policy to reliably distinguish frames that benefit from additional stages — content-aware scheduling reduces to budget-aware scheduling only, and the core research claim must be reconsidered. There is no secondary threshold; the criterion is binary.

**Expected result:** Motion magnitude should correlate with early-stage $\Delta Q(1), \Delta Q(2)$; latent energy should correlate with gains across all stages.

---

### Experiment 2 — k* vs. Content at Fixed Budget

**Purpose:** Verify that the system allocates more compute to harder frames at any fixed budget $B$. This is the central content-awareness claim.

**Method:** For each fixed $B \in \{0.3, 0.5, 0.7\}$, run the full system on all test frames. Record $k^\star$ selected for each frame and its content features (latent energy, motion magnitude). Compute the correlation between content features and $k^\star$.

**Feasibility check (required before applying criterion):** Before computing correlations, verify that the budget $B$ permits at least 2 depth values for more than 50% of frames. Specifically, compute the fraction of frames where $C_\text{full} \cdot B$ allows $k \ge 1$ vs $k \ge 2$. If fewer than 2 depth values are feasible for the majority of frames, the budget is too tight to differentiate by content, and the correlation test is not applicable at that $B$.

**Success criterion:** At each fixed $B$ where feasibility check passes, frames with higher latent energy or motion magnitude receive higher $k^\star$ on average (positive correlation, $r > 0.2$). If feasibility check fails, the budget level is excluded from the correlation analysis — the system correctly responds to budget but has insufficient depth range to differentiate by content.

**Interpretation:** This is the definitive content-awareness test. If it fails at budgets where multiple depths are feasible, the policy has not learned to condition on content despite the architecture suggesting it should. The experiment must also report the distribution of $k^\star$ values at each $B$ to verify that depth variance exists.

---

### Experiment 3 — K+1 Operating Points vs. Static Tiers

**Purpose:** Demonstrate that the adaptive system produces $K+1$ quality-compute operating points that cover or match static execution tiers at every budget level, and that quality (not stage count) increases with budget.

**Method:** For $B \in \{0.2, 0.3, 0.44, 0.5, 0.67, 0.8, 0.89, 0.95, 1.0\}$, evaluate: (a) the proposed system, (b) static tier T1 (always 1 stage), (c) static tier T2 (always 2 stages), (d) static tier T3 (always 3 stages), (e) static tier T4 (always 4 stages), and (f) oracle (best possible at each budget). Plot PSNR vs. FLOPs ratio.

**Budget value rationale:** The values {0.44, 0.67, 0.89, 1.0} are the stage cost boundaries for $C_1:C_2:C_3:C_4 = 4:2:2:1$. The values {0.2, 0.3, 0.5, 0.8, 0.95} fall between boundaries to demonstrate that the proposed system has at most $K+1 = 5$ distinct operating points — identical to the static tier at depths between boundaries, with advantage appearing only near boundaries where content-adaptive selection differs from the static choice. The curve will appear as a staircase; this is correct behavior for a discrete-depth system.

**Quality monotonicity verification:** For each consecutive pair of budget levels $B_i < B_{i+1}$, verify that $\text{PSNR}(B_{i+1}) \ge \text{PSNR}(B_i)$ for the proposed system. This confirms O4 holds empirically even when $k^\star$ is not monotonic in $B$ (as noted in Section 6.4). Report any budget pairs where quality decreases; if this occurs for more than 5% of pairs, the system violates O4 and decoder training must be revisited.

**Success criterion:** (1) At budget levels where at least 2 depth values are feasible within the budget, the proposed system's curve lies on or above all static tier curves. At budgets where only one depth value is feasible, the proposed system is necessarily identical to the corresponding static tier. (2) Quality monotonicity: PSNR increases with budget for more than 95% of consecutive budget pairs.

The oracle is the ceiling per metric (PSNR, MS-SSIM, LPIPS separately); the proposed system should approach it.

---

## 14. Limitations and Future Work

### Limitations

**Dependency on teacher quality.** The Budget-Adaptive Decoder is trained via distillation from DCVC. If DCVC itself has systematic quality deficiencies at certain content types or bitrates, those deficiencies may propagate to our decoder. This is mitigated by the stage-aligned distillation, which supervises intermediate outputs, but the teacher ceiling remains a fundamental limitation.

**Sequential scheduling assumption.** The framework assumes that refinement stages must execute as a prefix — stage $t$ cannot be executed without stages $1, \ldots, t-1$. This is appropriate for sequential refinement decoders but would not apply to architectures with independent or parallel refinement paths. Future work could explore arbitrary subset scheduling for non-sequential decoder architectures.

**Budget mapping to device class.** At inference, the budget $B$ must be provided by the deployment environment. We assume this mapping (e.g., phone → 0.3, desktop → 0.9) is known and accurate. Incorrect budget specification would lead to suboptimal scheduling decisions.

**Fixed stage costs.** The stage costs $C_t$ are treated as constant and analytically determined. In practice, FLOP counts may vary slightly with input resolution, batch size, or hardware caching effects. We assume these variations are negligible relative to the inter-stage cost differences.

**Single codec instantiation.** The framework is demonstrated with DCVC as the teacher codec. Generalization to other learned video codecs (COVTC, TT-VC, etc.) requires verifying that the sequential decoder structure and stage count are compatible with the distillation and scheduling approach.

**Resolution generalization.** Phase 2 ΔQ targets are computed at Vimeo90k's 448×256 resolution. No analytical normalization is applied. Per-stage gain distributions at 1080p may differ from 448×256 due to the spatial correlation structure of codec errors. After training, prediction error is measured separately on Vimeo90k resolution and on 1080p test sequences. If the mean absolute prediction error on 1080p exceeds the 448×256 error by more than 0.5 dB, Phase 2 targets should be recomputed on a high-resolution training set.

### Future Work

**Non-sequential refinement.** A natural extension is to explore independent refinement experts — where each stage produces a complementary residual and the scheduler selects an arbitrary subset. This would require a more complex scheduler (potentially learned) but could enable richer compute-quality tradeoffs.

**Learned stage costs.** Instead of analytically determined stage costs, a learned cost predictor could estimate the actual FLOPs consumed at runtime, accounting for hardware-specific variations.

**Multi-frame temporal scheduling.** The current framework operates on a single frame independently. A richer extension would condition the Policy Network on temporal features (e.g., scene cuts, fade transitions) to adapt depth selection across frames.

**Region-level scheduling.** Instead of frame-level scheduling, spatial segmentation could allow different image regions to receive different compute budgets — allocating more computation to complex regions and less to simple regions within the same frame.

**On the validity of content-adaptive scheduling.** The system's two-layer contribution (budget-adaptive quality frontier always valid; content-adaptive scheduling gated on ΔQ heterogeneity) is documented in full in §6.4. In brief: when the trained decoder produces near-uniform ΔQ across stages and frames, the policy degenerates to a budget-only rule and the content-adaptive benefit disappears. This is expected for single-QP, single-content-class video. The content-adaptive component is most impactful for mixed content and/or multi-QP evaluation scenarios. The paper honestly reports both layers and uses the |r| < 0.3 gate from Experiment 1 to determine whether the system qualifies as content-adaptive or budget-adaptive only.

---

# Appendix A — Design Rationale

| Decision | Alternatives Considered | Chosen | Reason |
|----------|--------------------------|--------|--------|
| Optimization variable | Arbitrary subset $S^\star \subseteq \{1,\ldots,K\}$ | Prefix depth $k^\star \in \{0,\ldots,K\}$ | Aligns with sequential refinement dependency chain; eliminates invalid execution paths |
| Scheduler algorithm | Greedy, RL-based, approximate DP, threshold-based (free parameter) | Cumulative ΔQ maximization: k* = argmax_{k: C_0+Σ_{t=1}^k C_t ≤ B·(C_0+C_full)} Σ ΔQ(t) | Directly optimizes stated objective; no free parameters; O(K); exact for prefix structure |
| Decoder family | Independent refinement experts (Family B) | Sequential refinement (Family A) | Isolates scheduler contribution; preserves DCVC compatibility; lower implementation risk |
| Policy output | Stage utility estimates $U_t$ | Marginal quality gains $\Delta Q(k)$ via regression | Directly interpretable; training is pure supervised regression; no discrete gradient problem |
| Policy input | Decoder state $s_t = (F_t, M, R_t)$ (post-decode) | Bitstream features pre-decoding (latent, motion, prev_frame) | No circular dependency; scheduling decision before any decode compute; zero extra FLOPs |
| Budget representation | Discrete tiers, continuous with arbitrary sampling | Continuous $\text{Beta}(2,2)$ during training; discrete at deployment | Prevents controller collapse; standard distribution; maps cleanly to device classes at inference |
| Stage costs | Equal across stages | $C_1 \ge C_2 \ge \cdots \ge C_K$ | Reflects that coarse reconstruction is expensive; fine refinement is cheap |
| Teacher instantiation | Generic learned codec | DCVC | Sequential decoder structure; publicly available; strong baseline quality |
| Feature aggregation | Global average pooling (fixed) | BitstreamContentExtractor: latent energy + motion complexity + temporal diff + frame type | All signals available pre-decoding; interpretable; no decoder compute required |
| Distillation mapping | Stage-aligned (student stage $k$ ↔ teacher stage $k$) | **Hybrid**: Stage 1 ↔ teacher reconstruction; Stages 2-4 ↔ ground truth (weights `[0.5, 0.75, 1.0]`) | DCVC has K=1 (single-stage output); stage-aligned mapping is therefore impossible; hybrid mapping makes Stage 1 a clean teacher replica and Stages 2-4 progressively exceed it. Requires gradient detach between Stage 1 and Stage 2 (C1 fix) to prevent Stage 2-4 GT-loss from corrupting Stage 1's teacher signal |
| Training phases | Joint end-to-end training | Four-phase: Phase 1 (decoder distill) → Phase 2 (ΔQ precompute) → Phase 3 (policy regression) → Phase 4 (decoder adaptation) | Decoder trained before ΔQ targets are computed; Phase 2 targets are data-driven; Phase 4 gradient does not flow to policy |
| Gradient through scheduling | Backprop through argmax / REINFORCE | No gradient through scheduling; policy trained via regression to precomputed ΔQ targets | Phase 3 is pure supervised regression; no discrete gradient approximation needed |
| Phase 4 decoder learning rate | Same as policy | $\eta_\text{decoder} = 0.05 \cdot \eta_\text{policy}$ | Adapter-style; validated on held-out set; A13 sweeps over {0.01, 0.05, 0.1, 0.5} |
| Monotonicity guarantee | Asserted from training dynamics | Empirical verification: after Phase 1, PSNR(k) ≥ PSNR(k-1) for >99% of frame-depth pairs | softplus gate does not guarantee PSNR improvement; δ is unconstrained; replaced with empirical check |
| Collapse prevention | Explicit entropy/regularization penalty | Budget-conditioned supervision (budget is input) + Phase 2 ΔQ precomputation | Cleaner; budget input prevents mode collapse; Phase 2 targets are data-driven, not policy-generated |
| Scheduler search space | $2^K = 16$ arbitrary subsets | $K+1 = 5$ depth values | Prefix constraint matches sequential dependency; 5 candidates is trivially small |
| Quality metric primary | LPIPS, MS-SSIM | PSNR | Industry standard; directly interpretable; matches training loss |
| Evaluation baselines | Only DCVC | Fixed-tier + heuristic + oracle + prior work | Isolates each contribution (scheduler vs. decoder vs. training) |
| Content-awareness experiments | Absent from earlier designs | Three required experiments: ΔQ-content correlation, k* vs content at fixed B, K+1 operating points vs tiers | Core claims must be tested directly, not assumed |