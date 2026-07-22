# Budget-Constrained Neural Decoder Scheduling for Learned Video Compression

## A Research and System Design Specification

---

# Part I — Research Foundation

## 1. Motivation

Modern video decoding spans heterogeneous client devices with widely varying compute budgets. The same encoded bitstream may be decoded on a battery-constrained mobile phone, a mid-range laptop, a desktop workstation, or an edge appliance — each tier providing a different envelope of available FLOPs, memory, and per-frame latency for the same reconstruction task. Learned video codecs have favorably reshaped rate-distortion performance, but they impose a structural assumption inherited from their classical counterparts: the decoder must execute the full reconstruction pipeline in order to reconstruct the frame. The full pipeline is run on every device, on every frame, whether the device can afford it or whether the content demands it. The result is a misalignment between deployment reality and the compute footprint of the decoder itself.

The mismatch is twofold. Constrained devices are forced to execute reconstruction work whose cost was never tailored to their available envelope; capable devices, conversely, do not differentiate their compute expenditure across content. A static talking-head frame and a fast-motion sports frame are processed with the same decoder compute, even though the content difficulty, the residual signal, and the resulting reconstruction sensitivity differ substantially. Adaptive mechanisms in today's learned codecs operate almost exclusively on the encoder side — through learned bitrate selection, frame-level quality-mode switching, or quantization adaptation — while the decoder itself behaves as a fixed computational object. The decoder does not know the device it runs on. It does not know the content it is reconstructing. It executes its full reconstruction graph regardless.

This problem becomes increasingly important as learned video codecs transition from research prototypes to deployment across increasingly heterogeneous client hardware. It suggests a reconsideration of the decoder's design. Rather than treating the decoder as a fixed pipeline that all devices must run clumsily, the decoder should adapt its compute allocation to the dual constraints imposed by deployment: the available compute budget on the host device, and the intrinsic difficulty of the frame currently being reconstructed. The central question this work asks is thus:

> *Given a runtime compute budget and the current decoder state, how should a neural video decoder allocate computation across its refinement stages to maximize reconstruction quality?*

The motivation is not anchored to any specific application. It applies equally to video-on-demand streaming, live conferencing, cloud-to-edge adaptive decoding, and edge-class inference. What unites them is the underlying phenomenon — heterogeneous devices, varying per-frame content — and the same architectural question arises across all of them.

---

## 2. Research Gap

The literature has produced a rich set of complementary approaches that each touch portions of this question without jointly addressing it.

**Adaptive inference.** Early-exit networks, slimmable networks, and conditional-computation methods demonstrate per-input modulation of compute in classification and recognition tasks. They establish that neural networks need not consume the same FLOPs for every input. They do not, however, address the specific structure of learned video decoding, where refinement operates across a sequential decoding graph and where the bitstream is itself already compressed under a coder.

**Efficient learned video codecs.** Compact-decoder designs, slim architectures, mixed-precision decoders, reduced-channel-count codecs, and knowledge-distilled codecs target the fixed-compute quality-compression frontier. These methods accept a single compute footprint and compress its quality-efficiency frontier. They do not adaptively allocate that footprint across hosts or frames.

**Content-adaptive coding.** Adaptive quantization, learned bitrate ladders, and content-conditional frame-level mode selection modulate the bitstream in response to content. The decoder, however, remains fixed. The adaptation runs on the encoder/orchestration side, not on the decoder execution side.

**Dynamic neural networks.** Conditional computation, mixture-of-experts, and input-dependent routing demonstrate per-input graph selection for deep networks. They generally assume a backbone whose compute at full execution is significantly larger than the budgeted subset, and they target default discriminative tasks — not the sequential, stateful reconstruction structure of video decoding with its latent, context, and motion couplings.

**Constraint-aware inference.** Budget-constrained neural network execution has been studied under energy, latency, and FLOPs budgets. These formulations typically target inference pipelines with homogeneous operations, single-network policies, and discriminative tasks — and rarely concern themselves with the bitstream-decoder coupling that defines video coding.

To the best of our knowledge, existing approaches generally do not formulate decoder execution as a constrained optimization problem that jointly considers runtime compute budget and current decoder state — applied to a sequential refinement decoder whose decoding graph is stateful across stages and whose reconstruction is conditioned on the motion and context produced earlier in the codec. The gap, in short, is the missing joint formulation: the decoder should be scheduled as an optimizer subject to a budget, conditioned on content.

---

## 3. Problem Formulation

We model the video codec's decoder as a sequential Budget-Adaptive Decoder. Given the latent, motion, and context already produced by the codec, the Budget-Adaptive Decoder reconstructs the frame through a sequence of refinement stages. Let:

- $K$ denote the number of refinement stages in the decoder. For the remainder of this work, we set $K = 4$. The formulation is stated generally over $K$; this work instantiates $K = 4$ to enable exhaustive scheduling and clean ablation studies.
- $s_t = (F_t, M, R_t)$ denote the decoder state at refinement stage $t \in \{1, \ldots, K\}$, where $F_t$ is the current decoder feature tensor, $M$ is the motion/context representation computed once earlier in the codec and reused across stages, and $R_t$ is the current reconstructed frame at stage $t$.
- $C_{\max}$ denote the maximum computation currently available to the decoder at runtime (FLOPs).
- $C_t$ denote the analytical FLOP count of executing refinement stage $t$. Stage costs satisfy $C_1 \ge C_2 \ge \cdots \ge C_K$; the earliest stage dominates compute, with later stages becoming progressively cheaper.
- $C_\text{full} = \sum_{t=1}^{K} C_t$ denote the full decoder compute.
- $B \in (0, 1]$ denote the runtime budget ratio, defined as the maximum compute currently available divided by the full decoder compute: $B = C_{\max} / C_\text{full}$. $B = 1$ corresponds to executing all stages; smaller $B$ corresponds to a more constrained device.

The Budget-Adaptive Decoder executes a *prefix* of refinement stages. Specifically, if $k \in \{0, 1, \ldots, K\}$ stages are executed, they are always stages $\{1, 2, \ldots, k\}$ — stage $t$ depends on the output of stage $t-1$, so skipping an intermediate stage is not feasible. The compute incurred is $C(k) = \sum_{t=1}^{k} C_t$.

Let $Q(k; s_1)$ denote the reconstruction quality realized by executing the first $k$ refinement stages sequentially, starting from initial decoder state $s_1$. The exact computation of $Q(k; s_1)$ is unavailable at inference. The *Policy Network* $f_\theta$, a lightweight module, produces stage utility estimates given current decoder state and runtime budget:
$$U_t = f_\theta(s_t, B),$$
one scalar per refinement stage. These utilities are surrogate estimates used by the scheduler to approximate each stage's marginal contribution — not an assumption of independence or additivity.

The *Scheduler* $\Omega$ takes the utility estimates $\{U_1, \ldots, U_K\}$ together with the stage costs $\{C_1, \ldots, C_K\}$ and budget $B$, and produces the execution depth:
$$k^\star = \Omega(\{U_t\}, \{C_t\}, B).$$
The scheduler approximates the solution of the constrained optimization problem:
$$
\begin{aligned}
k^\star \;=\; \arg\max_{k \in \{0, \ldots, K\}} \quad & Q(k; s_1) \\
\text{subject to} \quad & \sum_{t=1}^{k} C_t \;\le\; B \cdot C_\text{full},
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

**O2 — Minimal modification of the underlying codec.** The system should build upon an established video codec without redesigning its encoder, entropy model, or reconstruction pipeline. This isolates the contribution to the scheduling framework and preserves bitstream compatibility. In this work, the codec is instantiated as DCVC; the formulation generalizes to any codec with a sequential refinement decoder.

**O3 — Lightweight runtime overhead.** Decisions about which stages to execute must be made with negligible computational cost relative to the decoder itself. If scheduling overhead is comparable to decoding overhead, the adaptation provides no practical benefit. The Policy Network is therefore designed to be minimal — on the order of tens of thousands of parameters — and the Scheduler performs only simple arithmetic comparisons.

**O4 — Monotonic quality improvement with increasing compute.** If $B_1 > B_2$, the system executing under budget $B_1$ should produce reconstruction quality at least as good as that under $B_2$. This property ensures that the scheduling framework is monotonic with respect to available compute — a natural requirement for any budget-adaptive system — and follows from the sequential structure of the Budget-Adaptive Decoder.

**O5 — Exact scheduling under a small search space.** The constrained optimization problem over depth values $k \in \{0, \ldots, K\}$ admits $K + 1$ feasible solutions. For $K = 4$, the search space contains only 5 depths, enabling exact evaluation rather than approximation. This eliminates any concern about scheduler suboptimality and makes quality differences directly attributable to the Policy Network's utility estimates.

**O6 — Compatibility with teacher-guided training.** The Budget-Adaptive Decoder must be trainable via distillation from a teacher codec, and the Policy Network must be trainable via budget-conditioned supervision, without architectural modifications. Both training paradigms must coexist within the same framework.

These six objectives collectively determine the structure of every component in Part II. A component that violates any objective is rejected; a component that satisfies all six objectives is retained.

---

## 5. Overall System Architecture

The system consists of four principal components, arranged in a cascade:

```
                    ┌─────────────────────────────────┐
                    │         DCVC (Teacher)          │  (frozen)
                    │   Baseline codec for distillation │
                    └────────────────┬──────────────────┘
                                     │ intermediate features + final output
                                     ▼
┌──────────┐                ┌─────────────────────────────────┐
│  Budget  │────────────────│      Policy Network f_θ         │
│    B     │                │   s_t, B  →  {U_t} (utilities)  │
└──────────┘                └──────────────┬──────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────┐
                    │          Scheduler Ω             │
                    │  {U_t}, {C_t}, B  →  k^\star     │
                    └──────────────┬──────────────────┘
                                   │ selected depth k*
                                   ▼
                    ┌─────────────────────────────────┐
                    │   Budget-Adaptive Decoder       │
                    │   Base reconstruction + first k* │
                    │   sequential refinement stages  │
                    └──────────────┬──────────────────┘
                                   │
                                   ▼
                              Ŷ (reconstruction)
```

**DCVC (Teacher):** A pretrained learned video codec used exclusively during training. Provides intermediate reconstruction targets at each refinement depth and the final output. Not used during inference.

**Policy Network $f_\theta$:** A lightweight module that maps the current decoder state $s_t = (F_t, M, R_t)$ and budget $B$ to a utility estimate $U_t$ for each refinement stage. Satisfies O1 (budget-aware), O3 (lightweight), O5 (exact optimization downstream).

**Scheduler $\Omega$:** Takes the full set of utility estimates, stage costs, and budget, and selects the optimal depth $k^\star$ by solving the constrained selection problem over $\{0, \ldots, K\}$. Satisfies O5 (exact for $K=4$). Contains no learned parameters.

**Budget-Adaptive Decoder:** The execution engine. Consists of a base reconstruction stage followed by $K = 4$ sequential refinement stages. Only the first $k^\star$ stages are executed at inference. Satisfies O2 (minimal codec modification), O4 (monotonic with budget), O6 (distillation-compatible).

The Policy Network and Scheduler are execution-time components; they run during every inference forward pass. The teacher is used only during training.

---

## 6. Component Design

### 6.1 DCVC as the Teacher Codec

The teacher codec provides two functions during training: (i) intermediate reconstruction targets at each refinement depth, which supervise the Budget-Adaptive Decoder during Phase 1 (distillation), and (ii) the final reconstruction target, which supervises the full system during Phase 2–3 (policy training).

We instantiate the teacher as DCVC, a state-of-the-art learned video codec that decomposes video compression into motion estimation, context encoding, latent encoding, and sequential refinement decoding. DCVC's decoder already employs a multi-stage refinement pipeline, making it directly compatible with the stage-aligned distillation strategy described in Section 8.

**Inputs:** Previous decoded frame, current bitstream (entropy-decoded latent and motion representations).  
**Outputs:** Per-stage reconstruction targets $\{R_1^\star, R_2^\star, R_3^\star, R_4^\star\}$ and the final decoded frame.  
**Role in training:** Frozen. Produces teacher supervision signals without being modified by the training loss.  
**Role at inference:** None. The teacher is used exclusively during training.

The teacher is not involved at inference. The system runs independently of the bitstream's encoder side; only the decoder-side computation is being scheduled.

---

### 6.2 Policy Network $f_\theta$

The Policy Network is the only learned component that runs at inference. It must produce stage utility estimates $\{U_1, \ldots, U_K\}$ that guide the scheduler toward high-quality reconstructions.

**Inputs:**
- $F_t$: the decoder feature tensor after stage $t - 1$ (or after base reconstruction for $t = 1$)
- $M$: the motion/context representation (constant across all stages)
- $R_t$: the current reconstruction after stage $t - 1$
- $B$: the runtime budget ratio, provided by the deployment environment

These four inputs are each reduced via a lightweight global feature aggregation — which may be global average pooling, attention pooling, or a learned projection during implementation — to a scalar, concatenated, and passed through a small MLP. The aggregation ensures that the Policy Network's decision is based on global content characteristics — motion intensity, residual energy, reconstruction confidence — rather than spatial details that may not generalize across resolutions.

**Architecture:** The MLP consists of two hidden layers with approximately 32–64 units each, yielding a total parameter count in the range of 20k–50k. This is intentionally minimal. The Policy Network should not learn complex representations; it should learn a mapping from aggregated decoder state and budget to per-stage utility estimates.

**Output:** A vector $[U_1, \ldots, U_K]$ of stage utilities, one per refinement stage. The utilities are produced as unbounded real values; they are compared only by relative magnitude during scheduling.

**Role in training:** Trained via budget-conditioned supervision in Phase 2 and fine-tuned jointly in Phase 3. During Phase 1, gradients do not flow to $f_\theta$; the decoder learns via direct distillation loss.

**Complexity:**

| Component | Computational Complexity |
|-----------|--------------------------|
| Policy Network | $O(P)$ where $P$ is the number of parameters (~20k–50k) |

**Design rationale:** A single unified module consuming all four inputs jointly — rather than separate complexity-predictor and controller modules — avoids redundancy. The network learns to extract whatever features (motion magnitude, residual energy, feature variance) are predictive of stage utility, without explicit decomposition.

**Alternatives considered:** A two-module design with a complexity predictor followed by a budget-conditioned controller was rejected because it separates information that is inherently coupled (the budget must condition the utility estimate, so separating the modules offers no advantage and introduces extra parameters). Direct prediction of $k^\star$ via a classification head was rejected because it eliminates the interpretable stage-utility abstraction and makes the loss indirect. Heuristic complexity estimators (motion magnitude, residual variance) were rejected because they are not learnable end-to-end and cannot adapt to the codec's specific error patterns.

---

### 6.3 Scheduler $\Omega$

The Scheduler turns utility estimates into an execution depth. It solves the constrained optimization problem using the surrogate utilities provided by $f_\theta$ as proxies for the true quality function.

**Inputs:** Utility estimates $\{U_1, \ldots, U_K\}$, stage costs $\{C_1, \ldots, C_K\}$, budget $B$.

**Algorithm:** For $K = 4$, the scheduler evaluates all $K + 1 = 5$ feasible depths $\{0, 1, 2, 3, 4\}$. For each depth $k$, it checks the constraint $\sum_{t=1}^{k} C_t \le B \cdot C_\text{full}$. The scheduler ranks candidate execution policies using the surrogate utility estimates produced by the Policy Network; specifically, it selects the largest $k$ such that the constraint is satisfied. This is an exact algorithm for the sequential refinement case.

| Component | Computational Complexity |
|-----------|--------------------------|
| Scheduler | $O(K)$ — evaluates at most $K + 1$ depth values |

**Role during training:** The same as at inference. The scheduler is never trained; it is a deterministic function of its inputs.

**Design rationale:** A greedy scheduler that selects stages in descending utility order until budget is exhausted would produce the same result as the exact solver for the prefix case and is simpler to implement. An RL-based scheduler was rejected because it introduces training complexity and variance without benefit in the small-search-space regime ($K = 4$). Exact evaluation of all depths is optimal by definition and requires no approximation.

---

### 6.4 Budget-Adaptive Decoder

The Budget-Adaptive Decoder is the execution engine of the system. It consists of a base reconstruction stage followed by $K = 4$ sequential refinement stages, each with decreasing computational cost $C_1 \ge C_2 \ge C_3 \ge C_4$.

**Base Reconstruction:** Produces $s_1 = (F_1, M, R_1)$ from the codec's latent, motion, and context. This stage is always executed regardless of budget, because no reconstruction is possible without it.

**Refinement Stages $R_1, \ldots, R_K$:** Sequential stages, each operating on the feature tensor produced by the previous stage. Each refinement stage updates the decoder feature representation and reconstruction. Only stages $1, \ldots, k^\star$ selected by $\Omega$ are executed.

**Stage costs:** The base stage incurs cost $C_\text{base}$. Each refinement stage incurs cost $C_t$, with $C_1 \ge C_2 \ge C_3 \ge C_4$. Exact values are determined by analytical FLOP accounting once the convolution specifications are frozen during implementation; the relative ordering $C_1 \ge C_2 \ge C_3 \ge C_4$ is the only constraint required by the formulation.

| Component | Computational Complexity |
|-----------|--------------------------|
| Decoder (full) | $O(C_\text{base} + \sum_{t=1}^{K} C_t) = O(C_\text{full})$ |

**Outputs:** The final reconstruction $\widehat{Y} = R_{k^\star+1}$ where $k^\star$ is the depth selected by $\Omega$.

**Role during training:** In Phase 1, the Budget-Adaptive Decoder is trained via stage-aligned distillation from the DCVC teacher. In Phase 2, its parameters are frozen while the Policy Network trains. In Phase 3, both are fine-tuned jointly.

**Design rationale:** Keeping the decoder architecture fixed — and making only the *execution depth* adaptive — satisfies O2 (minimal codec modification) and O6 (distillation-compatible). The sequential structure satisfies O4 (monotonic quality with increasing budget) because executing additional stages can only improve or maintain quality, never degrade it. The decreasing stage costs satisfy O5 (exact scheduling): the search space $\{0, \ldots, K\}$ is small enough for exact evaluation.

**Alternatives considered:** Independent refinement experts — where each stage predicts a complementary residual independently and the scheduler selects a subset — were rejected for the first publication because they simultaneously introduce a new decoder architecture AND a new scheduling framework, making it impossible to isolate whether quality improvements come from the decoder design or from the scheduling policy. The sequential Budget-Adaptive Decoder, by contrast, uses DCVC's base architecture and adds only the scheduling layer, cleanly isolating the contribution to the scheduling framework.

---

## 7. Operational Flow

The complete system execution pipeline, end-to-end, is as follows:

**At training time (Phase 1 — distillation):**
1. The DCVC teacher produces intermediate reconstruction targets at each refinement depth: $R_1^\star, R_2^\star, R_3^\star, R_4^\star$.
2. The Budget-Adaptive Decoder produces its own reconstructions at each depth: $R_1, R_2, R_3, R_4$.
3. A per-stage distillation loss compares each Budget-Adaptive Decoder output to the corresponding teacher output.
4. The Policy Network is frozen; only the decoder learns via the distillation loss.

**At training time (Phase 2 — policy training):**
1. A runtime budget $B$ is sampled from a distribution (e.g., $\text{Beta}(2, 2)$).
2. The base reconstruction executes, producing decoder state $s_1$.
3. The Policy Network evaluates $f_\theta(s_1, B)$ to produce utility estimates $U_1, \ldots, U_K$.
4. The Scheduler selects depth $k^\star = \Omega(\{U_t\}, \{C_t\}, B)$.
5. The Budget-Adaptive Decoder executes stages $1, \ldots, k^\star$ and produces $\widehat{Y}$.
6. The decoder is frozen; the Policy Network receives gradients based on reconstruction quality under the selected budget.

**At training time (Phase 3 — joint fine-tuning):**
1. Steps 1–6 above are repeated.
2. Both the Policy Network and the Budget-Adaptive Decoder receive gradients, with the decoder learning rate much smaller than the Policy Network learning rate to preserve Phase 1 specialization.

**At inference time:**
1. A runtime budget $B$ is provided by the deployment environment (e.g., mapped from device class: phone → 0.3, desktop → 0.9).
2. The base reconstruction executes, producing decoder state $s_1$.
3. The Policy Network evaluates $f_\theta(s_1, B)$ to produce utility estimates.
4. The Scheduler selects depth $k^\star$.
5. The Budget-Adaptive Decoder executes stages $1, \ldots, k^\star$ and returns the final reconstruction.
6. The total FLOPs consumed are $\sum_{t=1}^{k^\star} C_t$, which never exceeds $B \cdot C_\text{full}$.

---

# Part III — Learning Framework

## 8. Training Methodology

The training program consists of three sequential phases. Each phase freezes components from the previous phase, preventing gradient interference and enabling staged specialization.

### Phase 1 — Budget-Adaptive Decoder Distillation

**Objective:** Train the Budget-Adaptive Decoder to match the DCVC teacher at each refinement depth.

In this phase, the Policy Network is detached (no gradients flow to it). The Budget-Adaptive Decoder learns purely through stage-aligned distillation from the DCVC teacher.

**Teacher supervision:** The DCVC teacher produces reconstruction targets at each stage: $R_1^\star, R_2^\star, R_3^\star, R_4^\star$. The Budget-Adaptive Decoder produces its own reconstructions $R_1, R_2, R_3, R_4$ by executing its stages sequentially (always executing all $K = 4$ stages during distillation). The loss at each stage $k$ is:
$$\mathcal{L}_\text{distill}^{(k)} = \|R_k - R_k^\star\|_2^2$$
where $\|\cdot\|_2^2$ denotes the mean squared error. The total Phase 1 loss is:
$$\mathcal{L}_\text{Phase 1} = \sum_{k=1}^{K} \mathcal{L}_\text{distill}^{(k)}.$$

**Why all stages during distillation:** The Budget-Adaptive Decoder must learn to produce high-quality reconstructions at every depth, not just at the depths it expects to use at inference. This ensures that when the Policy Network selects depth $k^\star$, the decoder is capable of producing the best possible reconstruction at that depth.

**Frozen components:** DCVC teacher (always frozen), Policy Network (detached).  
**Trainable components:** Budget-Adaptive Decoder.  
**Training signal:** Stage-aligned MSE to teacher reconstruction targets.

**Expected outcome:** After Phase 1, the Budget-Adaptive Decoder, when executing all 4 stages, produces reconstructions comparable to the DCVC teacher. At intermediate depths, the decoder's outputs approximately match the corresponding teacher stage outputs.

---

### Phase 2 — Policy Network Training

**Objective:** Train the Policy Network to select execution depths that maximize reconstruction quality under a given budget.

In this phase, the Budget-Adaptive Decoder is frozen. The Policy Network learns to predict utility estimates that guide the Scheduler toward high-quality reconstructions within the runtime budget.

**Budget sampling:** At each training step, a budget $B$ is sampled from a Beta distribution, $\text{Beta}(2, 2)$. This distribution is symmetric and peaks at $B = 0.5$, providing rich training signal across the full range of device capabilities while avoiding extreme values that could lead to degenerate policies.

**Forward pass:**
1. Budget $B$ is sampled.
2. Base reconstruction executes, producing decoder state $s_1 = (F_1, M, R_1)$.
3. The Policy Network evaluates $U_t = f_\theta(s_t, B)$ for each $t \in \{1, \ldots, K\}$, where $s_t$ is the decoder state at the start of stage $t$ (for $t > 1$, this is the state produced by executing the previous $t-1$ stages with gradients detached — the decoder runs in inference mode with respect to the Policy Network's gradient flow).
4. The Scheduler selects depth $k^\star = \Omega(\{U_t\}, \{C_t\}, B)$.
5. The Budget-Adaptive Decoder executes stages $1, \ldots, k^\star$ (with gradients flowing only to the Policy Network).

**Loss:** The reconstruction quality is compared to the ground-truth frame using mean squared error:
$$\mathcal{L}_\text{Phase 2} = \| \widehat{Y} - Y^\star \|_2^2,$$
where $\widehat{Y}$ is the reconstruction produced under budget $B$ and $Y^\star$ is the ground-truth frame. The Policy Network is updated via backpropagation through this loss; gradients do not flow to the Budget-Adaptive Decoder (frozen).

**Why no collapse penalty is needed:** The budget $B$ is provided explicitly as an input to the Policy Network. The Policy Network learns to select depths appropriate to the budget — it cannot "forget" the budget because the budget is a direct input. This is in contrast to approaches where a network must internally infer the budget, which can lead to mode collapse.

**Frozen components:** DCVC teacher, Budget-Adaptive Decoder.  
**Trainable components:** Policy Network.  
**Training signal:** Budget-conditioned MSE on reconstruction output.

**Expected outcome:** After Phase 2, the Policy Network has learned to predict utility estimates that result in appropriate depth selection across the full budget range. At low budgets it selects small depths; at high budgets it selects large depths.

---

### Phase 3 — Joint Fine-Tuning

**Objective:** Allow slight adaptation of the decoder to the policy's behavior, while preserving the specialization learned in Phase 1.

Both the Policy Network and the Budget-Adaptive Decoder are trainable, but with significantly different learning rates. The decoder's learning rate is set to a small fraction of the Policy Network's learning rate (e.g., $\eta_\text{decoder} = 0.05 \cdot \eta_\text{policy}$), following an adapter-style fine-tuning rationale. This prevents the decoder from forgetting the teacher-aligned representations learned in Phase 1 while allowing it to make small adjustments that improve quality under the policy's selected depths.

**Budget sampling:** Same as Phase 2 — $\text{Beta}(2, 2)$.

**Forward pass:** Same as Phase 2.

**Loss:** Same as Phase 2 — MSE on reconstruction under selected budget.

**Frozen components:** DCVC teacher.  
**Trainable components:** Policy Network (learning rate $\eta_\text{policy}$), Budget-Adaptive Decoder (learning rate $0.05 \cdot \eta_\text{policy}$).  
**Training signal:** Budget-conditioned MSE, same as Phase 2.

**Expected outcome:** After Phase 3, the system is jointly optimized. Small decoder adjustments improve quality under policy-selected depths without degrading the decoder's general capability.

---

## 9. Training Curriculum

The three phases are executed sequentially:

| Phase | Steps | Frozen Components | Trainable Components | Budget Sampling |
|-------|-------|-------------------|----------------------|-----------------|
| 1 — Distillation | Until decoder convergence | Teacher, Policy Network | Budget-Adaptive Decoder | N/A (all stages) |
| 2 — Policy Training | Until policy convergence | Teacher, Budget-Adaptive Decoder | Policy Network | $\text{Beta}(2, 2)$ |
| 3 — Joint Fine-Tuning | Fixed epoch count | Teacher | Policy + Decoder (diff. l.r.) | $\text{Beta}(2, 2)$ |

**Phase 1 termination:** Monitor distillation loss on a validation set. Stop when per-stage MSE falls below a threshold or when validation loss plateaus.

**Phase 2 termination:** Monitor reconstruction MSE under random budget sampling on a validation set. Stop when validation quality plateaus.

**Phase 3 termination:** Fixed number of epochs (determined empirically; typically fewer than Phase 1 or 2 due to adapter-style small learning rate).

---

## 10. Inference Pipeline

At inference time, the system operates entirely without the teacher and without gradient computation. The execution path is:

**Input:** Bitstream (entropy-decoded latent and motion), previous decoded frame, runtime budget $B$.

**Step 1 — Base reconstruction:** The base reconstruction stage executes, producing decoder state $s_1 = (F_1, M, R_1)$. Motion/context $M$ is obtained from the codec's motion estimation pipeline.

**Step 2 — Utility estimation:** The Policy Network evaluates $U_t = f_\theta(s_t, B)$ for $t = 1, \ldots, K$. The decoder state $s_t$ used here is the state at the start of stage $t$, obtained from a forward pass through the decoder with gradients disabled.

**Step 3 — Scheduling:** The Scheduler takes $\{U_t\}$, $\{C_t\}$, and $B$, and computes $k^\star = \Omega(\{U_t\}, \{C_t\}, B)$. This is a simple numerical comparison — find the largest $k$ such that $\sum_{t=1}^{k} C_t \le B \cdot C_\text{full}$.

**Step 4 — Execution:** The Budget-Adaptive Decoder executes stages $1, \ldots, k^\star$ in sequence and returns the final reconstruction $\widehat{Y}$.

**Runtime complexity:** The Policy Network and Scheduler together add negligible overhead compared to the decoder itself. For $K = 4$, the scheduler performs at most 5 cost comparisons; the Policy Network is a small MLP consuming ~20k–50k parameters.

**Memory:** No additional memory beyond the Policy Network parameters and the intermediate decoder states required for the selected depth. The system does not store results for unexecuted stages.

**Latency:** The Policy Network adds a single forward pass (milliseconds on modern hardware). The Scheduler is $O(1)$. Total decoding latency is the sum of base reconstruction plus the FLOPs of the selected stages.

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
| Custom hard dataset | Fast motion, occlusions, scene cuts | Failure mode analysis |

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
- **Quality-compute Pareto frontier:** The set of (FLOPs, quality) pairs for which no other method dominates.

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
- **Oracle-per-frame:** Optimal depth selection chosen with full knowledge of ground-truth quality (for RQ2 upper bound).
- **Oracle-budget:** Best quality achievable at exactly the given budget (for RQ1 upper bound).

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

All reported results are averaged over at least 3 independent runs with different random seeds. Significance testing uses a paired t-test with Bonferroni correction for multiple comparisons. Confidence intervals are reported at the 95% level.

---

## 12. Ablation Strategy

Each architectural and training design choice is validated individually. The ablation plan covers 11 components:

| Ablation | Description | Expected Outcome |
|----------|-------------|------------------|
| **A1 — No Policy Network (random)** | Random depth selection at each frame | Establishes baseline; RQ2 answer without learned policy |
| **A2 — No Policy Network (uniform utility)** | Policy Network outputs constant utility | Tests whether learned utility vs. fixed utility matters |
| **A3 — Greedy scheduler** | Replace exact solver with greedy stage selection | Tests whether exact scheduling matters vs. heuristic |
| **A4 — Uniform budget sampling** | Replace Beta(2,2) with uniform sampling | Tests whether biased sampling prevents collapse |
| **A5 — Phase 1 only** | Train decoder via distillation, evaluate with policy removed | Tests decoder quality without scheduling |
| **A6 — Phase 2 only** | Train policy without Phase 3 joint fine-tuning | Tests contribution of Phase 3 adaptation |
| **A7 — Phase 1+2, no Phase 3** | Same as A6 but with decoder distillation | Baseline before joint fine-tuning |
| **A8 — Stage cost equality** | Set $C_1 = C_2 = C_3 = C_4$ | Tests whether decreasing-cost design matters |
| **A9 — Fewer stages ($K=2$)** | Reduce decoder to 2 refinement stages | Tests sensitivity to stage count |
| **A10 — More stages ($K=5$)** | Increase decoder to 5 refinement stages | Tests whether more stages improve the Pareto frontier |
| **A11 — No teacher distillation** | Train decoder from scratch without DCVC teacher | Tests whether teacher guidance is essential |

**Expected pattern:** The largest quality drops should occur for A1 (random), A2 (uniform utility), and A5 (no scheduling). Smaller drops for A3 (greedy vs. exact) and A6 (Phase 3 contribution). This confirms that both the learned policy and the sequential training curriculum are contributing.

---

## 13. Limitations and Future Work

### Limitations

**Dependency on teacher quality.** The Budget-Adaptive Decoder is trained via distillation from DCVC. If DCVC itself has systematic quality deficiencies at certain content types or bitrates, those deficiencies may propagate to our decoder. This is mitigated by the stage-aligned distillation, which supervises intermediate outputs, but the teacher ceiling remains a fundamental limitation.

**Sequential scheduling assumption.** The framework assumes that refinement stages must execute as a prefix — stage $t$ cannot be executed without stages $1, \ldots, t-1$. This is appropriate for sequential refinement decoders but would not apply to architectures with independent or parallel refinement paths. Future work could explore arbitrary subset scheduling for non-sequential decoder architectures.

**Budget mapping to device class.** At inference, the budget $B$ must be provided by the deployment environment. We assume this mapping (e.g., phone → 0.3, desktop → 0.9) is known and accurate. Incorrect budget specification would lead to suboptimal scheduling decisions.

**Fixed stage costs.** The stage costs $C_t$ are treated as constant and analytically determined. In practice, FLOP counts may vary slightly with input resolution, batch size, or hardware caching effects. We assume these variations are negligible relative to the inter-stage cost differences.

**Single codec instantiation.** The framework is demonstrated with DCVC as the teacher codec. Generalization to other learned video codecs (COVTC, TT-VC, etc.) requires verifying that the sequential decoder structure and stage count are compatible with the distillation and scheduling approach.

### Future Work

**Non-sequential refinement.** A natural extension is to explore independent refinement experts — where each stage produces a complementary residual and the scheduler selects an arbitrary subset. This would require a more complex scheduler (potentially learned) but could enable richer compute-quality tradeoffs.

**Learned stage costs.** Instead of analytically determined stage costs, a learned cost predictor could estimate the actual FLOPs consumed at runtime, accounting for hardware-specific variations.

**Multi-frame temporal scheduling.** The current framework operates on a single frame independently. A richer extension would condition the Policy Network on temporal features (e.g., scene cuts, fade transitions) to adapt depth selection across frames.

**Region-level scheduling.** Instead of frame-level scheduling, spatial segmentation could allow different image regions to receive different compute budgets — allocating more computation to complex regions and less to simple regions within the same frame.

---

# Appendix A — Design Rationale

| Decision | Alternatives Considered | Chosen | Reason |
|----------|--------------------------|--------|--------|
| Optimization variable | Arbitrary subset $S^\star \subseteq \{1,\ldots,K\}$ | Prefix depth $k^\star \in \{0,\ldots,K\}$ | Aligns with sequential refinement dependency chain; eliminates invalid execution paths |
| Scheduler algorithm | Greedy, RL-based, approximate DP | Exact evaluation over $K+1$ depths | Optimal by definition; deterministic; negligible overhead for $K=4$ |
| Decoder family | Independent refinement experts (Family B) | Sequential refinement (Family A) | Isolates scheduler contribution; preserves DCVC compatibility; lower implementation risk |
| Policy output | Stage utility estimates $U_t$ | Stage utility estimates $U_t$ (surrogate for $Q(k;s_1)$) | Directly supports optimization objective; no ground-truth labels needed |
| Budget representation | Discrete tiers, continuous with arbitrary sampling | Continuous $\text{Beta}(2,2)$ during training; discrete at deployment | Prevents controller collapse; standard distribution; maps cleanly to device classes at inference |
| Stage costs | Equal across stages | $C_1 \ge C_2 \ge \cdots \ge C_K$ | Reflects that coarse reconstruction is expensive; fine refinement is cheap |
| Teacher instantiation | Generic learned codec | DCVC | Sequential decoder structure; publicly available; strong baseline quality |
| Policy input | Separate complexity predictor + controller | Unified $f_\theta(s_t, B)$ | Eliminates redundancy; single learning objective; fewer parameters |
| Feature aggregation | Global average pooling (fixed) | Lightweight global feature aggregation (flexible) | Preserves implementation flexibility; pooling is engineering, not research |
| Distillation mapping | Budget-aligned (0.25→stage 1, etc.) | Stage-aligned (student stage $k$ ↔ teacher stage $k$) | Architecture-dependent rather than deployment-dependent; remains valid as costs change |
| Training phases | Joint end-to-end training | Three-phase sequential (distill → policy → joint) | Prevents moving-target training; staged specialization is more stable |
| Phase 3 decoder learning rate | Same as policy | $\eta_\text{decoder} = 0.05 \cdot \eta_\text{policy}$ | Adapter-style; preserves Phase 1 specialization; enables small adaptation |
| Collapse prevention | Explicit entropy/regularization penalty | Budget-conditioned supervision (budget is input) | Cleaner; budget input prevents mode collapse without auxiliary losses |
| Scheduler search space | $2^K = 16$ arbitrary subsets | $K+1 = 5$ depth values | Prefix constraint matches sequential dependency; 5 candidates is trivially small |
| Quality metric primary | LPIPS, MS-SSIM | PSNR | Industry standard; directly interpretable; matches training loss |
| Evaluation baselines | Only DCVC | Fixed-tier + heuristic + oracle + prior work | Isolates each contribution (scheduler vs. decoder vs. training) |