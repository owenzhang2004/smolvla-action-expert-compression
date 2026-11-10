# Is the Action Expert Low-Rank? Course design document

> **This file is the design master.** It was exported from the Claude Doc's "Course design" tab at revision 179 on Nov 10, 2026, and that doc is frozen. Design changes are made here, by commit, with Owen's approval. The changes made at the export are listed in `research-record.md` §5, round 8. Where older text says "tab", read the file: Course design = this file, Research record = `research-record.md`, Archive = `archive/proposal-2026-11-05.md`.

Drafted Nov 5, 2026 · Owen · last design change Nov 10, 2026

Labels used throughout: **Verified** (Source-checked: the claim matches a cited passage opened in this project; or Reproduced: a derivation, code inspection or run checked here), **Secondary report** (only summaries of the source were read), **Hypothesis** (an experiment here tests it), **Estimate** (follows from stated assumptions), **Unverified** (awaiting confirmation), **Toy** (reproducible script result on a synthetic model, not evidence about SmolVLA). Every labelled claim has a row in the claim ledger of `research-record.md` §1.

**Contents**

- [1. Research question and hypotheses](#1-research-question-and-hypotheses)
   - [1.1 Central question](#11-central-question) · [1.2 The questions (Q1–Q5)](#12-the-questions-q1q5) · [1.3 Publication potential](#13-publication-potential)
- [2. Background the core depends on](#2-background-the-core-depends-on)
   - [2.1 The model](#21-the-model) · [2.2 Why the action expert](#22-why-the-action-expert) · [2.3 Prior compression work](#23-prior-compression-work) · [2.4 Spectra and parameterization](#24-spectra-and-parameterization)
- [3. Committed core (due Dec 20)](#3-committed-core-due-dec-20)
   - [3.1 Core commitments](#31-core-commitments) · [3.2 Latency prediction](#32-latency-prediction) · [3.3 Experimental configurations](#33-experimental-configurations)
- [4. Gated additions](#4-gated-additions)
   - [4.1 Additions and their gates](#41-additions-and-their-gates) · [4.2 Corrective-behavior hypothesis](#42-corrective-behavior-hypothesis)
- [5. Evaluation design](#5-evaluation-design)
   - [5.1 Nominal evaluation](#51-nominal-evaluation) · [5.2 Recovery protocol](#52-recovery-protocol) · [5.3 Recovery outcome and Δ\_m](#53-recovery-outcome-and-δ_m) · [5.4 Mechanism test](#54-mechanism-test)
- [6. Plan and budget](#6-plan-and-budget)
   - [6.1 Compute and schedule](#61-compute-and-schedule) · [6.2 Scope-cut order](#62-scope-cut-order) · [6.3 Episode arithmetic](#63-episode-arithmetic)
- [7. Pre-implementation verification checklist](#7-pre-implementation-verification-checklist)
   - [7.1 Model facts](#71-model-facts) · [7.2 Measurement setup](#72-measurement-setup) · [7.3 Open sources and protocols](#73-open-sources-and-protocols)
- [8. Risks to the core](#8-risks-to-the-core)
- [9. Appendix: toy results (label: Toy)](#9-appendix-toy-results-label-toy)
- [10. References cited in this tab](#10-references-cited-in-this-tab)

## 1. Research question and hypotheses

### 1.1 Central question

**Central question.** At matched action-expert storage budgets, how do low-rank and low-bit compression affect closed-loop success and measured inference speed in a flow-matching VLA, and which calibration criteria best preserve deployed behavior? Disturbance recovery is a gated extension of the same question.

Spectral characterization of the expert's weights is supporting evidence for interpreting results, not a test of compressibility on its own: a weight spectrum can be changed by function-preserving rescalings (ledger E4, §4.3 of the record), so functional compressibility is established only by closed-loop evaluation.

### 1.2 The questions (Q1–Q5)

**Five questions, each answerable on its own, each with a falsifier.** None assumes SVD or any hybrid is superior. Spectral prediction is exploratory, not a hypothesis: a single checkpoint cannot establish a spectrum-to-performance relationship, and the predictor and its test are not yet specified.

| # | Question | Falsified if | Status |
| --- | --- | --- | --- |
| Q1 Nominal comparison | Across stored-byte points from 2-bit to dense bf16, where does each family's non-inferiority threshold lie (§5.1), and at the W4-matched point does closed-loop nominal success differ between low-rank and low-bit compression of the action expert? | At the W4-matched point the paired difference is under 5 points, and the two families' thresholds fall at the same byte point | Core |
| Q2 Recovery effect | Does compression add a disturbance penalty (policy’s clean minus perturbed success) beyond its nominal effect (how much compression lowers success), i.e. is Δ\_m (§5) positive for compressed policies? | Δ\_m within the effect size of zero for all compressors | Gated |
| Q3 Family interaction | Does that penalty differ between the two families (low-rank vs low-bit) at comparable storage and nominal fidelity? | Δ\_m does not differ between families beyond the effect size, or the difference tracks nominal error | Gated (requires Q2) |
| Q4 Calibration benefit | Does calibrating to deployed behavior (executed prefix at the verified horizon; perturbed states for recovery) improve held-out closed-loop success over full-chunk and activation-aware calibration at equal bytes, data and teacher queries? | The deployed-behavior criterion does not beat ordinary distillation on the same states; then the result is diagnostic | Core (prefix) / Gated (perturbed) |
| Q5 Speed | Does reduced storage translate into measured speedup? Time per expert step and per policy call for each compressor with its best available kernel, CUDA graphs on, against dense bf16; measured break-even ranks for SVD against dense bf16 and W4A16 (§3.1, §3.2) | Measured results contradict the §3.2 predictions for the A100: W4A16 changes expert step time by more than 10% in either direction, or unfused SVD is faster than dense bf16 at any closed-loop rank | Core (raised by Prof. Zhang, Nov 9) |
| Exploratory | Do per-layer spectra, after parameterization controls, co-vary with which family wins? | — (reported, not tested) | Core, descriptive |

### 1.3 Publication potential

**Publication potential.** A reproducible answer to Q1 with the storage accounting and diagnostics is the course deliverable and could support a workshop submission; given the pace of this niche (ten or more compression papers on flow VLAs since mid-2026), a short arXiv write-up of the core result in late December is worth more than a polished spring draft, with reserve-funded extensions as a second version. Whether anything here reaches a conference depends on effect sizes, the strength of the quantization baselines, controls, and what a reader learns from the mechanism; positive findings on Q2–Q4 would strengthen the case but are neither required nor sufficient, and a strong, well-controlled result on one question could be enough. Nothing is promised, and the design is built so that null results on Q2–Q4 leave Q1 complete.

## 2. Background the core depends on

Only facts the committed experiments rest on. Ledger rows in brackets.

### 2.1 The model

**The model. Verified:** SmolVLA pairs a frozen SmolVLM2 prefix with a \~100M-parameter action expert trained from scratch on about 23K community episodes \[A1, A2\]. The expert interleaves cross-attention to the VLM prefix with causal self-attention over action tokens \[A9\]; cross-attention K/V projections read the VLM's cached key/value states \[A11\]. It runs 10 flow-matching steps and emits a 50-action chunk \[A3\]. The LIBERO checkpoint's config sets n\_action\_steps = 1, so in simulation one action is executed per 10-step denoise and 49 of 50 are discarded \[A4\]; the authors' ablation shows success falling from 80.3% at one executed step to 51.8% at fifty \[A5\]. **Read in code, runtime count pending \[A4, D15\]:** the policy's own action queue executes n\_action\_steps = 1, and `lerobot-eval` calls select\_action once per environment step, changing the horizon only if `--policy.n_action_steps` is passed; the first headless episode confirms it by counting policy calls per environment step. Resolved from the checkpoint config and the pinned code on Nov 10 \[A17\], pending confirmation by the loaded-model manifest: this checkpoint is not the paper's 0.45B configuration. It uses SmolVLM2-500M-Instruct with all 32 text layers (num\_vlm\_layers = 0), an expert width multiplier of 0.5, and an expert as deep as the VLM, so the expert has hidden size E = 480, intermediate size F = 1280 and 32 layers, alternating self-attention (even layers) and cross-attention (odd layers), about 97M parameters (61% MLP, 30% Q/O projections); the model card lists 0.6B parameters in total. Action tokens are causal: each starts its own attention block (att\_masks = \[1\] × chunk\_size under a cumulative-sum mask) \[A10\]. All byte and rank figures below use E = 480, F = 1280.

### 2.2 Why the action expert

**Why the expert. Verified:** on a Hopper GPU at batch 1, π0.5's action expert accounts for 8.8% of FLOPs and 57.2% of latency \[B1\]; SnapFlow measures the 10-step denoise at \~241 of 274 ms for π0.5 on an A800 \[B2\]. **Estimate:** at batch 1 with 50 tokens, each expert pass re-reads all expert weights while processing \~48 KB of activations per layer, so cost is set by weight traffic and kernel launches, not arithmetic. On the A100 launches should dominate: the expert's \~193 MB of bf16 weights take \~0.12 ms to read per step, while 32 layers at roughly 15 kernels each cost 0.5–1 ms at 1–2 µs per launch. The VLM also runs all 32 layers once per call, so the expert's share of a policy call is itself to be measured. This is the premise the week-2 profiling tests.

### 2.3 Prior compression work

**What is known about compressing it. Verified unless marked otherwise:** CHASE-VLA quantizes both MLP and attention projections to W4A4 and matches FP16 on π0.5 (97.3%) while losing 6.5 points on GR00T N1.6 \[C1\]; Ω-QVLA reportedly reaches 98.0% on π0.5 at uniform W4A4 using SVD only as a rotation \[C4, Secondary report: summary pages, paper not opened\]; ActQuant allocates precision by action sensitivity down to 3 bits \[C13\]. For low-rank: GLUESTICK found VLA language-backbone spectra flatter than LLM/VLM spectra, that fine-tuning flattened them (AUC/k 1.32 → 1.55), and that truncated SVD on OpenVLA gave 0% LIBERO-Spatial at ranks up to 1,000 \[C6\]; that evidence is from a backbone, not an action expert \[C7\]. QSVD's joint-QKV SVD with gradient-based rank allocation beats plain SVD on VLMs and names extension to other blocks as future work \[C5\]. PIVOT-Q showed quantization damage hides under distribution shift: GR00T-N1.5 drops from 82.1% to 67.0% on LIBERO-Plus while standard LIBERO stays at 97% \[C12\]. VLAQuantBench (Sep 2026) is the closest benchmark: 409 runs and \~94,600 episodes of post-training quantization on π0.5, π0, OpenVLA-OFT and X-VLA, varying the quantizer and layer scope at fixed bit-width, with no SmolVLA, no low-rank arm and no memory-matched comparison; it notes that equal nominal bit width does not make representations identical \[C20\]. This project varies the representation (bits vs rank) at fixed stored bytes on SmolVLA, which that benchmark leaves open. No study found applies weight SVD to a flow-matching action expert, compares it with quantization at matched storage in closed loop, or tests whether the execution horizon changes the answer (record §2). A W4 baseline on SmolVLA is therefore table stakes, reproduced for comparability, not a contribution.

### 2.4 Spectra and parameterization

**Spectra are parameterization-dependent. Verified (algebra):** for a gated MLP, W\_u' = DW\_u and W\_d' = W\_dD⁻¹ preserve the function for any invertible diagonal D while changing both matrices' singular values. Raw spectra are therefore checkpoint-specific. We report spectra after a canonical per-channel rescaling and after SVD-LLM-style activation whitening, and show how much conclusions move. Per-token expert inputs are a function of the sampled noise, which LeRobot draws at the padded action width (max\_action\_dim = 32 in this checkpoint, so 32 stochastic coordinates per token, not 7) plus the timestep \[A14, A17; noise shape and input path confirmed in code at LeRobot commit d490e47: action\_in\_proj (32 → 480), concatenated with a sinusoidal time embedding, then action\_time\_mlp\_in (960 → 480), SiLU, action\_time\_mlp\_out (480 → 480)\]; seven of those coordinates map to executed actions. Hidden-activation covariance is therefore measured per layer rather than inferred from the action dimension.

## 3. Committed core (due Dec 20)

### 3.1 Core commitments

| Element | Commitment |
| --- | --- |
| Model | SmolVLA, HuggingFaceVLA/smolvla\_libero (0.6B per its model card; 32-layer VLM and a 32-layer, 480-wide expert, not the paper's 0.45B configuration) at Hugging Face revision 6721902b (full hash in `env/pins.md`; read through a fetch summary on Nov 10, to be confirmed when the checkpoint is first downloaded \[A18\]), on LeRobot pinned to commit d490e47 (version 0.7.1), the version whose code the code-verified claims describe |
| Reference | FP16 baseline reproduced at 50 episodes per task on one suite, LIBERO-10 or LIBERO-Spatial, chosen after the throughput pilot (LIBERO-10 has more headroom, but its episodes cap at 520 steps vs 280 for LIBERO-Spatial in LeRobot at the pinned commit \[D15\]) (authors: 87.3% at 10 trials; independent: 74.0% \[A6, A8\]) |
| Benchmark | Standard LIBERO, closed loop, at the verified runtime horizon H, with the FP16 reference evaluated at the same H |
| Compressors | Weight-only quantization with the algorithm named: GPTQ as the closed-loop quantizer, with per-group round-to-nearest as an open-loop floor (bf16 scales; group size set by the fastest fused W4A16 kernel that accepts the expert's shapes (checklist §7.1), otherwise 120 along the 480 hidden dimension, which 128 does not divide, and 128 along the 1280 intermediate dimension; success and latency are measured on the same stored artifact), since a claim that SVD beats RTN alone would be weak; importance-aware SVD as the closed-loop low-rank method (joint gate-up and joint self-attention QKV, QSVD-style rank allocation with importance computed from LeRobot's own flow-matching loss, record §3.4), with plain truncated SVD as an open-loop floor. Published W4A4 results on other models (CHASE-VLA, Ω-QVLA) are related-work context, not numerically matched SmolVLA baselines |
| Storage accounting | Actual stored bytes per compressor including scales, metadata and alignment, from the manifest; comparisons at byte points matched to 2, 3, 4, 6 and 8 bits, plus SVD-only points at ≈ 0.6E and ≈ 0.8E (record §3.2), finalized after the manifest |
| Spectral analysis | Every expert linear layer, raw and after the two parameterization controls, plus per-layer activation covariance on calibration data; summarized by 90%-energy rank and AUC/k. The timestep path (action\_time\_mlp\_in/out, which fuses the sinusoidal time embedding with the noisy action; SmolVLA has no adaRMS) is tracked as a named component in spectra and sensitivity, because VLAQuantBench found π0.5's attention + adaRMS subset collapses to 1.5% under uncalibrated RTN \[C20\] and RL updates concentrate in timestep modules \[C9\] |
| Latency | Measured on the A100 at batch 1 with CUDA graphs on (eager also reported): (i) per-layer microbenchmarks of the MLP block and attention projections for dense bf16, W4A16 with a named fused kernel, and SVD over a rank sweep, both per-matrix and joint factorizations, giving measured break-even ranks against dense bf16 and W4A16; (ii) time per expert step and per policy call at every closed-loop point, p50 and p99; (iii) DRAM bytes moved and kernel count per step from the profiler. (iv) four inference paths per arm, latency only: eager; torch.compile default mode (Inductor fuses norms, activations and elementwise ops, no CUDA graphs); reduce-overhead (fusion plus CUDA graphs); max-autotune (fusion, CUDA graphs and autotuned Triton matmuls), with kernel count reported per path (modes Verified in torch 2.15 \[C24\]). No mode fuses SVD's two matmuls: Inductor's back-to-back GEMM pass is experimental, off by default, and its own cost model declines these shapes (§3.2), so SVD keeps its extra launches under compilation. This answers Q5. Thor figures are projections from bytes moved and kernel count, and are labelled so |
| Inference path | **Success episodes run on LeRobot's own (eager) inference path, for the FP16 reference and every compressed arm** (decided Nov 10). It is the reference implementation, so it needs no equivalence validation, and one path for all arms keeps the pairing valid. **The optimized path is used for latency (Q5) only:** CUDA graphs on, and the expert's cross-attention K/V projections of the cached VLM prefix computed once per policy call instead of at each denoising step. LeRobot recomputes them at every step (Verified in code at commit d490e47); hoisting is exact because their only input is the fixed prefix cache, and without it every speedup would be measured against a needlessly slow baseline. It is built after the FP16 baseline and the first RTN and plain-SVD artifacts exist, and is timed on the same saved model files. **Revisit after the throughput pilot:** if episodes are bound by policy compute rather than rendering, so the optimized path would save meaningful GPU-hours, every arm including R0 switches to it, after it passes the action-equivalence test and before any closed-loop arm runs. Arms are never mixed across paths, because compilation and fusion change the last bits of the numerics. |
| Calibration | Fixed set of LIBERO demonstrations from suites other than the evaluated one (\~256 samples), shared across all compressors; scores averaged over noise and timestep draws; full-chunk vs executed-prefix objectives compared (Q4, prefix part) |
| Diagnostics | Per-step velocity error and final executed-prefix error on recorded observations, correlated against closed-loop success. The transported-error coherence index (record §4.4) is optional and is the first diagnostic cut |

### 3.2 Latency prediction

**Prediction on latency (Estimate, record §4.2).** Per MLP block per step, t ≈ n\_launch · t\_ℓ + bytes\_moved / B. Joint gate-up plus down SVD runs 4 GEMMs where dense bf16 and W4A16 run 2, so it pays two extra launches, worth 2 · t\_ℓ · B in bytes.

| Hardware (t\_ℓ = 1–2 µs, CUDA graphs) | Launch penalty | SVD faster than dense bf16 | SVD faster than W4A16 |
| --- | --- | --- | --- |
| A100 40GB, \~1.555 TB/s (B6 Unverified) | 3.1–6.2 MB | rank ≤ 60 at 1 µs; never at 2 µs (penalty exceeds the dense 3.69 MB block) | never: penalty exceeds W4A16's whole 0.95 MB block |
| Jetson Thor, 273 GB/s | 0.55–1.09 MB | rank ≤ 327 at 1 µs, ≤ 270 at 2 µs (0.68E, 0.56E) | rank ≤ 42 at 1 µs (0.09E); never at 2 µs |

Reading the expert's weights takes \~0.12 ms per step on the A100 vs \~0.71 ms on Thor, so storage savings are worth far less time on the A100. With 32 thin layers the A100 step is expected to be launch-bound, so the prediction is that W4A16 moves step time by under 10% there and unfused SVD is slower than dense bf16 at every closed-loop rank; the speed case for either compressor is an edge-hardware projection. Quantization is not automatically faster either: without a fused dequantizing kernel W4 adds a dequantization pass and can be slower than bf16, and the 2-, 3- and 6-bit points have no fast kernel (success-only). The model omits arithmetic, kernel selection, occupancy and caching; the §3.1 microbenchmarks decide. Stored bytes, moved bytes, packing and fusion are different quantities. Computed at E = 480, F = 1280 (dense MLP block 3,686,400 B; W4A16 951,680 B with scales; 9,600 B per rank) with the model in `analysis/toy/svd_rank_budget.py` (defaults are these dimensions; `--legacy` reproduces the earlier illustrative 720/2048 run); `tests/test_latency_model.py` pins every number in the table.

**Fused low-rank kernel (Hypothesis; gated addition §4.1).** If one kernel computes U(Vx) with the intermediate Vx kept on chip, joint-factorized SVD runs 2 kernels per MLP block, like dense bf16 and W4A16, and the launch penalty above disappears. Bytes then decide alone: faster than dense bf16 below rank ≈ 384 and faster than fused W4A16 below the byte-matched rank ≈ 99 (E = 480, F = 1280). These are upper bounds. At 50 tokens a fused kernel must avoid re-reading V for every output tile: PyTorch Inductor's experimental back-to-back GEMM template, which does re-read, scores 0.03–0.07 of baseline load efficiency on these shapes in its own cost model and declines them (Reproduced, torch 2.15, Nov 9). A split-rank design, in which each program owns a slice of the rank, reads its slices of V and U once, and accumulates into the output with fp32 atomic adds, avoids re-reads but exposes only about r / slice programs of parallelism. Whether any fused kernel beats two cuBLAS calls at 50 tokens is what the microbenchmark measures; a negative result still separates SVD's launch cost from its byte cost.

### 3.3 Experimental configurations

**Experimental configurations.** Every separately calibrated model is a row, so no arm hides inside a phrase. Each closed-loop point is 10 tasks × 50 episodes = 500 episodes on the chosen suite; byte points are measured from stored artifacts (record §3.2). E is the expert hidden size.

| ID | Compressor | Calibration | Byte point | Episodes | Status |
| --- | --- | --- | --- | --- | --- |
| R0 | FP16 reference | — | dense bf16 | 500 | Core; never cut |
| G2 | GPTQ | nominal, full chunk | 2-bit | 500 | Core; third cut |
| G3 | GPTQ | nominal, full chunk | 3-bit | 500 | Core |
| G4 | GPTQ | nominal, full chunk | 4-bit | 500 | Core; never cut |
| S-W4 | Importance-aware SVD | nominal, full chunk | W4-matched (rank ≈ 0.21E) | 500 | Core; never cut |
| S-W8 | Importance-aware SVD | nominal, full chunk | W8-matched (rank ≈ 0.41E) | 500 | Core |
| S-0.6 | Importance-aware SVD | nominal, full chunk | rank ≈ 0.6E (≈ 12 bits-equivalent) | 500 | Core |
| S-x | Importance-aware SVD | nominal, full chunk | chosen after the open-loop screen | 500 | Core; second cut |
| S3 | Importance-aware SVD with executed-prefix calibration | nominal, prefix (Q4) | the SVD point the screen places between collapse and lossless | 500 | Core; first cut |
| Screen | RTN 2–8 bit; GPTQ 6, 8 bit; plain SVD at every point; importance-aware SVD at ≈ 0.8E | as above | all points | open loop only | Core |
| H1 | Residual hybrid (bf16 branch + sub-4-bit residual) | nominal, full chunk | W4-matched | 500 | Gated |
| S4 | Importance-aware SVD with perturbed-state calibration | perturbed states | W4-matched | 500 | Gated (Q4, recovery) |
| S5 | Importance-aware SVD with recovery-sensitive allocation | perturbed states | W4-matched | 500 | Gated (Q4, recovery) |

The core closed-loop rows total 4,500 episodes, 75 GPU-hours at 60 s per episode; with week-1 checks, spectra and the open-loop screen, about 91 GPU-hours, leaving about 149 of the 240 usable for the reserve-funded extensions in record §3.1–3.3. The throughput pilot sets which rows fit.

## 4. Gated additions

### 4.1 Additions and their gates

Each runs only if its gate opens, and displaces lower-priority sweeps rather than enlarging the project. Gate decisions are made after the throughput pilot (§6).

| Addition | Gate | What it adds |
| --- | --- | --- |
| Recovery study (Q2, Q3) | FP16 teacher recovers on a predefined fraction of a mid-execution perturbation pilot (protocol §5), and the measured episode cost leaves budget | Standard perturbation rollouts and common-state interventions for the core compressors |
| Residual hybrid (SVDQuant-style bf16 low-rank branch plus sub-4-bit residual) | Storage accounting shows a configuration that meets the INT4 budget; a usable low-bit kernel exists or the comparison is on success only | The one variant that keeps low-energy directions in low precision |
| Recovery-sensitive rank allocation (Q4, perturbed part) | Recovery study shows an effect beyond nominal error | Three arms at equal bytes and calibration access: nominal-state calibration, ordinary distillation on perturbed states, recovery-sensitive allocation on the same perturbed states |
| Second execution horizon | Runtime H verified and budget allows | Repeats the core comparison at one longer H to test whether the winner or the best calibration changes |
| Prefix-only sampling baseline | Causal-mask check passes and chunk-1 sampling reproduces the executed action under matched noise | A free latency baseline at H = 1; its latency change, read together with a profiler trace, helps separate weight bandwidth from launch overhead (timing alone cannot; checklist §7) |
| Fused low-rank Triton kernel (Q5) | Core compressors implemented and unit-tested by Dec 1; prior art checked first (Inductor's back-to-back GEMM pass, CUTLASS's two-GEMM fusion example, FlashSVD \[C23\], the last two Unverified); time box of 5 working days, developed for correctness on CPU with TRITON\_INTERPRET=1 and benchmarked on the A100 | Microbenchmarks only, since the math equals the unfused pair up to bf16 rounding (equivalence test, no new closed-loop episodes): fused vs unfused SVD at the screened ranks against dense bf16 and W4A16, separating SVD's launch cost from its byte cost (§3.2). About 1 GPU-hour from the unallocated reserve |

### 4.2 Corrective-behavior hypothesis

**Corrective-behavior hypothesis (Q2/Q3 mechanism, Hypothesis).** Demonstrations mostly show states near the intended trajectory, so directions that matter only after a slip or a wrong pose contribute little to nominal reconstruction loss and nominal-state calibration may discard them. Two policies can agree at the setpoint and differ in their restoring response. This does not by itself favor a residual: on a 480×480 layer a 4-bit residual costs the same as 60 bf16 ranks, so few identifiable corrective directions favor ranks and many diffuse ones favor the residual. The local diagnostic is V\_ℓᵀΔx\_ℓ per layer, where Δx\_ℓ is the teacher's layer-input difference between paired nominal and perturbed states; it is local, and only intervention connects it to executed actions.

**Reserve-funded extensions.** Research record §3.1 (steps × horizon × compression) and §3.3 (latency on dynamic tasks, post-Dec 20) hold the options the reserve can fund. They run at the threshold points the core identifies, so they reuse its harness and calibrated models.

## 5. Evaluation design

### 5.1 Nominal evaluation

**Nominal (core).** Closed-loop success on the chosen suite at the verified H, 50 episodes per task, fixed seeds and initial states. Official LIBERO initial states. Each configuration's success is reported with a 95% Wilson interval (the per-config convention of VLAQuantBench \[C20\]); differences against FP16 use the paired comparison on the same episodes, with the same initial state and the same flow-noise seed per decision in every arm (the harness generates each decision's noise from the seed, episode and step and passes it through select\_action's `noise` argument, because `lerobot-eval` draws from the global generator, whose stream stops matching between arms once their episode lengths differ \[D15\]), and a bootstrap that resamples paired episodes within each task (stratified by task, so conclusions are conditional on these 10 tasks); a bootstrap that resamples whole tasks is reported beside it as a sensitivity check and will be much wider with 10 tasks; a predefined meaningful effect size (5 points average success) stated before runs. Failure to reject a difference is not evidence of equivalence. Nominal fidelity for matched-budget selection uses final executed-prefix error on held-out demonstrations, not per-step velocity error; configurations are selected on separate validation data, and if no equal-storage pair with comparable fidelity exists, the trade-off is reported rather than forced.

**Thresholds and pre-registered predictions.** A configuration is non-inferior to FP16 if the lower bound of its one-sided 95% paired-bootstrap interval for (its success − FP16 success) is above −5 points; a family's threshold is the smallest byte point that is non-inferior together with every larger point of that family, so a single noisy pass below a failure does not set it. Only P1 and P2 are confirmatory; every other per-arm comparison is reported as an estimate with its interval, because eight arms each tested at 95% would produce false passes by chance, and the report says so. Estimate: at 500 paired episodes with \~20% discordant pairs, SE ≈ 2.0 points, so a point passes when its observed difference exceeds ≈ −1.7 points (power ≈ 0.80 if the true difference is 0, ≈ 0.44 if it is −2). A two-sided ±5-point equivalence test was rejected: it would pass only differences within ≈ 1.1 points. The 20% discordance is an assumption: closed-loop rollouts can diverge from tiny numerical differences, and if pairing buys little the discordance approaches 2p(1 − p) ≈ 35% at p ≈ 0.75, giving SE ≈ 2.6 points and power ≈ 0.60 at a true difference of 0. The measured discordance between the FP16 reference and its first compressed arm replaces the assumption, and an FP16 repeat on the same seeds (500 episodes from the reserve) measures the run-to-run noise floor if results are close.

- P1 (Hypothesis): GPTQ is non-inferior at 4 bits and not at 2.
- P2 (Hypothesis): importance-aware SVD is not non-inferior at any rank up to the W8-matched rank.
- P3 (Hypothesis): no crossover; quantization is at least as good at the W4-matched point closed loop and at every point of the open-loop screen.
- P4 (exploratory): the whitened 90%-energy rank predicts SVD's threshold to within one byte point.
- P5 (exploratory): open-loop executed-action error orders the closed-loop points; reported within each family, since the gap between families alone would inflate a pooled rank correlation.

### 5.2 Recovery protocol

**Recovery (gated).** Two complementary evaluations, because matched seeds and perturbation schedules do not guarantee matched states at disturbance onset:

1. *Standard perturbation rollouts.* Each policy runs under the benchmark's protocol with the same perturbation schedule.
2. *Common-state interventions.* Create each disturbed state once: save the simulator state at chosen points of successful FP16 rollouts, apply the disturbance under a fixed scripted controller (not any policy) through the whole perturbation window, and save the resulting state. Then fork: for every policy, restore that identical post-disturbance state, regenerate observations by rendering, reset action queues and caches, and let the policy control from there. Forking after the window matters because during a sustained perturbation policies act and would diverge even from identical starts. Flow noise is matched across policies at corresponding decisions and fresh across decisions; reusing one noise sample through an episode would change the controller being evaluated. This separates arriving at a bad state from responding badly to the same bad state.

Protocol candidate: LIBERO-ReAct's mid-execution perturbations (action noise, gripper flips, end-effector displacement over a 20-step window) \[D8\]; its implementation, units and LeRobot compatibility are unverified \[D11\] and must be checked before it is a dependency. Teacher recoverability is established in an independent pilot and the qualification rule frozen before any compressed policy is evaluated.

### 5.3 Recovery outcome and Δ\_m

**Primary recovery outcome:** task completion within a fixed post-disturbance budget. Reported per policy m: clean success, perturbed success, and the excess disturbance penalty relative to FP16,

Δ\_m = (S\_m^clean − S\_m^perturbed) − (S\_FP16^clean − S\_FP16^perturbed).

Recovery timing uses task milestones or completion time with censoring for episodes that never recover; re-entering the teacher's trajectory is not a metric, since a successful recovery may take another path. The teacher-forced action-response gap ‖(â\_pert − â\_nom) − (a\_pert − a\_nom)‖ under matched noise is a secondary diagnostic.

### 5.4 Mechanism test

**Mechanism test (gated, after a recovery effect).** Direction exchange at equal storage: restore candidate corrective singular components while removing the same number of other components; compare against random exchange and singular-value-ordered exchange; evaluate selected directions on held-out cases. Stronger than spectral-energy correlations because it intervenes.

## 6. Plan and budget

### 6.1 Compute and schedule

Compute: NYU HPC Cloud Bursting, 1× A100 40GB, 300 GPU-hours. GPU-hours below are placeholders until the throughput pilot in week 1 measures seconds per episode including rendering, resets, policy calls and any env parallelism; the matrix is then sized to fit with a 20% contingency. Implementation effort is separate from GPU time.

| Dates | Work | GPU-h (placeholder) |
| --- | --- | --- |
| Nov 9–15 | Critical path, in order: container with pinned versions and one headless LIBERO episode (which also counts policy calls per environment step, the runtime H); checkpoint manifest; throughput pilot (10 episodes each on Spatial and LIBERO-10). Compressor code and unit tests start on CPU in parallel | \~5 |
| Nov 16–22 | FP16 baseline (50 ep/task, chosen suite, eager path); spectral analysis with parameterization controls and activation covariance; causal-mask assertion and chunk-1 exactness test; W4A16 kernel compatibility, which sets the quantization group size and so precedes GPTQ calibration | \~10 |
| Nov 23–Dec 1 (Thanksgiving Nov 26) | Implement GPTQ and RTN at 2–8 bits, plain SVD, importance-aware SVD with joint decompositions; open-loop screen over all byte points; storage accounting; calibration set; unit tests against original layers; open-loop diagnostics | \~4 |
| Dec 2–10 (window is a placeholder until the pilot measures seconds per episode) | Closed-loop core rows from the configuration table (GPTQ at 2, 3, 4 bits; importance-aware SVD at four points; S3; R0 already run) on the chosen suite. Latency work for Q5, as separate jobs, starting once the baseline is done and RTN and plain-SVD artifacts exist: optimized path (K/V hoisting, static cache, CUDA graphs), per-component profile and launch counts, chunk-50 vs chunk-1 latency, per-layer microbenchmarks for dense bf16, W4A16 and an SVD rank sweep, torch.compile paths. Gate decisions; gated additions if opened; stretch: fused Triton low-rank kernel (≤ 5 working days, microbenchmarks only) | \~72 |
| Dec 11–13 | Analysis: paired statistics, diagnostic-to-success correlations, Pareto curves | 0 (CPU) |
| Dec 14–18 | Presentation | — |
| Dec 20 | Report | — |
| Contingency (20%) | 20% of 300 GPU-h, held back for reruns and debugging | 60 |
| Unallocated | Reserve-funded extensions (record §3.1–3.3) and gated additions (§4), in the order §3.3 gives | \~149 |

**Sensitivity (Estimate).** The column above is derived from §3.3: core ≈ 16 + 4,500 × s / 3,600 GPU-h at s seconds per episode (the 16 = checks and latency work \~10, spectra \~2, open-loop screen \~4), and unallocated = 240 − core. At 30 s per episode, core ≈ 54 and ≈ 186 are free; at 60 s, ≈ 91 and ≈ 149; at 120 s, ≈ 166 and ≈ 74. The throughput pilot's measured seconds per episode replaces these numbers.

### 6.2 Scope-cut order

**Scope-cut order if the pilot shows the matrix does not fit:** (1) S3; (2) the screen-chosen SVD point S-x; (3) GPTQ-2; (4) the suite to a 5-task subset; (5) episodes to 30 per task with wider intervals reported. Never cut R0, GPTQ-4, SVD at the W4-matched point, or the paired design.

### 6.3 Episode arithmetic

**Illustrative episode arithmetic (Estimate, record §4.7):** a recovery matrix of 20 tasks × 50 episodes × 4 policies × 4 conditions is 16,000 episodes, about 267 GPU-hours at 60 s each. That is why recovery is gated on the pilot, not committed.

## 7. Pre-implementation verification checklist

**Order (decided Nov 10).** Week 1 has three items on the critical path: one headless episode runs (§7.2), the checkpoint manifest (§7.1), and the throughput pilot (§7.2). Everything else on this list follows them, and the latency items wait until the FP16 baseline and the first RTN and plain-SVD artifacts exist.

### 7.1 Model facts

- [ ] **Checkpoint manifest** from the loaded model: every tensor shape; expert layer count and hidden/intermediate sizes (resolves A12, A13); parameter and byte counts per layer group (replaces A15); which projections are computed once per call vs per step; attention structure per layer. Pin the HF commit and LeRobot version.
- [ ] **Runtime horizon**: confirm the number of actions executed per policy call in the actual LIBERO eval path, not only n\_action\_steps in config (A4), by counting policy calls per environment step in the first headless episode.
- [ ] **Causal mask** and noise shape: first assert the mask itself on the loaded model, namely that in the 2-D attention mask action token i attends to the prefix and to action tokens ≤ i only, so a surprise fails as a clear assertion and not as a confusing exactness failure (expected from code \[A17\]); check the sampler's noise shape in modeling\_smolvla.py (expected (batch, chunk, max\_action\_dim)); then test that chunk\_size = 1 reproduces token 0 of chunk\_size = 50 under matched noise to floating-point tolerance (A10, A14, E8). If latency barely changes between chunk 50 and 1, use the profiler to tell weight-bandwidth from launch overhead; the timing alone cannot.
- [ ] **W4A16 kernel compatibility**: find a fused W4A16 kernel (Marlin, torchao int4 or similar) that accepts the expert's shapes (hidden 480, intermediate 1280; Q and O projections are 480↔960; 480 is not a multiple of 128) and a usable group size (Unverified which do). If none accepts the native shapes, choose between padding (bytes moved then exceed bytes stored) and a slower kernel, and record which. Sets the group size in §3.1 and gates Q5.
- [ ] **Optimized inference path**: hoist the cross-attention K/V projections out of the denoising loop and test that actions match the unhoisted path under matched noise to floating-point tolerance. Make shapes static for CUDA-graph capture: LeRobot's DynamicCache appends and crops self-attention K/V at every step, and the prefix length must be fixed or padded per call. If capture needs more than \~2 days of work, report eager and partial-graph timings and say so. Latency only, unless the throughput pilot triggers the switch described in §3.1; deferred until the FP16 baseline and the first RTN and plain-SVD artifacts exist.

### 7.2 Measurement setup

- [ ] **Throughput pilot**: seconds per episode on Spatial and LIBERO-10 with rendering, resets and policy calls, at the env parallelism to be used (E7). Decides the suite.
- [ ] **Launch count and gap**: count kernels per expert step from the profiler in eager and CUDA-graph modes; measure the per-launch gap on the A100 (B12).
- [ ] **A100 latency breakdown (check 7)**: per-component time per policy call with CUDA graphs on, p50 and p99; bytes moved vs kernel count per expert step; whether W4 changes the expert's step time by more than 10%. Decides whether compression is a latency lever on this hardware (record §3.1).
- [ ] **Storage accounting**: bytes per compressor as stored, including scales and metadata, from the saved artifacts.
- [ ] **Headless rendering** (week-1 critical path): one LIBERO episode in the container with `MUJOCO_GL=egl`.
- [ ] **torch.compile readiness**: count graph breaks and recompiles per path (torch.\_dynamo.explain; TORCH\_LOGS=graph\_breaks,recompiles); confirm reduce-overhead actually captures (TORCH\_LOGS=perf\_hints), since it skips graphs that mutate inputs and LeRobot's DynamicCache does, so the static-shape work in §7.1 comes first; confirm the chosen W4A16 kernel compiles without a graph break (a custom op may need registering through torch.library; Unverified which kernels need it); test each path's actions against eager under matched noise.

### 7.3 Open sources and protocols

- [ ] **LIBERO-ReAct**: locate code; verify perturbation semantics, action units, and LeRobot compatibility (D11). If unavailable, define a minimal perturbation protocol and state it before the pilot.
- [ ] **OGPO figures**: read the results table to determine whether 76.7%/81.8% is ReAct-only (D9).
- [ ] **NestedVLA**: read or drop (C8).
- [ ] **A100 bandwidth**: datasheet (B6). The LIBERO control rate is 20 Hz (control\_freq default in LeRobot's LIBERO integration at the pinned commit; D3 Verified in code Nov 10).
- [ ] **Delay wrapper (optional, after the harness works)**: asynchronous chunk execution with delay d; must reproduce normal results exactly at d = 0 (record §3.3).
- [ ] **Calibration data and GPTQ plumbing**: the LIBERO demonstrations in LeRobot format are `lerobot/libero` (1,693 episodes, 40 tasks, 1.9 GB as video) or `HuggingFaceVLA/libero` (the same data as PNG, 69.9 GB), per LeRobot's docs at the pinned commit \[A19\]; pin the dataset revision; confirm which suites the checkpoint was trained on (Unverified); confirm LeRobot's training loss can be called on it for importance gradients; decide how GPTQ's per-layer Hessians are collected on the expert, since GPTQ libraries target language models and the expert needs a custom calibration loop (Unverified which library adapts with least work).

## 8. Risks to the core

- **Headless rendering.** MuJoCo needs EGL on cluster nodes. Verified in week 1; fallbacks are OSMesa (slow) or open-loop error on recorded demonstrations (weaker evidence, reported as such).
- **Baseline drift.** Published SmolVLA success spans 74.0–87.3% \[A6, A8\]; we report our own FP16 reference at 50 episodes per task and compare only against it.
- **Architecture assumptions.** Hidden size 480, intermediate 1280 and 32 layers come from the checkpoint config and pinned code \[A17\], not yet from a loaded model; the manifest confirms them before any run, and the paper's published numbers describe a different (0.45B, 16-layer) configuration, so only our own FP16 reference is a valid baseline.
- **Launch overhead on the A100 may hide byte savings.** CUDA-graph baseline first; bytes moved reported beside wall-clock; edge figures labelled projections.
- **Quantization without a fused kernel moves more bytes than it saves.** Each compressor is measured with its best available kernel and the kernel named.
- **Plain SVD may collapse**, as GLUESTICK's backbone result suggests \[C6\]. That is a reportable result; the importance-aware variant is the positive path.
- **Rank allocation may starve flat layers.** Global importance ranking can under-allocate to layers whose importance is spread thin; mitigation is spectrum-gated selection with a per-layer minimum rank.
- **Budget.** The core matrix must fit after the pilot; the scope-cut order in §6 is binding.
- **Solo scope.** Gated additions are the first casualties of time; the core is sized to finish without them.
- **Compressor implementation time.** The schedule leaves nine days, over Thanksgiving, for GPTQ and importance-aware SVD on a non-standard architecture. Mitigation: write and unit-test both on CPU during weeks 1–2 while the GPU runs checks and the baseline; if GPTQ slips, RTN at the same byte points runs closed loop in its place and the report says so.
- **Compilation and kernel stretch may not pay off.** reduce-overhead may not capture (mutated cache) and a fused Triton kernel may lose to two cuBLAS calls at 50 tokens. Mitigation: closed-loop runs never wait on either; latency is reported for whichever paths work, and a losing fused kernel is reported as a result.

## 9. Appendix: toy results (label: Toy)

Two seeded scripts in the repo under `analysis/toy/` (torch 2.15 CPU, float64, seed 0; full settings and outputs in the Research record §6). They exercise a synthetic gated-MLP velocity field and the launch-vs-bytes model with illustrative dimensions. They support design choices; they are not evidence about SmolVLA and no claim in §§1–5 rests on them. All arithmetic is float64 and, by default, SVD factors and RTN scales are not rounded to bf16, so the comparison matches a storage budget rather than performing deployment-precision numerics; a rerun with bf16 rounding of factors and scales (the script's --round flag) produced small numerical changes (for example 0.2856 → 0.2842 for one endpoint error) without changing the qualitative conclusions. Neither script measures deployment memory or speed. The dropped-subspace shift is built from the SVD compressor's own discarded directions, a targeted stress test rather than a neutral sample of shifts, and the teacher is an untrained random field with an imposed spectrum, not a learned transport. The supported conclusions are narrow: deterministic quantization error can stay aligned across steps; changing the imposed spectrum reverses the error ordering; local and endpoint error respond differently to a shift.

- `compounding_harness.py`: at matched bytes over 10 Euler steps, with per-channel 4-bit RTN (2,221,472 B including scales) against rank-146 joint SVD (2,214,528 B), both gave error-alignment coherence C ≈ 0.94 and growth slope ≈ 1.06 (C lies in \[0, 1\]; 1/√K ≈ 0.32 is the reference for equal-scale uncorrelated errors, not a lower bound), so a random-walk assumption for quantization error was not supported even in this setting; SVD was about 6× worse than W4 under a flat imposed spectrum and about 2× better under a steep one; a conditioning shift hurt SVD modestly more at the steep spectrum (16% vs 9% increase in teacher-forced local velocity error; 35% vs 27% in endpoint error). The two metrics move differently, which is one reason the design measures both.
- `svd_rank_budget.py --legacy`: with E = 720, F = 2048, 273 GB/s and an assumed 2 µs launch gap, SVD factors match W4A16 bytes at rank 150 (group-128 scale layout; 146 with per-channel scales) and, under the two-term model, are strictly faster only at rank 78 or below; at 8 µs no rank qualifies. The default run, at the checkpoint's 480 / 1280, gives the §3.2 table.

## 10. References cited in this tab

Numbers match the full list in the Research record §8, which records how far each was verified.

\[1\] Shukor et al., SmolVLA, 2025. https://arxiv.org/abs/2506.01844 · \[3\] Wang, Wang & Zhang, QSVD, NeurIPS 2025 · \[4\] SnapFlow, 2026 · \[5\] PhyAI, 2026 · \[6\] Hyun et al., CHASE-VLA, 2026. https://arxiv.org/pdf/2610.02666 · \[8\] Jabbour et al., GLUESTICK, 2025. https://arxiv.org/pdf/2510.08464 · \[12\] HuggingFaceVLA/smolvla\_libero · \[14\] SMILE, 2026 · \[16\] Liu et al., LIBERO, NeurIPS 2023 · \[26\] PIVOT-Q, 2026. https://arxiv.org/abs/2610.05745 · \[27\] Ω-QVLA, 2026 (Secondary report) · \[31\] OGPO / LIBERO-ReAct, 2026. https://arxiv.org/abs/2610.12914 · \[35\] ActQuant, 2026. https://arxiv.org/abs/2605.24011 · \[36\] Coupling Execution Horizon and Sampling Depth, 2026. https://arxiv.org/abs/2610.32102 · \[48\] LeRobot smolvlm\_with\_expert.py · \[49\] VLAQuantBench, 2026. https://arxiv.org/abs/2609.25376 · \[50\] Wu et al., On-Device Diffusion Transformer Policy (LightDP), ICCV 2025. https://arxiv.org/abs/2508.00697.

`research-record.md` holds the claim ledger, related-work map, extension scope (π0.5, GR00T N1.6, LIBERO-Plus/PRO, Jetson Thor, step-scheduled rank, direction exchange), derivations, correction log, toy outputs, hardware notes and all 52 references. `archive/proposal-2026-11-05.md` holds the pre-restructure proposal unchanged.
