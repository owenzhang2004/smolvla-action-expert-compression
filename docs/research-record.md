# Research record

> **This file is the master copy of the record.** It was exported from the Claude Doc's "Research record" tab at revision 64 on Nov 10, 2026, and that doc is frozen. New ledger rows, corrections and references are added here, by commit. Where older text says "tab", read the file: Course design = `course-design.md`, Research record = this file, Archive = `archive/proposal-2026-11-05.md`. Read one section at a time: a task that cites a ledger ID (for example A17) or a section number needs only that part.

Drafted Nov 5, 2026 · Owen · last change Nov 10, 2026

The full record behind `course-design.md`: what we believe, why, and how sure. Labels: Verified (Source-checked or Reproduced), Secondary report, Hypothesis, Estimate, Unverified, Toy. `archive/proposal-2026-11-05.md` holds the pre-restructure proposal unchanged.

## 1. Claim ledger

One row per substantive claim. "Source" names the passage, table, code line or script that supports it; "Checked" is the date it was opened in this project. Verified is refined into Source-checked (the claim matches a cited passage opened in this project), Reproduced (the result was executed and checked here), and Secondary report (only summaries of the source were read). Rows labelled Verified are Source-checked unless the Source column names a secondary page; an abstract-level check covers only what the abstract states, not implementation details. Rows marked Unverified carry the action that resolves them. Nothing in the Course design tab should rest on a row that is not Verified, Toy, or explicitly labelled as hypothesis or estimate.

**A. SmolVLA model and checkpoint**

| # | Claim | Label | Source | Checked |
| --- | --- | --- | --- | --- |
| A1 | SmolVLA has \~450M parameters, \~100M in the action expert | Verified | SmolVLA paper §4.3: "450 million parameters, with approximately 100 million dedicated to the action expert" | Oct 29 |
| A2 | Expert trained from scratch with the VLM frozen; pretraining on 481 community datasets, 22.9K episodes | Verified | SmolVLA paper §3.2 Table 1 and §4.3 "We train only the action expert module, keeping the VLM frozen" | Oct 29 |
| A3 | Chunk size 50, 10 flow-matching steps at inference | Verified | SmolVLA paper §4.3; checkpoint config.json chunk\_size: 50, num\_steps: 10 | Nov 5 |
| A4 | LIBERO checkpoint configured with n\_action\_steps = 1 | Verified (config and code) / runtime count pending | config.json n\_action\_steps: 1; `lerobot-eval` calls select\_action once per environment step and changes the horizon only through `--policy.n_action_steps` (D15, Nov 10); the first headless episode counts policy calls per step | Nov 10 |
| A5 | Success falls from 80.3% at 1 executed action to 51.8% at 50 | Verified | SmolVLA paper Table 13 (action execution steps 1/10/30/50 = 80.3/82.8/70.8/51.8) | Nov 5 |
| A6 | Authors report 87.3% LIBERO average for the 0.45B model, 10 trials per task | Verified | SmolVLA paper Table 2 and §4.1 "10 trials per task" | Oct 29 |
| A7 | 88.75% is the 2.25B variant, not the 0.45B checkpoint | Verified | SmolVLA paper Table 2 | Oct 29 |
| A8 | Independent re-evaluation found 74.0% | Verified | SMILE paper \[14\] | Oct 28 |
| A9 | Expert interleaves cross-attention and causal self-attention; VLM layer skipping to N = L/2 | Verified | SmolVLA paper §3.1 "causal attention mask for the SA layers"; "N = L/2" | Oct 29 |
| A10 | Causal mask is implemented as every action token starting its own attention block | Superseded by A17 (Verified in code; was Unverified) | Paper states causal; code path make\_att\_2d\_masks / embed\_suffix not inspected. Action: read modeling\_smolvla.py and test token-0 invariance under chunk\_size 1 vs 50 with matched noise | — |
| A11 | Cross-attention K/V projections take the VLM's cached key/value states; expert k\_proj/v\_proj input dim = VLM kv dim | Verified | smolvlm\_with\_expert.py forward\_cross\_attn\_layer: key\_states/value\_states read from past\_key\_values, then expert k\_proj/v\_proj applied; re-applied at every denoising step (denoise\_step → forward → forward\_cross\_attn\_layer), so their outputs are identical across the 10 steps and can be hoisted (main, commit d490e47, Nov 9) | Nov 5 |
| A12 | Expert hidden size is 720 (0.75 × 960) | Superseded by A17 (480; was Conflict) | Paper §3.1 says 0.75 × d; checkpoint config.json says expert\_width\_multiplier: 0.5, which would give 480 if the VLM hidden is 960. LeRobot main (commit d490e47): SmolVLAConfig defaults to 0.75 and SmolVLMWithExpertModel's constructor to 0.5; the checkpoint's own config.json value is what loads, which points toward 480. Action: manifest from the loaded checkpoint. All 720×720 and E=720/F=2048 arithmetic is illustrative until resolved | Nov 5 |
| A13 | Expert has 16 layers | Superseded by A17 (32; was Unverified) | Paper uses first 16 VLM layers; config num\_vlm\_layers: 0 (no reduction) and num\_expert\_layers: -1. Either 16 or 32 expert layers gives \~100M at the two candidate widths, so parameter count does not resolve it. Action: manifest | Nov 5 |
| A14 | Per-token expert input is a projection of the sampled noise, whose shape is (batch, chunk, max\_action\_dim) with max\_action\_dim = 32, so 32 stochastic coordinates per token rather than 7, plus a time embedding, through a two-layer MLP | Superseded by A17 (Verified in code; was Unverified) | From memory of LeRobot modeling\_smolvla.py (action\_in\_proj, action\_time\_mlp\_in/out); max\_action\_dim: 32 in config; noise shape confirmed in modeling\_smolvla.py sample\_actions as (bsize, chunk\_size, max\_action\_dim) (main, commit d490e47, Nov 9); the two-layer MLP structure is still unconfirmed. Action: confirm in code | — |
| A15 | MLP ≈ 72% of expert bytes, Q/O ≈ 22%, self-attn K/V ≈ 4%, cross-attn K/V ≈ 2% | Superseded by A17 (MLP 61%, Q/O 30%; was Estimate at 720/2048/16) | From LeRobot defaults assuming hidden 720, intermediate 2048, 15 heads, 5 kv heads, head\_dim 64, 16 layers. Depends on A12/A13 | Nov 5 |
| A16 | Checkpoint HuggingFaceVLA/smolvla\_libero exists; OpenRAL card is a wrapper with no weights | Verified | HF model page; OpenRAL card "does not copy model weights" | Oct 29 |
| A17 | Checkpoint architecture (supersedes A10, A12, A13, A14, A15 and qualifies A1, A9): vlm\_model\_name SmolVLM2-500M-Instruct, num\_vlm\_layers 0 (all 32 text layers kept), num\_expert\_layers −1 (expert as deep as the VLM: 32), expert\_width\_multiplier 0.5, self\_attn\_every\_n\_layers 2, attention\_mode cross\_attn, chunk 50, n\_action\_steps 1, 10 steps, max\_action\_dim 32. Expert hidden 480; intermediate get\_intermediate\_size(480) = 1280; head\_dim 64 kept, so Q is 480→960, O 960→480, self-attention K/V 480→320, cross-attention K/V 320→320; even layers self-attention, odd layers cross-attention; ≈ 96.7M expert parameters (MLP 61%, Q/O 30%), ≈ 193 MB in bf16. Action tokens causal: att\_masks = \[1\] × chunk\_size with make\_att\_2d\_masks's cumulative-sum rule. Timestep enters through action\_time\_mlp\_in/out on concat(action embedding, sinusoidal time). select\_action executes n\_action\_steps from a queue. Model card: 0.6B parameters, so this is not the paper's 0.45B, 16-layer, 0.75-width configuration | Verified (config and code); parameter count Estimate; loaded-model manifest still to confirm | Checkpoint config.json (read through a fetch summary, not verbatim); SmolVLM2-500M-Video-Instruct text\_config (960 hidden, 32 layers, 15 heads, 5 KV heads, head\_dim 64; the Instruct variant's config assumed identical, Unverified); LeRobot d490e47 smolvlm\_with\_expert.py, modeling\_smolvla.py, common/vla\_utils.py | Nov 10 |
| A18 | Checkpoint revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 (last modified 2025-09-17). Files: config.json, model.safetensors, policy pre- and post-processor JSON with their normalizer safetensors. 604,934,176 parameters (600,902,304 in bf16, 4,031,872 in f32). Tagged as a fine-tune of lerobot/smolvla\_base, Apache-2.0, training dataset not stated | Verified (Hub API read through a fetch summary, not verbatim); confirm at first download | https://huggingface.co/api/models/HuggingFaceVLA/smolvla\_libero | Nov 10 |
| A19 | LIBERO demonstrations in LeRobot format: `lerobot/libero` (video, 1.9 GB) and `HuggingFaceVLA/libero` (PNG, 69.9 GB), the same 1,693 episodes / 273,465 frames / 40 tasks, two 256×256 cameras, state 8, action 7. LeRobot's docs say to pin the dataset revision. Which suites smolvla\_libero was trained on is not stated (Unverified) | Verified (docs at the pinned commit) | LeRobot docs/source/libero.mdx at d490e47 \[52\] | Nov 10 |
| A20 | LeRobot at commit d490e472a5d1a207cfe0255caaaf74b6d66e4656 is version 0.7.1 (commit dated Nov 6, 2026): Python ≥ 3.12, torch ≥ 2.7 and < 2.16, transformers 5.22.x; extras `smolvla` and `libero` (the latter pulls hf-libero and is Linux-only, so LIBERO does not install on macOS) | Verified | pyproject.toml at d490e47 \[52\] | Nov 10 |

**B. Latency and hardware**

| # | Claim | Label | Source | Checked |
| --- | --- | --- | --- | --- |
| B1 | π0.5 action expert: 8.8% of FLOPs, 57.2% of latency on a Hopper GPU at batch 1 | Verified | PhyAI \[5\] | Oct 28 |
| B2 | SnapFlow: 10-step denoising ≈ 241 of 274 ms end-to-end for π0.5 on A800 | Verified | SnapFlow \[4\] abstract and intro | Nov 5 |
| B3 | Jetson AGX Thor: 273 GB/s, T5000 128 GB / T4000 64 GB, same bandwidth | Verified | Forecr module comparison; Flopper spec pages | Oct 29 |
| B4 | AGX Orin: 204.8 GB/s; Orin NX/Nano \~102 GB/s | Verified | Forecr | Oct 29 |
| B5 | Orin NX 4.8 FP16 TFLOPs vs B200 4,500; Orin NX \~13 GB usable | Verified | GLUESTICK §3 and Appendix D.3 | Nov 5 |
| B6 | A100 40GB memory bandwidth ≈ 1.5 TB/s | Unverified | Not opened this project; from memory (HBM2, 1,555 GB/s). Action: NVIDIA datasheet | — |
| B7 | CHASE-VLA W4A4: π0.5 expert 26.957 → 21.069 ms (1.279×), GR00T N1.6 43.003 → 37.563 ms (1.145×) on A100 batch 1 | Verified | CHASE-VLA Table 5 | Oct 29 |
| B8 | GR00T N1.7 on Thor: 125 → 39.9 ms with mixed NVFP4; TRT bf16 action head 45.0 ms vs backbone 27.0 ms | Verified | Jetson AI Lab tutorial \[17\] | Oct 29 |
| B9 | π0.5 on Thor: \~163 → \~94 ms (Jetson AI Lab); \~44 ms (FlashRT) | Verified, discrepant | Jetson AI Lab openpi tutorial; FlashRT wiki. Different pipelines; which is comparable to ours is unresolved | Oct 29 |
| B10 | On Thor, vision, VLM and action stages are all memory-bound; on A100/4090 only the action stage is | Verified | VLA-Perf Table 3 | Oct 29 |
| B11 | 2:4 pruning gives 1.83–1.89× end-to-end on Thor | Verified | GLUESTICK Table 6 | Nov 5 |
| B12 | Kernel launch gap ≈ 2 µs under CUDA graphs, ≈ 8 µs eager | Estimate | Assumption; to measure on A100 and Thor | — |
| B13 | QCRL: W4A16 π0.5 reaches 99.3% of FP; NVFP4 1.73× on Thor | Verified | QCRL abstract \[30\] | Nov 5 |
| B14 | Jetson prices after July 2026 increase: Thor dev kit $5,499; T4000 module $2,999; AGX Orin dev kit $3,499 | Verified | CNX Software, Notebookcheck | Oct 29 |

**C. Compression literature**

| # | Claim | Label | Source | Checked |
| --- | --- | --- | --- | --- |
| C1 | CHASE-VLA: W4A4 on MLP + attention projections; π0.5 97.3% (FP16 97.1%), GR00T N1.6 91.5% (FP16 98.0%), CogACT 90.5%; 73.4% weight storage cut; 20 episodes/task | Verified | CHASE-VLA Tables 2, 3, Fig. 4 | Oct 29 |
| C2 | CHASE-VLA supplementary: π0.5 expert 18 blocks, 10 steps, 50×7 chunk; GR00T N1.6 32 blocks, 4 steps, 16×7 | Verified | CHASE-VLA Supp. Table 1 | Oct 29 |
| C3 | Earlier VLA PTQ (QuantVLA) quantized MLP only, attention in FP | Verified | CHASE-VLA §1 and Table 1; QuantVLA | Oct 29 |
| C4 | Ω-QVLA: uniform W4A4 on backbone + full DiT head via SVD-Hadamard rotation; π0.5 98.0%, GR00T N1.5 87.8%; 71.3% memory cut | Secondary report | Secondary pages (SOTAVerified, ChatPaper); paper PDF not opened; reference is the code repo | Oct 29 |
| C5 | QSVD: joint SVD over concatenated QKV sharing one down-projection; importance score E\[(σᵢ\[UᵀGV\]ᵢᵢ)²\] from task-loss gradients; QuaRot rotation + learned β; SmolVLM-2B ASVD 0.69% vs QSVD-noQ 79.57% at R1 = 80%; self-attention QKV only; 256 calibration samples; 13.1× is an RTX 4070 16K-cache offloading result | Verified | QSVD paper §3.1–3.3, Table 1, Fig. 4, §5 | Nov 5 |
| C6 | GLUESTICK: VLA backbone spectra flatter than LLM/VLM; fine-tuning shifted AUC/k 1.32→1.55 (LM) and 0.84→0.92 (vision); SVD on OpenVLA 0% at ranks 200–1,000, 0.4% at 2,000, 21% at 2,500 | Verified | GLUESTICK Fig. 2, Appendix B, Table 18 | Oct 29 |
| C7 | GLUESTICK's spectral comparison is on language backbones, not an action expert | Verified | GLUESTICK §4.2 | Oct 29 |
| C8 | NestedVLA found no usable low-rank structure in fine-tuning deltas | Unverified | Abstract describes hypernetwork skill consolidation; claim not found. Action: read full paper or drop | Nov 5 |
| C9 | RL post-training updates are low-rank and concentrated in timestep modules (π0.5, GR00T N1.5/N1.6) | Verified | Low-Rank Structure of VLA RL \[10\] abstract | Oct 29 |
| C10 | Velocity trajectory across denoising steps is low-rank; 10 → 2 steps | Verified | Reducing Temporal Redundancy \[11\] | Oct 29 |
| C11 | π0 LoRA study: action-expert MLP fine-tuning deltas use \~31% of spectrum at 95% energy vs \~47% for VLM | Verified | \[2\] §3.5 | Oct 29 |
| C12 | PIVOT-Q: QuantVLA drops GR00T-N1.5 LIBERO-Plus 82.1 → 67.0% while LIBERO stays 97%; vulnerable-state on-policy distillation recovers to 81.3% | Verified | PIVOT-Q abstract and GitHub README | Oct 29 |
| C13 | ActQuant: action-guided mixed-precision PTQ, ≤ 3 bpw, OpenVLA-OFT 95.0% and π0.5 94.8% retained | Verified | ActQuant abstract \[35\] | Nov 5 |
| C14 | Traj-MC: trajectory-aware low-rank calibration via activation second moments, diffusion language models | Verified | Traj-MC abstract \[29\] | Nov 5 |
| C15 | CTEC: cross-timestep error propagation in quantized image diffusion | Verified (title/venue only) | AAAI 2025 listing \[28\] | Nov 5 |
| C16 | SARA adapts SVD rank per step and block at inference; SlimDiff timestep-stratified rank allocation; AdaTSQ per-step bit-width | Verified | NeurIPS 2026 poster; SlimDiff; AdaTSQ abstracts | Nov 5 |
| C17 | Any-Precision LLM: bit-plane storage enables runtime bit-width selection with a custom kernel | Verified | \[25\] abstract | Nov 5 |
| C18 | SVDQuant keeps a 16-bit low-rank branch plus 4-bit residual; its Nunchaku engine fuses the branch | From memory | Not opened this project | — |
| C19 | ASVD = diagonal activation-magnitude scaling before SVD; SVD-LLM = Cholesky whitening of activation covariance | From memory (standard) | QSVD §2.2 describes both in one line | Nov 5 |
| C20 | VLAQuantBench: closed-loop PTQ benchmark on π0.5, π0, OpenVLA-OFT, X-VLA; 409 runs, \~94,600 episodes; 200 episodes/config (190 X-VLA), official LIBERO initial states, 95% Wilson intervals, task-clustered two-stage bootstrap for selected contrasts; π0.5 attention + adaRMS subset 1.5% \[1, 4\] under uncalibrated RTN; "equal nominal bit width does not make their representations identical" (about quantizers at matched bits); no SmolVLA, no low-rank, no memory-matched comparison | Verified | arXiv 2609.25376, §3, §4, Tables 1 and 4; SmolVLA appears only in related work | Nov 9 |
| C21 | LightDP (On-Device Diffusion Transformer Policy, ICCV 2025): block pruning of small diffusion-policy denoisers (DP-T 8.97M → 2.65M; MDT-V 22.5M → 7.45M) with SVD reconstruction error used only to initialize Gumbel-Softmax block gates, plus consistency distillation to ≤ 4 steps; iPhone 13 and Orin NX; pruning, not weight factorization, and no quantization-vs-rank comparison | Verified (abstract and method summary) | arXiv 2508.00697 | Nov 9 |
| C22 | QCRL does not report SmolVLA: models named are π0.5 (INT4 W4A16, NVFP4 W4A4) and GR00T N1.5; the 4,050 trials are real-world across "five VLA models" the abstract does not name. An external review's claim that QCRL "covers SmolVLA quantization" is unsupported | Verified (abstract level); full paper not opened | arXiv 2610.15675 abstract and page; amends B13 | Nov 9 |
| C23 | Further flow-VLA compression work named by an external review but not opened: Quantize by Doing (arXiv 2610.38661, exists, same group as VLAQuantBench), SACQ, DA-PTQ, VLA-Rolling, REACT, GeoAAC, Dobi-SVD, FlashSVD, GLOWQ, a budget-aware quant+low-rank LLM method (IJCAI 2026) | Unverified | Action: open before the related-work section of the report is final | — |
| C24 | torch.compile modes: default = Inductor fusion, no CUDA graphs; reduce-overhead = CUDA graphs, applied only to graphs that do not mutate inputs; max-autotune = autotuned matmul templates plus CUDA graphs and coordinate-descent tuning; max-autotune-no-cudagraphs exists. Inductor's back-to-back GEMM pass (fx\_passes/b2b\_gemm.py) is off by default (config.b2b\_gemm\_pass = False), handles 2-D matmuls only, and fires only when its load-ratio model exceeds 1 | Verified (code) | torch 2.15.0 torch.compile docstring, torch.\_inductor.list\_mode\_options(), torch/\_inductor/config.py and fx\_passes/b2b\_gemm.py | Nov 9 |

**D. Benchmarks and robustness**

| # | Claim | Label | Source | Checked |
| --- | --- | --- | --- | --- |
| D1 | LIBERO: 130 tasks, four standard suites of 10 tasks; standard protocol 50 episodes/task | Verified | LIBERO paper; GLUESTICK Appendix C.1 "each task repeated 50 times" | Oct 29 |
| D2 | Actions are 7-dimensional (6-DoF EEF delta + gripper) | Verified | HF discussion on smol-libero action dim 7; LIBERO docs | Oct 29 |
| D3 | LIBERO control rate 20 Hz | Verified (code) | control\_freq defaults to 20 in LeRobot's LiberoEnv and in docker/sim/configs/libero.yaml at d490e47 \[52\] | Nov 10 |
| D4 | LIBERO-Plus: seven perturbation dimensions; models drop from \~95% to below 30% | Verified | LIBERO-Plus CVPR 2026 abstract | Oct 29 |
| D5 | LIBERO-PRO: π0.5 position-shift success 0.08–0.38 by suite; π0 and OpenVLA 0 | Verified | LIBERO-PRO Fig. 2 | Oct 29 |
| D6 | VLASH: π0.5 LIBERO 96.8 → 94.6% at 3-step delay; SmolVLA 79.0 → 79.1% | Verified | VLASH Table I | Oct 29 |
| D7 | ReflexBench: six dynamic tasks, latency-aware sync/async evaluation | Verified | ReflexBench abstract | Oct 29 |
| D8 | OGPO introduces LIBERO-ReAct: 20-step mid-execution perturbations (action noise ±0.1, gripper ±0.5, EEF ±0.2) | Verified | OGPO page \[31\] | Nov 5 |
| D9 | OGPO "total robustness" 76.7% SFT / 81.8% OGPO is ReAct-only | Unverified | Page calls it "total" in a paragraph that also reports LIBERO-Plus categories; likely an aggregate. Action: read results table | Nov 5 |
| D10 | CoRe tests mid-episode disruptions on π0.5, LIBERO-Long; bare VLA 12.8–43.5% → 69.8–92.7% | Verified | CoRe page \[32\] | Nov 5 |
| D11 | LIBERO-ReAct is released and compatible with LeRobot | Unverified | Paper describes it; no package located. Action: find code | — |
| D12 | Coupling Execution Horizon and Sampling Depth: fewer steps hurt under frequent replanning, help under long open-loop; π0, π0.5, GR00T | Verified | \[36\] abstract | Nov 5 |
| D13 | DART injects noise into the supervisor's demonstrations during collection | Verified (standard) | DART abstract \[34\] | Nov 5 |
| D14 | Jacobian matching for distillation: Srinivas & Fleuret ICML 2018; Sobolev training: Czarnecki et al. 2017 | From memory | Links supplied by review; not opened | — |
| D15 | LeRobot's LIBERO integration at d490e47: episode caps are 280 steps for libero\_spatial and libero\_object, 300 for libero\_goal, 520 for libero\_10, 400 for libero\_90; fixed initial states (init\_states) and hard resets are the defaults; 10 settling steps after reset. `lerobot-eval` seeds once (set\_seed, then environment seeds start\_seed + i) and calls policy.select\_action(observation) once per environment step without a noise argument, so flow noise comes from the global generator. SmolVLAPolicy.select\_action accepts `noise=` and passes it to sample\_actions (shape (batch, chunk\_size, max\_action\_dim), cast to float32) | Verified (code) | src/lerobot/sim/integrations/libero.py, src/lerobot/scripts/lerobot\_eval.py, modeling\_smolvla.py at d490e47 \[52\] | Nov 10 |

**E. Derived and toy results**

| # | Claim | Label | Source | Checked |
| --- | --- | --- | --- | --- |
| E1 | 720×720 at INT8 = 518,400 B; bf16 SVD matches at r = 180 | Estimate (illustrative) | Derivation §4.1; depends on A12 | Nov 5 |
| E2 | MLP block E=720, F=2048: 4-bit raw 2,211,840 B; with per-channel bf16 scales 2,221,472 B; with group-128 bf16 scales 2,280,960 B; bf16 factors 15,168 B per rank; match at r ≈ 146 (raw or per-channel) / 150 (group-128) | Estimate (illustrative) | Derivation §4.1; per-channel is what the toy harness implements | Nov 5 |
| E3 | Truncation error has rank n − r (540 at r = 180) with fixed row space | Verified (algebra) | Derivation §4.3 | Nov 5 |
| E4 | E‖Ex‖² = Tr(E C\_x Eᵀ): input orientation matters for both compressors | Verified (algebra) | Derivation §4.3 | Nov 5 |
| E5 | Toy, per-channel RTN (2,221,472 B) vs rank-146 SVD (2,214,528 B): both compressors give coherence C ≈ 0.94, growth slope ≈ 1.06; SVD 6× worse at flat spectrum, 2× better at steep; conditioning shift raises teacher-forced local velocity error 16% (SVD) vs 9% (W4) at the steep spectrum, and endpoint error 35% vs 27%; independently reproduced by GPT review 4 on the same environment | Toy (Reproduced) | compounding\_harness.py, seed 0, §6; rerun Nov 5 after the storage-accounting fix | Nov 5 |
| E6 | Toy: under a two-term launch-plus-bytes model at Thor bandwidth (273 GB/s), launch penalty 1.09 MB-equivalent; SVD faster than W4A16 only below r = 78 at 2 µs launches; no rank at 8 µs. At A100 bandwidth (\~1.555 TB/s) the penalty is 3.1–6.2 MB at 1–2 µs, so SVD is never faster than a fused W4A16 kernel and is faster than dense bf16 only below r ≈ 173–378 | Toy / Estimate, conditional on the model | svd\_rank\_budget.py; depends on B12; omits arithmetic, kernel selection, occupancy, caching | Nov 5 |
| E7 | Recovery matrix 20 tasks × 50 episodes × 4 policies × 4 conditions = 16,000 episodes; ≈ 267 GPU-h at 60 s/episode | Estimate | Arithmetic; per-episode time unmeasured | Nov 5 |
| E8 | At H = 1 with causal action attention, token 0's output is independent of tokens 1–49 | Hypothesis (exact if A10 holds) | Derivation §4.5 | Nov 5 |
| E9 | Core nominal matrix (R0; GPTQ at 2, 3, 4 bits; importance-aware SVD at four points; S3; 500 episodes each) = 4,500 episodes ≈ 75 GPU-h at 60 s, ≈ 91 with week-1 checks, spectra and the open-loop screen, before recovery, a second horizon or repetitions | Estimate | Course design configuration table; §4.7 | Nov 5 |
| E10 | Inductor's back-to-back GEMM cost model, run on SmolVLA MLP shapes (50 tokens; gate-up and down; ranks 77–432), gives average load ratios 0.029–0.046 at E = 720 and 0.041–0.068 at E = 480, so the pass declines. A fused low-rank kernel that avoids re-reads would, under the two-term model, beat dense bf16 below rank ≈ 583 and fused W4A16 below ≈ 150 (upper bounds, E = 720, F = 2048) | Reproduced (cost model) / Estimate (bounds) | load\_ratio\_left over b2b\_gemm\_configs in torch 2.15; §4.2 | Nov 9 |

Rows E1, E2, E6 and E10 are computed at the earlier illustrative dimensions (E = 720, F = 2048) and are kept as history. The figures at the checkpoint's dimensions (480 / 1280; A17) are in §3.2 and in `course-design.md` §3.2, and `tests/test_storage.py` and `tests/test_latency_model.py` reproduce them.

## 2. Related work map

Every paper found, placed by what it compresses, how, and how it evaluates. "Gap" says what it leaves for this project. Ledger rows in brackets.

| Paper | Component | Method | Evaluation | What it shows | Gap it leaves |
| --- | --- | --- | --- | --- | --- |
| QuantVLA (CVPR 2026) \[C3\] | VLA backbone + expert MLP | PTQ, W4A8, attention kept FP | LIBERO | First VLA PTQ framework; π0.5, GR00T N1.5 | Attention projections; low-rank |
| CHASE-VLA (Oct 2026) \[C1, C2, B7\] | Expert MLP + attention | W4A4 with chunk-aware activation scales | LIBERO, 20 ep/task; A100 latency | FP16-level on π0.5; 91.5% on GR00T N1.6; 1.28× A100 | No low-rank; no edge measurement; GR00T loses 6.5 pts |
| Ω-QVLA (May 2026) \[C4\] | Backbone + full DiT head | Uniform W4A4 via SVD-Hadamard rotation + per-step scaling | LIBERO | 98.0% π0.5; SVD used only as rotation | Rank truncation untested |
| ActQuant (May 2026) \[C13\] | VLA weights | Action-guided mixed-precision, ≤ 3 bpw | LIBERO, real UR3 | 95% retained at 3 bpw | Precision allocation only; no low-rank |
| QSVD (NeurIPS 2025) \[C5\] | VLM self-attention QKV | Joint SVD + gradient rank allocation + quantization | VQA accuracy | Beats ASVD/SVD-LLM at equal cost | VLMs only; no MLP; no closed loop |
| GLUESTICK (Oct 2025) \[C6, C7, B5, B11\] | VLA language backbone | 2:4 pruning + low-rank gap correction | LIBERO, VLN, Thor/Orin NX latency | VLA spectra flat; plain SVD 0%; pruning 1.8× on Thor | Action expert spectra; SVD on expert |
| PIVOT-Q (Oct 2026) \[C12\] | Quantized VLAs (GR00T N1.5, π0.5) | Vulnerable-state on-policy distillation | LIBERO-Plus, 560 episodes | Quantization hides damage under shift; recovers 66.6 → 81.3% | Low-rank; LIBERO-PRO; mid-execution disturbance |
| QCRL (Oct 2026) \[B13\] | Quantized VLAs | Offline RL from FP rollouts | Real-world, 3 visual-shift conditions; Thor | W4A16 π0.5 99.3% of FP; NVFP4 1.73× on Thor | Nominal and static-shift only |
| Low-Rank Structure of VLA RL \[C9\] | RL weight updates | Spectral analysis | — | Updates low-rank, concentrated in timestep modules | Trained weights themselves |
| Reducing Temporal Redundancy \[C10\] | Velocity trajectory | Low-rank in time | LIBERO | 10 → 2 steps | Weights |
| LoRA for π0 \[C11\] | Fine-tuning deltas | SVD of deltas | UR5e tasks | Expert MLP deltas use 31% of spectrum | Weights |
| Traj-MC (Oct 2026) \[C14\] | Diffusion LLMs | Trajectory-aware low-rank calibration | Math benchmarks | Calibrate on trajectory states | VLA; closed loop; horizon |
| CTEC (AAAI 2025) \[C15\] | Quantized image diffusion | Cross-timestep error correction | FID | Errors propagate across steps | VLA; low-rank |
| SARA (NeurIPS 2026), SlimDiff, AdaTSQ \[C16\] | Image/video DiTs | Per-step rank or bit-width | FID | Rank/bits as a per-step knob | VLA; closed loop |
| Any-Precision LLM \[C17\] | LLM weights | Bit-plane storage | Perplexity | Runtime bit selection needs custom kernel | — |
| PhyAI \[B1\], SnapFlow \[B2\], VLA-Perf \[B10\], rMuscle, FlashRT \[B9\], Jetson AI Lab \[B8\] | Inference profiling | — | Latency on H100/A100/4090/Thor | Expert is latency-dominant and memory-bound | Compressed-expert latency on Thor |
| LIBERO-Plus \[D4\], LIBERO-PRO \[D5\] | Benchmarks | Static perturbations | — | Near-saturated models collapse under shift | — |
| VLASH \[D6\], ReflexBench \[D7\], RTC/RAVEL | Benchmarks, async inference | Latency injection; dynamic tasks | LIBERO; Kinetix; DOM | Latency barely matters on static LIBERO | Compression × latency × dynamics |
| OGPO / LIBERO-ReAct \[D8, D9\], CoRe \[D10\] | Recovery | Mid-execution perturbation; counterfactual realignment | LIBERO-ReAct, LIBERO-Long, real xArm6 | Recovery is measurable and trainable | Compression × recovery |
| Horizon × sampling depth \[D12\] | Flow VLAs | Joint H and step-count tuning | LIBERO, RoboTwin, Franka | Fewer steps hurt at short H, help at long H | Compression × H |
| DART \[D13\], Jacobian matching, Sobolev \[D14\] | Imitation / distillation | Perturbed-state supervision | — | Prior for any response-matching criterion | — |
| VLAQuantBench \[C20\] | Whole VLA, layer subsets (π0.5, π0, OpenVLA-OFT, X-VLA) | PTQ: RTN, AWQ, NF4, int8, SmoothQuant; W4A4 and W3; calibration recipes | Closed loop, LIBERO and three other sim families; 200 ep/config, Wilson intervals; real kernels and a physical robot | Layer scope × format × calibration interact; attention + adaRMS most fragile on π0.5; equal bits ≠ equal representation | Varies the quantizer at fixed bits; no low-rank arm, no memory-matched comparison, no SmolVLA. This project varies bits vs rank at fixed bytes |
| LightDP \[C21\] | Small diffusion-policy denoisers (9M, 22M) | SVD-initialized block pruning + consistency distillation | Push-T, CALVIN, LIBERO, real arms; iPhone and Orin NX latency | Up to \~93× speedup with small success drops | Pruning of whole blocks, with training; no weight factorization, no quantization comparison, no VLA-scale expert |

**Open cells, as far as found (Nov 5):** SVD vs quantization vs hybrid on a flow-matching action expert at matched storage in closed loop; whether the execution horizon changes the winner; whether compression damages recovery at matched nominal error; transported-error alignment measured in a VLA; step-scheduled rank in a VLA. Each is open as a combination; every piece has a neighbor in the table.

## 3. Extension scope (beyond Dec 20)

Everything the course deliverable does not commit to, with status and what each needs. Items here are not dropped; they are sequenced after the core.

| Extension | Status | Needs | Notes |
| --- | --- | --- | --- |
| π0.5 (expert 18 blocks, 10 steps, \~430M est.) and GR00T N1.6 (32-block DiT, 4 steps, \~1.4B est.) \[C2\] | Planned, spring | Parameter manifests; CHASE-VLA baselines exist for both | Both score 97–98% on LIBERO in FP16, so damage shows only under shift. π0/π0.5 use bidirectional action attention (from memory, unverified), so the H = 1 prefix-only shortcut does not transfer |
| GR00T N1.7 | Possible | Resolve the 16- vs 32-layer DiT discrepancy by counting | Newer backbone (Cosmos-Reason2-2B); TensorRT export pipeline eases Thor deployment |
| LIBERO-Plus (primary robustness measure) and LIBERO-PRO (object/semantic only) | Planned, spring | Episode budget | Moved out of the course core by the episode arithmetic \[E7\]; compare directly with PIVOT-Q |
| Jetson AGX Thor T4000 latency | Planned, spring | Board access (none found for rent; ask Zhang's lab or buy) | TensorRT, FlashRT and QCRL Thor numbers are baselines; the project's claim is compressed-expert latency, not Thor measurement itself |
| Step-scheduled rank | Gated on per-step importance scores | Per-step sensitivity sweep | Rank is a native runtime knob for factorized weights (contiguous column slice); bit-width needs bit-plane storage. Matched test: schedule with mean rank r̄ vs fixed r̄. Direction measured, not assumed: late steps have the stiffest Jacobian, early errors propagate longest |
| Direction-exchange experiment | Gated on a recovery effect | Recovery pilot | Restore candidate corrective singular components while removing the same number of others; controls: random exchange, singular-value-ordered exchange; evaluate on held-out cases |
| Transported-Jacobian coherence diagnostic | Optional | JVPs along teacher trajectory | Full version of the C index; finite-deletion scores first |
| Recovery-sensitive rank allocation as a method | Gated on recovery pilot | Perturbed-state calibration pool | Counts as a method only if it beats plain distillation on the same perturbed states at equal bytes |
| Prefix-only sampling as a latency baseline | Gated on A10 | Causal-mask verification | If exact, chunk\_size = 1 at H = 1 is free; its latency change also diagnoses whether the expert is weight-bound |
| Dynamic tasks (ReflexBench, DOM) | Future | Adapters for SmolVLA | Static LIBERO cannot show latency mattering \[D6\] |
| Non-NVIDIA edge (Qualcomm IQ10/IQ-9075, Intel B60, Ascend) | Future | Porting off CUDA | Cross-hardware VLA studies exist; a project in itself |
| Real-robot experiments | Future | Robot access | — |
| Custom fused kernels (SVDQuant-style) | Future | CUDA/CUTLASS or Triton on aarch64 | Needed before the residual hybrid can be a speed candidate |

### 3.1 Proposed reframing under evaluation (Nov 6, not adopted)

A brief dated Nov 6 proposes replacing the matched-bytes question with: given a latency budget, which of three efficiency levers (denoising steps K, execution horizon H, weight compression) buys the most closed-loop success, and do they interact? The Course design tab is unchanged; this section records the proposal and its evaluation so the decision can be made after the week-1 checks.

**Cost model behind the three levers (Estimate).** Per episode of T robot steps, time ≈ (T/H) · (t\_VLM + K · t\_step), with t\_step ≈ n\_launch · t\_ℓ + bytes\_moved / B. Steps K scale the expert term linearly and trade discretization error; H scales the number of calls and trades open-loop drift (A5: 80.3% at H = 1 → 51.8% at H = 50); compression scales only bytes\_moved, so its latency effect is bounded by the bandwidth share of t\_step. K and H are inference-time hyperparameters that leave the weights unchanged; compression is the only lever that changes the function computed. Fine-tuning for any lever (few-step distillation, chunk-noise training, QAT) is out of scope: it costs GPU-hours the evaluation needs and confounds lever with training. Calibration (GPTQ, importance-aware SVD on \~256 samples) is the only parameter fitting retained.

| Proposal element | Verdict | Reason |
| --- | --- | --- |
| Old answer is predictable | Partly | W4 near-lossless on other VLAs (B13, C1); SVD collapse was on a backbone, not an expert (C7). A prior, not a result |
| Storage is not a bottleneck | Valid | \~200 MB expert fits anywhere; bytes matter only as bytes moved per step |
| Compression is second-order to K and H | Likely valid, to be measured | H = 5 cuts calls 5×; K 10→4 cuts expert work 2.5×; W4 cuts only the bandwidth share |
| Mechanism: fewer steps → less compounding → ranking may flip | Rejected | δx ≈ Σ h·e\_k. Aligned errors (toy C ≈ 0.94) sum to ≈ e regardless of K; uncorrelated errors shrink as 1/√K with more steps. Replacement hypothesis: at K = 4 the FP16 discretization error grows and masks compressor differences |
| Novelty of steps × H | Overlaps D12 | Coupling Execution Horizon and Sampling Depth (2026) covers steps × H on π0, π0.5, GR00T. New axis here is compression and its interaction; read D12 in full first |
| A100 latency gain from compression | Unverified, decisive | Estimate: \~200 MB bf16 at \~1.5 TB/s ≈ 0.13 ms/step vs \~0.5 ms/step launch overhead. If W4 saves < 10% of t\_step, compression is a success-for-memory lever on A100 and latency claims become Thor projections |
| Power for interactions | Marginal | \~±3.9 pt CI per main effect at 500 paired episodes; interaction is a difference of differences, \~±5.5 pt. Report interactions as CI estimates, not tests |
| Deliverable 3 (prefix calibration at H = 5) | No arm in the matrix | Needs 1–2 prefix-calibrated arms, 500–1,000 episodes |
| Marlin kernel, fused low-rank path, GPTQ-W3 latency | Risky | Hidden size 720/480 is not a 128-multiple (Unverified against Marlin's shape rules); no 3-bit Marlin path; fused low-rank kernel is weeks of work. Use two cuBLAS matmuls under CUDA graphs; W3 = bytes-moved projection only |
| SVD at the W4 byte budget | Likely collapses | Rank ≈ 0.2E. Screen several ranks open-loop; promote survivors. Matched bytes is a leftover of the old framing on a latency frontier |
| Budget | Fits without slack | 133 + 30 + 20 + prefix arms 8–17 + calibration/screens/debug 20–30 ≈ 210–230 of 240 usable GPU-h |

**Cut-down version that fits (Estimate).** Steps {10, 4} × H {1, 5} × {FP16, GPTQ-W4, best SVD rank} = 12 arms + 1 prefix-calibrated arm at H = 5 ≈ 6,500 episodes ≈ 110 GPU-h at 60 s. Scope-cut order: LIBERO-Plus, W3, second SVD rank, K = 4. Never cut FP16, GPTQ-W4, the best SVD arm, the prefix arm, or the paired design. Fallback if week 1 slips past Nov 15: the current Q1 comparison at H = 1, a strict subset of this matrix.

**Decision gate.** Adopt only after week-1 check 7 (A100 latency breakdown with CUDA graphs) shows whether compression moves t\_step, and after D12 is read in full.

**Bridge scope (option, not adopted; superseded by §3.2, adopted Nov 7).** A smaller version of the current core that finishes on time whichever way the decision gate goes. The current core's H = 1, K = 10 rows are a strict subset of the proposed matrix, so the bridge can grow into it.

| Keep | Drop or defer | Add |
| --- | --- | --- |
| R0 FP16 reference | Budget B1 (INT8-equivalent): W8 is near-lossless and low information | Check 7: A100 latency breakdown under CUDA graphs, week 1, \~5 GPU-h |
| Q1 RTN and Q2 GPTQ-W4 at B2 | S3 prefix calibration at H = 1 (move to H = 5 if the reframing is adopted) | Lever test: FP16 at K = 10, 4, 2 on 3 tasks × 10 episodes, \~1 GPU-h |
| S1 plain SVD and S2 importance-aware SVD at B2, after an open-loop rank screen | Gated rows H1, S4, S5 and the recovery study | One H = 5 FP16 arm, 500 episodes, \~8 GPU-h |
| Paired design, storage accounting, spectra, manifest, causal-mask test |  |  |

Cost (Estimate, 60 s/episode): 5 arms × 1,000 episodes = 5,000 episodes ≈ 83 GPU-h, plus \~15 GPU-h of additions, leaving \~140 GPU-h of the 240 usable. If check 7 shows compression saves real time, the reserve funds the K = 4 and H = 5 rows and the prefix arm. If not, the Q1 answer is complete and the step and horizon numbers serve as context.

**Deployment-relevant metrics (tail latency adopted into Course design Q5 and §3.1 on Nov 9; action smoothness still an option).** Two additions computed from runs already in the plan, with no extra episodes. Deployed VLAs treat latency as a pass/fail budget set by the control rate (inference must finish within H/f, the time the current chunk takes to execute), so the worst case matters more than the mean; and compressed weights can add small action noise that shows up as jerky motion before it shows up as failure.

| Metric | Definition | Why it matters | Cost |
| --- | --- | --- | --- |
| Tail latency | p50 and p99 wall-clock per policy call (VLM prefix + K expert steps), CUDA graphs on, per compressor; report jitter as p99 − p50 | A late action is a missed control cycle; deployment budgets are set against the worst case (e.g. Helix's fast loop runs at 200 Hz, a 5 ms budget per cycle) | Timing hooks in the eval harness; no extra GPU-h beyond the latency profile |
| Action smoothness | Mean squared jerk of executed actions, ∑‖a\_{t+1} − 2a\_t + a\_{t−1}‖² / T per episode, paired against FP16 on the same seeds; also the per-step L2 difference from the FP16 action under matched noise | Jerk wears hardware and looks unsafe; it can separate compressors even where success rates tie | Logged actions from existing rollouts; post-hoc analysis only |

Both are Estimate-level rationale until measured; neither changes the success comparison. Sources: [PhAIL](https://arxiv.org/abs/2605.29710) (real-robot evaluation practice, time-to-success), [Helix](https://www.aiwiki.ai/wiki/helix_vla) (control rates).

### 3.2 Rate–distortion curves (adopted into Course design Nov 7)

Replaces the two fixed budgets (B1, B2) with one plot: stored expert bytes, including scales and metadata, against closed-loop success, one curve per compression family. The result is each family's non-inferiority threshold, the smallest byte point whose one-sided 95% paired lower bound on (success − FP16 success) is above −5 points, rather than one matched-bytes comparison. A two-sided ±5-point equivalence test was rejected: with a ±3.9-point CI it passes only when the observed difference is within ≈ 1.1 points. "Knee" is not used because it has no test.

**Byte arithmetic (Estimate; illustrative at E = 720, F = 2048, N = 3EF = 4,423,680 MLP weights; recomputed after the manifest resolves A12/A13).** B\_q(b) = N·b/8 + scale bytes, with bf16 scales in groups of 120 along E for gate and up (128 divides neither 720 nor 480) and 128 along F for down; matched rank r(b) = ⌊B\_q(b) / 2(2E + 3F)⌋ for joint gate-up plus down factors in bf16.

| Bits | MLP bytes (scales as above) | Matched rank | Rank / E |
| --- | --- | --- | --- |
| 2 | 1,178,112 | 77 | 0.11 |
| 3 | 1,731,072 | 114 | 0.16 |
| 4 | 2,284,032 | 150 | 0.21 |
| 6 | 3,389,952 | 223 | 0.31 |
| 8 | 4,495,872 | 296 | 0.41 |
| dense bf16 | 8,847,360 | 583 | 0.81 |

**At the checkpoint's dimensions (E = 480, F = 1280; A17), which replace the table above for planning.** N = 1,843,200 MLP weights; scales 30,080 B; 9,600 B per rank; dense bf16 3,686,400 B.

| Bits | MLP bytes | Matched rank | Rank / E |
| --- | --- | --- | --- |
| 2 | 490,880 | 51 | 0.11 |
| 3 | 721,280 | 75 | 0.16 |
| 4 | 951,680 | 99 | 0.21 |
| 6 | 1,412,480 | 147 | 0.31 |
| 8 | 1,873,280 | 195 | 0.41 |
| dense bf16 | 3,686,400 | 384 | 0.80 |

Rank fractions are unchanged, so the configuration table's byte points stand. Latency bounds under the two-term model (§4.2) at these dimensions: on the A100 the two-launch penalty (3.1–6.2 MB) exceeds W4A16's 0.95 MB block at any rank and exceeds the dense 3.69 MB block at 2 µs, leaving SVD faster than dense only below rank 60 at 1 µs; on Thor (0.55–1.09 MB) SVD is faster than dense below rank 270–327 and faster than W4A16 only below rank 42 at 1 µs. Expert weight read per step: \~0.12 ms on the A100, \~0.71 ms on Thor, against an estimated 0.5–1 ms of launch overhead for 32 layers, so the A100 step is predicted launch-bound and W4A16 is predicted to change it by under 10%.

On a flat spectrum the relative truncation error is √(1 − r/n) ≈ 0.77 at the 8-bit-matched rank, so SVD may have no threshold inside the 2–8-bit byte range. The SVD curve therefore adds points at ≈ 0.6E and ≈ 0.8E; "SVD cannot compress this expert below X bytes" is itself a finding.

**Methods.** Closed loop: GPTQ, and importance-aware SVD (joint gate-up, joint self-attention QKV; importance from the flow-matching loss, §3.4). Open loop only: RTN and plain SVD as floors; GPTQ at 6 and 8 bits, expected near-lossless by inference from W4 results on other VLAs (B13, C1, C13, which report W4, not W8); importance-aware SVD at ≈ 0.8E, which costs as much as dense bf16 and is a diagnostic, not a compression point. Executed-prefix calibration (S3) runs at the SVD point the screen places between collapse and lossless, not at the W4-matched point, where P2 predicts collapse and a calibration difference would be invisible.

| Item | Episodes | GPU-h (Estimate, 60 s/ep) |
| --- | --- | --- |
| Week-1 checks incl. A100 latency breakdown and throughput pilot | — | \~10 |
| Spectra, raw and activation-whitened | — | \~2 |
| Open-loop screen, \~20 configurations | — | \~4 |
| FP16 reference, chosen suite | 500 | 8.3 |
| GPTQ closed loop, 2/3/4 bits | 1,500 | 25 |
| Importance-aware SVD closed loop: W4-matched, W8-matched, ≈ 0.6E, one screen-chosen point | 2,000 | 33.3 |
| S3, executed-prefix calibration at the in-between SVD point | 500 | 8.3 |
| Total | 4,500 | \~91 (reserve \~149 of 240 usable) |

Reserve order: LIBERO-Plus at the two threshold points plus its own FP16 reference (3 × 500 episodes ≈ 25 GPU-h), then the §3.1 steps × horizon rows at those points. Rejected: 20 episodes per task (CI ≈ ±6 points, wider than the 5-point margin). Suite: LIBERO-10 for headroom unless the throughput pilot shows its longer episodes (520-step cap vs 280 on Spatial in LeRobot at the pinned commit, D15) break the budget.

**Pre-registered predictions.** P1 (Hypothesis): GPTQ non-inferior at 4 bits and not at 2. P2 (Hypothesis): importance-aware SVD not non-inferior at any rank up to the 8-bit-matched rank. P3 (Hypothesis): no crossover, tested at the W4-matched point closed loop (the only point where both families run closed loop) and at every point of the open-loop screen. P4 (exploratory): whitened 90%-energy rank predicts SVD's threshold to within one point. P5 (exploratory): open-loop executed-action error orders the closed-loop points, reported within each family because the gap between families alone would inflate a pooled Spearman ρ. Power (Estimate): SE ≈ 2.0 points at 500 paired episodes and \~20% discordance; a point passes non-inferiority when its observed difference exceeds ≈ −1.7 points (power ≈ 0.80 at a true difference of 0, ≈ 0.44 at −2).

**Figure.** x: stored bytes, log scale, top axis in bits per weight. y: success %. FP16 as a line with CI band and a shaded ±5-point zone. GPTQ and SVD as filled markers with paired-bootstrap 95% bars, labelled by bits or rank/E; open-loop configurations as hollow markers in a strip panel on the same x; dashed vertical at each threshold; side panel with per-layer-group 90%-energy rank, raw vs whitened.

**Caveats.** Bytes are not latency: 2-, 3- and 6-bit points have no fast kernel and are success-only. SVD without fine-tuning is its worst case (§3.4). Spectra move under function-preserving rescalings (§4.3). Mixed-precision quantization (ActQuant, C13) would dominate a uniform curve and is out of scope. One model, one suite.

**Relation to §3.1.** Replaces the bridge scope's compressor core at similar cost (\~91 vs \~98 GPU-h, including S3), leaving \~149 GPU-h of reserve.

### 3.3 Stage 2 extension: when does the clock matter? (Option, not adopted; post-Dec 20)

From a brief dated Nov 6. Premise (Verified): standard LIBERO pauses the simulator while the policy computes, so latency never affects success; VLASH measured SmolVLA at 79.0% vs 79.1% under a 3-step delay (D6). Claim to test (Hypothesis): under injected latency on dynamic tasks, the success-vs-latency frontier over K × H × compression reorders relative to the paused clock. A null result is reportable.

**Mechanism to pre-register (Estimate).** With delay d = ⌈latency / control period⌉ and asynchronous execution, the chunk computed from observation o\_t applies from step t + d; steps t..t+d−1 execute the previous chunk's later tokens. The effective open-loop horizon is therefore ≥ d + 1 regardless of the nominal H, which couples latency to horizon and is the route by which the frontier can reorder. Alignment: the new chunk's token d executes at step t + d, since tokens 0..d−1 were planned for steps already past; executing token 0 at t + d would add an artificial d-step staleness. This is the alignment VLASH and real-time chunking describe (Unverified against their code). The control rate is Unverified (D3) and must be read from the env config before d is computed.

**Key risk.** smolvla\_libero was trained on static tasks only; on moving-object tasks FP16 at d = 0 may be near zero, leaving no signal. Mitigation order: (a) tunable dynamics, a target that moves once mid-episode or drifts at speed v, sweeping v and keeping the largest at which FP16 at d = 0 retains ≥ \~70% of its static success; (b) fine-tune the expert on scripted demonstrations (VLM frozen; 2–3 weeks plus GPU-h, Unverified effort); (c) fall back to "direction 2" (cross-step error accumulation plus a per-step compensator), not yet specified.

**Implementation.** Delay wrapper in the LeRobot/LIBERO eval loop (LeRobot's Euler sampler already accepts real-time-chunking arguments, inference\_delay, prev\_chunk\_left\_over and execution\_horizon, in sample\_actions on main at commit d490e47; their semantics are Unverified and may cover part of this): observation o\_t, new chunk applies from t + d, previous chunk keeps executing; optional per-call d sampled from a measured distribution for jitter; must reproduce normal results exactly at d = 0; every policy runs at d = 0 and at its own d to separate "task is hard" from "policy is slow"; sensor noise is a separate knob. Hardware: all episodes on the A100; a Jetson Thor, if available, only times each configuration's forward pass on recorded inputs to supply d; otherwise d is projected from the two-term model calibrated on the A100 profiler and published Thor numbers (B8, B9, B13), reported at d and 1.5d. B9 is discrepant (94 vs 44 ms) so the projection must name its pipeline. Tasks: 2–4 moving-object LIBERO/MuJoCo variants with the speed knob; ReflexBench (D7) is the published alternative and a comparison point. Optional second model: π0.5 via openpi, +3–4 weeks.

**Effort (Estimate, solo, 15–20 h/week after Dec 20).** Delay wrapper 1–2 weeks; dynamic tasks and speed sweep 3–5; fine-tune only if needed 2–3; Thor timing 1–2; main runs 2–3; writing 3–4. Total 3–5 months. Milestones: late Jan wrapper validated, tasks built, sweep done; Feb–Mar fine-tune if needed and timings; Mar–Apr main runs; late Apr–May 2027 draft (June with π0.5). Go/no-go by mid-January, both required: spring compute secured; some dynamic setting gives FP16 a usable d = 0 success.

**Before Dec 20.** After the course harness works (not in week 1), build the delay wrapper beside it and run it on static LIBERO at a few d; little effect is the evidence for the premise. If time allows, prototype one moving-target variant and run FP16 at a few speeds. Stage 1 comes first.

### 3.4 Low-rank method notes (Nov 6)

- **Importance score (required if importance-aware SVD runs).** Gradients for QSVD-style rank allocation come from the flow-matching training loss on calibration demonstrations, computed with LeRobot's own loss function because SmolVLA's sign convention for the velocity target is Unverified, never from a teacher–student reconstruction loss, whose gradient is zero at the uncompressed weights (§4.6).
- **Fine-tuned SVD arm (Option).** CP and Tucker compression of CNNs (Lebedev et al. 2014; Kim et al. 2015, both from memory, not opened) recovered accuracy only after fine-tuning, so post-training SVD is the low-rank family's worst case. An optional arm distills the SVD factors toward the original expert for a few GPU-h; for balance, quantization receives equal calibration budget (GPTQ already is its calibration). State the limitation if the arm is not run.
- **VBMF rank selection (Option).** Variational Bayesian matrix factorization picks per-layer ranks with no gradients or training (Kim et al. 2015); a cheap baseline beside gradient-based allocation.
- **Quantized factors (Estimate).** 4-bit factors match W4 bytes at rank ≈ 0.81E rather than 0.21E; on a flat spectrum the relative error is √(1 − 0.81) ≈ 0.44 against ≈ 0.10–0.13 for 4-bit RTN on Gaussian weights (group-wise to per-row scales), so dense W4 still leads ≈ 3–4× unless the spectrum is steep. Same conclusion as the toy run; the spectrum remains the deciding measurement.
- **Cross-attention K/V (check).** These projections read the VLM cache, fixed for a whole policy call (A11). If the code recomputes them at each of the K steps, caching once per call is a lossless speedup and compressing them saves little time. Already covered by the manifest item "computed once per call vs per step".
- **Cross-layer Tucker (extension).** On a single 2-D weight, CP and Tucker reduce to SVD; extra structure exists only across layers (a 32 × E × F tensor at this checkpoint's 32 layers, A17) or across heads (E × heads × head\_dim, of which joint QKV is the flattened form). Pays off only if layers are similar, which is a measurement.

## 4. Derivations

The dimensions below (720, 2048, 16 layers) are the earlier illustrative ones; A17 gives 480, 1280 and 32 layers, pending the loaded-model manifest, and §3.2 holds the byte and latency figures recomputed at those dimensions. The algebra does not depend on them.

**4.1 Storage arithmetic.** A single n×n layer at b bits stores n²b/8 bytes; bf16 factors of rank r store 2r(n+n) = 4rn bytes. For n = 720, INT8 = 518,400 B and the match is r = 518,400/2,880 = 180. For a gated MLP with hidden E and intermediate F, 4-bit weights store 3EF/2 bytes (2,211,840 at E=720, F=2048) plus scale metadata, which depends on the quantizer: group-wise scales cost 3EF/g × 2 B (69,120 at g = 128, total 2,280,960); one bf16 scale per output channel costs (2F + E) × 2 B (9,632, total 2,221,472). Joint gate-up factors (E×r, r×2F) and down factors (F×r, r×E) in bf16 store 2r(E + 2F + F + E) = 2r(2E + 3F) = 15,168r bytes. Match: r = 145.8 raw, 146.5 with per-channel scales, 150.4 with group-128 scales. The toy harness implements per-channel RTN, so its matched rank is 146; the latency model keeps group-128 because that is the layout a fused kernel reads. A QSVD-style hybrid that quantizes the factors to b bits stores 4rE·b/8 for a joint E×3E QKV block. An SVDQuant-style hybrid stores a bf16 rank-k branch plus a b-bit residual over the full matrix: for n = 720, k = 32, b = 4 that is 92,160 + 259,200 = 351,360 B, 36% above INT4. Because the dense 4-bit residual alone already equals the 4-bit budget, any positive-rank bf16 branch exceeds it, and reducing k cannot close the gap. The hybrid meets an INT4 budget only with a sub-4-bit residual (a 3-bit residual is 194,400 B, leaving 64,800 B ≈ 22 bf16 ranks), which is why the residual hybrid is gated on a sub-4-bit residual. A 4-bit 720×720 residual (259,200 B) equals 90 bf16 ranks.

**4.2 Launch and bandwidth model.** Per block per step, t = n\_launch · t\_ℓ + bytes\_moved / B. Stored bytes equal moved bytes only for a fused kernel that reads packed weights and dequantizes in registers; an unfused path writes a bf16 copy and moves more. For W4A16 with fused gate-up and down GEMMs, n = 2; per-matrix SVD of three matrices is 6 GEMMs, joint gate-up plus down is 4. With t\_ℓ = 2 µs and B = 273 GB/s, two extra launches cost 2 · 2e−6 · 273e9 = 1.09 MB of traffic. SVD is strictly faster than W4A16 only when 15,168r < 2,280,960 − 1,092,000, i.e. r < 78.4. At t\_ℓ = 8 µs the penalty is 4.37 MB, larger than W4A16's total, so no rank qualifies. Achievable bandwidth below peak shrinks the penalty in byte terms (0.76 MB at 70%, r < 99). These are conclusions of a two-term model (launches plus bytes moved) that omits arithmetic, kernel selection, occupancy and caching; the model does not establish that eager SVD can never win, only that under its assumptions no rank does. Both t\_ℓ and achieved bandwidth are to be measured. With per-channel scale accounting (§4.1) W4A16 stores 2,221,472 B and the bound becomes r < 74.5 at 2 µs; the group-128 figure is kept above because it matches the fused-kernel layout. The same model at A100 bandwidth (\~1.555 TB/s, B6 Unverified) gives a launch penalty of 2 · t\_ℓ · B = 3.1–6.2 MB at t\_ℓ = 1–2 µs, larger than W4A16's entire 2.28 MB block, so no rank beats a fused W4A16 kernel; against dense bf16 (8.85 MB, same launch count as W4A16) SVD is faster only below r ≈ 378 at 1 µs and r ≈ 173 at 2 µs. Reading \~200 MB of expert weights takes \~0.13 ms per step on the A100 vs \~0.73 ms on Thor. The A100 is therefore the hardest case for any storage-saving method to show speedup, which is why Course design Q5 measures it directly. A kernel that fuses each factor pair (U applied to Vx with Vx kept on chip) removes the two extra launches, leaving bytes alone: faster than dense bf16 when 15,168r < 8,847,360, i.e. r ≤ 583, and faster than fused W4A16 when 15,168r < 2,280,960, i.e. r ≤ 150. These bounds assume each factor is read once; Inductor's own back-to-back template re-reads V per output tile and its cost model declines these shapes (E10), so the bound is reachable only with a design that avoids re-reads, such as split-rank accumulation.

**4.3 Error geometry.** For either compressor let E = W − Ŵ. Then E‖Ex‖² = Tr(E C\_x Eᵀ) with C\_x = E\[xxᵀ\]: the input covariance orientation matters for both. Truncation to rank r gives E = Σ\_{i>r} σᵢuᵢvᵢᵀ, of rank n − r, with row space span{v\_{r+1..n}}; Ex = 0 iff x ⊥ that row space, so the error is a deterministic function of the input's projection onto dropped directions. Round-to-nearest error has entries bounded by half a scale step and correlated with W through the scale; the common "isotropic noise scaled by ‖x‖" description holds only in expectation over a random E and is retracted as a claim (see §5). Which family's realized E C\_x Eᵀ is larger under a given shift is a measurement.

**4.4 Transported error.** For Euler x\_{k+1} = x\_k + h v(x\_k, t\_k), a per-step perturbation e\_k = ṽ(x\_k) − v(x\_k) at the teacher's states produces terminal error δx\_N ≈ Σ\_k Φ\_{N,k+1} h e\_k to first order, where Φ is the product of (I + h ∂v/∂x) over later steps. With output selection P\_H and q\_k = P\_H Φ\_{N,k+1} h e\_k, ‖Σ q\_k‖² = Σ‖q\_k‖² + 2Σ\_{i\<j}⟨q\_i, q\_j⟩. Mean per-step velocity MSE omits both Φ and the cross terms. Gronwall bounds (Error Bounds for Flow Matching, 2305.16860) depend on sup‖e‖ and the Lipschitz constant only; of the papers reviewed here (that paper, PTQD, CTEC), none gives a bound that distinguishes error families by their alignment across steps, and a wider search was not done. The coherence index C = ‖Σ q\_k‖ / Σ‖q\_k‖ lies in \[0, 1\] when the denominator is nonzero: 1 for aligned errors, 0 when they cancel; 1/√K is its value for K equal-norm uncorrelated errors in expectation, a reference point rather than a lower bound. The toy run (§6) found C ≈ 0.94 for both families on a smooth trajectory, so alignment is not assumed to differ.

**4.5 Executed-prefix causality (SmolVLA).** If self-attention over action tokens is causal and cross-attention reads only the VLM prefix, token 0's hidden state at every layer depends only on token 0's input, the prefix, and the timestep; the MLP and output head are per-token; shared weights add no cross-token path. Hence at H = 1, a chunk of length 1 with the same position-0 noise and schedule reproduces the executed action exactly up to floating point. Consequences: (i) unexecuted tokens affect the executed one only through calibration, not inference; (ii) full-chunk calibration spends budget on 49 tokens irrelevant at H = 1, so the calibration criterion depends on H; (iii) measuring latency at chunk 50 vs 1 holds weight bytes fixed and cuts activation work 50×; little change is consistent with a weight-bandwidth-bound expert but equally with launch-bound execution, so a profiler trace (bytes moved against kernel count and gaps) is needed to tell them apart. Requires A10. Does not hold for models with bidirectional action attention.

**4.6 Reconstruction-loss gradients.** For L(θ̂) = ‖F\_θ̂ − F\_θ‖², ∂L/∂θ̂ = 0 at θ̂ = θ, so first-order importance scores on a teacher–student reconstruction loss cannot rank deletions at the uncompressed point. QSVD's score uses the task loss, whose gradient is nonzero. For prefix or perturbed-state reconstruction objectives, use finite deletions of singular-value groups scored per byte saved, or a Gauss–Newton approximation.

**4.7 Episode budget.** 20 tasks × 50 episodes × 4 policies × (3 perturbations + 1 clean) = 16,000 episodes. An 80 GPU-hour budget implies 18 s per episode; at 60 s the matrix takes 267 GPU-hours. LIBERO-Spatial episodes cap at 280 steps and LIBERO-10 at 520 in LeRobot at the pinned commit (D15); at H = 1 every step is a full policy call. Per-episode time must be measured, including rendering, resets and any env parallelism, before the matrix is sized. Separately, the core nominal matrix in the Course design configuration table (R0; GPTQ at 2, 3, 4 bits; importance-aware SVD at four points; S3) is 4,500 episodes, about 75 GPU-hours at 60 s and about 91 with week-1 checks, spectra and the open-loop screen, before any recovery run, second horizon or repetition.

## 5. Correction log

Dated record of claims that were wrong or overstated, who caught them, and what replaced them. Kept so the review history is auditable.

| Date | Claim as it stood | Caught by | Replacement |
| --- | --- | --- | --- |
| Oct 29 | GLUESTICK cited only for the Orin NX spec; "no study has measured VLA spectra" | Other chat (Gemini) | GLUESTICK's spectral analysis and SVD table added; claim narrowed to action-expert weights |
| Oct 29 | "88.8% reported" for the 0.45B SmolVLA | Other chat | 87.3% at 10 trials; 88.75% is the 2.25B variant |
| Oct 29 | \[1\] cited the HF blog; \[2\] an Arm tutorial for chunk/step facts; \[12\] linked the OpenRAL wrapper; \[16\]/\[17\]/\[19\] used blog/retail pages | Other chat | arXiv paper; LIBERO paper; official checkpoint page; NVIDIA/Forecr specs |
| Nov 5 | "Joint K/V decomposition for cross-attention" | Self, on reading LeRobot code | Cross-attention Q, K, V take different inputs; only per-matrix SVD applies there; joint gate-up for the MLP instead |
| Nov 5 | QSVD "uses Taylor importance to automatically preserve flat cross-attention layers"; "rotation makes rounding vastly more stable"; "solves the hardware problem" (Gemini brief) | Self, on reading QSVD | Global ranking can starve flat layers (risk, not feature); rotation is baseline-standard; the 13.1× result is an offloading artifact |
| Nov 5 | "Quantization error is incoherent, O(√T); SVD coherent, O(T)" (Gemini brief) | Toy run | Both C ≈ 0.94 on a smooth trajectory; alignment is a measurement |
| Nov 5 | "SVD fails gracefully; quantization corrupts primary dimensions" (Gemini brief) | Self, vs toy data and GLUESTICK | Rejected; SVD was 6× worse at flat spectrum |
| Nov 5 | Step schedule r = 33 → 122 "where the Jacobian is stiffest" (Gemini brief) | Self | The schedule came from an invented sensitivity profile; direction is measured |
| Nov 5 | Matched-bytes rank 0.06E (Gemini brief) | Self | 0.2E; the brief's byte formula overcounted |
| Nov 5 | "Truncation error is low-rank" | GPT review | Rank n − r with fixed row space |
| Nov 5 | "Expert trains only a low-dimensional input subspace" (mechanism) | GPT review (zero-action counterexample) | Removed; then over-corrected to "full-rank by construction" |
| Nov 5 | "Expert inputs are full-rank by construction" | GPT review 2 + self (action\_in\_proj) | True only at the 7-dim action; per-token hidden input is an 8-scalar function; later layers mix VLM features. Measure per-layer activation covariance |
| Nov 5 | Corrective-response diagnostic projected the action delta onto a hidden layer's singular basis | GPT review 2 | Project per-layer input deltas Δx\_ℓ onto V\_ℓ; local only |
| Nov 5 | "Unexecuted tokens matter through cross-attention and shared weights" | GPT review 2 + causal-mask argument | False for SmolVLA at inference; matters only for calibration and H > 1 |
| Nov 5 | "Quantization error is isotropic, shift-sensitive only via ‖x‖" | GPT review 2 | Tr(E C\_x Eᵀ); orientation matters for both |
| Nov 5 | "Activation whitening as in ASVD" | GPT review 2 | ASVD scales; SVD-LLM whitens |
| Nov 5 | "Matched bytes ⇒ equal bandwidth; SVD can at best tie" | GPT review 2 | Kernel-contingent; stored, moved, packed and fused are different quantities |
| Nov 5 | "One extra launch per factorized layer" | GPT review 2 | Count from the real graph: joint vs per-matrix differ (4 vs 6 for the MLP) |
| Nov 5 | Recovery experiment "\~130 GPU-hours" | GPT review 2 | Unsupported; 16,000 episodes at 60 s is 267 h; pilot required |
| Nov 5 | OGPO 76.7%/81.8% presented as LIBERO-ReAct results | GPT review 2 | "Total robustness", likely aggregate; unverified |
| Nov 5 | "DART-style state-noise distillation" | GPT review 2 | DART injects noise into supervisor demonstrations |
| Nov 5 | NestedVLA "found no usable low-rank structure in deltas" | Self, on reading abstract | Unverified; abstract describes hypernetwork skill consolidation |
| Nov 5 | "4rE·b bytes" | Self (audit) | 4rE·b/8 |
| Nov 5 | Expert hidden size 720 used throughout | Self, on reading config.json | Config says expert\_width\_multiplier 0.5; conflict with paper's 0.75; manifest required |

Round 3 (GPT review 3, Nov 5), applied after validating each point against the scripts and derivations:

| Date | Claim as it stood | Caught by | Replacement |
| --- | --- | --- | --- |
| Nov 5 | Toy storage budget used group-128 scales (2,280,960 B, rank 150) while the harness quantizer is per-channel RTN | GPT review 3 | Per-channel accounting (2,221,472 B, rank 146); harness rerun, conclusions unchanged; svd\_rank\_budget.py now prints both layouts |
| Nov 5 | "Per-token hidden input is an 8-scalar function (7 actions + time)" | GPT review 3 | Sampler noise has max\_action\_dim = 32 coordinates per token; covariance is measured, not inferred (A14) |
| Nov 5 | H1–H3 "each independent" | GPT review 3 | H1 bundled family difference with spectral prediction; H2 bundled recovery effect with family interaction. Split into Q1–Q4; spectral prediction exploratory until a predictor and test are specified |
| Nov 5 | Coherence C "ranges from 1 to about 1/√K" | GPT review 3 | C ∈ \[0, 1\]; 1/√K is a reference for equal-norm uncorrelated errors, not a lower bound |
| Nov 5 | SVDQuant hybrid "meets an INT4 budget only with a sub-4-bit residual or a smaller k" | GPT review 3 | Any positive-rank bf16 branch plus a dense 4-bit residual exceeds the 4-bit budget; sub-4-bit residual required |
| Nov 5 | "No published bound distinguishes error families by frequency" | GPT review 3 | Bounded to the papers reviewed |
| Nov 5 | Eager-mode "no rank qualifies" stated as a result | GPT review 3 | Conditional on the two-term model; measured latency includes arithmetic, kernel selection, occupancy, caching |
| Nov 5 | Chunk 50 vs 1 latency "diagnoses whether the expert is weight-bound" | GPT review 3 | Launch overhead gives the same signature; profiler needed |
| Nov 5 | Transported coherence committed in the course tab but optional here; gated additions called the first casualties while the cut order removed a budget first | GPT review 3 | Coherence optional in both tabs; cut order starts with diagnostics and gated rows; one configuration table lists every calibrated model (10,000 core episodes) |
| Nov 5 | Common-state intervention applied the disturbance under each policy after restoring state; "fix the flow noise" unspecified | GPT review 3 | Create the disturbed state once under a scripted controller and fork after the window; noise matched across policies per decision, fresh across decisions |
| Nov 5 | C4 "Verified at abstract level only" from secondary pages | GPT review 3 | Secondary report; Verified split into Source-checked and Reproduced |
| Nov 5 | "Confirming H2 or H3 would make it a conference paper" | GPT review 3 | Necessary, not sufficient; depends on effect size, baseline strength, controls, generalization |
| Nov 5 | W4A16 baseline specified by kernel only | GPT review 3 | Algorithm named: per-group RTN plus a calibrated GPTQ/AWQ-style method; CHASE-VLA and Ω-QVLA numbers are related-work context, not matched baselines |

Round 4 (GPT review 4, Nov 5; reviewer independently reran both scripts and reproduced the results):

| Date | Claim as it stood | Caught by | Replacement |
| --- | --- | --- | --- |
| Nov 5 | Harness docstring: decay = 0 "leaves the Marchenko–Pastur random spectrum" | GPT review 4 | d = 0 sets all singular values equal; retained subspace non-unique; docstring corrected |
| Nov 5 | "16% vs 9%" stated without its metric | GPT review 4 | Teacher-forced local velocity error; endpoint error is 35% vs 27% |
| Nov 5 | "W4A16 vs bf16 SVD" read as performed numerics | GPT review 4 | Float64 throughout, factors and scales unrounded by default; --round check: small numerical changes, qualitative conclusions unchanged (an earlier "nothing past the third decimal" was wrong, caught by GPT final review); stated in §6 and the course appendix |
| Nov 5 | Course tab: old Verified definition; Ω-QVLA inside a "Verified" paragraph; H2/H3 references in §§3–4; gated prefix row "diagnoses weight-bound"; "confirming Q2–Q4 is necessary" | GPT review 4 | Labels aligned with this tab; Secondary report marked; Q1–Q4 throughout; profiler qualification; positive findings strengthen but are neither required nor sufficient |
| Nov 5 | svd\_rank\_budget.py comment "Jacobian stiffens near t = 1" | GPT review 4 | Removed; the profile is invented |
| Nov 5 | "Storage was the one real bug" (reply summary) | GPT review 4 | Storage-accounting bug fixed; remaining limitations are numerical idealization, interpretation, and document consistency |

Round 5 (validation of the rate–distortion design before adoption, Nov 7):

| Date | Claim as it stood | Caught by | Replacement |
| --- | --- | --- | --- |
| Nov 7 | Course design: two suites, budgets B1/B2, 10,000-episode core | Owen's approval | Record §3.2 adopted: one suite chosen after the pilot, byte-point curves, 4,500 closed-loop episodes ≈ 91 GPU-h |
| Nov 7 | Lossless threshold as a two-sided 95% CI within ±5 points | Self (validation) | One-sided non-inferiority, margin 5; power ≈ 0.80 at a true difference of 0 |
| Nov 7 | Quantization group 128 along E | Self | 128 divides neither 720 nor 480; group 120 along E, 128 along F; 3-bit matched rank 113 → 114 |
| Nov 7 | P3 "no crossover in the 2–8-bit range" | Self | Families meet closed loop only at the W4 point; P3 stated there and across the open-loop screen |
| Nov 7 | SVD at ≈ 0.8E as a closed-loop point | Self | Costs as much as dense bf16; open-loop diagnostic only |
| Nov 7 | LIBERO-Plus at two points ≈ 17 GPU-h | Self | Needs its own FP16 reference; ≈ 25 GPU-h |
| Nov 7 | P5 as a tested Spearman ρ ≥ 0.8 | Self | Gap between families inflates a pooled ρ; exploratory, reported within family |
| Nov 7 | S3 implicitly at the W4-matched SVD point | Self | P2 predicts collapse there; S3 placed at the screen's in-between SVD point |
| Nov 7 | Bridge dropped S3 while the proposed Course edits kept it | Self | Kept (Owen, Nov 7) as the first scope cut |
| Nov 7 | Stage 2: "new chunk applies from t + d" | Self | Token d executes at t + d |
| Nov 7 | Delay wrapper in week 1 | Self | After the core harness; optional item in Course design §7.3 |
| Nov 7 | "W8 near-lossless on every VLA in the ledger" | Self | Inferred from W4 results; the cited rows report W4 |
| Nov 7 | Quantized-factor comparison "dense W4 leads ≈ 3×" | Self | RTN error 0.10–0.13 depending on scale granularity; ≈ 3–4× |

Round 6 (instructor feedback on the initial proposal, Nov 9). Prof. Sai Qian Zhang: "I suggest carefully investigating whether reducing weight storage translates into actual inference speedup, as low-rank decomposition introduces additional matrix operations and kernel overhead."

| Date | Claim as it stood | Caught by | Replacement |
| --- | --- | --- | --- |
| Nov 9 | Speedup treated as a caveat ("bytes are not latency") rather than a question | Prof. Zhang | New core question Q5: measured speedup per expert step and per policy call, with per-layer microbenchmarks and measured SVD break-even ranks against dense bf16 and W4A16 |
| Nov 9 | Latency prediction quoted Thor-bandwidth break-even (r < 78) without saying so | Self, prompted by the feedback | Prediction table for both A100 and Thor; on the A100 SVD never beats fused W4A16 and beats dense bf16 only below r ≈ 173–378 |
| Nov 9 | Group size fixed at 120 for byte accounting | Self | Set by the fastest fused W4A16 kernel that accepts the expert's shapes (new checklist item), so success and latency use the same artifact; 120 only as fallback |
| Nov 9 | svd\_rank\_budget.py modelled Thor only | Self | Script now prints A100 regimes and the dense-bf16 break-even rank |

Round 7 (optimized-baseline check against LeRobot source, Nov 9):

| Date | Claim as it stood | Caught by | Replacement |
| --- | --- | --- | --- |
| Nov 9 | FP16 reference timed on LeRobot's default inference path | Self, prompted by the world-models evaluation doc (GR00T N1.7 Thor pipeline hoists K/V; secondary report) | LeRobot recomputes the cross-attention K/V projections at every denoising step (Verified in code); the reference and every arm hoist them and run under CUDA graphs (Course design §3.1, §7.1) |
| Nov 9 | "CUDA graphs on" treated as a switch | Self, on reading denoise\_step | DynamicCache appends and crops per step and prefix length varies per call; static shapes needed, time-boxed checklist item |
| Nov 9 | A12 framed as paper (0.75) vs checkpoint (0.5) | Self, on reading the code | Code defaults also disagree (config 0.75, constructor 0.5); the checkpoint value loads, pointing toward 480; still pending the manifest |
| Nov 9 | A14 noise shape from a review's citation | Self | Confirmed in code: (bsize, chunk\_size, max\_action\_dim) |
| Nov 9 | Stage 2 delay wrapper to be built from scratch | Self | LeRobot's sampler already takes real-time-chunking arguments; semantics to check before building |
| Nov 9 | Course design §6.1 GPU-h placeholders (15/40/20/150/5, reserve 70) summing to 300 and leaving nothing for extensions, against §3.3's ≈ 91 core and ≈ 149 free | Owen | §6.1 GPU-h placeholders reconciled with the §3.3 derivation (Nov 9): \~5/\~15/\~4/\~67/0, contingency 60, unallocated \~149, plus a sensitivity line at 30/60/120 s per episode; ledger E9 updated to match |
| Nov 9 | Audit after the budget reconciliation: §4.7 still cited the 10,000-episode core; Course design §2.4 called the noise shape unverified; the Model row named no versions; the deployment-metrics status predated Q5 | Self | §4.7 now 4,500 episodes ≈ 75 GPU-h (≈ 91 with overhead); noise shape marked confirmed in code; LeRobot d490e47 and a recorded HF revision named; tail latency marked adopted, smoothness still an option. Older log rows keep their original figures as history |
| Nov 9 | External rigor review (Fable): "QCRL already covers SmolVLA quantization"; "track adaRMS timestep modules"; adopt VLAQuantBench's protocol; position against VLAQuantBench | Self, on opening the papers | QCRL names π0.5 and GR00T N1.5 only (C22); SmolVLA has no adaRMS, its timestep path is action\_time\_mlp\_in/out, now a named component in §3.1; Wilson intervals added per config beside the paired bootstrap; VLAQuantBench and LightDP added (C20, C21) and the gap stated in Course design §2.3. No change to the question, arms, byte points, budget or schedule |
| Nov 9 | Pre-start review: §5.1 named a "task-level bootstrap" while its power estimate assumed episode-level pairing; the 20% discordance was stated as if known; the central question still led with recovery and omitted speed; calibration data and GPTQ plumbing had no checklist item | Self | Bootstrap resamples paired episodes within task, with a task-resampling interval as sensitivity; matched flow-noise seeds stated; discordance marked an assumption (power ≈ 0.60 at 35%) with an FP16 repeat as noise floor; central question names success and speed, recovery gated; checklist and risk items added for calibration data, GPTQ on the expert, and the nine-day implementation window |
| Nov 9 | Proposed in chat before entering the doc: "if compile fuses away much of the kernel count, SVD's launch penalty shrinks"; "a fused Triton kernel keeps Vx on chip, one launch, no round trip" as if the gain were free | Self, on reading torch 2.15 source | Compile fuses elementwise ops, not matmul pairs; the experimental back-to-back GEMM pass is off and declines these shapes (C24, E10), so SVD keeps its extra launches under compile. The fused kernel is a gated Hypothesis with re-read and parallelism caveats (Course design §3.2, §4.1). reduce-overhead skips graphs that mutate inputs, so the static-cache work precedes compile (checklist §7.2) |
| Nov 10 | Design arithmetic at E = 720, F = 2048, 16 layers, "SmolVLA 0.45B"; causal mask and timestep path Unverified; Q5 falsifier assumed W4A16 is faster than dense on the A100 | Self, on reading the checkpoint config and pinned code before coding | Checkpoint is a 0.6B variant: 32-layer VLM, expert 480 wide, 1280 intermediate, 32 layers (A17). Byte points and latency bounds recomputed; rank fractions unchanged. Causal mask and timestep MLP confirmed in code. Q5 prediction restated: A100 step launch-bound, W4A16 within 10% of dense, unfused SVD slower at every closed-loop rank; speed gains are an edge projection. Paper success figures describe a different configuration |

Round 8 (repository setup and a full recheck before coding, Nov 10). From this round on, the repository is the master and these files are edited directly.

| Date | Claim as it stood | Caught by | Replacement |
| --- | --- | --- | --- |
| Nov 10 | The Claude Doc tabs are the source of truth | Owen | The repository's `main` branch is the single source of truth for design, code and results. The Claude Doc is frozen at Course design revision 179 and Research record revision 64 |
| Nov 10 | One optimized inference path (K/V hoisting, CUDA graphs) for the FP16 reference and every arm, for success and latency, built in weeks 1–2 | External review, accepted by Owen | Success episodes run on LeRobot's eager path for every arm; the optimized path is for latency only and is built after the FP16 baseline and the first RTN and plain-SVD artifacts exist; revisit after the throughput pilot; arms are never mixed across paths (`course-design.md` §3.1, §6.1, §7.1) |
| Nov 10 | Week 1 held seven checks | External review | Three critical-path items: one headless episode, the manifest, the throughput pilot. The rest follow. The causal mask is asserted directly before the chunk-1 exactness test |
| Nov 10 | Planned in chat: HPC jobs commit per-episode results to the repository | External review | Per-episode outputs stay on HPC scratch; only `results/LOG.md` and per-arm summary files are committed, by hand |
| Nov 10 | LIBERO-Spatial episodes cap at 220 steps | Self, on reading LeRobot's libero.py at d490e47 | 280 steps (D15). LIBERO-10's 520 was right |
| Nov 10 | LIBERO control rate Unverified (D3) | Self, same file | 20 Hz, Verified in code |
| Nov 10 | Matched flow noise per decision assumed to follow from a shared seed | Self, on reading lerobot\_eval.py | `lerobot-eval` passes no noise, so the global generator's stream stops matching between arms once episode lengths differ. The harness must generate each decision's noise from (seed, episode, step) and pass it through select\_action(noise=…), which the policy supports (D15) |
| Nov 10 | Course design §2.4: the input-projection structure "still unverified" | Self, rereading modeling\_smolvla.py | Confirmed: action\_in\_proj (32 → 480), concatenated with the sinusoidal time embedding, action\_time\_mlp\_in (960 → 480), SiLU, action\_time\_mlp\_out (480 → 480) |
| Nov 10 | Course design §3.2 table: SVD faster than dense at "rank < 60", than W4A16 at "rank < 42", Thor "rank < 270–327" | Self, rerunning the model at 480 / 1280 | The largest qualifying ranks are 60, 42, and 327 / 270 at 1 / 2 µs, inclusive; the table now says ≤ |
| Nov 10 | Hugging Face revision "recorded at setup", not yet known | Self | 6721902b… (A18), 604.9M parameters, read through a fetch summary; confirm at first download |
| Nov 10 | Calibration dataset not located | Self, LeRobot docs | `lerobot/libero` or `HuggingFaceVLA/libero` (A19); the checkpoint's training suites remain Unverified |
| Nov 10 | Schedule: R0 listed in the Dec 2–10 row; latency work in Nov 16–22 (\~15 GPU-h) with \~67 in Dec | Self | R0 runs Nov 16–22 (\~10 GPU-h with spectra); latency moves to the December row (\~72). The total stays \~91 |
| Nov 10 | `svd_rank_budget.py` defaults at the illustrative 720 / 2048 | Self | Defaults are 480 / 1280 with scale groups of 120 and 128; `--legacy` reproduces the old run; tests pin both |
| Nov 10 | Course design footer: "all 48 references" | Self | 52 |

## 6. Toy experiments

Both scripts were run on Nov 5, 2026 on CPU with torch 2.15.0, float64, `torch.manual_seed(0)`. They live in the project repo under `analysis/toy/`. They exercise a synthetic model and are evidence about the mechanism's plausibility, not about SmolVLA. All arithmetic is float64; by default the SVD factors and RTN scales are not rounded to bf16, so "W4A16 vs bf16 SVD" names the storage budget matched, not the numerics performed. A rerun with the harness's --round flag (factors and scales rounded to bf16 before reconstruction) produced small numerical changes (for example the d = 0.5 quantized conditioning-shift endpoint error, 0.2856 → 0.2842) without changing the qualitative conclusions. Neither script measures deployment memory or speed. GPT review 4 ran both scripts independently on the same environment and reproduced the reported conclusions, with small differences at d = 0 (see 6.1).

**6.1 `compounding_harness.py`.** Teacher: a residual gated MLP velocity field v(x, t, c) with hidden 720, intermediate 2048, action dim 350 (= 50 × 7), conditioning dim 64; the output head is rescaled so mean ‖v‖ ≈ ‖x₀‖, so the 10-step Euler trajectory moves O(‖x₀‖) as a real noise-to-action flow does. Singular values of gate/up/down are reset to σᵢ ∝ i^(−d) at equal Frobenius norm, d ∈ {0, 0.5, 1}; d = 0 makes every singular value equal (it replaces the random init's spectrum rather than keeping it), so the retained rank-146 subspace is non-unique and small cross-platform differences at d = 0 are expected. The teacher is an untrained random field with an imposed spectrum, not a learned noise-to-action transport. Students at matched bytes: per-output-channel symmetric RTN 4-bit fake quantization with one bf16 scale per row (2,221,472 B for the block); joint gate-up and down truncated SVD at rank 146 (2,214,528 B). An earlier run budgeted group-128 scales (2,280,960 B, rank 150), which did not match the per-channel quantizer; the numbers below are from the per-channel rerun and differ from that run in the third decimal only. n = 256 samples, K = 10 steps, shared x₀. Two shifts: conditioning c → 1.5c + 2, and a shift of the MLP input along the directions rank-146 SVD drops, scaled to the mean input norm. Metrics: teacher-forced local relative error and cosine; coherence C = ‖Σe\_k‖/Σ‖e\_k‖ on raw (untransported) errors; free-running log-log growth slope; endpoint error.

| Spectrum d | Metric | SVD ID | SVD OOD(c) | SVD OOD(sub) | W4 ID | W4 OOD(c) | W4 OOD(sub) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (flat) | local rel err | 0.0486 | 0.0590 | 0.0698 | 0.0083 | 0.0100 | 0.0120 |
| 0 | coherence C | 0.947 | 0.956 | 0.953 | 0.947 | 0.955 | 0.953 |
| 0 | growth slope | 1.064 | 1.041 | 1.060 | 1.063 | 1.044 | 1.061 |
| 0 | endpoint err | 0.989 | 1.389 | 1.992 | 0.169 | 0.237 | 0.342 |
| 0.5 | local rel err | 0.0370 | 0.0450 | 0.0417 | 0.0099 | 0.0121 | 0.0114 |
| 0.5 | endpoint err | 0.751 | 1.065 | 1.176 | 0.202 | 0.286 | 0.322 |
| 1 (steep) | local rel err | 0.0053 | 0.0062 | 0.0055 | 0.0108 | 0.0118 | 0.0106 |
| 1 | coherence C | 0.931 | 0.945 | 0.924 | 0.935 | 0.951 | 0.919 |
| 1 | growth slope | 1.064 | 1.078 | 1.079 | 1.071 | 1.068 | 1.093 |
| 1 | endpoint err | 0.107 | 0.145 | 0.155 | 0.218 | 0.276 | 0.294 |

Reference: C lies in \[0, 1\]; 1/√K ≈ 0.316 is its value for K = 10 equal-norm uncorrelated errors, a reference point rather than a lower bound (opposing errors can cancel to 0); slope 1.0 coherent, 0.5 random walk.

What it shows: (i) on a smooth trajectory both compressors' errors stay aligned across steps, so an O(√T) random-walk assumption for quantization is not supported even in the simplest setting; (ii) at matched bytes the spectrum decides the in-distribution winner, SVD 6× worse at d = 0 and 2× better at d = 1; (iii) shifts hurt SVD somewhat more at d = 1 (on the conditioning shift, teacher-forced local velocity error rises 16% vs 9%, endpoint error 35% vs 27%), the predicted direction, but the effect is modest because a steep-spectrum model has little energy in the dropped directions; (iv) local and endpoint error respond differently to the same shift, which is why the design measures both. The dropped-subspace shift is built from the SVD compressor's own discarded directions: a targeted stress test, not a neutral sample of environmental shifts. What it does not show: anything about attention, timestep conditioning, real activation covariance, or SmolVLA. The coherence index here is on untransported errors; the transported version (§4.4) is not implemented in the script.

**6.2 `svd_rank_budget.py`.** Implements §4.2. The table below is the run at the illustrative E = 720, F = 2048, reproduced by `svd_rank_budget.py --legacy` (output in `analysis/toy/outputs/svd_rank_budget.legacy_720_2048.txt`). Since Nov 10 the script defaults to the checkpoint's 480 / 1280 with scale groups of 120 and 128; that output is in `analysis/toy/outputs/svd_rank_budget.txt` and its numbers are the ones in §3.2 and `course-design.md` §3.2. Default group-128 scales (2,280,960 B, matched rank 150) model the layout a fused kernel reads; group = 0 gives per-channel scales (2,221,472 B, rank 146), matching the harness, and the script prints both. The latency rows below use group-128 and are conditional on the two-term model. Output:

| Regime | W4A16 ms/step | Launch penalty (MB-eq) | Matched-bytes r | Max r strictly faster |
| --- | --- | --- | --- | --- |
| CUDA graphs, 2 µs, peak BW | 0.012 | 1.09 | 150 | 78 (0.11E) |
| CUDA graphs, 2 µs, 70% BW | 0.016 | 0.76 | 150 | 99 (0.14E) |
| Eager, 8 µs, peak BW | 0.024 | 4.37 | 150 | none |

The script also emits rank schedules from a user-supplied sensitivity profile under per-step or per-call budgets; the profile used in the demo run (1 + 0.3k) was invented to exercise the function and carries no evidential weight.

## 7. Hardware and landscape notes

Gathered Oct 29 for scoping; not used by the course core.

**Edge hardware.** Thor T5000: 128 GB, 273 GB/s, 130 W, 517 FP8 TFLOPS dense; T4000: 64 GB, 273 GB/s, 90 W, 300 FP8 TFLOPS; AGX Orin 64 GB: 204.8 GB/s, \~42 BF16 TFLOPS, no FP8; Orin NX/Nano \~102 GB/s. Prices after NVIDIA's July 2026 increase: Thor dev kit $5,499 (was $3,499), T5000 module $4,999, T4000 module $2,999, AGX Orin dev kit $3,499 (was $1,999). The T4000 is sold as a bare module (Arrow 900-13834-0000-000) needing a carrier board; partner T4000 systems run $3,749–$5,959. No hourly rental of Thor was found; access is a borrowed board or a purchase. For the action expert the T5000 dev kit is a close proxy for T4000 latency because bandwidth is identical. DGX Spark also has 273 GB/s but a different GPU. Jetsons can fine-tune (LoRA on 9B, QLoRA on 27B on a 128 GB Thor) but are not training machines; a GB200 has \~19× Thor's FP8 throughput.

**What deployed systems run on.** Figure Helix runs entirely onboard on two embedded NVIDIA GPUs; π0 was timed at 73 ms on an RTX 4090; NVIDIA's reference humanoid and OxyGen's π0.5 humanoid run on Jetson Thor. Non-NVIDIA options (Qualcomm Dragonwing IQ10 at 700 TOPS, IQ-9075 with public VLA demos; Intel Arc B60 at 456 GB/s; Huawei Ascend) appear in cross-hardware VLA studies but require porting off CUDA.

**Model landscape (Oct 2026).** Closed: π0.7 (PI, Apr 2026, API beta Oct 13), Gemini Robotics 2 (Aug 2026). Open: π0.5 (3.3B, \~430M expert est.), GR00T N1.6 (\~1.4B expert est., 32-layer DiT, 16-step chunks) and N1.7 (Cosmos-Reason2-2B backbone, 40-step chunks, 132-dim actions, ONNX/TensorRT export; DiT depth listed as 16 in the README and 32 on the Jetson tutorial, unresolved), SmolVLA (0.45B). LIBERO is near-saturated for the large models (97–99%); SmolVLA has headroom.

**Benchmarks for latency mattering.** Static LIBERO pauses the simulator during inference, so success cannot show that lower latency helps (VLASH: π0.5 96.8 → 94.6% at 3-step delay). ReflexBench, DOM and Kinetix are the latency-aware options; LeRobot's real-time chunking exposes an inference\_delay parameter.

Compute fallback and edge hardware (Nov 9; Estimate unless sourced). If course hours run short, renting A100s keeps the design A100-consistent: on-demand A100 rates on Nov 3, 2026 were $0.59/h (Vast.ai 40GB median), $1.09/h (Thunder 80GB), $1.79/h (RunPod), $1.99–2.79/h (Lambda), so 91–300 GPU-hours costs about $54–330. Buying a desktop GPU breaks even only past \~2,700–5,900 hours (RTX 5080 desktop \~$2,900, RTX 5090 desktop \~$5,000 at Nov 2026 street prices; ConEd at \~35¢/kWh adds \~$0.18–0.25/h). Jetson prices after NVIDIA's July 2026 increase: AGX Thor dev kit $5,499, AGX Orin dev kit $3,499, Orin Nano Super $399. Thor can fine-tune (Jetson AI Lab: full SFT of a 4B model at 1.68 samples/s, LoRA 9B, QLoRA 27B in 128 GB), but its ridge point (\~830–945 FLOPs/byte from an unofficial \~227–258 BF16 TFLOPS against 273 GB/s) exceeds the maximum arithmetic intensity of the expert's layers, so training expert-sized layers on Thor is bandwidth-bound; Thor is for measurement and deployment, not training throughput. The $399 Orin Nano Super is the cheapest real test of the bandwidth-bound regime at SmolVLA scale. Funding path: Professor Zhang's lab first; the NVIDIA Academic Grant Program (faculty PI applies, quarterly, DGX Spark units and H100 hours, inference efficiency and edge AI among focus areas) second; personal purchase only once a spring project depends on it.

## 8. Full reference list

Numbering continues from the archived proposal. "Opened" = full text or the cited table/section was read in this project; "Abstract" = only the abstract or a summary page; "Memory" = cited from prior knowledge, not opened.

1. Shukor et al., *SmolVLA*, 2025. https://arxiv.org/abs/2506.01844 — Opened (§3, §4.3, Tables 2, 13)
2. *On the Efficiency of LoRA Fine-Tuning for VLAs in Industrial Robotic Manipulation*, 2026. https://arxiv.org/html/2607.10172v1 — Abstract + §3.5
3. Wang, Wang & Zhang, *QSVD*, NeurIPS 2025. https://www.saiqianzhang.com/gallery/NeurIPS\_2025\_QSVD\_camera\_ready.pdf — Opened
4. *SnapFlow*, 2026. https://arxiv.org/html/2604.05656 — Abstract + intro
5. *PhyAI: Real-Time Physical AI at the Edge*, 2026. https://arxiv.org/pdf/2608.03682 — Abstract
6. Hyun et al., *CHASE-VLA*, 2026. https://arxiv.org/pdf/2610.02666 — Opened
7. Zhang et al., *QuantVLA*, CVPR 2026. https://arxiv.org/pdf/2602.20309 — Abstract
8. Jabbour et al., *Don't Run with Scissors (GLUESTICK)*, 2025. https://arxiv.org/pdf/2510.08464 — Opened
9. *NestedVLA*, 2026. https://arxiv.org/abs/2610.23271 — Abstract; cited claim unverified
10. Oh et al., *The Low-Rank Structure of VLA RL*, 2026. https://arxiv.org/abs/2609.34599 — Abstract
11. *Reducing Temporal Redundancy for Efficient VLA Inference*, 2026. https://arxiv.org/pdf/2607.12287 — Abstract + §C
12. HuggingFaceVLA/smolvla\_libero. https://huggingface.co/HuggingFaceVLA/smolvla\_libero — Opened (config.json)
13. *VLA-Adapter*, 2025. https://arxiv.org/pdf/2509.09372 — Table only
14. *SMILE*, 2026. https://arxiv.org/pdf/2608.29432 — Table only
15. Black et al., *π0*, 2024. https://arxiv.org/abs/2410.24164 — Table I; PI, *Knowledge Insulating VLAs* — Memory
16. Liu et al., *LIBERO*, NeurIPS 2023 D&B. https://arxiv.org/abs/2306.03310 — Memory (standard)
17. Jetson AI Lab, *GR00T N1.7 on Jetson Thor*. https://www.jetson-ai-lab.com/tutorials/groot\_n17\_on\_thor/ — Opened
18. RLinf LIBERO documentation. https://rlinf.readthedocs.io/en/release-v0.3/rst\_source/examples/embodied/libero.html — Abstract-level
19. Forecr, *NVIDIA Jetson comparison*. https://www.forecr.io/blogs/all/comparison-of-nvidia-jetson-modules — Opened
20. NVIDIA, *Develop for All Six Jetson Orin Modules*. https://developer.nvidia.com/blog/?p=55019 — Abstract
21. *SARA*, NeurIPS 2026. https://neurips.cc/virtual/2026/poster/151173 — Abstract
22. *SlimDiff*, 2025. https://arxiv.org/abs/2509.21498 — Abstract
23. *AdaTSQ*, 2026. https://arxiv.org/abs/2602.09883 — Abstract
24. Zhang, Liu et al., *TGATE*, 2024. https://arxiv.org/abs/2404.02747 — Abstract
25. Park et al., *Any-Precision LLM*, ICML 2024. https://arxiv.org/abs/2402.10517 — Abstract
26. *PIVOT-Q*, 2026. https://arxiv.org/abs/2610.05745 — Abstract + GitHub README
27. Wang et al., *Ω-QVLA*, 2026. https://github.com/UCMP13753/Omega-QVLA — Secondary summaries only; paper link to add
28. *CTEC*, AAAI 2025. https://ojs.aaai.org/index.php/AAAI/article/view/34039 — Listing only
29. *Traj-MC*, 2026. https://arxiv.org/abs/2610.03326 — Abstract
30. *QCRL*, 2026. https://arxiv.org/abs/2610.15675 — Abstract
31. *OGPO* (introduces LIBERO-ReAct), 2026. https://arxiv.org/abs/2610.12914 — Abstract page
32. Zhang et al., *CoRe: Imagining Recovery*, 2026. https://arxiv.org/abs/2608.14822 — Abstract page
33. Czarnecki et al., *Sobolev Training for Neural Networks*, NeurIPS 2017. https://arxiv.org/abs/1706.04859 — Memory
34. Laskey et al., *DART*, CoRL 2017. https://arxiv.org/abs/1703.09327 — Abstract
35. Akbari et al., *ActQuant: Sub-4-bit Action-Guided Quantization for VLAs*, 2026. https://arxiv.org/abs/2605.24011 — Abstract
36. Yang et al., *Coupling Execution Horizon and Sampling Depth in Flow-Matching VLA Policies*, 2026. https://arxiv.org/abs/2610.32102 — Abstract
37. Srinivas & Fleuret, *Knowledge Transfer with Jacobian Matching*, ICML 2018. https://proceedings.mlr.press/v80/srinivas18a.html — Memory
38. Li et al., *SVDQuant*, 2024. https://arxiv.org/abs/2411.05007 — Memory
39. *VLASH*, 2025. https://arxiv.org/pdf/2512.01031 — Table I
40. *ReflexBench / Reflex*, 2026. https://arxiv.org/html/2608.14379 — Abstract
41. *VLA-Perf*, 2026. https://arxiv.org/pdf/2602.18397 — Table 3
42. *LIBERO-PRO*, 2025. https://arxiv.org/pdf/2510.03827 — Fig. 2
43. *LIBERO-Plus*, CVPR 2026. https://arxiv.org/html/2510.13626v1 — Abstract
44. *Error Bounds for Flow Matching Methods*, 2023. https://arxiv.org/pdf/2305.16860v2.pdf — Abstract
45. *PTQD*, NeurIPS 2023. https://arxiv.org/pdf/2305.10657v2 — Abstract
46. FlashRT. https://github.com/LiangSu8899/FlashRT/wiki — Opened
47. Jetson AI Lab, *OpenPi π0.5 on Jetson Thor*. https://www.jetson-ai-lab.com/tutorials/openpi\_on\_thor — Opened
48. LeRobot, smolvlm\_with\_expert.py. https://github.com/huggingface/lerobot/blob/d490e472a5d1a207cfe0255caaaf74b6d66e4656/src/lerobot/policies/smolvla/smolvlm\_with\_expert.py — Opened (at the pinned commit)
49. VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for VLA Models, 2026. https://arxiv.org/abs/2609.25376 — Opened (§3, §4, Tables 1, 4)
50. Wu et al., On-Device Diffusion Transformer Policy (LightDP), ICCV 2025. https://arxiv.org/abs/2508.00697 — Abstract + method summary
51. Quantize by Doing: Task-Guided Mixed-Precision Quantization of VLA Models, 2026. https://arxiv.org/html/2610.38661 — Listing only; cited by an external review, not opened
52. LeRobot at commit d490e47 (version 0.7.1): pyproject.toml, docs/source/libero.mdx, src/lerobot/sim/integrations/libero.py, src/lerobot/scripts/lerobot\_eval.py, src/lerobot/policies/smolvla/modeling\_smolvla.py. https://github.com/huggingface/lerobot/tree/d490e472a5d1a207cfe0255caaaf74b6d66e4656 — Opened
