# Handoff

Last updated: Nov 10, 2026 (repository setup). Update this file whenever the state of the work changes.

## Where things stand

- The design is frozen in `course-design.md`. **No model has been loaded and no episode has run.**
- The repository holds the design, the record, the byte arithmetic with its tests (29 CPU tests pass),
  the toy scripts with their recorded outputs, and the version pins.
- Everything the design says about the model's shape (expert hidden 480, intermediate 1280, 32 layers,
  about 96.7M expert parameters) comes from reading the checkpoint's config and the pinned LeRobot code
  (ledger A17). The loaded-model manifest has to confirm it before any run.
- The HPC environment does not exist yet. `env/pins.md` holds an untested draft.

## Next: week 1 critical path (Nov 9–15), in this order

Only these three are on the critical path. Getting headless rendering to work can take most of the week.

### 1. Environment and one headless episode (HPC)

Done when all of these hold:

- One LIBERO episode completes inside the container with `MUJOCO_GL=egl`, using the pinned LeRobot and
  the checkpoint at the pinned revision.
- The snapshot directory confirms the checkpoint revision `6721902b…` (ledger A18, currently read through
  a summary).
- Policy calls per environment step are counted. Expected: one call per step, so H = 1 (ledger A4).
- `env/freeze-hpc.txt` is committed, and the "HPC setup" section of `env/pins.md` is replaced by the
  commands that actually worked.
- `results/LOG.md` has an entry.

### 2. Checkpoint manifest (`src/aec/manifest.py` and its test)

Records every tensor's name, shape and dtype, grouped per layer (MLP gate / up / down; attention
Q / K / V / O; self-attention or cross-attention layer; timestep path), with parameters and bytes per group.

- Shapes can be read from the header of `model.safetensors` alone, without LeRobot or a GPU, so this runs
  on any machine that can download the file (about 1.2 GB). A second check on HPC compares against the
  loaded model's `named_parameters()`.
- The test asserts what the design assumes: 32 expert layers; E = 480; F = 1280; Q 480→960; O 960→480;
  K and V 480→320 in even (self-attention) layers and 320→320 in odd (cross-attention) layers;
  604,934,176 parameters in total.
- **If anything differs, stop and report it.** The byte points, `tests/test_storage.py` and
  `course-design.md` §2.1 and §3 all rest on these numbers.
- Done when the manifest file is committed (it is small), the test passes, and ledger A17 says
  "Verified (loaded model)" with the commit.

### 3. Throughput pilot (HPC)

Ten episodes each on `libero_spatial` and `libero_10`, eager path, with time split into rendering, resets
and policy calls.

- Done when `results/LOG.md` has the seconds per episode and `course-design.md` §6.1's sensitivity line is
  replaced by the measured figure.
- Owen then decides: the suite (§3.1 Reference row), whether the matrix fits (§6.2 scope-cut order), and
  whether every arm switches to the optimized inference path (§3.1 Inference path row).

### In parallel, on CPU (laptop or web session)

Start the compressors with unit tests on synthetic matrices of the real shapes, RTN and plain SVD first,
then GPTQ and importance-aware SVD. The schedule leaves only nine days for these in late November
(`course-design.md` §8), so starting early is the mitigation.

## After week 1, in dependency order

1. **Paired evaluation harness** (`src/aec/eval/`): a seed manifest fixing initial state and per-decision
   noise for every arm; per-episode JSONL with git SHA, config hash and pins; resumable after preemption.
2. **FP16 baseline R0**, 500 episodes on the chosen suite, eager path.
3. **Spectra** with the two parameterization controls and activation covariance (§3.1).
4. **Causal-mask assertion, then the chunk-1 exactness test** (§7.1).
5. **W4A16 kernel compatibility** (§7.1). It sets the quantization group size, so it must be settled
   **before** any GPTQ arm is calibrated; changing the group size later changes the byte points.
6. **Compressors, storage measured from saved files, calibration set, open-loop screen** (§3.3 "Screen").
7. **Closed-loop arms** (§3.3) and, as separate jobs, the **latency work for Q5** (§3.1 Latency row).
8. **Analysis**: thresholds, P1–P5, Q5 (§5.1).

## Decisions made on Nov 10 (already applied to the docs; log in `research-record.md` §5, round 8)

- The repository is the single source of truth. The Claude Doc is frozen.
- Success episodes run on LeRobot's eager path for every arm. The optimized path is for latency only,
  built later. Revisit after the pilot. Never mix paths across arms.
- Week 1 is the three items above. The other checks follow.
- Per-episode outputs stay on HPC scratch. Only `results/LOG.md` and per-arm summaries are committed, by hand.
- Commit to `main`; branch only when two sessions work at once.

## Traps a new session is likely to fall into

- **This is not the paper's SmolVLA.** The checkpoint is a 0.6B variant with a 32-layer, 480-wide expert.
  The paper's 0.45B / 16-layer / 0.75-width figures and its success numbers describe another configuration.
- **`lerobot-eval` does not give matched flow noise.** It draws from the global generator. The harness has
  to pass per-decision noise through `select_action(noise=...)` (ledger D15).
- **LIBERO is Linux-only** (ledger A20). Nothing that imports it runs on the Mac.
- **128 does not divide 480.** Quantization groups are 120 along the hidden dimension and 128 along the
  intermediate dimension, until the kernel check replaces them.
- **Joint SVD is not available everywhere.** Gate and up share an input, and self-attention Q, K, V share
  an input, so those can be factored jointly. In cross-attention layers Q reads the action tokens while
  K and V read the VLM cache, so only per-matrix SVD applies there.
- **Importance scores need the flow-matching training loss.** A teacher–student reconstruction loss has
  zero gradient at the uncompressed weights (`research-record.md` §4.6). Use LeRobot's own loss function.
- **Use hard resets and fixed initial states** (LeRobot's defaults) for anything reported.
- **Statistics:** Wilson interval per configuration; paired bootstrap within task for differences against
  FP16; one-sided non-inferiority at −5 points; only P1 and P2 are confirmatory (§5.1).

## Open items

| Item | Who | Matters for |
| --- | --- | --- |
| Do course GPU-hours expire at term end? Is a multi-environment job billed per GPU-hour? | Owen asks NYU HPC | Sizing the matrix |
| Scratch purge policy; partition names; whether Apptainer images can be built or pulled | Owen asks NYU HPC | Task 1 |
| Which LIBERO suites the checkpoint was trained on (Unverified, ledger A19) | Read the model card and config at download | Calibration data is meant to come from suites other than the evaluated one (§3.1) |
| Config values were read through a fetch summary, not verbatim (ledger A17, A18) | Manifest task | Everything in §2.1 |
| The SmolVLM2-500M-Instruct text config is assumed equal to the Video-Instruct variant's | Manifest task | A17 |
| Dataset revision of `lerobot/libero` | Whoever first downloads it | `env/pins.md` |
| Fused W4A16 kernel that accepts 480 / 1280 / 960 shapes; GPTQ library for a non-LLM expert | Week 2 | Group size; §7.1, §7.3 |
| A100 memory bandwidth from the datasheet (ledger B6) | Any session with web access | §3.2 predictions |
| Unopened related work (ledger C23), D12 in full, C8, D9, D11 | Before the report | Related work; gated additions |

## Proposed design changes (awaiting Owen)

None.
