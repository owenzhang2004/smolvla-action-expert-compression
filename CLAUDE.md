# CLAUDE.md

Instructions for any Claude session working in this repository. Keep this file under 200 lines.

## What this project is

Owen's solo final project for NYU Efficient AI (Prof. Sai Qian Zhang). Presentation Dec 14–18, 2026; report Dec 20, 2026.

**Question:** at matched stored bytes, how do low-rank (SVD) and low-bit (quantization) compression of
SmolVLA's flow-matching action expert affect closed-loop LIBERO success and measured inference speed?

Owen is new to VLAs, flow matching and quantization internals. Explain from first principles, with the
math where it helps, and keep answers short.

## Source of truth

**This repository's `main` branch is the single source of truth** for design, code and results.

| What | Where |
| --- | --- |
| Design (questions, arms, statistics, schedule) | `docs/course-design.md` |
| Claim ledger, derivations, correction log, references | `docs/research-record.md` |
| Current state, next task, open items | `docs/HANDOFF.md` |
| What has been run | `results/LOG.md` |
| Versions | `env/pins.md` |

The Claude Doc the design came from is frozen. Do not treat it as current and do not edit it.

## What to read, and when

1. **Every session:** this file, then `docs/HANDOFF.md`.
2. **When a task needs the design:** the matching section of `docs/course-design.md`. Find it by heading;
   do not read the whole file by default. Useful entry points: §3.1 commitments, §3.3 arms,
   §5.1 statistics, §6.1 schedule, §7 checklist.
3. **Only when a task cites it:** one section or ledger row of `docs/research-record.md` (search for the
   ID, for example `A17`). It is about 16,000 words; never load it whole.
4. `docs/archive/` is history. Do not read it unless asked.

## Pins (details in `env/pins.md`)

- LeRobot commit `d490e472a5d1a207cfe0255caaaf74b6d66e4656` (version 0.7.1). Never upgrade it.
- Checkpoint `HuggingFaceVLA/smolvla_libero` at revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.
- Python 3.12 or newer.

The model facts the design rests on (expert hidden 480, intermediate 1280, 32 layers) come from reading
the config and the pinned code. They are not yet confirmed from a loaded model; the manifest task does that.

## Rules

1. **Labels.** Every factual claim you add to `docs/` carries one of: Verified, Secondary report,
   Hypothesis, Estimate, Unverified, Toy (definitions at the top of `docs/course-design.md`). Do not
   write a claim as Verified unless you opened the source or ran the code in this session.
2. **Design changes need Owen's approval.** If code or results suggest changing anything
   `docs/course-design.md` decides (question, arms, byte points, statistics, scope-cut order), write the
   proposal and its evidence under "Proposed design changes" in `docs/HANDOFF.md` and ask. Corrections of
   fact (a wrong number, a stale label) may be fixed directly, with a row in `docs/research-record.md` §5.
3. **Numbers come from scripts.** Never type a result into a document from memory or from chat. Run the
   script, then cite the commit and the output file.
4. **Every task ships with the test that proves it.** A checklist item is done when its test passes.
   If a test of a documented number fails, find out whether the code or the document is wrong; never
   edit the expected value just to pass.
5. **The pairing invariant.** Every closed-loop arm, FP16 included, uses the same inference path, the same
   initial states and the same flow noise per decision. Noise is generated from (seed, episode, step) and
   passed through `select_action(noise=...)`; `lerobot-eval` on its own does not do this (ledger D15).
   Anything that breaks this invalidates the comparison between arms.
6. **Success episodes run on LeRobot's eager path.** The optimized path (K/V hoisting, CUDA graphs,
   `torch.compile`) is for latency only, unless Owen switches every arm after the throughput pilot
   (`docs/course-design.md` §3.1).
7. **Never commit:** per-episode outputs (`*.jsonl`), weights or compressed artifacts, container images,
   tokens. Run outputs live on HPC scratch. Only `results/LOG.md` and small per-arm summary files are
   committed, by hand, not by a Slurm job.
8. **Git.** Commit to `main` in small commits. Use a branch only when another session is working in the
   repository at the same time. Start every task with `git pull`. Do not force-push.
9. **Do not add scaffolding ahead of need.** No empty modules, no speculative abstractions. A module is
   created by the task that needs it.
10. **Say what you did not verify.** If something was assumed, skipped or could not be run (for example
    no GPU in this session), say so in the commit message or in `docs/HANDOFF.md`.

## Commands

```bash
pip install -e ".[dev]"        # once per clone; add --no-build-isolation if the install hangs offline
pre-commit install             # once per clone: ruff + CPU tests run before every commit
pytest                         # CPU tests (GPU tests are deselected by default)
pytest -m gpu                  # GPU tests, on HPC inside the pinned environment
ruff check --fix . && ruff format .
python -m aec.compress.storage                      # byte-point table
python analysis/toy/svd_rank_budget.py              # two-term latency model at 480 / 1280
```

There is no GPU and no LIBERO in a laptop or web session (LIBERO is Linux-only, ledger A20). Anything that
loads the checkpoint or runs an episode runs on NYU HPC; see `env/pins.md` for the setup, which is still
an untested draft.

## Layout

```
CLAUDE.md                 this file
docs/                     design master, record, handoff, archive
env/                      pins and HPC requirements
src/aec/                  the package ("action expert compression")
  compress/storage.py     byte arithmetic for matched-bytes points
tests/                    one test file per checklist item; CPU by default
analysis/toy/             synthetic-model scripts (label: Toy) and their recorded outputs
results/LOG.md            append-only run log
```

Planned, created by the task that needs each one: `src/aec/manifest.py` (checkpoint shapes and bytes),
`src/aec/eval/` (paired evaluation harness), `src/aec/compress/{rtn,gptq,svd,importance_svd}.py`,
`src/aec/inference/` and `src/aec/bench/` (latency, later), `arms/` (one YAML per configuration),
`slurm/` (job templates).

## Keeping the documents current

- After a run: append to `results/LOG.md` (template at its top).
- When a ledger row's status changes (for example the manifest confirms A17): update the row in
  `docs/research-record.md` §1, citing the commit and output, and fix any text in `docs/course-design.md`
  that quoted the old status.
- When a claim turns out wrong: fix it and add a row to `docs/research-record.md` §5.
- When the state of the work changes: update `docs/HANDOFF.md` ("Where things stand" and "Next task").
