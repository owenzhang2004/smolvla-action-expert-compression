# Toy experiments (label: Toy)

Synthetic-model checks referenced by `docs/course-design.md` §9 and `docs/research-record.md` §6.
They are evidence about mechanism plausibility, not about SmolVLA.

Recorded outputs are in `outputs/`. Environment: CPU, torch 2.15.0, float64, `torch.manual_seed(0)`
(`outputs/environment.txt`).

## `compounding_harness.py`

Gated-MLP velocity field (hidden 720, intermediate 2048, action 350, cond 64), 10 Euler steps,
matched-bytes SVD vs 4-bit RTN, three imposed spectra (decay 0 / 0.5 / 1), two shifts (conditioning;
dropped-subspace). Prints local error, cosine, coherence C, growth slope, endpoint error.

- **Its dimensions are the earlier illustrative ones and are left as they were**, so the file stays
  byte-identical to the run recorded on Nov 5, 2026 (it is excluded from ruff for the same reason).
  The model is synthetic, so the dimensions carry no claim about SmolVLA.
- Storage accounting matches the quantizer the script implements: per-output-channel symmetric RTN with
  one bf16 scale per row (2,221,472 B for the block), so the matched SVD rank is 146 (2,214,528 B).
- Numerics: everything is float64 and, by default, SVD factors and RTN scales are not rounded to bf16, so
  the comparison matches a storage budget rather than performing deployment-precision arithmetic.
  `--round` rounds factors and scales to bf16 before reconstruction; that produced small numerical
  changes (for example 0.2856 -> 0.2842) without changing the qualitative conclusions.
- `decay=0` sets all singular values equal, so the retained subspace is non-unique and small
  cross-platform differences are expected at d = 0. The teacher is an untrained random field.
- The dropped-subspace shift is built from the SVD compressor's own discarded directions: a targeted
  stress test, not a neutral sample of shifts.
- Coherence is computed on untransported errors; the transported version (record §4.4) is not implemented.

## `svd_rank_budget.py`

Two-term latency model for one MLP block: kernel launches plus bytes moved. Prints the matched-bytes rank
and the largest rank at which SVD is strictly faster than dense bf16 and than W4A16, for Jetson Thor and
A100 regimes.

- Defaults are the checkpoint's dimensions (E = 480, F = 1280; ledger A17) with scale groups of 120 along
  the hidden dimension and 128 along the intermediate dimension. Output: `outputs/svd_rank_budget.txt`.
  These are the numbers in `docs/course-design.md` §3.2, pinned by `tests/test_latency_model.py`.
- `--legacy` reproduces the earlier illustrative run (E = 720, F = 2048, uniform group 128).
  Output: `outputs/svd_rank_budget.legacy_720_2048.txt`.
- The sensitivity profile in the demo schedule (1 + 0.3k) is invented and carries no evidential weight.
- The model omits arithmetic, kernel selection, occupancy and caching; its conclusions are conditional
  on it. The measured microbenchmarks (Q5) decide.

## Run

```bash
python analysis/toy/compounding_harness.py [--round]     # needs torch
python analysis/toy/svd_rank_budget.py [--legacy]        # pure Python
```
