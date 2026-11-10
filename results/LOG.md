# Run log

Append-only. One entry per run or batch of runs, newest at the bottom. Numbers are copied from script
output, never typed from memory. Per-episode files stay on HPC scratch; this file and small per-arm
summaries are what gets committed.

Template:

```
## YYYY-MM-DD  short title
- What: one line
- Command: exact command or Slurm script
- Code: git commit SHA            Config: arm file and its hash (if any)
- Environment: machine, LeRobot commit, checkpoint revision, torch version
- Outputs: where they live (scratch path) and what was committed
- Result: the headline numbers, with intervals where they exist
- Notes: anything unexpected; what was not verified
```

---

## 2026-11-10  Repository setup (no model runs)
- What: byte-point table and two-term latency model recomputed at the checkpoint's dimensions (E = 480, F = 1280).
- Command: `python -m aec.compress.storage`; `python analysis/toy/svd_rank_budget.py`; `python analysis/toy/svd_rank_budget.py --legacy`; `pytest`
- Code: the initial setup commit. Config: none.
- Environment: cloud container, CPU, Python 3.13.18, torch 2.15.0 (only the toy harness needs torch).
- Outputs: `analysis/toy/outputs/svd_rank_budget.txt`, `analysis/toy/outputs/svd_rank_budget.legacy_720_2048.txt` (committed).
- Result: 2 / 3 / 4 / 6 / 8-bit points = 490,880 / 721,280 / 951,680 / 1,412,480 / 1,873,280 B, matched ranks 51 / 75 / 99 / 147 / 195; dense bf16 3,686,400 B = rank 384. 29 CPU tests pass.
- Notes: formula bytes only; nothing has been loaded or measured. `compounding_harness.py` was not rerun; its outputs are the Nov 5 recordings.
