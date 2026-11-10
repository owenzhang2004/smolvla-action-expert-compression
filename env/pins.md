# Pins

Three outside ingredients can change underneath the project: the LeRobot code, the checkpoint weights,
and the software environment. Each is frozen here. A difference between two arms must come from the
compression and nothing else.

| Ingredient | Pin | How it was established |
| --- | --- | --- |
| LeRobot | commit `d490e472a5d1a207cfe0255caaaf74b6d66e4656`, version 0.7.1, dated Nov 6, 2026 | Verified: fetched from GitHub on Nov 10, 2026 (ledger A20) |
| Checkpoint | `HuggingFaceVLA/smolvla_libero` at revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` (last modified 2025-09-17) | Read from the Hub API through a fetch summary on Nov 10, 2026 (ledger A18). **Confirm at first download** |
| Python | 3.12 or newer | Verified: LeRobot's `requires-python` at the pinned commit |
| torch | LeRobot allows `>=2.7,<2.16`. The exact version is whatever the first working HPC install resolves; record it in `env/freeze-hpc.txt` | The toy scripts and the `torch.compile` claims (ledger C24) used 2.15.0 |
| transformers | 5.22.x, pulled in by `lerobot[smolvla]` | Verified: `pyproject.toml` at the pinned commit |
| Calibration dataset | `lerobot/libero` (1.9 GB, video) at a revision **to be recorded** | Dataset located in LeRobot's docs (ledger A19); revision not yet read |
| Container image | **to be recorded** | Not built yet |

The checkpoint is public and not gated, so downloading it needs no Hugging Face token.

## HPC setup (untested draft)

Nothing below has been run. It is the starting point for the first week-1 task, which replaces this
section with the commands that actually worked.

Known from Owen (confirm on the cluster): NYU HPC Cloud Bursting through Open OnDemand, Slurm account
`csci_ga_3033-2026fa`, A100 40GB partitions, conda inside a Singularity/Apptainer overlay.

```bash
# inside the container / overlay, in a fresh Python >= 3.12 environment
pip install -r env/requirements-hpc.txt
pip install -e ".[dev]"
python -c "import lerobot; print(lerobot.__version__)"     # expect 0.7.1
pip freeze > env/freeze-hpc.txt                            # commit this file

# headless rendering (LeRobot's LIBERO docs)
export MUJOCO_GL=egl
```

Download the checkpoint at the pinned revision and check it is the one recorded above:

```python
from huggingface_hub import snapshot_download

path = snapshot_download(
    "HuggingFaceVLA/smolvla_libero",
    revision="6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
)
print(path)  # the snapshot directory name is the revision hash
```

Smoke test, one episode of one task (flags from LeRobot's `docs/source/libero.mdx` at the pinned commit;
the exact combination is untested):

```bash
lerobot-eval \
  --policy.path="$CKPT_DIR" \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --seed=0
```

## Open questions for NYU HPC

- Do the course GPU-hours expire at the end of term?
- Is a job that runs several simulator environments on one GPU billed per GPU-hour?
- What is the scratch purge policy? Per-episode outputs live on scratch, so summaries must be copied off
  promptly.
- Exact partition names and whether building or pulling an Apptainer image is allowed on the bursting nodes.
