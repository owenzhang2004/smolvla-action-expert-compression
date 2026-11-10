# smolvla-action-expert-compression

**Is the action expert low-rank?** Bytes-matched compression of a flow-matching VLA under closed-loop evaluation.

Final project for NYU Efficient AI (Fall 2026). At equal stored bytes, does low-rank (SVD) or low-bit
(quantization) compression of SmolVLA's action expert keep more closed-loop LIBERO success, and does the
storage saved turn into measured speed?

**Status (Nov 10, 2026):** design frozen, repository set up, no model runs yet.

| Read | For |
| --- | --- |
| [`docs/course-design.md`](docs/course-design.md) | The design: questions, arms, statistics, schedule |
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | Where things stand and what is next |
| [`docs/research-record.md`](docs/research-record.md) | Claim ledger, derivations, correction log, references |
| [`env/pins.md`](env/pins.md) | Pinned versions and HPC setup |
| [`results/LOG.md`](results/LOG.md) | What has been run |

```bash
pip install -e ".[dev]" && pre-commit install
pytest
```

Model: [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero).
Code base: [LeRobot](https://github.com/huggingface/lerobot) at a pinned commit.
