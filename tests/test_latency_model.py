"""The two-term latency model must reproduce the prediction table in docs/course-design.md section 3.2.

The model is an Estimate (launches plus bytes moved); these tests pin its arithmetic, not its truth.
The measured microbenchmarks (Q5) are what confirm or falsify it.
"""

import importlib.util
from pathlib import Path

import pytest

from aec.compress import storage as st

_SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "toy" / "svd_rank_budget.py"
_spec = importlib.util.spec_from_file_location("svd_rank_budget", _SCRIPT)
model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model)

A100_BW = 1.555e12  # ledger B6, Unverified until the datasheet is opened
THOR_BW = 273e9


def hw(bandwidth, t_launch_us):
    return model.Hardware(bandwidth_Bps=bandwidth, t_launch_s=t_launch_us * 1e-6)


def test_model_bytes_agree_with_storage_module():
    m, d = model.MLPBlock(), st.MLPDims()
    assert model.bytes_quant(m) == st.mlp_quant_bytes(d, 4)
    assert model.bytes_per_rank(m) == st.mlp_bytes_per_rank(d)
    assert model.bytes_dense(m) == st.mlp_dense_bytes(d)
    assert model.matched_bytes_rank(m) == st.matched_rank(d, 4)


@pytest.mark.parametrize(
    "bandwidth, t_us, penalty_mb",
    [(A100_BW, 1, 3.11), (A100_BW, 2, 6.22), (THOR_BW, 1, 0.55), (THOR_BW, 2, 1.09)],
)
def test_launch_penalty(bandwidth, t_us, penalty_mb):
    assert model.launch_penalty_bytes(model.MLPBlock(), hw(bandwidth, t_us)) / 1e6 == pytest.approx(
        penalty_mb, abs=0.005
    )


def test_a100_svd_never_beats_w4a16():
    m = model.MLPBlock()
    assert model.max_rank_faster_than_quant(m, hw(A100_BW, 1)) is None
    assert model.max_rank_faster_than_quant(m, hw(A100_BW, 2)) is None


def test_a100_svd_vs_dense():
    m = model.MLPBlock()
    assert model.max_rank_faster_than_dense(m, hw(A100_BW, 1)) == 60
    assert model.max_rank_faster_than_dense(m, hw(A100_BW, 2)) is None


def test_thor_break_even_ranks():
    m = model.MLPBlock()
    assert model.max_rank_faster_than_dense(m, hw(THOR_BW, 1)) == 327
    assert model.max_rank_faster_than_dense(m, hw(THOR_BW, 2)) == 270
    assert model.max_rank_faster_than_quant(m, hw(THOR_BW, 1)) == 42
    assert model.max_rank_faster_than_quant(m, hw(THOR_BW, 2)) is None


def test_break_even_is_strict():
    m, h = model.MLPBlock(), hw(THOR_BW, 1)
    r = model.max_rank_faster_than_quant(m, h)
    assert model.latency_svd(r, m, h) < model.latency_quant(m, h) <= model.latency_svd(r + 1, m, h)


def test_legacy_run_matches_record_section_6_2():
    m = model.MLPBlock(E=720, F=2048, scale_layout="uniform")
    assert model.bytes_quant(m) == 2_280_960
    assert model.matched_bytes_rank(m) == 150
    assert model.max_rank_faster_than_quant(m, hw(THOR_BW, 2)) == 78
    assert model.max_rank_faster_than_quant(m, model.Hardware(efficiency=0.7)) == 99
    assert model.max_rank_faster_than_quant(m, hw(THOR_BW, 8)) is None
