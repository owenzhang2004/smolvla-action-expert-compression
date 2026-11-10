"""Byte arithmetic must reproduce the tables in docs/ (course design sections 3.2, 4.2; record section 3.2).

If one of these fails, either the code or the document is wrong. Fix whichever it is and say so
in the commit; never edit an expected number just to make the test pass.
"""

import pytest

from aec.compress import storage as st

# Record section 3.2, "At the checkpoint's dimensions (E = 480, F = 1280; A17)".
DOC_TABLE = {
    2: (490_880, 51),
    3: (721_280, 75),
    4: (951_680, 99),
    6: (1_412_480, 147),
    8: (1_873_280, 195),
}


def test_block_constants():
    d = st.MLPDims()
    assert d.n_weights == 1_843_200
    assert st.mlp_scale_bytes(d) == 30_080
    assert st.mlp_bytes_per_rank(d) == 9_600
    assert st.mlp_dense_bytes(d) == 3_686_400


@pytest.mark.parametrize("bits", sorted(DOC_TABLE))
def test_byte_points_match_doc(bits):
    expected_bytes, expected_rank = DOC_TABLE[bits]
    d = st.MLPDims()
    assert st.mlp_quant_bytes(d, bits) == expected_bytes
    assert st.matched_rank(d, bits) == expected_rank


@pytest.mark.parametrize("bits", sorted(DOC_TABLE))
def test_matched_rank_is_the_largest_that_fits(bits):
    d = st.MLPDims()
    r = st.matched_rank(d, bits)
    budget = st.mlp_quant_bytes(d, bits)
    assert st.mlp_lowrank_bytes(d, r) <= budget < st.mlp_lowrank_bytes(d, r + 1)


def test_rank_fractions_quoted_in_config_table():
    # Course design section 3.3: W4-matched rank ~ 0.21E, W8-matched ~ 0.41E.
    d = st.MLPDims()
    assert round(st.matched_rank(d, 4) / d.E, 2) == 0.21
    assert round(st.matched_rank(d, 8) / d.E, 2) == 0.41


def test_dense_bf16_matches_rank_0p8E():
    d = st.MLPDims()
    assert st.mlp_dense_bytes(d) // st.mlp_bytes_per_rank(d) == 384
    assert 384 / d.E == pytest.approx(0.80)


def test_rank_0p6E_is_about_12_bits():
    d = st.MLPDims()
    assert st.bits_equivalent(d, rank=288) == pytest.approx(11.87, abs=0.01)


def test_residual_on_square_layer_costs_60_ranks():
    # Course design section 4.2: on a 480x480 layer a 4-bit residual costs the same as 60 bf16 ranks.
    residual = 480 * 480 * 4 // 8
    assert residual == st.lowrank_bytes(480, 480, 60)


def test_group_128_does_not_divide_480():
    with pytest.raises(ValueError):
        st.quant_bytes(1280, 480, bits=4, group=128)
    assert st.quant_bytes(1280, 480, bits=4, group=120) == 1280 * 480 // 2 + 1280 * 4 * 2


def test_legacy_illustrative_dimensions():
    # Record section 3.2, illustrative table at E = 720, F = 2048 (group 120 along E, 128 along F).
    d = st.MLPDims(E=720, F=2048)
    assert st.mlp_quant_bytes(d, 4) == 2_284_032
    assert st.mlp_bytes_per_rank(d) == 15_168
    assert [st.matched_rank(d, b) for b in (2, 3, 4, 6, 8)] == [77, 114, 150, 223, 296]


def test_byte_points_rows():
    rows = st.byte_points()
    assert [r["bits"] for r in rows] == [2, 3, 4, 6, 8]
    assert all(r["lowrank_bytes"] <= r["quant_bytes"] for r in rows)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        st.lowrank_bytes(480, 480, 481)
    with pytest.raises(ValueError):
        st.quant_bytes(480, 480, bits=0, group=120)
    with pytest.raises(ValueError):
        st.mlp_lowrank_bytes(st.MLPDims(), 481)
