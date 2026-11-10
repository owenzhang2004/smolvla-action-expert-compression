"""Stored-byte arithmetic for matched-bytes comparisons.

Design: docs/course-design.md section 3.1; tables: docs/research-record.md section 3.2.

Pure Python on purpose: these are the numbers every arm is matched on, so they must be
checkable on any machine without torch or the checkpoint.

Conventions
-----------
* A linear layer has weight shape (out_features, in_features).
* Quantized storage: `bits` per weight plus one scale per (output row, group of `group`
  consecutive input columns), symmetric (no zero-points). Scales are bf16 (2 bytes).
* Low-rank storage: factors (out x r) and (r x in) in bf16, so r * (out + in) * 2 bytes.
* The gated MLP block has gate and up (F x E each) and down (E x F). Its joint low-rank form
  factors gate and up together as one (2F x E) matrix, plus down on its own, which costs
  2 * (2E + 3F) bytes per rank.

These are formula bytes. Bytes measured from saved files (which add metadata and alignment
padding) are compared against them in a later test, once compressors write artifacts.
The dimensions default to the checkpoint's (E = 480, F = 1280; ledger A17) and are confirmed
by the loaded-model manifest before any run.
"""

from dataclasses import dataclass

BF16_BYTES = 2


def dense_bytes(out_features: int, in_features: int, dtype_bytes: int = BF16_BYTES) -> int:
    """Bytes of an uncompressed weight matrix."""
    return out_features * in_features * dtype_bytes


def quant_bytes(
    out_features: int, in_features: int, bits: int, group: int, scale_bytes: int = BF16_BYTES
) -> int:
    """Bytes of a `bits`-bit weight matrix with one scale per (row, group of input columns).

    Raises ValueError if `group` does not divide `in_features`: a ragged last group changes
    both the byte count and what a fused kernel can read, so it must be an explicit decision.
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    if group < 1 or in_features % group != 0:
        raise ValueError(f"group {group} does not divide in_features {in_features}")
    n_weights = out_features * in_features
    weight_bytes = -(-n_weights * bits // 8)  # ceil; no alignment padding modelled
    n_scales = out_features * (in_features // group)
    return weight_bytes + n_scales * scale_bytes


def lowrank_bytes(out_features: int, in_features: int, rank: int, dtype_bytes: int = BF16_BYTES) -> int:
    """Bytes of rank-`rank` factors (out x r) and (r x in)."""
    if rank < 0 or rank > min(out_features, in_features):
        raise ValueError(f"rank {rank} outside [0, {min(out_features, in_features)}]")
    return rank * (out_features + in_features) * dtype_bytes


@dataclass(frozen=True)
class MLPDims:
    """Gated-MLP block of the action expert.

    group_E groups the hidden (input) dim of gate and up; group_F groups the intermediate
    (input) dim of down. 128 does not divide 480, hence 120. The group size is replaced by
    whatever the chosen fused W4A16 kernel reads (docs/course-design.md section 7.1).
    """

    E: int = 480
    F: int = 1280
    group_E: int = 120
    group_F: int = 128

    @property
    def n_weights(self) -> int:
        return 3 * self.E * self.F


def mlp_dense_bytes(d: MLPDims = MLPDims()) -> int:
    return 2 * dense_bytes(d.F, d.E) + dense_bytes(d.E, d.F)


def mlp_scale_bytes(d: MLPDims = MLPDims()) -> int:
    """Scale metadata only (identical at every bit width)."""
    return mlp_quant_bytes(d, bits=8) - d.n_weights


def mlp_quant_bytes(d: MLPDims = MLPDims(), bits: int = 4) -> int:
    gate_up = 2 * quant_bytes(d.F, d.E, bits, d.group_E)
    down = quant_bytes(d.E, d.F, bits, d.group_F)
    return gate_up + down


def mlp_bytes_per_rank(d: MLPDims = MLPDims()) -> int:
    """Joint gate-up (2F x E) plus down (E x F), both at the same rank."""
    return lowrank_bytes(2 * d.F, d.E, 1) + lowrank_bytes(d.E, d.F, 1)


def mlp_lowrank_bytes(d: MLPDims = MLPDims(), rank: int = 1) -> int:
    if rank < 0 or rank > d.E:
        raise ValueError(f"rank {rank} outside [0, {d.E}]")
    return rank * mlp_bytes_per_rank(d)


def matched_rank(d: MLPDims = MLPDims(), bits: int = 4) -> int:
    """Largest rank whose factors cost no more bytes than `bits`-bit quantization."""
    return mlp_quant_bytes(d, bits) // mlp_bytes_per_rank(d)


def bits_equivalent(d: MLPDims = MLPDims(), rank: int = 1) -> float:
    """Bit width whose quantized block (with scales) costs the same as rank-`rank` factors."""
    return (mlp_lowrank_bytes(d, rank) - mlp_scale_bytes(d)) * 8 / d.n_weights


def byte_points(d: MLPDims = MLPDims(), bit_widths: tuple[int, ...] = (2, 3, 4, 6, 8)) -> list[dict]:
    """The comparison points of docs/course-design.md section 3.1: bytes and matched rank per bit width."""
    rows = []
    for b in bit_widths:
        r = matched_rank(d, b)
        rows.append(
            {
                "bits": b,
                "quant_bytes": mlp_quant_bytes(d, b),
                "matched_rank": r,
                "lowrank_bytes": mlp_lowrank_bytes(d, r),
                "rank_over_E": r / d.E,
            }
        )
    return rows


if __name__ == "__main__":
    d = MLPDims()
    print(f"E = {d.E}, F = {d.F}: {d.n_weights:,} MLP weights, scales {mlp_scale_bytes(d):,} B, ")
    print(f"{mlp_bytes_per_rank(d):,} B per rank, dense bf16 {mlp_dense_bytes(d):,} B")
    print(f"{'bits':>5} {'quant bytes':>12} {'rank':>5} {'rank/E':>7} {'low-rank bytes':>15}")
    for row in byte_points(d):
        print(
            f"{row['bits']:>5} {row['quant_bytes']:>12,} {row['matched_rank']:>5} "
            f"{row['rank_over_E']:>7.2f} {row['lowrank_bytes']:>15,}"
        )
    r_dense = mlp_dense_bytes(d) // mlp_bytes_per_rank(d)
    print(f"dense {mlp_dense_bytes(d):>12,} {r_dense:>5} {r_dense / d.E:>7.2f}")
