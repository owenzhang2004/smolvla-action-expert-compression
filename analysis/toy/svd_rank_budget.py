"""Two-term latency model for a gated-MLP block: launches plus bytes moved (label: Toy / Estimate).

Model (per block, per denoising step):
  dense  : t_d   = n_q   * t_launch + bytes_dense / B
  W4A16  : t_q   = n_q   * t_launch + bytes_q / B
  SVD(r) : t_svd = n_svd * t_launch + r * bytes_per_rank / B
with bytes_q = 3EF*w_bits/8 + (scale metadata) and
bytes_per_rank = dtype_bytes * (2E + 3F)   (joint gate-up factors E x r, r x 2F; down factors F x r, r x E).

Scale metadata depends on the quantizer layout:
  "split"       groups of group_E along the hidden dim for gate and up, group_F along the
                intermediate dim for down (the layout in docs/course-design.md section 3.1;
                120 / 128 because 128 does not divide 480)
  "uniform"     3EF / group scales (the earlier illustrative layout, group 128)
  "per-channel" one scale per output channel (what compounding_harness.py implements)

Strict inequality t_svd < t_q gives
  r < (bytes_q - (n_svd - n_q) * t_launch * B) / bytes_per_rank.
If the right-hand side is <= 0, no rank beats W4A16 on latency: the two extra launches cost more
than all of W4A16's bytes. The same bound with bytes_dense gives the break-even against dense bf16.

The model omits arithmetic, kernel selection, occupancy and caching. Its conclusions are conditional
on it; the microbenchmarks in docs/course-design.md section 3.1 decide.

Defaults are the checkpoint's dimensions (E = 480, F = 1280; ledger A17).
`--legacy` reproduces the earlier illustrative run (E = 720, F = 2048, uniform group 128).
"""

import math
import sys
from dataclasses import dataclass


@dataclass
class Hardware:
    bandwidth_Bps: float = 273e9  # Jetson AGX Thor LPDDR5X, peak
    t_launch_s: float = 2e-6  # GPU-side gap per kernel under CUDA graphs; ~8e-6 eager (ledger B12, Estimate)
    efficiency: float = 1.0  # achievable fraction of peak bandwidth for small batch-1 GEMMs


@dataclass
class MLPBlock:
    E: int = 480  # hidden dim (SmolVLA expert, ledger A17)
    F: int = 1280  # intermediate dim
    w_bits: int = 4  # weight-only quantization bits
    scale_layout: str = "split"  # "split", "uniform" or "per-channel"
    group_E: int = 120  # split layout: group size along the hidden dim (gate, up)
    group_F: int = 128  # split layout: group size along the intermediate dim (down)
    group: int = 128  # uniform layout: one group size for every matrix
    scale_bits: int = 16
    svd_dtype_bytes: int = 2  # bf16 factors
    launches_quant: int = 2  # fused gate-up GEMM + down GEMM (dense bf16 has the same count)
    launches_svd: int = 4  # two GEMMs per factorized matrix


def n_scales(m: MLPBlock) -> float:
    if m.scale_layout == "split":
        return 2 * m.F * (m.E / m.group_E) + m.E * (m.F / m.group_F)
    if m.scale_layout == "uniform":
        return 3 * m.E * m.F / m.group
    if m.scale_layout == "per-channel":
        return 2 * m.F + m.E  # gate (F rows), up (F rows), down (E rows)
    raise ValueError(m.scale_layout)


def bytes_quant(m: MLPBlock) -> float:
    return 3 * m.E * m.F * m.w_bits / 8 + n_scales(m) * m.scale_bits / 8


def bytes_per_rank(m: MLPBlock) -> float:
    return m.svd_dtype_bytes * (2 * m.E + 3 * m.F)


def bytes_dense(m: MLPBlock) -> float:
    return 3 * m.E * m.F * 2  # bf16, no scales


def latency_quant(m: MLPBlock, hw: Hardware) -> float:
    return m.launches_quant * hw.t_launch_s + bytes_quant(m) / (hw.bandwidth_Bps * hw.efficiency)


def latency_svd(r: float, m: MLPBlock, hw: Hardware) -> float:
    return m.launches_svd * hw.t_launch_s + r * bytes_per_rank(m) / (hw.bandwidth_Bps * hw.efficiency)


def launch_penalty_bytes(m: MLPBlock, hw: Hardware) -> float:
    return (m.launches_svd - m.launches_quant) * hw.t_launch_s * hw.bandwidth_Bps * hw.efficiency


def _max_rank_below(budget_bytes: float, m: MLPBlock) -> int | None:
    if budget_bytes <= 0:
        return None
    r = math.ceil(budget_bytes / bytes_per_rank(m)) - 1  # strictly below the bound
    return r if r >= 1 else None


def max_rank_faster_than_dense(m: MLPBlock, hw: Hardware) -> int | None:
    """Largest r with SVD strictly faster than dense bf16, or None if no rank qualifies."""
    return _max_rank_below(bytes_dense(m) - launch_penalty_bytes(m, hw), m)


def max_rank_faster_than_quant(m: MLPBlock, hw: Hardware) -> int | None:
    """Largest r with latency_svd(r) < latency_quant, or None if no rank qualifies."""
    return _max_rank_below(bytes_quant(m) - launch_penalty_bytes(m, hw), m)


def matched_bytes_rank(m: MLPBlock) -> int:
    """Rank at which SVD factors cost the same bytes as W4A16 (ignores launches)."""
    return int(bytes_quant(m) // bytes_per_rank(m))


def rank_schedule(
    m: MLPBlock, hw: Hardware, sensitivity: list[float], per_step: bool = True
) -> list[int] | None:
    """Allocate ranks across K steps in proportion to a sensitivity profile.

    per_step=True : every step individually beats W4A16  (r_t <= r_max for all t).
    per_step=False: only the per-call total beats W4A16    (sum r_t <= K * r_max);
                    high-sensitivity steps may exceed r_max if others fall below it.
    Returns None if no feasible schedule exists.
    """
    r_max = max_rank_faster_than_quant(m, hw)
    if r_max is None:
        return None
    K = len(sensitivity)
    full_rank = min(m.E, 2 * m.F)  # rank cap of the joint gate-up matrix
    s = [max(x, 0.0) for x in sensitivity]
    tot = sum(s) or 1.0
    if per_step:
        return [min(r_max, int(r_max * K * x / tot)) if max(s) > 0 else r_max for x in s]
    sched = [min(full_rank, int(K * r_max * x / tot)) for x in s]
    # re-distribute anything clipped by full_rank so the total stays <= K * r_max
    while sum(sched) > K * r_max:
        i = max(range(K), key=lambda j: sched[j])
        sched[i] -= 1
    return sched


REGIMES = [
    ("Thor, CUDA graphs 1 us, peak BW", Hardware(t_launch_s=1e-6)),
    ("Thor, CUDA graphs 2 us, peak BW", Hardware()),
    ("Thor, CUDA graphs 2 us, 70% BW", Hardware(efficiency=0.7)),
    ("Thor, eager PyTorch 8 us, peak BW", Hardware(t_launch_s=8e-6)),
    ("A100 40GB, CUDA graphs 1 us", Hardware(bandwidth_Bps=1.555e12, t_launch_s=1e-6)),
    ("A100 40GB, CUDA graphs 2 us", Hardware(bandwidth_Bps=1.555e12, t_launch_s=2e-6)),
]


def main(legacy: bool = False) -> None:
    if legacy:
        dims = dict(E=720, F=2048)
        layouts = [("group-128 scales", "uniform"), ("per-channel scales", "per-channel")]
        m = MLPBlock(scale_layout="uniform", **dims)
    else:
        dims = dict(E=480, F=1280)
        layouts = [("group-120/128 scales", "split"), ("per-channel scales", "per-channel")]
        m = MLPBlock(**dims)
    print(f"E = {m.E}, F = {m.F}; dense bf16 = {bytes_dense(m):,.0f} B; {bytes_per_rank(m):,.0f} B per rank")
    for label, layout in layouts:
        mm = MLPBlock(scale_layout=layout, **dims)
        print(f"{label}: W4A16 = {bytes_quant(mm):,.0f} B, matched-bytes rank = {matched_bytes_rank(mm)}")
    for name, hw in REGIMES:
        r_bytes = matched_bytes_rank(m)
        r_fast = max_rank_faster_than_quant(m, hw)
        r_dense = max_rank_faster_than_dense(m, hw)
        print(f"\n[{name}]")
        print(f"  W4A16 block: {bytes_quant(m) / 1e6:.2f} MB, {latency_quant(m, hw) * 1e3:.4f} ms/step")
        print(f"  launch penalty of 2 extra kernels = {launch_penalty_bytes(m, hw) / 1e6:.2f} MB-equivalent")
        print(
            f"  matched-bytes rank (equal bytes, slower by the launch penalty): r = {r_bytes}, "
            f"{latency_svd(r_bytes, m, hw) * 1e3:.4f} ms/step"
        )
        if r_dense is None:
            print("  max rank strictly faster than dense bf16: none (launch penalty exceeds the dense block)")
        else:
            print(f"  max rank strictly faster than dense bf16: r = {r_dense} ({r_dense / m.E:.2f} E)")
        if r_fast is None:
            print("  max rank strictly faster than W4A16: none (launch penalty exceeds W4A16's bytes)")
        else:
            print(f"  max rank strictly faster than W4A16: r = {r_fast} ({r_fast / m.E:.2f} E)")
            # example schedule from an invented, monotone sensitivity profile; it exercises the function only
            sens = [1.0 + 0.3 * k for k in range(10)]
            print(f"  per-step schedule (each step faster):  {rank_schedule(m, hw, sens, per_step=True)}")
            print(f"  per-call schedule (total faster):      {rank_schedule(m, hw, sens, per_step=False)}")


if __name__ == "__main__":
    main(legacy="--legacy" in sys.argv)
