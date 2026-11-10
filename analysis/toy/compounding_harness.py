"""Coherence index and trajectory growth exponent for compressed flow-matching
velocity fields under Euler integration.

Teacher: float64 dummy velocity field v(x, t, c) with a gated MLP.
Students: (a) truncated SVD of gate/up/down weights, (b) per-channel RTN
fake-quantization of the same weights. Both are compared at matched bytes.

Metrics (per sample, then averaged):
  local error      ||v_hat(x_k) - v(x_k)|| / ||v(x_k)||   (teacher-forced)
  cosine           cos(v_hat(x_k), v(x_k))                  (teacher-forced)
  coherence index  C = ||sum_k e_k|| / sum_k ||e_k||         (teacher-forced)
  growth exponent  slope of log ||x_hat_k - x_k|| vs log k  (free-running)
  shift ratio      metric_OOD / metric_ID
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn

torch.manual_seed(0)
DTYPE = torch.float64
# All arithmetic is float64. By default the SVD factors and RTN scales are NOT rounded to bf16, so
# "W4A16 vs bf16 SVD" names the storage budget being matched, not the numerics performed. Set
# ROUND_STORAGE = True (or pass --round) to round factors and scales to bf16 before reconstruction.
ROUND_STORAGE = False


def _store(t):
    return t.to(torch.bfloat16).to(DTYPE) if ROUND_STORAGE else t


class VelocityField(nn.Module):
    """v(x, t, c): x action chunk (B, D), t in [0,1], c conditioning (B, Dc)."""

    def __init__(self, d_action=350, d_cond=64, hidden=720, inter=2048):
        super().__init__()
        self.inp = nn.Linear(d_action + d_cond + 1, hidden)
        self.gate = nn.Linear(hidden, inter, bias=False)
        self.up = nn.Linear(hidden, inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)
        self.out = nn.Linear(hidden, d_action)
        self.h_shift = None   # optional (B, hidden) shift of the MLP input: a distribution shift
                              # defined in the expert's own input space, applied to every model

    def forward(self, x, t, c):
        h = torch.cat([x, c, t.expand(x.shape[0], 1)], dim=-1)
        h = torch.tanh(self.inp(h))
        if self.h_shift is not None:
            h = h + self.h_shift
        h = h + self.down(Fn.silu(self.gate(h)) * self.up(h))  # residual gated MLP
        return self.out(h)


def impose_spectrum_(linear: nn.Linear, decay: float):
    """Rescale singular values to sigma_i ~ i^(-decay), keeping singular vectors and Frobenius norm.
    decay=0 sets every singular value equal (exactly flat; this replaces, not keeps, the random
    init's Marchenko-Pastur spectrum, and makes the retained rank-r subspace non-unique, so small
    cross-platform differences are expected at d=0); decay=1 is steep."""
    W = linear.weight.data
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    i = torch.arange(1, S.numel() + 1, dtype=DTYPE)
    S_new = i ** (-decay)
    S_new = S_new * (S.norm() / S_new.norm())
    linear.weight.data = (U * S_new) @ Vh


# ---------------------------------------------------------------- compression
def svd_truncate_(linear: nn.Linear, rank: int):
    W = linear.weight.data
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    linear.weight.data = _store(U[:, :rank] * S[:rank]) @ _store(Vh[:rank])


def joint_svd_truncate_(gate: nn.Linear, up: nn.Linear, rank: int):
    """Joint gate-up decomposition: [W_gate; W_up] shares one input-side factor."""
    W = torch.cat([gate.weight.data, up.weight.data], dim=0)  # (2F, E)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    Wr = _store(U[:, :rank] * S[:rank]) @ _store(Vh[:rank])
    F_ = gate.weight.shape[0]
    gate.weight.data, up.weight.data = Wr[:F_], Wr[F_:]


def rtn_quantize_(linear: nn.Linear, bits: int = 4):
    """Symmetric per-output-channel round-to-nearest, dequantized back (fake quant)."""
    W = linear.weight.data
    qmax = 2 ** (bits - 1) - 1
    scale = _store(W.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax)
    linear.weight.data = torch.clamp(torch.round(W / scale), -qmax, qmax) * scale


def mlp_bytes_svd(E, F, r, dtype_bytes=2):
    return dtype_bytes * r * (2 * E + 3 * F)        # joint gate-up + down


def mlp_bytes_quant(E, F, bits=4, scale_bits=16):
    """Per-output-channel symmetric RTN, matching rtn_quantize_: one scale per row of
    gate (F rows), up (F rows) and down (E rows). Group-wise scales would cost more."""
    return 3 * E * F * bits / 8 + (2 * F + E) * scale_bits / 8


def matched_rank(E, F, bits=4):
    return int(mlp_bytes_quant(E, F, bits) // (2 * (2 * E + 3 * F)))


def make_students(teacher: VelocityField, bits=4):
    E, F = teacher.gate.in_features, teacher.gate.out_features
    r = matched_rank(E, F, bits)
    svd = VelocityField(teacher.out.out_features, teacher.inp.in_features
                        - teacher.out.out_features - 1, E, F).to(DTYPE)
    svd.load_state_dict(teacher.state_dict())
    joint_svd_truncate_(svd.gate, svd.up, r)
    svd_truncate_(svd.down, r)

    quant = VelocityField(teacher.out.out_features, teacher.inp.in_features
                          - teacher.out.out_features - 1, E, F).to(DTYPE)
    quant.load_state_dict(teacher.state_dict())
    for lin in (quant.gate, quant.up, quant.down):
        rtn_quantize_(lin, bits)
    print(f"matched bytes (per-channel RTN): W{bits}A16 = {mlp_bytes_quant(E, F, bits):,.0f} B, "
          f"SVD rank {r} = {mlp_bytes_svd(E, F, r):,.0f} B")
    return {"svd": svd, "quant": quant}


def dropped_direction_shift(teacher, rank, n, scale):
    """Random shift of the MLP input lying in the input-side subspace that rank-r joint SVD drops."""
    W = torch.cat([teacher.gate.weight.data, teacher.up.weight.data], dim=0)
    _, _, Vh = torch.linalg.svd(W, full_matrices=False)
    basis = Vh[rank:]                                   # (E - r, E) dropped right singular vectors
    coef = torch.randn(n, basis.shape[0], dtype=DTYPE)
    shift = coef @ basis
    return scale * shift / shift.norm(dim=-1, keepdim=True)


# ----------------------------------------------------------------- integration
@torch.no_grad()
def euler_states(model, x0, c, K=10):
    """Free-running Euler rollout; returns states x_0..x_K and velocities v_0..v_{K-1}."""
    xs, vs, x = [x0], [], x0
    for k in range(K):
        t = torch.full((1, 1), k / K, dtype=DTYPE)
        v = model(x, t, c)
        vs.append(v)
        x = x + v / K
        xs.append(x)
    return torch.stack(xs), torch.stack(vs)


@torch.no_grad()
def teacher_forced_errors(student, teacher_xs, teacher_vs, c, K=10):
    """e_k = v_hat(x_k) - v(x_k) at the teacher's states."""
    es, cos = [], []
    for k in range(K):
        t = torch.full((1, 1), k / K, dtype=DTYPE)
        v_hat = student(teacher_xs[k], t, c)
        e = v_hat - teacher_vs[k]
        es.append(e)
        cos.append(Fn.cosine_similarity(v_hat, teacher_vs[k], dim=-1))
    return torch.stack(es), torch.stack(cos)        # (K, B, D), (K, B)


# --------------------------------------------------------------------- metrics
def coherence_index(es):
    """C = ||sum_k e_k|| / sum_k ||e_k||, per sample. 1 = aligned, ~1/sqrt(K) = random."""
    num = es.sum(0).norm(dim=-1)
    den = es.norm(dim=-1).sum(0)
    return num / den.clamp_min(1e-12)


def growth_exponent(err_k):
    """Least-squares slope of log err_k vs log k for k = 1..K, per sample. err_k: (K, B)."""
    K = err_k.shape[0]
    lk = torch.log(torch.arange(1, K + 1, dtype=DTYPE))[:, None]
    le = torch.log(err_k.clamp_min(1e-12))
    lk_c, le_c = lk - lk.mean(0), le - le.mean(0)
    return (lk_c * le_c).sum(0) / (lk_c ** 2).sum(0)


@torch.no_grad()
def evaluate(teacher, students, c, K=10, n=256, x0=None):
    """Returns dict metric -> {student: mean}. Same x0 (noise seed) for every model."""
    x0 = x0 if x0 is not None else torch.randn(n, teacher.out.out_features, dtype=DTYPE)
    t_xs, t_vs = euler_states(teacher, x0, c, K)
    out = {}
    for name, s in students.items():
        es, cos = teacher_forced_errors(s, t_xs, t_vs, c, K)
        rel = (es.norm(dim=-1) / t_vs.norm(dim=-1).clamp_min(1e-12)).mean()
        s_xs, _ = euler_states(s, x0, c, K)                      # free-running
        err_k = (s_xs[1:] - t_xs[1:]).norm(dim=-1)              # (K, B)
        out[name] = {
            "local_rel_err": rel.item(),
            "cosine": cos.mean().item(),
            "coherence_C": coherence_index(es).mean().item(),
            "growth_exp": growth_exponent(err_k).mean().item(),
            "endpoint_err": err_k[-1].mean().item(),
        }
    return out


def run(spectrum_decay: float, K=10, n=256):
    torch.manual_seed(0)
    teacher = VelocityField().to(DTYPE)
    for lin in (teacher.gate, teacher.up, teacher.down):
        impose_spectrum_(lin, spectrum_decay)
    # Real flow matching moves x from noise to data, i.e. O(||x0||) over K steps. Scale the output
    # head so mean ||v|| ~ ||x0||; otherwise x barely moves and every error looks coherent.
    with torch.no_grad():
        x0p = torch.randn(n, 350, dtype=DTYPE); cp = torch.randn(n, 64, dtype=DTYPE)
        g = x0p.norm(dim=-1).mean() / teacher(x0p, torch.zeros(1, 1, dtype=DTYPE), cp).norm(dim=-1).mean()
        teacher.out.weight.mul_(g); teacher.out.bias.mul_(g)
    students = make_students(teacher, bits=4)

    c_id = torch.randn(n, 64, dtype=DTYPE)            # in-distribution conditioning
    c_ood = torch.randn(n, 64, dtype=DTYPE) * 1.5 + 2.0  # shifted conditioning (stand-in for LIBERO-Plus)
    x0 = torch.randn(n, 350, dtype=DTYPE)             # shared noise seed

    res_id = evaluate(teacher, students, c_id, K, n, x0)
    res_ood = evaluate(teacher, students, c_ood, K, n, x0)

    # OOD defined in the expert's input space: energy along the directions rank-r SVD dropped.
    E_ = teacher.gate.in_features
    with torch.no_grad():  # shift magnitude comparable to the MLP input norm
        h_norm = torch.tanh(teacher.inp(torch.cat([x0, c_id, torch.zeros(n, 1, dtype=DTYPE)], -1))).norm(dim=-1).mean()
    shift = dropped_direction_shift(teacher, matched_rank(E_, teacher.gate.out_features), n, scale=h_norm.item())
    for m in [teacher, *students.values()]:
        m.h_shift = shift
    res_sub = evaluate(teacher, students, c_id, K, n, x0)
    for m in [teacher, *students.values()]:
        m.h_shift = None

    print(f"\n===== teacher spectrum decay = {spectrum_decay} (0 = flat/random, 1 = steep) =====")
    for label, res in [("OOD = shifted conditioning", res_ood), ("OOD = shift along dropped directions", res_sub)]:
        print(f"--- {label} ---")
        hdr = f"{'metric':16s}" + "".join(f"{s+'_ID':>12s}{s+'_OOD':>12s}{s+'_ratio':>12s}" for s in students)
        print(hdr)
        for m in res_id["svd"]:
            row = f"{m:16s}"
            for s in students:
                a, b = res_id[s][m], res[s][m]
                row += f"{a:12.4f}{b:12.4f}{b/max(a,1e-12):12.3f}"
            print(row)
        print()


if __name__ == "__main__":
    import sys
    if "--round" in sys.argv:
        ROUND_STORAGE = True
    print(f"storage rounding to bf16: {ROUND_STORAGE}")
    print(f"K=10. Reference: C_random ~ 1/sqrt(K) = {1/math.sqrt(10):.3f}; growth exp 1.0 = coherent, 0.5 = random walk")
    for d in (0.0, 0.5, 1.0):
        run(spectrum_decay=d)
