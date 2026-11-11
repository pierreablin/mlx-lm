# Copyright © 2026 Apple Inc.

"""A quantized SwitchGLU for one token, with its experts' outputs weighted and
summed, and optionally a gated shared expert added, in two kernels."""

from functools import lru_cache

import mlx.core as mx

# x[16] as floats and their sum; for 4 bits, x[i] is prescaled by 1 / 16^(i % 4) so
# that the masked nibbles of a uint16 need no shift (as mlx's qmv).
_HEADER = """
template <int BITS, typename U>
inline float load16(const device U* x, thread float* xs) {
    float s = 0.0f;
    for (int i = 0; i < 16; i += 4) {
        float a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
        s += a + b + c + d;
        if (BITS == 4) {
            xs[i] = a; xs[i + 1] = b / 16.0f; xs[i + 2] = c / 256.0f; xs[i + 3] = d / 4096.0f;
        } else {
            xs[i] = a; xs[i + 1] = b; xs[i + 2] = c; xs[i + 3] = d;
        }
    }
    return s;
}

template <int ACT>
inline float glu_act(float g) {
    if (ACT == 1) {
        return 0.5f * g * (1.0f + metal::precise::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g)));
    }
    return g / (1.0f + metal::exp(-g));
}

template <int BITS>
inline float qdot16(const device uint8_t* w, thread const float* xs) {
    float a = 0.0f;
    if (BITS == 4) {
        auto ws = (const device uint16_t*)w;
        for (int i = 0; i < 4; i++) {
            uint16_t q = ws[i];
            a += xs[4 * i] * float(q & 0x000f) + xs[4 * i + 1] * float(q & 0x00f0) +
                xs[4 * i + 2] * float(q & 0x0f00) + xs[4 * i + 3] * float(q & 0xf000);
        }
    } else {
        auto ws = (const device uchar4*)w;
        for (int i = 0; i < 4; i++) {
            uchar4 q = ws[i];
            a += xs[4 * i] * q.x + xs[4 * i + 1] * q.y + xs[4 * i + 2] * q.z + xs[4 * i + 3] * q.w;
        }
    }
    return a;
}

// RPS rows from row on of a gate and an up projection (D values each) on x: each lane
// adds its part of the dot products to ga and ua.
template <int BITS, int GS, int D, int RPS, typename T, typename S>
inline void gate_up_rows(
    const device T* x,
    const device uint8_t* gq,
    const device S* gs,
    const device S* gb,
    const device uint8_t* uq,
    const device S* us,
    const device S* ub,
    size_t row,
    uint lane,
    thread float* ga,
    thread float* ua) {
    constexpr int RB = D * BITS / 8;
    constexpr int NG = D / GS;
    float xs[16];
    for (int k0 = lane * 16; k0 < D; k0 += 512) {
        float xsum = load16<BITS>(x + k0, xs);
        for (int r = 0; r < RPS; r++) {
            size_t g = (row + r) * NG + k0 / GS;
            size_t w = (row + r) * RB + k0 * BITS / 8;
            ga[r] += float(gs[g]) * qdot16<BITS>(gq + w, xs) + float(gb[g]) * xsum;
            ua[r] += float(us[g]) * qdot16<BITS>(uq + w, xs) + float(ub[g]) * xsum;
        }
    }
}

// RPS rows from row on of a down projection (NN values each), at the 16 columns from
// j on: adds their dot products with xs (from load16, which returned xsum) to acc.
template <int BITS, int GS, int NN, int RPS, typename S>
inline void down_rows(
    const device uint8_t* dq,
    const device S* ds,
    const device S* db,
    size_t row,
    int j,
    thread const float* xs,
    float xsum,
    thread float* acc) {
    constexpr int RB = NN * BITS / 8;
    constexpr int NG = NN / GS;
    for (int r = 0; r < RPS; r++) {
        size_t g = (row + r) * NG + j / GS;
        acc[r] += float(ds[g]) * qdot16<BITS>(dq + (row + r) * RB + j * BITS / 8, xs) + float(db[g]) * xsum;
    }
}
"""

# The gate and up rows of the K experts (K * N rows of D values), the GLU, times
# each expert's weight. A threadgroup takes RPS * SPT rows of one expert: RPS rows
# per simdgroup, 16 values per lane.
_GATE_UP_SOURCE = """
    constexpr int ROWS = RPS * SPT;
    constexpr int TPE = (N + ROWS - 1) / ROWS;
    uint lane = thread_index_in_simdgroup;
    uint slot = threadgroup_position_in_grid.x / TPE;
    int row0 = (threadgroup_position_in_grid.x % TPE) * ROWS + simdgroup_index_in_threadgroup * RPS;
    if (row0 >= N) {
        return;
    }
    float ga[RPS] = {0.0f};
    float ua[RPS] = {0.0f};
    gate_up_rows<BITS, GS, D, RPS>(
        x, (const device uint8_t*)gw, gs, gb, (const device uint8_t*)uw, us, ub,
        size_t(inds[slot]) * N + row0, lane, ga, ua);
    float we = weights[slot];
    for (int r = 0; r < RPS; r++) {
        float g = simd_sum(ga[r]);
        float u = simd_sum(ua[r]);
        if (lane == 0) {
            h[slot * N + row0 + r] = static_cast<T>(glu_act<ACT>(g) * u * we);
        }
    }
"""

# The same, then the SN rows of a shared expert weighted by the sigmoid of its gate's
# logit sgate, after the experts' rows in h: threadgroups K * TPE and up.
_GATE_UP_SHARED_SOURCE = """
    constexpr int ROWS = RPS * SPT;
    constexpr int TPE = (N + ROWS - 1) / ROWS;
    uint lane = thread_index_in_simdgroup;
    uint tg = threadgroup_position_in_grid.x;
    bool shared = tg >= K * TPE;
    uint slot = shared ? K : tg / TPE;
    int row0 = (shared ? tg - K * TPE : tg % TPE) * ROWS + simdgroup_index_in_threadgroup * RPS;
    if (row0 >= (shared ? SN : N)) {
        return;
    }
    float ga[RPS] = {0.0f};
    float ua[RPS] = {0.0f};
    float we;
    if (shared) {
        gate_up_rows<BITS, GS, D, RPS>(
            x, (const device uint8_t*)sgw, sgs, sgb, (const device uint8_t*)suw, sus, sub,
            row0, lane, ga, ua);
        we = 1.0f / (1.0f + metal::exp(-float(sgate[0])));
    } else {
        gate_up_rows<BITS, GS, D, RPS>(
            x, (const device uint8_t*)gw, gs, gb, (const device uint8_t*)uw, us, ub,
            size_t(inds[slot]) * N + row0, lane, ga, ua);
        we = weights[slot];
    }
    for (int r = 0; r < RPS; r++) {
        float g = simd_sum(ga[r]);
        float u = simd_sum(ua[r]);
        if (lane == 0) {
            h[slot * N + row0 + r] = static_cast<T>(glu_act<ACT>(g) * u * we);
        }
    }
"""

# The down rows of the K experts side by side are one row of K * N values, and h
# (K x N) the matching vector: the weighted sum over the experts is one dot
# product per output row. A lane's 16 values are of one expert (N % 16 == 0).
_DOWN_SOURCE = """
    uint lane = thread_index_in_simdgroup;
    int row0 = (threadgroup_position_in_grid.x * SPT + simdgroup_index_in_threadgroup) * RPS;
    if (row0 >= D) {
        return;
    }
    float acc[RPS] = {0.0f};
    float xs[16];
    for (int j0 = lane * 16; j0 < K * N; j0 += 512) {
        int slot = j0 / N;
        float xsum = load16<BITS>(h + j0, xs);
        down_rows<BITS, GS, N, RPS>(
            (const device uint8_t*)dw, ds, db, size_t(inds[slot]) * D + row0,
            j0 - slot * N, xs, xsum, acc);
    }
    for (int r = 0; r < RPS; r++) {
        float a = simd_sum(acc[r]);
        if (lane == 0) {
            y[row0 + r] = static_cast<O>(a);
        }
    }
"""

# The same, with the shared expert's down rows (SN values) after the experts'.
_DOWN_SHARED_SOURCE = """
    uint lane = thread_index_in_simdgroup;
    int row0 = (threadgroup_position_in_grid.x * SPT + simdgroup_index_in_threadgroup) * RPS;
    if (row0 >= D) {
        return;
    }
    float acc[RPS] = {0.0f};
    float xs[16];
    for (int j0 = lane * 16; j0 < K * N + SN; j0 += 512) {
        float xsum = load16<BITS>(h + j0, xs);
        if (j0 < K * N) {
            int slot = j0 / N;
            down_rows<BITS, GS, N, RPS>(
                (const device uint8_t*)dw, ds, db, size_t(inds[slot]) * D + row0,
                j0 - slot * N, xs, xsum, acc);
        } else {
            down_rows<BITS, GS, SN, RPS>(
                (const device uint8_t*)sdw, sds, sdb, row0, j0 - K * N, xs, xsum, acc);
        }
    }
    for (int r = 0; r < RPS; r++) {
        float a = simd_sum(acc[r]);
        if (lane == 0) {
            y[row0 + r] = static_cast<O>(a);
        }
    }
"""

_GATE_UP_INPUTS = ["x", "inds", "weights", "gw", "gs", "gb", "uw", "us", "ub"]
_gate_up_kernel = mx.fast.metal_kernel(
    name="switch_glu_gate_up",
    input_names=_GATE_UP_INPUTS,
    output_names=["h"],
    header=_HEADER,
    source=_GATE_UP_SOURCE,
)

_gate_up_shared_kernel = mx.fast.metal_kernel(
    name="switch_glu_gate_up_shared",
    input_names=_GATE_UP_INPUTS + ["sgw", "sgs", "sgb", "suw", "sus", "sub", "sgate"],
    output_names=["h"],
    header=_HEADER,
    source=_GATE_UP_SHARED_SOURCE,
)

_DOWN_INPUTS = ["h", "inds", "dw", "ds", "db"]
_down_kernel = mx.fast.metal_kernel(
    name="switch_glu_down_combine",
    input_names=_DOWN_INPUTS,
    output_names=["y"],
    header=_HEADER,
    source=_DOWN_SOURCE,
)

_down_shared_kernel = mx.fast.metal_kernel(
    name="switch_glu_down_combine_shared",
    input_names=_DOWN_INPUTS + ["sdw", "sds", "sdb"],
    output_names=["y"],
    header=_HEADER,
    source=_DOWN_SHARED_SOURCE,
)

# Rows per simdgroup and simdgroups per threadgroup, the fastest of those tried
# with Qwen3-30B-A3B's experts (D = 2048, N = 768).
_GATE_UP_SHAPE = (2, 2)
_DOWN_SHAPE = (1, 4)


def _quantized(projs, bits: int, group_size: int, dtype, ndim: int) -> bool:
    """Whether the projections are affine-quantized with ``bits`` and ``group_size``,
    with ``ndim``-dimensional weights, scales and biases in ``dtype``, and no bias."""
    return all(
        getattr(p, "mode", None) == "affine"
        and getattr(p, "bits", None) == bits
        and getattr(p, "group_size", None) == group_size
        and p.weight.ndim == ndim
        and p.scales.dtype == dtype
        and p.get("biases") is not None
        and "bias" not in p
        and getattr(p, "global_scale", None) is None
        for p in projs
    )


def can_combine(switch_glu, x: mx.array, shared=None) -> bool:
    """Whether ``switch_glu_combine`` takes ``x``: one token, bf16 or fp16, on Metal,
    not training, and a SwitchGLU whose activation has a ``metal_act`` (SwiGLU, GeGLU)
    and whose experts are affine 4- or 8-bit (group size 32, 64 or 128, scales in the
    dtype of ``x``, no bias). A ``shared`` expert must be quantized as the experts."""
    projs = (switch_glu.gate_proj, switch_glu.up_proj, switch_glu.down_proj)
    if not (
        mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and x.size == x.shape[-1]
        and x.dtype in (mx.bfloat16, mx.float16)
        and not switch_glu.training
        and getattr(switch_glu.activation, "metal_act", None) in (0, 1)
    ):
        return False
    bits, group_size = getattr(projs[0], "bits", None), getattr(
        projs[0], "group_size", None
    )
    if not (
        bits in (4, 8)
        and group_size in (32, 64, 128)
        and _quantized(projs, bits, group_size, x.dtype, 3)
        and x.shape[-1] % 16 == 0
        and switch_glu.gate_proj.weight.shape[1] % 16 == 0
    ):
        return False
    if shared is None:
        return True
    sprojs = (shared.gate_proj, shared.up_proj, shared.down_proj)
    return (
        _quantized(sprojs, bits, group_size, x.dtype, 2)
        and shared.gate_proj.weight.shape[0] % 16 == 0
    )


@lru_cache
def _combine_fn(act: int, group_size: int, bits: int):
    """The two kernels for one configuration, compiled: shapes come from the inputs, so a
    model's layers share one compiled function, and a call costs less on the host."""

    def combine(x, inds, weights, gw, gs, gb, uw, us, ub, dw, ds, db, *shared):
        # shared: the shared expert's gate, up and down weights, scales and biases,
        # then its gate's logit.
        K, N, D = inds.size, gw.shape[1], x.shape[-1]
        SN = shared[0].shape[0] if shared else 0
        inds = inds.reshape(-1).astype(mx.uint32)
        weights = weights.reshape(-1)
        out_dtype = mx.result_type(x.dtype, weights.dtype, *shared[-1:])
        template = [
            ("T", x.dtype),
            ("O", out_dtype),
            ("ACT", act),
            ("D", D),
            ("N", N),
            ("K", K),
            ("SN", SN),
            ("GS", group_size),
            ("BITS", bits),
        ]
        rps, spt = _GATE_UP_SHAPE
        threadgroups = K * -(-N // (rps * spt)) + -(-SN // (rps * spt))
        h = (_gate_up_shared_kernel if shared else _gate_up_kernel)(
            inputs=[x, inds, weights, gw, gs, gb, uw, us, ub, *shared[:6], *shared[9:]],
            template=template + [("RPS", rps), ("SPT", spt)],
            grid=(threadgroups * 32 * spt, 1, 1),
            threadgroup=(32 * spt, 1, 1),
            output_shapes=[(K * N + SN,)],
            output_dtypes=[x.dtype],
        )[0]
        rps, spt = _DOWN_SHAPE
        return (_down_shared_kernel if shared else _down_kernel)(
            inputs=[h, inds, dw, ds, db, *shared[6:9]],
            template=template + [("RPS", rps), ("SPT", spt)],
            grid=(-(-D // (rps * spt)) * 32 * spt, 1, 1),
            threadgroup=(32 * spt, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[out_dtype],
        )[0]

    return mx.compile(combine)


def switch_glu_combine(
    switch_glu,
    x: mx.array,
    inds: mx.array,
    weights: mx.array,
    shared=None,
    shared_gate=None,
) -> mx.array:
    """The sum of ``switch_glu``'s experts ``inds`` on the one token ``x``, weighted by
    ``weights``: ``(switch_glu(x, inds) * weights[..., None]).sum(-2)`` and of its
    dtype, plus ``mx.sigmoid(shared_gate(x)) * shared(x)`` with a ``shared`` expert (a
    GLU MLP with the activation of ``switch_glu``), in two kernels. Needs
    ``can_combine(switch_glu, x, shared)``."""
    gp, up, dp = switch_glu.gate_proj, switch_glu.up_proj, switch_glu.down_proj
    extra = []
    if shared is not None:
        for p in (shared.gate_proj, shared.up_proj, shared.down_proj):
            extra += [p.weight, p.scales, p.biases]
        extra.append(shared_gate(x))
    fn = _combine_fn(switch_glu.activation.metal_act, gp.group_size, gp.bits)
    return fn(
        x,
        inds,
        weights,
        gp.weight,
        gp.scales,
        gp.biases,
        up.weight,
        up.scales,
        up.biases,
        dp.weight,
        dp.scales,
        dp.biases,
        *extra,
    )
