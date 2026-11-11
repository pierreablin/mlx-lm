# Copyright © 2026 Apple Inc.

"""A quantized SwitchGLU for one token, with its experts' outputs weighted and
summed, in two kernels: affine 4- or 8-bit experts, or MXFP4 experts with biases
(gpt-oss)."""

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

// The GLU of gate g and up u: SiLU (ACT 0), tanh GELU (1), or gpt-oss's SwigluOAI
// with alpha 1.702 and limit 7 (2).
template <int ACT>
inline float glu(float g, float u) {
    if (ACT == 1) {
        return 0.5f * g * (1.0f + metal::precise::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g))) * u;
    }
    if (ACT == 2) {
        g = metal::min(g, 7.0f);
        u = metal::clamp(u, -7.0f, 7.0f);
        return (u + 1.0f) * (g / (1.0f + metal::exp(-1.702f * g)));
    }
    return g / (1.0f + metal::exp(-g)) * u;
}

// MXFP4: 16 e2m1 values (low nibbles first) dotted with xs, and a group's e8m0
// scale, decoded as mlx's fp4_e2m1 and fp8_e8m0.
inline float fp4dot16(const device uint8_t* w, thread const float* xs) {
    auto ws = (const device uint32_t*)w;
    float a = 0.0f;
    for (int i = 0; i < 2; i++) {
        uint q = ws[i];
        for (int j = 0; j < 8; j++) {
            uint n = q >> (4 * j);
            half v = as_type<half>(ushort((n & 7) << 9)) * 16384.0h;
            a += xs[8 * i + j] * ((n & 8) ? -float(v) : float(v));
        }
    }
    return a;
}

inline float e8m0(uint8_t s) {
    return as_type<float>(s == 0 ? 0x400000u : uint(s) << 23);
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
"""

# The gate and up rows of the K experts (K * N rows of D values), the GLU, times
# each expert's weight. A threadgroup takes RPS * SPT rows of one expert: RPS rows
# per simdgroup, 16 values per lane.
_GATE_UP_SOURCE = """
    constexpr int ROWS = RPS * SPT;
    constexpr int TPE = (N + ROWS - 1) / ROWS;
    constexpr int RB = D * BITS / 8;
    constexpr int NG = D / GS;
    uint lane = thread_index_in_simdgroup;
    uint slot = threadgroup_position_in_grid.x / TPE;
    int row0 = (threadgroup_position_in_grid.x % TPE) * ROWS + simdgroup_index_in_threadgroup * RPS;
    if (row0 >= N) {
        return;
    }
    size_t base = size_t(inds[slot]) * N + row0;
    auto gq = (const device uint8_t*)gw;
    auto uq = (const device uint8_t*)uw;
    float ga[RPS] = {0.0f};
    float ua[RPS] = {0.0f};
    float xs[16];
    for (int k0 = lane * 16; k0 < D; k0 += 512) {
        float xsum = load16<BITS>(x + k0, xs);
        for (int r = 0; r < RPS; r++) {
            size_t row = base + r;
            size_t g = row * NG + k0 / GS;
            ga[r] += float(gs[g]) * qdot16<BITS>(gq + row * RB + k0 * BITS / 8, xs) + float(gb[g]) * xsum;
            ua[r] += float(us[g]) * qdot16<BITS>(uq + row * RB + k0 * BITS / 8, xs) + float(ub[g]) * xsum;
        }
    }
    float we = weights[slot];
    for (int r = 0; r < RPS; r++) {
        float g = simd_sum(ga[r]);
        float u = simd_sum(ua[r]);
        if (lane == 0) {
            h[slot * N + row0 + r] = static_cast<T>(glu<ACT>(g, u) * we);
        }
    }
"""

# The down rows of the K experts side by side are one row of K * N values, and h
# (K x N) the matching vector: the weighted sum over the experts is one dot
# product per output row. A lane's 16 values are of one expert (N % 16 == 0).
_DOWN_SOURCE = """
    constexpr int RB = N * BITS / 8;
    constexpr int NG = N / GS;
    uint lane = thread_index_in_simdgroup;
    int row0 = (threadgroup_position_in_grid.x * SPT + simdgroup_index_in_threadgroup) * RPS;
    if (row0 >= D) {
        return;
    }
    auto dq = (const device uint8_t*)dw;
    float acc[RPS] = {0.0f};
    float xs[16];
    for (int j0 = lane * 16; j0 < K * N; j0 += 512) {
        int slot = j0 / N;
        int j = j0 - slot * N;
        float xsum = load16<BITS>(h + j0, xs);
        size_t base = size_t(inds[slot]) * D + row0;
        for (int r = 0; r < RPS; r++) {
            size_t row = base + r;
            size_t g = row * NG + j / GS;
            acc[r] += float(ds[g]) * qdot16<BITS>(dq + row * RB + j * BITS / 8, xs) + float(db[g]) * xsum;
        }
    }
    for (int r = 0; r < RPS; r++) {
        float a = simd_sum(acc[r]);
        if (lane == 0) {
            y[row0 + r] = static_cast<O>(a);
        }
    }
"""

_gate_up_kernel = mx.fast.metal_kernel(
    name="switch_glu_gate_up",
    input_names=["x", "inds", "weights", "gw", "gs", "gb", "uw", "us", "ub"],
    output_names=["h"],
    header=_HEADER,
    source=_GATE_UP_SOURCE,
)

_down_kernel = mx.fast.metal_kernel(
    name="switch_glu_down_combine",
    input_names=["h", "inds", "dw", "ds", "db"],
    output_names=["y"],
    header=_HEADER,
    source=_DOWN_SOURCE,
)

# MXFP4 experts with biases (gpt-oss): the same two kernels, with an e8m0 scale per
# 32 values and the experts' biases; the down biases are weighted in the second.
_GATE_UP_MXFP4_SOURCE = """
    constexpr int ROWS = RPS * SPT;
    constexpr int TPE = (N + ROWS - 1) / ROWS;
    constexpr int RB = D / 2;
    constexpr int NG = D / 32;
    uint lane = thread_index_in_simdgroup;
    uint slot = threadgroup_position_in_grid.x / TPE;
    int row0 = (threadgroup_position_in_grid.x % TPE) * ROWS + simdgroup_index_in_threadgroup * RPS;
    if (row0 >= N) {
        return;
    }
    size_t base = size_t(inds[slot]) * N + row0;
    auto gq = (const device uint8_t*)gw;
    auto uq = (const device uint8_t*)uw;
    float ga[RPS] = {0.0f};
    float ua[RPS] = {0.0f};
    float xs[16];
    for (int k0 = lane * 16; k0 < D; k0 += 512) {
        for (int i = 0; i < 16; i++) {
            xs[i] = x[k0 + i];
        }
        for (int r = 0; r < RPS; r++) {
            size_t row = base + r;
            size_t g = row * NG + k0 / 32;
            ga[r] += e8m0(gs[g]) * fp4dot16(gq + row * RB + k0 / 2, xs);
            ua[r] += e8m0(us[g]) * fp4dot16(uq + row * RB + k0 / 2, xs);
        }
    }
    float we = weights[slot];
    for (int r = 0; r < RPS; r++) {
        float g = simd_sum(ga[r]) + float(gbias[base + r]);
        float u = simd_sum(ua[r]) + float(ubias[base + r]);
        if (lane == 0) {
            h[slot * N + row0 + r] = static_cast<T>(glu<ACT>(g, u) * we);
        }
    }
"""

_DOWN_MXFP4_SOURCE = """
    constexpr int RB = N / 2;
    constexpr int NG = N / 32;
    uint lane = thread_index_in_simdgroup;
    int row0 = (threadgroup_position_in_grid.x * SPT + simdgroup_index_in_threadgroup) * RPS;
    if (row0 >= D) {
        return;
    }
    auto dq = (const device uint8_t*)dw;
    float acc[RPS] = {0.0f};
    float xs[16];
    for (int j0 = lane * 16; j0 < K * N; j0 += 512) {
        int slot = j0 / N;
        int j = j0 - slot * N;
        for (int i = 0; i < 16; i++) {
            xs[i] = h[j0 + i];
        }
        size_t base = size_t(inds[slot]) * D + row0;
        for (int r = 0; r < RPS; r++) {
            size_t row = base + r;
            acc[r] += e8m0(ds[row * NG + j / 32]) * fp4dot16(dq + row * RB + j / 2, xs);
        }
    }
    for (int r = 0; r < RPS; r++) {
        float a = simd_sum(acc[r]);
        if (lane == 0) {
            for (int k = 0; k < K; k++) {
                a += float(weights[k]) * float(dbias[size_t(inds[k]) * D + row0 + r]);
            }
            y[row0 + r] = static_cast<O>(a);
        }
    }
"""

_gate_up_mxfp4_kernel = mx.fast.metal_kernel(
    name="switch_glu_gate_up_mxfp4",
    input_names=["x", "inds", "weights", "gw", "gs", "gbias", "uw", "us", "ubias"],
    output_names=["h"],
    header=_HEADER,
    source=_GATE_UP_MXFP4_SOURCE,
)

_down_mxfp4_kernel = mx.fast.metal_kernel(
    name="switch_glu_down_combine_mxfp4",
    input_names=["h", "inds", "weights", "dw", "ds", "dbias"],
    output_names=["y"],
    header=_HEADER,
    source=_DOWN_MXFP4_SOURCE,
)

# Rows per simdgroup and simdgroups per threadgroup, the fastest of those tried
# with Qwen3-30B-A3B's experts (D = 2048, N = 768).
_GATE_UP_SHAPE = (2, 2)
_DOWN_SHAPE = (1, 4)


def can_combine(switch_glu, x: mx.array) -> bool:
    """Whether ``switch_glu_combine`` takes ``x``: one token, bf16 or fp16, on Metal,
    not training, and a SwitchGLU whose activation has a ``metal_act`` (SwiGLU, GeGLU,
    SwigluOAI) and whose experts are affine 4- or 8-bit (group size 32, 64 or 128,
    scales in the dtype of ``x``, no bias) or MXFP4 with biases (gpt-oss)."""
    projs = (switch_glu.gate_proj, switch_glu.up_proj, switch_glu.down_proj)
    if not (
        mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and x.size == x.shape[-1]
        and x.dtype in (mx.bfloat16, mx.float16)
        and not switch_glu.training
        and getattr(switch_glu.activation, "metal_act", None) in (0, 1, 2)
    ):
        return False
    if getattr(projs[0], "mode", None) == "mxfp4":
        return (
            all(
                getattr(p, "mode", None) == "mxfp4"
                and p.weight.ndim == 3
                and p.scales.dtype == mx.uint8
                and "bias" in p
                for p in projs
            )
            and x.shape[-1] % 32 == 0
            and switch_glu.gate_proj.weight.shape[1] % 32 == 0
        )
    bits, group_size = getattr(projs[0], "bits", None), getattr(
        projs[0], "group_size", None
    )
    return (
        bits in (4, 8)
        and group_size in (32, 64, 128)
        and all(
            getattr(p, "mode", None) == "affine"
            and getattr(p, "bits", None) == bits
            and getattr(p, "group_size", None) == group_size
            and p.weight.ndim == 3
            and p.scales.dtype == x.dtype
            and p.get("biases") is not None
            and "bias" not in p
            and getattr(p, "global_scale", None) is None
            for p in projs
        )
        and x.shape[-1] % 16 == 0
        and switch_glu.gate_proj.weight.shape[1] % 16 == 0
    )


@lru_cache
def _combine_fn(act: int, mode: str, group_size: int, bits: int):
    """The two kernels for one configuration, compiled: shapes come from the inputs, so a
    model's layers share one compiled function, and a call costs less on the host. gb,
    ub and db are the scales' biases (affine) or the experts' biases (MXFP4)."""
    mxfp4 = mode == "mxfp4"

    def combine(x, inds, weights, gw, gs, gb, uw, us, ub, dw, ds, db):
        K, N, D = inds.size, gw.shape[1], x.shape[-1]
        inds = inds.reshape(-1).astype(mx.uint32)
        weights = weights.reshape(-1)
        out_dtype = mx.result_type(x.dtype, weights.dtype)
        template = [
            ("T", x.dtype),
            ("O", out_dtype),
            ("ACT", act),
            ("D", D),
            ("N", N),
            ("K", K),
            ("GS", group_size),
            ("BITS", bits),
        ]
        rps, spt = _GATE_UP_SHAPE
        tpe = -(-N // (rps * spt))
        h = (_gate_up_mxfp4_kernel if mxfp4 else _gate_up_kernel)(
            inputs=[x, inds, weights, gw, gs, gb, uw, us, ub],
            template=template + [("RPS", rps), ("SPT", spt)],
            grid=(K * tpe * 32 * spt, 1, 1),
            threadgroup=(32 * spt, 1, 1),
            output_shapes=[(K, N)],
            output_dtypes=[x.dtype],
        )[0]
        rps, spt = _DOWN_SHAPE
        return (_down_mxfp4_kernel if mxfp4 else _down_kernel)(
            inputs=[h, inds, weights, dw, ds, db] if mxfp4 else [h, inds, dw, ds, db],
            template=template + [("RPS", rps), ("SPT", spt)],
            grid=(-(-D // (rps * spt)) * 32 * spt, 1, 1),
            threadgroup=(32 * spt, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[out_dtype],
        )[0]

    return mx.compile(combine)


def switch_glu_combine(
    switch_glu, x: mx.array, inds: mx.array, weights: mx.array
) -> mx.array:
    """The sum of ``switch_glu``'s experts ``inds`` on the one token ``x``, weighted by
    ``weights``: ``(switch_glu(x, inds) * weights[..., None]).sum(-2)`` and of its
    dtype, in two kernels. Needs ``can_combine(switch_glu, x)``."""
    gp, up, dp = switch_glu.gate_proj, switch_glu.up_proj, switch_glu.down_proj
    mode = gp.mode
    fn = _combine_fn(switch_glu.activation.metal_act, mode, gp.group_size, gp.bits)
    biases = "bias" if mode == "mxfp4" else "biases"
    return fn(
        x,
        inds,
        weights,
        gp.weight,
        gp.scales,
        gp[biases],
        up.weight,
        up.scales,
        up[biases],
        dp.weight,
        dp.scales,
        dp[biases],
    )
