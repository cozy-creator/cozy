#!/usr/bin/env python3
"""Writes svdq.nvfp4's cases into vectors/cases/svdq.json: the parts in natural layout, as TensorFS's Nunchaku import
and the Runtime's SVDQuant producer write them (E2M1 codes, E4M3 scales on row blocks of 16, `scale_2`, the
smoothing's reciprocal `pre_quant_scale`, Nunchaku's per-row `wcscales` as `channel_scale`, and the 16-bit low-rank
factors), and refusals. Expected bytes are the spec's f32 decode, here in numpy; each is checked against the same
formula in f64 (within a few steps of L).

    uv run --no-project --python 3.12 --with numpy python -I conformance/mine/svdq.py
"""

import base64
import json
from pathlib import Path

import numpy as np

CASES = Path(__file__).resolve().parents[1] / "vectors" / "cases" / "svdq.json"
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
PARAMS = {"block_size": 16, "global_scale": "multiply", "nibble_order": "low_first", "scale_layout": "row_major"}
N, K, R = 32, 64, 4


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def e4m3(b):
    b = np.asarray(b, np.uint8)
    e, m = (b >> 3) & 15, (b & 7).astype(np.float32)
    v = np.where(e == 0, m * 2.0 ** -9, (1 + m / 8) * np.exp2(e.astype(np.float32) - 7))
    return np.where(b & 0x80, -v, v).astype(np.float32)


def bf16(x) -> np.ndarray:
    """The BF16 bits of f32 values, nearest even."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def stored(dtype, x) -> tuple:
    """(bytes, the f32 values those bytes hold)."""
    x = np.asarray(x, np.float32)
    if dtype == "BF16":
        bits = bf16(x)
        return bits.astype("<u2").tobytes(), (bits.astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        h = x.astype(np.float16)
        return h.astype("<f2").tobytes(), h.astype(np.float32)
    return x.astype("<f4").tobytes(), x


def round_l(dtype, x) -> bytes:
    return stored(dtype, x)[0]


def part(role, dtype, shape, bits):
    return {"bits": b64(bits), "dtype": dtype, "role": role, "shape": list(shape)}


def case(key, dtype, shape, params, parts, expect=None):
    c = {"component": "svdq" if expect is not None else "refuse", "dtype": dtype,
         "encoding": {"params": params, "scheme": "svdq.nvfp4"}, "key": key, "parts": parts, "shape": list(shape)}
    c.update({"expect": b64(expect)} if expect is not None else {"refuse": "decode"})
    return c


def tile(s, n, w):
    """Row-major block scales [n, w] in cuBLAS's 128x4 tiles (spec §6)."""
    out = np.zeros((-(-n // 128) * 128) * (-(-w // 4) * 4), np.uint8)
    for i in range(n):
        for b in range(w):
            out[512 * ((i // 128) * -(-w // 4) + b // 4) + 16 * (i % 32) + 4 * (i % 128 // 32) + b % 4] = s[i, b]
    return out, [-(-n // 128) * 128, -(-w // 4) * 4]


def weight(seed, n=N, k=K, r=R, *, divide=False, high_first=False, tiled=False, channel=None, pdtype="F32",
           lrdtype="BF16", input_scale=False):
    """One weight's parts and its f32 decode (spec §6 `svdq.nvfp4`), and the same in f64."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, (n, k)).astype(np.uint8)
    s = rng.integers(0x28, 0x50, (n, k // 16)).astype(np.uint8)  # E4M3 2^-2 … 2^2
    g = np.float32(rng.uniform(0.002, 0.02))
    p_bytes, p = stored(pdtype, 1 / rng.uniform(0.25, 4, k))
    down_bytes, down = stored(lrdtype, rng.normal(0, 0.05, (r, k)))
    up_bytes, up = stored(lrdtype, rng.normal(0, 0.05, (n, r)))
    packed = (codes[:, 1::2] << 4 | codes[:, 0::2]) if not high_first else (codes[:, 0::2] << 4 | codes[:, 1::2])
    scale = tile(s, n, k // 16) if tiled else (s.reshape(-1), [n, k // 16])
    parts = [part("weight", "F4", [n, k], packed.astype(np.uint8).tobytes()),
             part("scale", "F8_E4M3", scale[1], scale[0].tobytes()),
             part("scale_2", "F32", [], np.float32(1 / g if divide else g).astype("<f4").tobytes())]
    if input_scale:
        parts.append(part("input_scale", "F32", [], np.float32(0.0125).astype("<f4").tobytes()))
    parts.append(part("pre_quant_scale", pdtype, [k], p_bytes))
    c = None
    if channel == "F32":
        c = rng.uniform(0.5, 2, n).astype(np.float32)
        parts.append(part("channel_scale", "F32", [n], c.astype("<f4").tobytes()))
    elif channel == "F8_E4M3":
        raw = rng.integers(0x30, 0x40, n).astype(np.uint8)
        c = e4m3(raw)
        parts.append(part("channel_scale", "F8_E4M3", [n], raw.tobytes()))
    parts += [part("lowrank_down", lrdtype, [r, k], down_bytes), part("lowrank_up", lrdtype, [n, r], up_bytes)]
    q, sv = E2M1[codes].reshape(n, k // 16, 16), e4m3(s).reshape(n, k // 16, 1)
    gs = np.float32(1 / g) if divide else g
    v = (q * (sv / gs) if divide else (q * sv) * gs).reshape(n, k)
    if c is not None:
        v = v * c[:, None]
    v = v * p[None, :]
    for t in range(r):
        v = v + up[:, t:t + 1] * down[t:t + 1, :]
    exact = (E2M1[codes].astype(np.float64) * np.repeat(e4m3(s), 16, 1) * float(g)
             * (c[:, None] if c is not None else 1) * p[None, :] + up.astype(np.float64) @ down)
    worst = float(np.max(np.abs(v - exact)) / np.max(np.abs(exact)))
    assert worst < 1e-5, worst
    return parts, v


def main():
    greens, reds = [], []
    variants = [
        ("multiply-bf16", "BF16", {}, {}),
        ("multiply-f16", "F16", {}, {"channel": "F32"}),
        ("multiply-f32", "F32", {}, {"channel": "F32", "pdtype": "BF16", "lrdtype": "F32"}),
        ("divide-bf16", "BF16", {"global_scale": "divide"}, {"divide": True, "pdtype": "F16", "lrdtype": "F16"}),
        ("comfy-layout-bf16", "BF16", {"nibble_order": "high_first", "scale_layout": "cublas_128x4"},
         {"high_first": True, "tiled": True, "channel": "F8_E4M3", "input_scale": True}),
        ("rank-1-bf16", "BF16", {}, {"r": 1, "n": 16, "k": 32}),
    ]
    for seed, (key, dtype, params, how) in enumerate(variants):
        parts, v = weight(seed, **how)
        shape = [how.get("n", N), how.get("k", K)]
        greens.append(case(key, dtype, shape, dict(PARAMS, **params), parts, round_l(dtype, v)))

    base, _ = weight(99)
    roles = {p["role"]: p for p in base}

    def red(key, parts, shape=(N, K), params=PARAMS):
        reds.append(case(key, "BF16", shape, params, parts))

    def without(*gone):
        return [p for p in base if p["role"] not in gone]

    def swap(role, new):
        return [new if p["role"] == role else p for p in base]

    red("missing-lowrank-up", without("lowrank_up"))
    red("missing-lowrank-down", without("lowrank_down"))
    red("missing-pre-quant-scale", without("pre_quant_scale"))
    red("rank-mismatch", swap("lowrank_up", part("lowrank_up", "BF16", [N, R + 1], bytes(2 * N * (R + 1)))))
    red("rank-zero", swap("lowrank_down", part("lowrank_down", "BF16", [0, K], b""))[:-1]
        + [part("lowrank_up", "BF16", [N, 0], b"")])
    down = base64.b64decode(roles["lowrank_down"]["bits"])
    red("lowrank-down-transposed", swap("lowrank_down", part("lowrank_down", "BF16", [K, R], down)))
    red("lowrank-int8", swap("lowrank_up", part("lowrank_up", "I8", [N, R], bytes(N * R))))
    red("channel-scale-columns", base[:-2] + [part("channel_scale", "F32", [K], bytes(4 * K))] + base[-2:])
    red("scale-2-two", swap("scale_2", part("scale_2", "F32", [2], bytes(8))))
    stack = [dict(p, shape=[2, *p["shape"]], bits=b64(base64.b64decode(p["bits"]) * 2))
             if p["role"] in ("weight", "scale") else p for p in base]
    red("stack", stack, shape=(2, N, K))
    red("unknown-part", base + [part("smooth_factor", "F32", [K], bytes(4 * K))])
    CASES.write_text('{"cases":[\n' + ",\n".join(json.dumps(c, sort_keys=True, separators=(",", ":"))
                                                   for c in greens + reds) + "\n]}\n")
    print(f"{len(greens)} cases and {len(reds)} refusals into {CASES.name}")


if __name__ == "__main__":
    main()
