#!/usr/bin/env python3
"""Mines vectors/cases/comfy.json: comfy.asym_w4a8_int8, comfy.w6a8_int8 and int4.convrot (kitchen's convrot_w4a4).

Producers, on CPU (PyTorch 2.14.1, one thread):
- comfy-kitchen 0.2.39's eager quantizers: quantize_w4a8_int8_weight at 4 and 6 bits (its defaults, as ComfyUI
  writes; no codebook; asymmetric with a correction; F32 group scales; W6A8 without the scale search, as its
  requantize runs) and quantize_convrot_w4a4_weight. The weights are small integers x 2^-11, whose rotations are
  exact in f32, so kitchen's matmul order cannot matter; a heavy-tailed one makes kitchen fit its codebook.
- Real rows of real files, range-read: AX1Y2JP/MiniMax-H3-W4A8-ConvRot (asym_w4a8_int8, codebook),
  Comfy-Org/MiniMax-H3 (w6a8_int8, group 32) and Merserk/MiniMax-H3-INT4-ConvRot (convrot_w4a4).

Expected bytes are the spec's decode: q8 from kitchen's own _dequant_int4_grouped_to_int8 (or its int4 unpack),
then y = q8 . H_B in integers and ((y x s) + z) x 2^-k in f32; kitchen's dequantize is printed beside it. Cases
kitchen quantized with its defaults carry `source` (the F32 values: encode::quantize must give their parts); the
others carry `kitchen` (source and options, for encode::quantize_with).

    uv run --no-project --python 3.12 --index https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match --with torch==2.14.1+cpu --with comfy-kitchen==0.2.39 \
      --with packaging python -I conformance/mine/comfy.py
"""

import base64
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from comfy_kitchen.backends.eager.convrot_w4a4 import dequantize_convrot_w4a4_weight, quantize_convrot_w4a4_weight
from comfy_kitchen.backends.eager.quantization import rotate_int8_convrot_weight
from comfy_kitchen.backends.eager.svdquant import _unpack_int4_row_major
from comfy_kitchen.backends.eager.w4a8_int8 import (
    _dequant_int4_grouped_to_int8, dequantize_w4a8_int8_weight, quantize_w4a8_int8_weight)
from comfy_kitchen.tensor.w4a8_packed import _FIXED_LUT

CASES = Path(__file__).resolve().parent.parent / "vectors" / "cases" / "comfy.json"
TORCH = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
NAME = {torch.float8_e4m3fn: "F8_E4M3", torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16"}
H3 = ("Comfy-Org/MiniMax-H3", "e5eb578a89295337b8ff433a035929ce0279e0b6")
REAL = {
    "asym_w4a8_int8": ("AX1Y2JP/MiniMax-H3-W4A8-ConvRot", "61a33b9224f9aa119e2332cb24a884051c9cf810",
                       "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors", "blocks.20.attn.qkv_proj"),
    "w6a8_int8": H3 + ("diffusion_models/minimax_h3_fl2va_pruned_w6a8.safetensors", "blocks.20.mlp.fc1"),
    "convrot_w4a4": ("Merserk/MiniMax-H3-INT4-ConvRot", "3c167bc916cab6cf3b85b9e3769952757ebb2e5d",
                     "minimax_h3_fl2va_pruned_int4_convrot.safetensors", "blocks.20.attn.out_proj"),
}
ROWS, COLS = (2048, 2056), 256
torch.set_num_threads(1)


def b64(b) -> str:
    return base64.b64encode(bytes(b)).decode()


def raw(t) -> bytes:
    return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def part(role, t_or_dtype, shape=None, bits=None):
    if isinstance(t_or_dtype, torch.Tensor):
        t = t_or_dtype
        dtype = "I8" if t.dtype == torch.int8 else NAME[t.dtype]
        return {"bits": b64(raw(t)), "dtype": dtype, "role": role, "shape": list(t.shape)}
    return {"bits": b64(bits), "dtype": t_or_dtype, "role": role, "shape": list(shape)}


def rot(b):
    return {"rotation": "kron_regular_h4", "rotation_size": b}


def case(key, scheme, dtype, shape, params, parts, expect=None, **extra):
    c = {"component": scheme if expect is not None else "refuse", "dtype": dtype,
         "encoding": {"params": params, "scheme": scheme}, "key": key, "parts": parts, "shape": list(shape)}
    c.update({"expect": b64(expect)} if expect is not None else {"refuse": "decode"})
    c.update(extra)
    return c


# ------------------------------------------------------------------ the spec's decode


def kron_regular_h4(b):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.int64)
    h = h4
    while h.shape[0] < b:
        h = torch.kron(h, h4)
    return h


def spec(L, q8, s, b, corr=None, g=None) -> bytes:
    """round_L(((y x s[i]) + z) x 2^-k), y = q8 . H_B exact, z = sum_b corr(i, b) x M_b left to right."""
    r, c = q8.shape
    k = b.bit_length() // 2
    h = kron_regular_h4(b)
    v = (q8.long().reshape(r, c // b, b) @ h).float() * s.float().reshape(r, 1, 1)
    if corr is not None:
        ct, z = corr.float().t(), torch.empty_like(v)
        for blk in range(c // b):
            first = True
            for grp in range(blk * b // g, -(-(blk + 1) * b // g)):
                m = (((torch.arange(blk * b, (blk + 1) * b) // g) == grp).long() @ h).float()
                t = ct[:, grp:grp + 1] * m[None, :]
                z[:, blk] = t if first else z[:, blk] + t
                first = False
        v = v + z
    return raw((v * 2.0 ** -k).reshape(r, c).to(TORCH[L]))


def differ(L, a, b) -> str:
    """How far kitchen's dequantize is from the spec's decode: elements that differ, relative L2."""
    x, y = (torch.frombuffer(bytearray(v), dtype=TORCH[L]).double() for v in (a, b))
    rel = float((x - y).norm() / y.norm())
    return f"{int((x != y).sum())}/{x.numel()} differ, relative L2 {rel:.1e}"


# ------------------------------------------------------------------ kitchen


def weight(n, k, L, seed, heavy=False):
    """Small integers x 2^-11 (exact in L): every rotation of them is exact in f32."""
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 40).round().clamp(-255, 255)
    if heavy:  # sparse spikes in the rotated basis, rotated back (multiples of 16 stay integers)
        w = (w * (torch.rand(n, k, generator=g) < 0.05)).round() * 16
        w = rotate_int8_convrot_weight(w, 256).round()
    return (w / 2048).to(TORCH[L])


def grouped(key, L, n, k, gs, b, bits=4, seed=1, heavy=False, **opts):
    """quantize_w4a8_int8_weight, its parts, and the spec's decode."""
    scheme = "comfy.asym_w4a8_int8" if bits == 4 else "comfy.w6a8_int8"
    w = weight(n, k, L, seed, heavy)
    kw = dict(group_size=gs, convrot_groupsize=b, bits=bits, symmetric=opts.get("symmetric", True),
              codebook=opts.get("codebook", True), scale_search=opts.get("scale_search", True),
              scale_dtype=torch.float32 if opts.get("f32_scales") else torch.float8_e4m3fn)
    qd, s_rel, s_ch, corr, cb = quantize_w4a8_int8_weight(w, **kw)
    if heavy:
        assert not torch.equal(cb, torch.tensor(_FIXED_LUT)), "kitchen fits a codebook"
    q8 = _dequant_int4_grouped_to_int8(qd, s_rel, cb, gs)
    expect = spec(L, q8, s_ch, b, corr, gs)
    kit = dequantize_w4a8_int8_weight(qd, s_rel, s_ch, cb, corr, gs, b, TORCH[L])
    print(f"{key}: against kitchen's dequantize in {L}: {differ(L, expect, raw(kit))}")
    parts = [part("weight", qd), part("group_scale", s_rel), part("scale", s_ch)]
    if cb is not None:
        parts.append(part("codebook", cb))
    if corr is not None:
        parts.append(part("correction", corr))
    source = b64(raw(w.float()))
    extra = {"source": source} if not opts else {"kitchen": dict(opts, source=source)}
    return case(key, scheme, L, [n, k], {"group_size": gs, **rot(b)}, parts, expect, **extra)


def w4a4(key, L, n, k, b, seed):
    w = weight(n, k, L, seed)
    qd, s = quantize_convrot_w4a4_weight(w, b, 64)
    expect = spec(L, _unpack_int4_row_major(qd), s, b)
    kit = dequantize_convrot_w4a4_weight(qd, s, b, 64, TORCH[L])
    print(f"{key}: against kitchen's dequantize in {L}: {differ(L, expect, raw(kit))}")
    return case(key, "int4.convrot", L, [n, k], rot(b),
                [part("weight", qd), part("scale", s)], expect, source=b64(raw(w.float())))


# ------------------------------------------------------------------ real files


def get(repo, rev, path, lo, hi) -> bytes:
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def real(fmt):
    """Rows ROWS x columns 0..COLS of a real module: whole rotation blocks and groups, so a valid tensor."""
    repo, rev, path, module = REAL[fmt]
    n = struct.unpack("<Q", get(repo, rev, path, 0, 8))[0]
    h, base = json.loads(get(repo, rev, path, 8, 8 + n)), 8 + n
    r0, r1 = ROWS

    def rows(suffix, width):
        t = h[f"{module}.{suffix}"]
        lo = base + t["data_offsets"][0]
        return torch.frombuffer(bytearray(get(repo, rev, path, lo + r0 * width, lo + r1 * width)), dtype=torch.uint8)

    w = h[f"{module}.weight"]
    width = w["shape"][1]
    q = rows("weight", width).reshape(-1, width)
    scheme, b = "int4.convrot" if fmt == "convrot_w4a4" else "comfy." + fmt, 256
    if fmt == "convrot_w4a4":
        qd = q[:, :COLS // 2].contiguous().view(torch.int8)
        s = rows("weight_scale", 4).view(torch.float32)
        expect = spec("BF16", _unpack_int4_row_major(qd), s, b)
        kit = dequantize_convrot_w4a4_weight(qd, s, b, 64, torch.bfloat16)
        parts, params = [part("weight", qd), part("scale", s)], rot(b)
    else:
        k = width * 8 // (4 if fmt == "asym_w4a8_int8" else 6)
        g = k // h[f"{module}.weight_s_rel"]["shape"][1]
        planes = [q[:, :COLS // 2]] + ([q[:, k // 2:k // 2 + COLS // 4]] if fmt == "w6a8_int8" else [])
        qd = torch.cat(planes, 1).contiguous().view(torch.int8)
        s_rel = rows("weight_s_rel", k // g).reshape(-1, k // g)[:, :COLS // g].contiguous().view(torch.float8_e4m3fn)
        s_ch = rows("weight_s_channel", 4).view(torch.float32)
        cb = None
        if f"{module}.weight_codebook" in h:
            t = h[f"{module}.weight_codebook"]
            cb = torch.frombuffer(bytearray(get(repo, rev, path, base + t["data_offsets"][0],
                                                base + t["data_offsets"][1])), dtype=torch.float32)
        expect = spec("BF16", _dequant_int4_grouped_to_int8(qd, s_rel, cb, g), s_ch, b)
        kit = dequantize_w4a8_int8_weight(qd, s_rel, s_ch, cb, None, g, b, torch.bfloat16)
        parts = [part("weight", qd), part("group_scale", s_rel), part("scale", s_ch)]
        parts += [part("codebook", cb)] if cb is not None else []
        params = {"group_size": g, **rot(b)}
    key = f"real-{fmt.replace('_', '-')}"
    print(f"{key} ({repo}@{rev[:9]} {module} rows {r0}..{r1}): against kitchen: {differ('BF16', expect, raw(kit))}")
    return case(key, scheme, "BF16", [r1 - r0, COLS], params, parts, expect)


# ------------------------------------------------------------------ cases


def refusals(lut, corr, w6, w4, cv) -> list:
    reds = []

    def red(key, base, params=None, parts=None, shape=None):
        c = dict(base, key=key, component="refuse", refuse="decode")
        c.pop("expect", None), c.pop("source", None), c.pop("kitchen", None)
        enc = dict(base["encoding"])
        enc["params"] = params if params is not None else enc["params"]
        c["encoding"] = enc
        c["parts"] = parts if parts is not None else base["parts"]
        c["shape"] = shape if shape is not None else base["shape"]
        reds.append(c)

    def swap(base, role, new):
        return [new if p["role"] == role else p for p in base["parts"] if new is not None or p["role"] != role]

    z = lambda n: bytes(n)  # noqa: E731
    asym = {"group_size": 48, **rot(16)}
    red("uneven-groups", lut, asym, [part("weight", "I8", [8, 32], z(256)), part("group_scale", "F8_E4M3", [8, 2], z(16)),
                                     part("scale", "F32", [8], z(32))], [8, 64])
    red("partial-rotation-block", lut, {"group_size": 16, **rot(256)},
        [part("weight", "I8", [8, 160], z(1280)), part("group_scale", "F8_E4M3", [8, 20], z(160)),
         part("scale", "F32", [8], z(32))], [8, 320])
    red("group-size-12", lut, {"group_size": 12, **rot(256)})
    red("rotation-missing", lut, {"group_size": 16, "rotation_size": 256})
    red("codebook-8", lut, parts=swap(lut, "codebook", part("codebook", "F32", [8], z(32))))
    rows = corr["shape"][0]
    red("correction-row-major", corr, parts=swap(corr, "correction",
                                                 part("correction", "BF16", [rows, 16], z(rows * 32))))
    sr = next(p for p in lut["parts"] if p["role"] == "group_scale")
    red("s-rel-bf16", lut, parts=swap(lut, "group_scale", part("group_scale", "BF16", sr["shape"], z(2 * np.prod(sr["shape"])))))
    nan = bytearray(base64.b64decode(sr["bits"]))
    nan[5] = 0x7F
    red("s-rel-nan", lut, parts=swap(lut, "group_scale", dict(sr, bits=b64(nan))))
    red("mma-packed", lut, dict(lut["encoding"]["params"], mma_packed=True))
    red("w6a8-group-8", w6, dict(w6["encoding"]["params"], group_size=8))
    n, k = w6["shape"]
    red("w6a8-nibble-width", w6, parts=swap(w6, "weight", part("weight", "I8", [n, k // 2], z(n * k // 2))))
    red("w6a8-codebook", w6, parts=w6["parts"] + [part("codebook", "F32", [16], z(64))])
    n, k = w4["shape"]
    red("w4a4-group-scales", w4, parts=swap(w4, "scale", part("scale", "F32", [n, k // 64], z(4 * n * k // 64))))
    red("w4a4-quant-group-size", w4, dict(w4["encoding"]["params"], quant_group_size=64))
    red("w4a4-scale-bf16", w4, parts=swap(w4, "scale", part("scale", "BF16", [n], z(2 * n))))
    red("w4a4-weight-scale-role", w4, parts=swap(w4, "scale", dict(next(p for p in w4["parts"] if p["role"] == "scale"),
                                                                   role="weight_scale")))
    # w4a8.convrot admits kitchen's 4-bit scheme at one profile only: g16, E4M3 group scales, no correction.
    n, k = cv["shape"]
    red("w4a8-convrot-group-32", cv, dict(cv["encoding"]["params"], group_size=32),
        parts=swap(cv, "group_scale", part("group_scale", "F8_E4M3", [n, k // 32], z(n * k // 32))))
    red("w4a8-convrot-f32-scales", cv, parts=swap(cv, "group_scale", part("group_scale", "F32", [n, k // 16], z(n * k // 4))))
    red("w4a8-convrot-correction", cv, parts=cv["parts"] + [part("correction", "F32", [k // 16, n], z(n * k // 4))])
    return reds


def experts(c) -> dict:
    """A stack [2, R/2, C] of the same parts: leading axes are rows."""
    n, k = c["shape"]
    stack = lambda p, last: dict(p, shape=[2, n // 2, last])  # noqa: E731
    parts = [stack(p, p["shape"][-1]) if p["role"] in ("weight", "group_scale") else p for p in c["parts"]]
    out = dict(c, key=c["key"].replace("lut", "experts"), shape=[2, n // 2, k], parts=parts)
    out.pop("source", None)
    return out


def main():
    lut = grouped("w4a8-lut-bf16", "BF16", 8, 256, 16, 256)
    greens = [lut, experts(lut),
              grouped("w4a8-fit-bf16", "BF16", 16, 256, 16, 256, seed=2, heavy=True),
              grouped("w4a8-uniform-f16", "F16", 8, 256, 16, 64, seed=3, codebook=False),
              grouped("w4a8-g48-bf16", "BF16", 4, 768, 48, 256, seed=4),
              grouped("w4a8-g8-bf16", "BF16", 4, 256, 8, 16, seed=5),
              grouped("w4a8-f32-scales-f32", "F32", 8, 256, 32, 256, seed=6, f32_scales=True)]
    corr = grouped("w4a8-correction-g48-bf16", "BF16", 4, 768, 48, 256, seed=7, symmetric=False)
    w6 = grouped("w6a8-g32-bf16", "BF16", 8, 256, 32, 256, bits=6, seed=9)
    w4 = w4a4("w4a4-bf16", "BF16", 8, 256, 256, 13)
    greens += [corr, grouped("w4a8-correction-g16-f32", "F32", 4, 256, 16, 64, seed=8, symmetric=False), w6,
               grouped("w6a8-g16-f16", "F16", 8, 256, 16, 64, bits=6, seed=10),
               grouped("w6a8-f32-scales-bf16", "BF16", 4, 256, 32, 256, bits=6, seed=11, f32_scales=True),
               grouped("w6a8-no-search-bf16", "BF16", 4, 256, 32, 256, bits=6, seed=12, scale_search=False),
               w4, w4a4("w4a4-f32", "F32", 4, 256, 64, 14)]
    greens += [real(f) for f in REAL]
    # w4a8.convrot is kitchen's 4-bit scheme at g16, E4M3 and no correction: the same parts and decode renamed.
    greens += [dict(c, key=c["key"].replace("w4a8", "w4a8-convrot", 1), component="w4a8.convrot",
                    encoding=dict(c["encoding"], scheme="w4a8.convrot"))
               for c in greens if c["encoding"]["scheme"] == "comfy.asym_w4a8_int8"
               and c["encoding"]["params"]["group_size"] == 16 and "kitchen" not in c
               and all(p["role"] != "correction" and (p["role"] != "group_scale" or p["dtype"] == "F8_E4M3")
                       for p in c["parts"])]
    cases = greens + refusals(lut, corr, w6, w4, next(c for c in greens if c["key"] == "w4a8-convrot-lut-bf16"))
    CASES.write_text('{"cases":[\n' + ",\n".join(json.dumps(c, sort_keys=True, separators=(",", ":"))
                                                   for c in cases) + "\n]}\n")
    print(f"{len(greens)} cases and {len(cases) - len(greens)} refusals into {CASES.name}")


if __name__ == "__main__":
    main()
