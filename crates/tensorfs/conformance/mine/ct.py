#!/usr/bin/env python3
"""Writes vectors/cases/ct.json: `ct.int1` … `ct.int8` decode cases (spec §6) from compressed-tensors on both sides
of its packing change (#715, 2026-05-29), on CPU. Run once per version; each run replaces its own cases:

    CUDA_VISIBLE_DEVICES= uv run --no-project --python 3.12 --with torch==2.14.1 --with compressed-tensors==0.16.0 --with psutil \\
        python -I conformance/mine/ct.py
    CUDA_VISIBLE_DEVICES= uv run --no-project --python 3.12 --with torch==2.14.1 --with compressed-tensors==0.19.0 \\
        python -I conformance/mine/ct.py

- llm-compressor's RTN: per-group min/max, compressed-tensors' `calculate_qparams`, then its `pack-quantized`
  compressor (`PackedQuantizationCompressor.compress`); its `decompress` gives the expected bytes. 0.16.0 packs
  `⌊32/b⌋` codes to a word (`packing: "word"`) and applies `weight_g_idx` (group act order, since removed); 0.17.0
  on packs one dense stream (`packing: "dense"`). For `b` dividing 32 the two are the same bytes.
- Real 8-row bands by HTTP range: RedHatAI Qwen3-0.6B w4a16 (asymmetric, group 64) read by 0.19.0, and RedHatAI
  Qwen2.5-0.5B w4a16 (act order `group`, with `weight_g_idx`) read by 0.16.0. A band's `weight_shape` states the
  band's shape.

The 0.19.0 cases carry `source`, the values quantized (in the logical dtype): TensorFS's closed-form encoder
reproduces their parts bit for bit (tests/format_pack.rs). Producer cases carry `pack`, the codes and zero points
the producer packed (`I16`, the logical shape and `[lead…, N, W]`).
"""

import base64
import json
import math
import struct
import sys
import urllib.request
from pathlib import Path

import compressed_tensors
import torch
from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.utils import calculate_qparams

OUT = Path(__file__).resolve().parents[1] / "vectors" / "cases" / "ct.json"
VERSION = compressed_tensors.__version__
MINED = f"compressed-tensors {VERSION}"
PACKING = "word" if tuple(int(v) for v in VERSION.split(".")[:2]) < (0, 17) else "dense"
DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16", torch.int32: "I32", torch.int64: "I64"}
REAL_ASYM = ("RedHatAI/Qwen3-0.6B-quantized.w4a16", "3bceedd23534ac8ef59952f72c94790b27938e42", 64)
REAL_ACT_ORDER = ("RedHatAI/Qwen2.5-0.5B-quantized.w4a16", "81996352fa5c5b2154927b2fc0b3f58cbe59ddcc", 64)
gen = torch.Generator().manual_seed(715)


def get(url, start=None, end=None) -> bytes:
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end - 1}")
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    assert start is None or len(data) == end - start, "short read"
    return data


def b64(t: torch.Tensor) -> str:
    return base64.b64encode(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).decode()


def scheme(bits, strategy, g, sym, act_order=False):
    extra = {"actorder": "group"} if act_order else {}
    args = QuantizationArgs(num_bits=bits, type="int", strategy=strategy, group_size=g if strategy == "group" else None,
                            symmetric=sym, dynamic=False, **extra)
    return QuantizationScheme(targets=["Linear"], weights=args)


def case(key, sd, sch, dtype, expect, producer, source=None, pack=None):
    w = sch.weights
    shape = [int(v) for v in sd["weight_shape"]]
    g = w.group_size if w.strategy == "group" else shape[-1]
    roles = [r for r in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape", "weight_g_idx")
             if r in sd]
    c = {
        "component": "ct", "key": key, "dtype": DTYPES[dtype], "shape": shape,
        "encoding": {"scheme": f"ct.int{w.num_bits}",
                     "params": {"group_size": g, "strategy": w.strategy, "symmetric": w.symmetric,
                                "packing": PACKING}},
        "parts": [{"role": r, "dtype": DTYPES[sd[r].dtype], "shape": list(sd[r].shape), "bits": b64(sd[r])}
                  for r in roles],
        "expect": b64(expect.to(dtype)), "producer": producer, "mined": MINED,
    }
    if source is not None:
        c["source"] = b64(source.float())
    if pack is not None:
        c["pack"] = {k: b64(v.to(torch.int16)) for k, v in pack.items()}
    return c


def decompressed(sd, sch):
    """compressed-tensors' decompress, with the scale read in F32: exact (q − zp)·s."""
    exact = PackedQuantizationCompressor.decompress({**sd, "weight_scale": sd["weight_scale"].float()}, sch)
    native = PackedQuantizationCompressor.decompress(dict(sd), sch)["weight"]
    assert torch.equal(native, exact["weight"].to(native.dtype)), "the producer rounds once"
    return exact["weight"]


def rtn_case(key, shape, dtype, bits, strategy, g, sym, act_order=False, unset=False):
    """llm-compressor's RTN in the weight's dtype: min/max per group, calculate_qparams, then compress."""
    sch = scheme(bits, strategy, g, sym, act_order)
    x = (torch.randn(*shape, generator=gen) * torch.logspace(-2.5, -1, shape[-1])).to(dtype)
    k = shape[-1]
    g_eff = g if strategy == "group" else k
    w = -(-k // g_eff)
    g_idx = (torch.argsort(torch.randperm(k, generator=gen)) // g_eff).to(torch.int32)
    if act_order and not unset:
        cols = [x[..., g_idx == v] for v in range(w)]
        lo = torch.stack([c.amin(-1) for c in cols], -1)
        hi = torch.stack([c.amax(-1) for c in cols], -1)
    else:
        groups = x.reshape(*shape[:-1], w, g_eff)
        lo, hi = groups.amin(-1), groups.amax(-1)
    scale, zp = calculate_qparams(lo, hi, sch.weights)
    sd = {"weight": x, "weight_scale": scale, "weight_zero_point": zp}
    if act_order:
        sd["weight_g_idx"] = torch.full((k,), -1, dtype=torch.int32) if unset else g_idx
    sd = PackedQuantizationCompressor.compress(sd, sch)
    q = unpack_from_int32(sd["weight_packed"], bits, torch.Size(shape))
    pack = {"q": q} if sym else {"q": q, "zeros": zp}
    expect = decompressed(sd, sch)
    source = x if PACKING == "dense" else None
    return case(key, sd, sch, dtype, expect, f"{MINED} pack-quantized, RTN", source, pack)


def real_case(key, repo, rev, g, sym, act_order):
    """Rows 0..8 of layer 0's q_proj: packed rows, scale rows, the zero points' first word row, the whole g_idx."""
    url = f"https://huggingface.co/{repo}/resolve/{rev}/model.safetensors"
    n = struct.unpack("<Q", get(url, 0, 8))[0]
    hdr, off = json.loads(get(url, 8, 8 + n)), 8 + n
    dts = {"I32": (torch.int32, 4), "BF16": (torch.bfloat16, 2), "I64": (torch.int64, 8)}

    def rows(name, r1):
        t = hdr[f"model.layers.0.self_attn.q_proj.{name}"]
        dt, size = dts[t["dtype"]]
        a, _ = t["data_offsets"]
        data = get(url, off + a, off + a + r1 * math.prod(t["shape"][1:]) * size)
        return torch.frombuffer(bytearray(data), dtype=dt).reshape(r1, *t["shape"][1:])

    k = hdr["model.layers.0.self_attn.q_proj.weight_packed"]["shape"][1] * 32 // 4
    sd = {"weight_packed": rows("weight_packed", 8), "weight_scale": rows("weight_scale", 8),
          "weight_shape": torch.tensor([8, k], dtype=torch.int64)}
    if not sym:
        sd["weight_zero_point"] = rows("weight_zero_point", 1)
    if act_order:
        sd["weight_g_idx"] = rows("weight_g_idx", hdr["model.layers.0.self_attn.q_proj.weight_g_idx"]["shape"][0])
    sch = scheme(4, "group", g, sym, act_order)
    expect = decompressed(sd, sch)
    producer = f"{repo}@{rev} model.layers.0.self_attn.q_proj rows 0..8, by {MINED}"
    return case(key, sd, sch, torch.bfloat16, expect, producer)


def halved(bits: str) -> str:
    """A part's first half: the length of the same shape in a dtype half as wide."""
    raw = base64.b64decode(bits)
    return base64.b64encode(raw[:len(raw) // 2]).decode()


def refusals(base):
    def twin(key, **change):
        c = json.loads(json.dumps(base))
        for k in ("expect", "pack", "source"):
            c.pop(k, None)
        c.update(key=key, component="refuse", refuse="decode", producer="")
        for path, v in change.items():
            *where, last = path.split("__")
            d = c
            for w in where:
                d = d[int(w)] if w.isdigit() else d[w]
            if v is None:
                del d[last]
            else:
                d[last] = v
        return c
    shape = torch.tensor(base["shape"], dtype=torch.int64)
    wrong = shape.clone()
    wrong[0] += 1
    return [
        twin("ct-symmetric-with-zero-point", encoding__params__symmetric=True),
        twin("ct-zero-point-missing", parts=[p for p in base["parts"] if p["role"] != "weight_zero_point"]),
        twin("ct-weight-shape-differs", parts__3__bits=b64(wrong)),
        twin("ct-channel-group-size-not-c", encoding__params__strategy="channel"),
        twin("ct-packing-planar", encoding__params__packing="planar"),
        twin("ct-word-shape-is-not-dense", encoding__params__packing="word"),
        twin("ct-group-size-0", encoding__params__group_size=0),
        twin("ct-actorder-param", encoding__params__actorder="group"),
        twin("ct-scale-u8", parts__1__dtype="U8", parts__1__bits=halved(base["parts"][1]["bits"])),
    ]


def main():
    if PACKING == "word":
        cases = [
            rtn_case("word-int3-asym-16x64-bf16", (16, 64), torch.bfloat16, 3, "group", 32, False),
            rtn_case("word-int5-sym-16x64-f16", (16, 64), torch.float16, 5, "group", 16, True),
            rtn_case("word-int6-asym-channel-16x64-bf16", (16, 64), torch.bfloat16, 6, "channel", None, False),
            rtn_case("word-int7-asym-16x64-f32", (16, 64), torch.float32, 7, "group", 32, False),
            rtn_case("word-act-order-int4-sym-16x64-bf16", (16, 64), torch.bfloat16, 4, "group", 16, True, True),
            rtn_case("word-act-order-int3-asym-16x64-bf16", (16, 64), torch.bfloat16, 3, "group", 32, False, True),
            rtn_case("word-g-idx-unset-int4-16x64-bf16", (16, 64), torch.bfloat16, 4, "group", 32, True, True, True),
            real_case("word-real-qwen2.5-0.5b-act-order-q_proj-8x896", *REAL_ACT_ORDER, True, True),
        ]
    else:
        cases = [
            rtn_case("dense-int4-asym-16x64-bf16", (16, 64), torch.bfloat16, 4, "group", 32, False),
            rtn_case("dense-int4-sym-16x64-f16", (16, 64), torch.float16, 4, "group", 16, True),
            rtn_case("dense-int3-asym-16x64-bf16", (16, 64), torch.bfloat16, 3, "group", 32, False),
            rtn_case("dense-int5-sym-16x64-bf16", (16, 64), torch.bfloat16, 5, "group", 16, True),
            rtn_case("dense-int6-asym-channel-16x64-f32", (16, 64), torch.float32, 6, "channel", None, False),
            rtn_case("dense-int7-sym-16x64-bf16", (16, 64), torch.bfloat16, 7, "group", 64, True),
            rtn_case("dense-int1-sym-16x64-bf16", (16, 64), torch.bfloat16, 1, "group", 32, True),
            rtn_case("dense-int2-asym-16x64-bf16", (16, 64), torch.bfloat16, 2, "group", 16, False),
            rtn_case("dense-int8-asym-channel-8x64-bf16", (8, 64), torch.bfloat16, 8, "channel", None, False),
            rtn_case("dense-experts-int4-asym-2x16x64-bf16", (2, 16, 64), torch.bfloat16, 4, "group", 32, False),
            real_case("dense-real-qwen3-0.6b-asym-q_proj-8x1024", *REAL_ASYM, False, False),
        ]
        cases += refusals(cases[2])
    old = json.loads(OUT.read_text())["cases"] if OUT.exists() else []
    kept = [c for c in old if c.get("mined") != MINED]
    merged = sorted(kept + cases, key=lambda c: c["mined"].split()[-1].split(".")[:2] != ["0", "16"])
    OUT.write_text(json.dumps({"cases": merged}))
    print(f"{len(cases)} cases from {MINED}, {len(merged)} in {OUT}")


if __name__ == "__main__":
    sys.exit(main())
