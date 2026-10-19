#!/usr/bin/env python3
"""Mines mxfp8's expert-stack cases into vectors/cases/mxfp8.json (its own keys; the other cases are
kept): a stack [E, N, C] with N not a multiple of 128, so tiling each matrix (spec §6
`cublas_128x4`) and tiling every row of the stack at once place the scales differently.

The producer is comfy-kitchen 0.2.39's own quantize_mxfp8, run per expert on CPU as ComfyUI's
TensorCoreMXFP8Layout quantizes a matrix; its tiles stacked are what ComfyUI's MoEExperts loads
(`weight_scale [E, 128·⌈N/128⌉, 4·⌈W/4⌉]`, one tiling per expert). The row-major twin holds the
same scales through kitchen's from_blocked. Expected bytes are the spec's decode, compared with
kitchen's dequantize_mxfp8 per expert. No published ComfyUI mxfp8 file holds an expert stack
(42 ComfyUI-style mxfp8 files read by header, 2026-10-19), so these are the only stacked ones.

    uv run --no-project --python 3.12 --index https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match --with torch==2.14.1+cpu --with comfy-kitchen==0.2.39 \
      --with packaging --with numpy python -I conformance/mine/mxfp8.py
"""

import base64
import json
from pathlib import Path

import numpy as np
import torch
from comfy_kitchen.backends.eager.quantization import dequantize_mxfp8, from_blocked, quantize_mxfp8

CASES = Path(__file__).resolve().parents[1] / "vectors" / "cases" / "mxfp8.json"
E, N, C = 3, 80, 96  # W = 3 blocks, tiled as 128 × 4
torch.set_num_threads(1)


def b64(t) -> str:
    return base64.b64encode(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).decode()


def part(role, dtype, t):
    return {"bits": b64(t), "dtype": dtype, "role": role, "shape": list(t.shape)}


def case(key, params, parts, expect=None):
    c = {"component": "mxfp8" if expect is not None else "refuse", "dtype": "BF16",
         "encoding": {"params": dict(block_size=32, **params), "scheme": "mxfp8"},
         "key": key, "parts": parts, "shape": [E, N, C]}
    c.update({"expect": b64(expect)} if expect is not None else {"refuse": "decode"})
    return c


def main():
    g = torch.Generator().manual_seed(49)
    w = torch.randn(E, N, C, generator=g) * 0.05
    w[:, :, 32:64] *= torch.logspace(-6, 3, N).reshape(1, N, 1)  # blocks across the E8M0 range
    w[1, 7, :32] = 0  # an all-zero block
    w = w.to(torch.bfloat16)
    q, tiles = zip(*(quantize_mxfp8(w[e]) for e in range(E)))
    q, tiles = torch.stack(q), torch.stack([t.view(torch.uint8) for t in tiles])
    rows = torch.stack([from_blocked(tiles[e], num_rows=N, num_cols=C // 32) for e in range(E)])
    # The spec's decode: weight × 2^(e − 127) in f32, rounded once.
    scale = torch.exp2(rows.to(torch.float32) - 127).repeat_interleave(32, dim=-1)
    expect = (q.to(torch.float32) * scale).to(torch.bfloat16)
    kitchen = torch.stack([dequantize_mxfp8(q[e], tiles[e].view(torch.float8_e8m0fnu)) for e in range(E)])
    assert torch.equal(expect.view(torch.int16), kitchen.view(torch.int16)), "kitchen's dequantize"
    weight = part("weight", "F8_E4M3", q)
    tiled, flat = {"scale_layout": "cublas_128x4"}, {"scale_layout": "row_major"}
    mine = [
        case("experts-cublas_128x4-3x80x96-bf16", tiled, [weight, part("scale", "F8_E8M0", tiles)], expect),
        case("experts-row_major-3x80x96-bf16", flat, [weight, part("scale", "F8_E8M0", rows)], expect),
    ]
    # One tiling over all E·N rows ([256, 4]), and tiles never padded to 128 rows: neither layout.
    one = torch.zeros(256, 4, dtype=torch.uint8)
    one.reshape(-1)[: E * N * 4] = tiles[:, :N].reshape(-1)
    refusals = [
        case("mxfp8-experts-tiled-as-one", tiled, [weight, part("scale", "F8_E8M0", one)]),
        case("mxfp8-experts-tiles-unpadded", tiled, [weight, part("scale", "F8_E8M0", tiles[:, :N])]),
    ]
    keys = {c["key"] for c in mine + refusals}
    kept = [c for c in json.loads(CASES.read_text())["cases"] if c["key"] not in keys]
    at = max(i for i, c in enumerate(kept) if c["component"] == "mxfp8") + 1
    cases = kept[:at] + mine + kept[at:] + refusals  # make_vectors.py's one form
    CASES.write_text('{"cases":[\n' + ",\n".join(json.dumps(c, sort_keys=True, separators=(",", ":"))
                                                   for c in cases) + "\n]}\n")
    print(f"{len(mine)} cases and {len(refusals)} refusals into {CASES.name}; kitchen's dequantize matches")


if __name__ == "__main__":
    main()
