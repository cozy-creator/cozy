#!/usr/bin/env python3
"""Writes vectors/cores.json: cases for the decode building blocks (spec §6, building blocks), checked by every
seat (tests/format_cores.rs, reader.py, the Runtime's GPU cores).

- blocks: GGUF q4_0 and q8_0 from gguf-py's own quantize and dequantize (the real producer).
- affine and table: codes packed bit by bit as spec §6 states, values computed by its formulas in
  float32, over the layouts producers use (GPTQ, AWQ, compressed-tensors, hqq, bnb, cuBLAS tiles).

    uv run --no-project --with gguf==0.19.0 --with numpy python -I conformance/mine/cores.py
"""

import base64
import importlib.metadata
import json
from pathlib import Path

import gguf
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "vectors" / "cores.json"
rng = np.random.default_rng(49)
AWQ = [0, 2, 4, 6, 1, 3, 5, 7]


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def pack(q, bits, word, transposed=False, axis=1, fill="word", order="low_first", signed=False, offset=0):
    """The packed bytes of the [R, C] codes q, one bit at a time."""
    stored = q.T if transposed else q
    lanes = stored.T if axis == 0 else stored  # [along, n]
    along, n = lanes.shape
    per = word // bits
    nw = {"word": -(-n // per), "dense": -(-n * bits // word), "planar": n // per}[fill]
    if order == "low_first":
        slot = list(range(per))
    elif order == "high_first":
        slot = list(range(per))[::-1]
    else:
        slot = [order.index(e) for e in range(per)]
    words = np.zeros((along, nw), np.uint64)
    for a in range(along):
        for t in range(n):
            raw = int(lanes[a, t]) & ((1 << bits) - 1) if signed else int(lanes[a, t]) + offset
            assert 0 <= raw < 1 << bits
            if fill == "dense":
                first = t * bits
            else:
                k, e = (t // per, t % per) if fill == "word" else (t % nw, t // nw)
                first = k * word + slot[e] * bits
            for i in range(bits):
                if raw >> i & 1:
                    words[a, (first + i) // word] |= np.uint64(1 << ((first + i) % word))
    words = words.T if axis == 0 else words  # back to the stored array's orientation
    return words.astype(f"<u{word // 8}")


def at(grid, rows, cols):
    i, j = np.arange(rows)[:, None], np.arange(cols)[None, :]
    kind = grid["kind"]
    if kind == "tensor":
        return np.zeros((rows, cols), np.int64)
    if kind in ("row", "col"):
        return np.broadcast_to(i if kind == "row" else j, (rows, cols))
    w = -(-cols // grid["size"])
    b = j // grid["size"] if "index" not in grid else np.array(grid["index"])[None, :]
    if grid["layout"] == "row_major":
        return i * w + b
    if grid["layout"] == "transposed":
        return b * rows + i
    return ((i // 128) * -(-w // 4) + b // 4) * 512 + (i % 32) * 16 + (i % 128 // 32) * 4 + b % 4


def grid_size(grid, rows, cols):
    if grid.get("layout") == "cublas_128x4":
        w = -(-cols // grid["size"])
        return 128 * -(-rows // 128) * 4 * -(-w // 4)
    return int(at(grid, rows, cols).max()) + 1


def codes(rows, cols, bits, signed=False, offset=0):
    lo = -(1 << (bits - 1)) if signed else -offset
    return rng.integers(lo, lo + (1 << bits), (rows, cols))


def scales(n):
    return (rng.standard_normal(n) * 0.05).astype(np.float32)


def affine(name, rows, cols, packing, grid, term=None):
    q = codes(rows, cols, packing["bits"], packing.get("signed", False), packing.get("offset", 0))
    n = grid_size(grid, rows, cols)
    s = scales(n)
    case = {"name": name, "core": "affine", "rows": rows, "cols": cols, "packing": packing, "grid": grid,
            "words": b64(pack(q, **packing)), "scale": b64(s)}
    v, idx = q.astype(np.float32), at(grid, rows, cols)
    if term == "zero":
        z = rng.integers(0, 1 << packing["bits"], n).astype(np.float32)
        case["zero"], out = b64(z), (v - z[idx]) * s[idx]
    elif term == "bias":
        bias = scales(n)
        case["bias"], out = b64(bias), v * s[idx] + bias[idx]
    else:
        out = v * s[idx]
    return {**case, "expect": b64(out.astype("<f4"))}


def table(name, rows, cols, packing, grid, values):
    q = rng.integers(0, len(values), (rows, cols))
    s = scales(grid_size(grid, rows, cols))
    t = np.array(values, np.float32)
    out = t[q] * s[at(grid, rows, cols)]
    return {"name": name, "core": "table", "rows": rows, "cols": cols, "packing": packing, "grid": grid,
            "words": b64(pack(q, **packing)), "table": b64(t), "scale": b64(s), "expect": b64(out.astype("<f4"))}


def p(bits, word, **kw):
    return {"bits": bits, "word": word, **kw}


g_idx = rng.permutation(np.repeat(np.arange(4), 8)).tolist()
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
NF4 = [-1.0, -0.6961928, -0.52507305, -0.3949175, -0.28444138, -0.18477343, -0.09105004, 0.0,
       0.0795803, 0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1.0]
CASES = [
    affine("gptq-int4-act-order", 6, 32, p(4, 32, transposed=True, axis=0),
           {"kind": "blocks", "size": 8, "layout": "transposed", "index": g_idx}, "zero"),
    affine("gptq-int3-dense", 3, 64, p(3, 32, transposed=True, axis=0, fill="dense"),
           {"kind": "blocks", "size": 32, "layout": "transposed"}, "zero"),
    affine("awq-int4", 16, 24, p(4, 32, transposed=True, axis=1, order=AWQ),
           {"kind": "blocks", "size": 8, "layout": "transposed"}, "zero"),
    affine("ct-int3-dense-padded", 5, 40, p(3, 32, fill="dense", offset=4),
           {"kind": "blocks", "size": 8, "layout": "row_major"}),
    affine("int6-word-signed-bias", 4, 15, p(6, 32, signed=True), {"kind": "row"}, "bias"),
    affine("hqq-int4-planar-high-first", 8, 16, p(4, 8, axis=0, fill="planar", order="high_first"),
           {"kind": "blocks", "size": 16, "layout": "row_major"}, "zero"),
    affine("int2-u16-high-first-signed", 3, 16, p(2, 16, order="high_first", signed=True), {"kind": "col"}),
    affine("int1-u8", 2, 24, p(1, 8), {"kind": "tensor"}),
    affine("int5-dense-u16", 3, 10, p(5, 16, fill="dense", signed=True), {"kind": "row"}),
    affine("int7-word-u8", 2, 9, p(7, 8, offset=64), {"kind": "blocks", "size": 4, "layout": "row_major"}),
    affine("int8-signed-u32", 2, 8, p(8, 32, signed=True), {"kind": "row"}),
    table("nf4-high-first", 4, 64, p(4, 8, order="high_first"),
          {"kind": "blocks", "size": 64, "layout": "row_major"}, NF4),
    table("e2m1-cublas-130-rows", 130, 48, p(4, 8), {"kind": "blocks", "size": 16, "layout": "cublas_128x4"}, E2M1),
    table("e2m1-ragged-blocks", 3, 40, p(4, 8), {"kind": "blocks", "size": 32, "layout": "row_major"}, E2M1),
]

x = rng.standard_normal((2, 64)).astype(np.float32)
x[0, 5], x[1, 40], x[1, 3] = 9.5, -7.25, 0.0
for kind, size in ((gguf.GGMLQuantizationType.Q4_0, 18), (gguf.GGMLQuantizationType.Q8_0, 34)):
    blocks = gguf.quants.quantize(x, kind)
    CASES.append({"name": f"gguf-{kind.name.lower()}", "core": "blocks", "type": kind.name.lower(), "rows": 2,
                  "cols": 64, "elements": 32, "size": size, "blocks": b64(blocks),
                  "expect": b64(gguf.quants.dequantize(blocks, kind).astype("<f4"))})

producers = f"gguf-py {importlib.metadata.version('gguf')}, numpy {np.__version__}"
OUT.write_text('{"producers":' + json.dumps(producers) + ',"cases":[\n'
               + ",\n".join(json.dumps(c, separators=(",", ":")) for c in CASES) + "\n]}\n")
print(f"{len(CASES)} core cases -> {OUT}")
