#!/usr/bin/env python3
"""Writes the header vectors: vectors/green/header-*.{cbor,json} from the projections below, and
vectors/red/header-*.cbor, each a green's twin or a single defect. Then checks every decode-case
file, vectors/cases/<family>.json (its cases come from the family's producer, mine/<family>.py),
and rewrites it in its one form: one case per line, keys sorted. Deterministic; run from anywhere:

    uv run --no-project --with numpy --with cbor2==6.1.4 python -I conformance/make_vectors.py

`cargo test -p tensorfs --test format_conformance -- --ignored regenerate` then writes each
family's repo, Conformance/<family>, into vectors/store.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import cbor2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reader import encode, unproject  # noqa: E402

TAG = "cozytensors/1"


def digest(label: str) -> dict:
    return {"hex": hashlib.sha256(label.encode()).hexdigest()}


ENCODINGS = [
    ["fp8.rowwise", {"axis": 0, "scale_dtype": "F32"}],
    ["fp8.tensorwise", {"scale_dtype": "F32"}],
    ["int8.convrot", {"rotation": "hadamard", "rotation_size": 256, "scale_dtype": "F32"}],
    ["int8.rowwise", {"axis": 0, "scale_dtype": "F32"}],
    ["mxfp8", {"block_size": 32, "scale_layout": "row_major"}],
    ["nvfp4", {"block_size": 16, "global_scale": "multiply", "nibble_order": "low_first",
               "scale_layout": "row_major"}],
    ["int8.tensorwise", {"scale_dtype": "F32"}],
]


def part(role, dtype, shape, chunk, offset, length):
    return [role, dtype, shape, [chunk, offset, length]]


BASIC = [
    TAG,
    [[digest("chunk 0"), 65536], [digest("chunk 1"), 12292], [digest("chunk 2"), 288]],
    ENCODINGS,
    [
        ["transformer", [
            ["blocks.0.attn.to_q.weight", "BF16", [64, 64], 1, [
                part("weight", "F8_E4M3", [64, 64], 0, 0, 4096), part("scale", "F32", [64], 0, 4096, 256)]],
            ["blocks.0.attn.to_k.weight", "BF16", [64, 64], 2, [
                part("weight", "F8_E4M3", [64, 64], 0, 8192, 4096), part("scale", "F32", [], 0, 12288, 4),
                part("input_scale", "F32", [], 0, 16384, 4)]],
            ["blocks.0.attn.to_v.weight", "BF16", [64, 256], 3, [
                part("weight", "I8", [64, 256], 0, 20480, 16384), part("scale", "F32", [64], 0, 36864, 256)]],
            ["blocks.0.ff.up.weight", "BF16", [64, 64], 4, [
                part("weight", "I8", [64, 64], 0, 40960, 4096), part("scale", "F32", [64, 1], 0, 45056, 256)]],
            ["blocks.0.ff.down.weight", "BF16", [64, 64], 5, [
                part("weight", "F8_E4M3", [64, 64], 0, 49152, 4096),
                part("scale", "F8_E8M0", [64, 2], 0, 53248, 128)]],
            ["blocks.0.ff.gate.weight", "BF16", [64, 64], 6, [
                part("weight", "F4", [64, 64], 0, 57344, 2048), part("scale", "F8_E4M3", [64, 4], 0, 61440, 256),
                part("scale_2", "F32", [], 1, 0, 4)]],
            ["blocks.0.norm.weight", "F32", [64], 0, [1, 4096, 256]],
            ["blocks.0.proj.weight", "BF16", [8, 64], 7, [
                part("weight", "I8", [8, 64], 1, 8192, 512), part("scale", "F32", [], 1, 12288, 4)]],
        ]],
        ["vae", [["decoder.conv_in.weight", "F16", [4, 4, 3, 3], 0, [2, 0, 288]]]],
    ],
    {"n": 1, "ok": True, "name": "basic", "tags": ["a", "b"]},
]


def plain(key, dtype, shape, chunk, offset, length):
    return [key, dtype, shape, 0, [chunk, offset, length]]


def one(tensors, chunks, encodings=(), metadata=None):
    return [TAG, chunks, list(encodings), [["unet", tensors]], metadata or {}]


A = plain("a.weight", "F32", [2], 0, 0, 8)
B = plain("b.weight", "F32", [2], 0, 4096, 8)
SIZES = {"F4": [2], "F6_E2M3": [4], "F6_E3M2": [4]}
DTYPES = ["BOOL", "F4", "F6_E2M3", "F6_E3M2", "U8", "I8", "F8_E5M2", "F8_E4M3", "F8_E8M0", "F8_E4M3FNUZ",
          "F8_E5M2FNUZ", "I16", "U16", "F16", "BF16", "I32", "U32", "F32", "C64", "F64", "I64", "U64"]
WIDTH = {"F4": 1, "F6_E2M3": 3, "F6_E3M2": 3, "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4,
         "F32": 4, "C64": 8, "F64": 8, "I64": 8, "U64": 8}  # the rest: one byte


def appended():
    h = copy.deepcopy(BASIC)
    h[0] = TAG + ".7"
    h.append({"future": [1, {"hex": "00ff"}]})
    h[1][0].append(True)
    h[2][0].append("note")
    h[3][0].append({"k": []})
    t = h[3][0][1][0]
    t.append(-1)
    t[4][0].append(0)
    t[4][0][3].append({"hex": ""})
    h[3][0][1][6][4].append([])
    return h


GREEN = {
    "basic": BASIC,
    "empty": [TAG, [], [], [], {}],
    "order-a": one([A, B], [[digest("order"), 4104]]),
    "order-b": one([B, A], [[digest("order"), 4104]]),
    "span": one([plain("x", "U8", [10000], 0, 4096, 10000)],
                [[digest("span 0"), 8192], [digest("span 1"), 4096], [digest("span 2"), 2000]]),
    "zero-length": one([plain("empty", "F32", [0, 4], 0, 0, 0)], []),
    "minor-version": [TAG + ".12", *BASIC[1:]],
    "appended-fields": appended(),
    "unknown-dtype": one([plain("x", "F3_E1M1", [8], 0, 0, 6)], [[digest("future"), 6]]),
    "unknown-scheme": one([["q.weight", "F32", [2, 32], 1, [part("q", "U8", [24], 0, 0, 24)]]],
                          [[digest("acme"), 24]], [["acme.int3", {"bits": 3, "group": [64, {"pack": "lsb"}]}]]),
    "values": [TAG, [], [], [], {
        "": "", "a": 0, "b": 23, "c": 24, "d": 255, "e": 256, "f": 65535, "g": 65536, "h": 2**32 - 1,
        "i": 2**32, "j": 2**64 - 1, "k": -1, "l": -24, "m": -25, "n": -(2**63), "aa": True, "ab": False,
        "ü→😀": "ü→😀", "x" * 23: "23", "x" * 24: "24", "list": [[], [[]], {"z": {"y": [1, -1]}}]}],
    "dtypes": one([plain(f"t{i}", d, SIZES.get(d, [1]), 0, 4096 * i, WIDTH.get(d, 1))
                   for i, d in enumerate(DTYPES)], [[digest("dtypes"), 4096 * 21 + 8]]),
}


def edit(path, value):
    """BASIC with the item at `path` (indices into the tree) replaced or appended; None deletes it."""
    h = copy.deepcopy(BASIC)
    *parents, last = path
    node = h
    for i in parents:
        node = node[i]
    if value is None:
        del node[last]
    elif last == len(node):
        node.append(value)
    else:
        node[last] = value
    return h


TENSORS = (3, 0, 1)  # BASIC[3][0][1]: the transformer's tensors
Q = (*TENSORS, 0)  # its first tensor
NORM = (*TENSORS, 6)


def swap_encodings():
    h = copy.deepcopy(BASIC)
    h[2][0], h[2][1] = h[2][1], h[2][0]
    tensors = h[3][0][1]
    tensors[0][3], tensors[1][3] = 2, 1
    return h


def reordered_params():
    """nvfp4's params with their keys in reverse order."""
    good = encode(BASIC)
    params = ENCODINGS[5][1]
    canonical = cbor2.dumps(unproject(params), canonical=True)
    raw = b"\xa4" + b"".join(cbor2.dumps(k) + cbor2.dumps(params[k]) for k in sorted(params, reverse=True))
    assert raw != canonical and len(raw) == len(canonical)
    return swap(good, canonical, raw)


def swap(data: bytes, old: bytes, new: bytes) -> bytes:
    assert data.count(old) == 1, old
    return data.replace(old, new)


def reds() -> dict[str, bytes]:
    good = encode(BASIC)
    meta = cbor2.dumps(unproject(BASIC[4]), canonical=True)
    assert meta == bytes.fromhex("a4616e01626f6bf5646e616d6565626173696364746167738261616162")
    out = {
        # Twins: BASIC's content in bytes other than its one encoding.
        "twin-nonpreferred-uint": swap(good, b"\x61n\x01", b"\x61n\x18\x01"),
        "twin-nonpreferred-uint32": swap(good, b"\x1a\x00\x01\x00\x00", b"\x1b" + bytes(5) + b"\x01\x00\x00"),
        "twin-nonpreferred-array": b"\x98\x05" + good[1:],
        "twin-nonpreferred-text": swap(good, b"\x6dcozytensors/1", b"\x78\x0dcozytensors/1"),
        "twin-indefinite-array": b"\x9f" + good[1:] + b"\xff",
        "twin-indefinite-map": swap(good, meta, b"\xbf" + meta[1:] + b"\xff"),
        "twin-indefinite-text": swap(good, b"\x65basic", b"\x7f\x63bas\x62ic\xff"),
        "twin-map-order": swap(good, b"\x61n\x01\x62ok\xf5", b"\x62ok\xf5\x61n\x01"),
        "twin-params-order": reordered_params(),
        "twin-encodings-order": encode(swap_encodings()),
        "twin-repeated-key": swap(good, meta, b"\xa5\x61n\x01" + meta[1:]),
        # Items outside the allowed kinds.
        "tag": swap(good, b"\x61n\x01", b"\x61n\xc1\x01"),
        "float": swap(good, b"\x61n\x01", b"\x61n\xf9\x3e\x00"),
        "float-extent": swap(good, b"\x84\x04\x04\x03\x03", b"\x84\xf9\x44\x00\x04\x03\x03"),
        "null": swap(good, b"\x61n\x01", b"\x61n\xf6"),
        "undefined": swap(good, b"\x61n\x01", b"\x61n\xf7"),
        "simple": swap(good, b"\x61n\x01", b"\x61n\xf0"),
        "bytes-value": swap(good, b"\x61n\x01", b"\x61n\x41\x01"),
        "int-key": swap(good, b"\x61n\x01", b"\x00\x01"),
        "below-i64": swap(good, b"\x61n\x01", b"\x61n\x3b\x80\x00\x00\x00\x00\x00\x00\x00"),
        "bignum": swap(good, b"\x61n\x01", b"\x61n\xc2\x41\x01"),
        "bad-utf8": swap(good, b"\x65basic", b"\x65bas\xffc"),
        "trailing-byte": good + b"\x00",
        "truncated": good[:-1],
        "not-array": cbor2.dumps({"format": TAG}, canonical=True),
        "short-root": encode(BASIC[:4]),
        # The format tag.
        "format-major-2": encode(["cozytensors/2", *BASIC[1:]]),
        "format-kind": encode(["tensorfs.manifest/1", *BASIC[1:]]),
        "format-empty-minor": encode([TAG + ".", *BASIC[1:]]),
        "format-not-text": encode([1, *BASIC[1:]]),
        # Structure and bounds.
        "digest-31": encode(edit((1, 0, 0), {"hex": "00" * 31})),
        "digest-text": encode(edit((1, 0, 0), "00" * 32)),
        "chunk-length-text": encode(edit((1, 0, 1), "65536")),
        "chunk-index": encode(edit((*NORM, 4, 0), 3)),
        "offset-unaligned": encode(edit((*NORM, 4, 1), 4100)),
        "offset-past-chunk": encode(edit((*NORM, 4, 1), 16384)),
        "length-mismatch": encode(edit((*NORM, 4, 2), 260)),
        "past-chunks": encode(edit((3, 1, 1, 0, 4), [2, 0, 576])),
        "location-short": encode(edit((*NORM, 4), [1, 4096])),
        "encoding-index": encode(edit((*Q, 3), 8)),
        "encoding-unused": encode(edit((2, 7), ["fp8.rowwise", {"axis": 1, "scale_dtype": "F32"}])),
        "encoding-repeated": encode(edit((2, 6), ENCODINGS[0])),
        "encoding-first-use": encode(edit((*Q, 3), 2)),
        "scheme-empty": encode(edit((2, 0, 0), "")),
        "params-array": encode(edit((2, 0, 1), [])),
        "plain-with-parts": encode(edit((*NORM, 4), [part("", "F32", [64], 1, 4096, 256)])),
        "encoded-no-parts": encode(edit((*Q, 4), [])),
        "encoded-location": encode(edit((*Q, 4), [0, 0, 4096])),
        "component-slash": encode(edit((3, 1, 0), "va/e")),
        "component-dotdot": encode(edit((3, 1, 0), "..")),
        "component-empty": encode(edit((3, 1, 0), "")),
        "component-repeated": encode(edit((3, 1, 0), "transformer")),
        "key-empty": encode(edit((*NORM, 0), "")),
        "key-repeated": encode(edit((*NORM, 0), "blocks.0.ff.gate.weight")),
        "role-empty": encode(edit((*Q, 4, 1, 0), "")),
        "role-repeated": encode(edit((*Q, 4, 1, 0), "weight")),
        "dtype-empty": encode(edit((*NORM, 1), "")),
        "dtype-code": encode(edit((*NORM, 1), 17)),
        "shape-negative": encode(edit((*NORM, 2), [-64])),
        "shape-bool": encode(edit((*NORM, 2), [True])),
        "tensor-short": encode(edit((*NORM, 4), None)),
        "metadata-array": encode([*edit((4,), [])[:5], 0]),  # six fields: not the pre-cut shape
    }
    return out


def check_cases(path: Path) -> list:
    """A family's decode cases: each names its component and key once per lane, holds its parts'
    (or a plain tensor's) bits, and expects bytes or a refusal code."""
    cases = json.loads(path.read_text())["cases"]
    seen = set()
    for c in cases:
        where = f"{path.name}: {c.get('component')}/{c.get('key')}"
        ident = (c.get("lane", "all"), c["component"], c["key"])
        assert ident not in seen, f"{where} is listed twice"
        seen.add(ident)
        assert ("expect" in c) != ("refuse" in c), f"{where}: one of expect and refuse"
        assert ("encoding" in c) == ("parts" in c) != ("bits" in c), f"{where}: parts with an encoding, else bits"
        for p in c.get("parts", []):
            assert {"role", "dtype", "shape", "bits"} <= p.keys(), f"{where}: part {p.get('role')}"
    path.write_text('{"cases":[\n' + ",\n".join(json.dumps(c, sort_keys=True, separators=(",", ":")) for c in cases)
                    + "\n]}\n")
    return cases


def main():
    green, red = HERE / "vectors" / "green", HERE / "vectors" / "red"
    for path in [*green.glob("header-*"), *red.glob("header-*")]:
        if not path.name.startswith("header-pre-cut"):
            path.unlink()
    for name, projection in GREEN.items():
        (green / f"header-{name}.cbor").write_bytes(encode(projection))
        (green / f"header-{name}.json").write_text(json.dumps(projection, ensure_ascii=False) + "\n")
    for name, data in reds().items():
        (red / f"header-{name}.cbor").write_bytes(data)
    print(f"{len(GREEN)} green and {len(reds())} red header vectors")
    for path in sorted((HERE / "vectors" / "cases").glob("*.json")):
        print(f"{path.stem}: {len(check_cases(path))} decode cases")


if __name__ == "__main__":
    main()
