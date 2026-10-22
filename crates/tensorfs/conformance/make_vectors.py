#!/usr/bin/env python3
"""Writes the header vectors: vectors/green/header-*.{cbor,json} from the projections below, and
vectors/red/header-*.cbor, each a green's twin or a single defect; the repo documents
vectors/{green,red}/repo-*.json and vectors/resolve.json, revisions resolved to commits. Then checks every decode-case
file, vectors/cases/<family>.json (its cases come from the family's producer, mine/<family>.py),
and rewrites it in its one form: one case per line, keys sorted. Deterministic; run from anywhere:

    uv run --no-project --with numpy --with cbor2==6.1.4 --with packaging==25.0 python -I conformance/make_vectors.py

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
TREE, REPO = "cozytensors.tree/1", "cozytensors.repo/1"


def digest(label: str) -> dict:
    return {"hex": hashlib.sha256(label.encode()).hexdigest()}


ENCODINGS = [
    ["fp8.rowwise", {}],
    ["fp8.tensorwise", {}],
    ["int8.convrot", {"rotation": "kron_regular_h4", "rotation_size": 256}],
    ["int8.rowwise", {}],
    ["mxfp8", {"block_size": 32, "scale_layout": "row_major"}],
    ["nvfp4", {"block_size": 16, "global_scale": "multiply", "nibble_order": "low_first",
               "scale_layout": "row_major"}],
    ["int8.tensorwise", {}],
]


def part(role, dtype, shape, chunk, offset, length):
    return [role, dtype, shape, [chunk, offset, length]]


PIPELINE = [["diffusers", "0.30.0", "FluxPipeline"], {
    "_class_name": "FluxPipeline", "_diffusers_version": "0.30.0", "scheduler": [None, None],
    "tokenizer": ["transformers", "CLIPTokenizer"], "transformer": ["diffusers", "FluxTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKL"]}]
TOKENIZER = ["tokenizer", ["transformers", "4.44.0", "CLIPTokenizer"],
             {"clean_up_tokenization_spaces": False, "do_lower_case": True, "model_max_length": 77,
              "pad_token": "<|endoftext|>"},
             [],
             {"merges.txt": [digest("merges"), 524619], "special_tokens_map.json": [digest("special"), 588],
              "vocab.json": [digest("vocab"), 1059962]},
             "tokenizer_config.json"]

BASIC = [
    TAG,
    [[digest("chunk 0"), 65536], [digest("chunk 1"), 12292], [digest("chunk 2"), 288]],
    ENCODINGS,
    PIPELINE,
    [
        ["transformer", ["diffusers", "0.30.0", "FluxTransformer2DModel"],
         {"_class_name": "FluxTransformer2DModel", "axes_dims_rope": [16, 56, 56], "eps": 1e-06,
          "guidance_embeds": True, "out_channels": None, "patch_size": 1, "ratio": 100000.0, "theta": 10000.0}, [
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
        ], {}],
        ["vae", ["diffusers", "0.30.0", "AutoencoderKL"],
         {"_class_name": "AutoencoderKL", "force_upcast": True, "latents_mean": None, "scaling_factor": 0.3611,
          "shift_factor": 0.1159},
         [["decoder.conv_in.weight", "F16", [4, 4, 3, 3], 0, [2, 0, 288]]], {}],
        TOKENIZER,
    ],
    {"n": 1, "ok": True, "name": "basic", "tags": ["a", "b"]},
]


UNKNOWN = ["", "", ""]


def plain(key, dtype, shape, chunk, offset, length):
    return [key, dtype, shape, 0, [chunk, offset, length]]


def one(tensors, chunks, encodings=(), metadata=None):
    """No pipeline; one component of unknown class with no config."""
    return [TAG, chunks, list(encodings), [], [["unet", UNKNOWN, {}, tensors, {}]], metadata or {}]


A = plain("a.weight", "F32", [2], 0, 0, 8)
B = plain("b.weight", "F32", [2], 0, 4096, 8)
SIZES = {"F4": [2], "F6_E2M3": [4], "F6_E3M2": [4]}
DTYPES = ["BOOL", "F4", "F6_E2M3", "F6_E3M2", "U8", "I8", "F8_E5M2", "F8_E4M3", "F8_E8M0", "F8_E4M3FNUZ",
          "F8_E5M2FNUZ", "I16", "U16", "F16", "BF16", "I32", "U32", "F32", "C64", "F64", "I64", "U64"]
WIDTH = {"F4": 1, "F6_E2M3": 3, "F6_E3M2": 3, "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4,
         "F32": 4, "C64": 8, "F64": 8, "I64": 8, "U64": 8}  # the rest: one byte


def appended():
    """Later fields. A component's follow its config_file, which is then present (`config.json`
    for the default)."""
    h = copy.deepcopy(BASIC)
    h[0] = TAG + ".7"
    h.append({"future": [1, {"hex": "00ff"}]})
    h[1][0].append(True)
    h[2][0].append("note")
    h[3].append([])
    h[4][2].append({"k": []})
    h[4][0] += ["config.json", 0]
    h[4][2][4]["vocab.json"].append(0)
    t = h[4][0][3][0]
    t.append(-1)
    t[4][0].append(0)
    t[4][0][3].append({"hex": ""})
    h[4][0][3][6][4].append([])
    return h


CONFIG = {
    "f16": 0.5, "f16-max": 65504.0, "f16-subnormal": 2.0**-24, "f32": 100000.0, "f32-subnormal": 2.0**-149,
    "f64": 0.1, "f64-subnormal": 5e-324, "f64-max": 1.7976931348623157e308, "negative-zero": -0.0, "one": 1.0,
    "int": 1, "null": None, "u64": 2**64 - 1, "i64": -(2**63), "text": "x", "bool": False,
    "nested": [0.00085, None, {"beta_end": 0.012, "betas": [[]]}],
}


GREEN = {
    "basic": BASIC,
    "empty": [TAG, [], [], [], [], {}],
    "configs": [TAG, [], [], [UNKNOWN, CONFIG],
                [["scheduler", ["diffusers", "0.30.0", "EulerDiscreteScheduler"], CONFIG, [], {},
                  "scheduler_config.json"]], {}],
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
    "values": [TAG, [], [], [], [], {
        "": "", "a": 0, "b": 23, "c": 24, "d": 255, "e": 256, "f": 65535, "g": 65536, "h": 2**32 - 1,
        "i": 2**32, "j": 2**64 - 1, "k": -1, "l": -24, "m": -25, "n": -(2**63), "aa": True, "ab": False,
        "ü→😀": "ü→😀", "x" * 23: "23", "x" * 24: "24", "list": [[], [[]], {"z": {"y": [1, -1]}}]}],
    "dtypes": one([plain(f"t{i}", d, SIZES.get(d, [1]), 0, 4096 * i, WIDTH.get(d, 1))
                   for i, d in enumerate(DTYPES)], [[digest("dtypes"), 4096 * 21 + 8]]),
}


DELETE = object()


def edit(path, value):
    """BASIC with the item at `path` (indices and keys into the tree) replaced, appended or
    (DELETE) deleted."""
    h = copy.deepcopy(BASIC)
    *parents, last = path
    node = h
    for i in parents:
        node = node[i]
    if value is DELETE:
        del node[last]
    elif type(node) is list and last == len(node):
        node.append(value)
    else:
        node[last] = value
    return h


TENSORS = (4, 0, 3)  # BASIC[4][0][3]: the transformer's tensors
Q = (*TENSORS, 0)  # its first tensor
NORM = (*TENSORS, 6)
VAE = (4, 1)
ASSETS = (4, 2, 4)  # the tokenizer's
GREEN["class-unversioned"] = edit((4, 2, 1, 1), "")  # a known library whose version nothing records


def swap_encodings():
    h = copy.deepcopy(BASIC)
    h[2][0], h[2][1] = h[2][1], h[2][0]
    tensors = h[4][0][3]
    tensors[0][3], tensors[1][3] = 2, 1
    return h


def reordered(m: dict) -> bytes:
    """BASIC with the map `m` (found once) written with its keys in reverse order."""
    canonical = cbor2.dumps(unproject(m), canonical=True)
    raw = bytes([0xa0 + len(m)]) + b"".join(cbor2.dumps(k) + cbor2.dumps(unproject(m[k]), canonical=True)
                                            for k in sorted(m, reverse=True))
    assert raw != canonical and len(raw) == len(canonical)
    return swap(encode(BASIC), canonical, raw)


def swap(data: bytes, old: bytes, new: bytes) -> bytes:
    assert data.count(old) == 1, old
    return data.replace(old, new)


def reds() -> dict[str, bytes]:
    good = encode(BASIC)
    meta = cbor2.dumps(unproject(BASIC[5]), canonical=True)
    assert meta == bytes.fromhex("a4616e01626f6bf5646e616d6565626173696364746167738261616162")
    out = {
        # Twins: BASIC's content in bytes other than its one encoding.
        "twin-nonpreferred-uint": swap(good, b"\x61n\x01", b"\x61n\x18\x01"),
        "twin-nonpreferred-uint32": swap(good, b"\x1a\x00\x01\x00\x00", b"\x1b" + bytes(5) + b"\x01\x00\x00"),
        "twin-nonpreferred-array": b"\x98\x06" + good[1:],
        "twin-nonpreferred-text": swap(good, b"\x6dcozytensors/1", b"\x78\x0dcozytensors/1"),
        "twin-indefinite-array": b"\x9f" + good[1:] + b"\xff",
        "twin-indefinite-map": swap(good, meta, b"\xbf" + meta[1:] + b"\xff"),
        "twin-indefinite-text": swap(good, b"\x65basic", b"\x7f\x63bas\x62ic\xff"),
        "twin-map-order": swap(good, b"\x61n\x01\x62ok\xf5", b"\x62ok\xf5\x61n\x01"),
        "twin-params-order": reordered(ENCODINGS[5][1]),
        "twin-assets-order": reordered(TOKENIZER[4]),
        "twin-config-order": reordered(PIPELINE[1]),
        "twin-float-single": swap(good, b"\xf9\x70\xe2", b"\xfa\x46\x1c\x40\x00"),
        "twin-float-double": swap(good, b"\xf9\x70\xe2", b"\xfb\x40\xc3\x88" + bytes(5)),
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
        "float-later-field": encode([*BASIC, 1.5]),
        "float-asset-length": encode(edit((*ASSETS, "vocab.json", 1), 1059962.0)),
        "null-params": encode(edit((2, 2, 1, "rotation"), None)),
        "null-class": encode(edit((*VAE, 1), None)),
        "nan": swap(good, b"\xf9\x70\xe2", b"\xf9\x7e\x00"),
        "infinity": swap(good, b"\xf9\x70\xe2", b"\xf9\x7c\x00"),
        "nan-single": swap(good, b"\xf9\x70\xe2", b"\xfa\x7f\xc0\x00\x00"),
        "config-tag": swap(good, b"\x6cforce_upcast\xf5", b"\x6cforce_upcast\xc1\xf5"),
        "config-undefined": swap(good, b"\x6cforce_upcast\xf5", b"\x6cforce_upcast\xf7"),
        "config-bytes": encode(edit((*VAE, 2, "force_upcast"), {"hex": "01"})),
        "config-array": encode(edit((*VAE, 2), [])),
        "config-int-key": swap(good, b"\x6cforce_upcast\xf5", b"\x01\xf5"),
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
        "past-chunks": encode(edit((*VAE, 3, 0, 4), [2, 0, 576])),
        "location-short": encode(edit((*NORM, 4), [1, 4096])),
        "encoding-index": encode(edit((*Q, 3), 8)),
        "encoding-unused": encode(edit((2, 7), ["fp8.columnwise", {}])),
        "encoding-repeated": encode(edit((2, 6), ENCODINGS[0])),
        "encoding-first-use": encode(edit((*Q, 3), 2)),
        "scheme-empty": encode(edit((2, 0, 0), "")),
        "params-array": encode(edit((2, 0, 1), [])),
        "plain-with-parts": encode(edit((*NORM, 4), [part("", "F32", [64], 1, 4096, 256)])),
        "pipeline-one-field": encode(edit((3,), PIPELINE[:1])),
        "pipeline-not-array": encode(edit((3,), {})),
        "pipeline-class-int": encode(edit((3, 0), 7)),
        "pipeline-config-array": encode(edit((3, 1), [])),
        "component-short": encode(edit((*VAE, 4), DELETE)),
        "component-class-int": encode(edit((*VAE, 1), 7)),
        "class-text": encode(edit((*VAE, 1), "diffusers.AutoencoderKL")),
        "class-short": encode(edit((*VAE, 1), ["diffusers", "AutoencoderKL"])),
        "class-long": encode(edit((*VAE, 1), ["diffusers", "0.30.0", "AutoencoderKL", ""])),
        "class-name-int": encode(edit((*VAE, 1, 2), 7)),
        "class-no-library": encode(edit((*VAE, 1, 0), "")),
        "class-version-only": encode(edit((*VAE, 1), ["", "0.30.0", ""])),
        "pipeline-class-no-library": encode(edit((3, 0, 0), "")),
        "config-file-default": encode(edit((4, 2, 5), "config.json")),
        "config-file-int": encode(edit((4, 2, 5), 7)),
        "config-file-dotdot": encode(edit((4, 2, 5), "../tokenizer_config.json")),
        "asset-path": encode(edit((*ASSETS, "a//b"), TOKENIZER[4]["vocab.json"])),
        "asset-is-config": encode(edit((*ASSETS, "tokenizer_config.json"), TOKENIZER[4]["vocab.json"])),
        "asset-digest-31": encode(edit((*ASSETS, "vocab.json", 0), {"hex": "00" * 31})),
        "asset-array": encode(edit(ASSETS, [])),
        "encoded-no-parts": encode(edit((*Q, 4), [])),
        "encoded-location": encode(edit((*Q, 4), [0, 0, 4096])),
        "component-slash": encode(edit((*VAE, 0), "va/e")),
        "component-dotdot": encode(edit((*VAE, 0), "..")),
        "component-empty": encode(edit((*VAE, 0), "")),
        "component-repeated": encode(edit((*VAE, 0), "transformer")),
        "key-empty": encode(edit((*NORM, 0), "")),
        "key-repeated": encode(edit((*NORM, 0), "blocks.0.ff.gate.weight")),
        "role-empty": encode(edit((*Q, 4, 1, 0), "")),
        "role-repeated": encode(edit((*Q, 4, 1, 0), "weight")),
        "dtype-empty": encode(edit((*NORM, 1), "")),
        "dtype-code": encode(edit((*NORM, 1), 17)),
        "shape-negative": encode(edit((*NORM, 2), [-64])),
        "shape-bool": encode(edit((*NORM, 2), [True])),
        "tensor-short": encode(edit((*NORM, 4), DELETE)),
        "metadata-array": encode(edit((5,), [])),
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


def canonical(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def ref(data: bytes) -> dict:
    return {"length": len(data), "sha256": hashlib.sha256(data).hexdigest()}


# The example trees' refs: tree-unnamed.json and tree-basic.json.
T1 = ref((HERE / "vectors" / "green" / "tree-unnamed.json").read_bytes())
T2 = ref((HERE / "vectors" / "green" / "tree-basic.json").read_bytes())


def repo(*tags, name="acme/tiny", **extra) -> dict:
    return {"format": REPO, "name": name, "tags": [dict(t) for t in tags], **extra}


def tag(name, commit=T2, public=None, yanked=None, **extra) -> dict:
    flags = {k: v for k, v in (("public", public), ("yanked", yanked)) if v is not None}
    return {"name": name, "commit": commit, **flags, **extra}


REPO_BASIC = repo(tag("1.0", T1, public=True), tag("1.1", public=True), tag("main"))
REPOS_GREEN = {
    "basic": REPO_BASIC,
    "mixed-case": {**REPO_BASIC, "name": "Acme/Tiny-SDXL"},
    "no-tags": repo(),
    "minor-version": {**REPO_BASIC, "format": REPO + ".3"},
    "unknown-fields": repo(tag("1.0", T1, public=True, future=1), tag("main"), future={"x": [1, True, "ok"]}),
    "yanked-private": repo(tag("1.0", T1, public=True), tag("1.1", public=True, yanked=True), tag("run-42", T1),
                           tag("main", yanked=True), tag("2.0")),
    "private-names": repo(tag("nightly"), tag("Run_2026.10.20 draft"), tag("Main"), tag("main"), tag("sha256")),
    "pep440-forms": repo(*(tag(v, public=True) for v in ("1.0rc1", "1.0.post1", "2!0.1", "1.0+local.7", "v1.2",
                                                         "1.0.dev3", "1.0.0", "01.2"))),
    "shared-commit": repo(tag("1.0", public=True), tag("main"), tag("1.0.1", public=True)),
    "explicit-false": repo(tag("1.0", T1, public=False, yanked=False)),
    "public-any-name": repo(tag("1.0.0", public=True), tag("1.0.0-audit.1", public=True), tag("main", public=True)),
}
REPOS_RED = {
    "bad-name": {**REPO_BASIC, "name": "acme"},
    "deep-name": {**REPO_BASIC, "name": "acme/tiny/x"},
    "old-kind": {**REPO_BASIC, "format": "tensorfs.repo/1"},
    "releases": {"format": REPO, "name": "acme/tiny", "releases": [{"tree": T1, "version": "1.0"}]},
    "duplicate-tag": repo(tag("main", T1), tag("main")),
    "duplicate-public-private": repo(tag("1.0", T1, public=True), tag("1.0")),
    "empty-tag": repo(tag("")),
    "dot-tag": repo(tag("..")),
    "slash-tag": repo(tag("runs/42")),
    "digest-tag": repo(tag("sha256:" + T1["sha256"])),
    "no-commit": repo({"name": "main"}),
    "public-not-bool": repo(tag("1.0", public="true")),
    "yanked-not-bool": repo(tag("1.0", public=True, yanked=1)),
    "tags-not-list": {**repo(), "tags": {"main": T1}},
    "bad-commit": repo(tag("main", {"length": 1, "sha256": T1["sha256"][:63]})),
}


def header_ref(label: str) -> dict:
    return {"length": 1000, "sha256": hashlib.sha256(label.encode()).hexdigest()}


def resolution() -> dict:
    """A repo whose tags cover every §7 rule, its commits' trees, and revisions resolved."""
    trees = [canonical({"format": TREE, "files": {f"model.{v}.cozytensors": header_ref(h) for v, h in files.items()}})
             for files in ({"bf16": "h1"}, {"bf16": "h2", "fp8": "h3"}, {"bf16": "h4"}, {"bf16": "h2"},
                           {"bf16": "h5", "fp8": "h3"}, {"bf16": "h6", "fp8": "h3"})]
    c1, c2, c3, c4, c5, c6 = (ref(t) for t in trees)
    doc = repo(tag("1.0", c1, public=True), tag("1.9", c2, public=True), tag("2.0", c3, public=True, yanked=True),
               tag("1.10", c4, public=True), tag("main", c5), tag("exp", c6))
    h = lambda label: "sha256:" + header_ref(label)["sha256"]  # noqa: E731
    cases = [
        {"why": "the highest public, non-yanked version (1.10 > 1.9)", "commit": c4},
        {"why": "a yanked tag by name", "revision": "2.0", "commit": c3},
        {"why": "a private tag by name", "revision": "main", "commit": c5},
        {"why": "a tag's commit", "revision": "sha256:" + c2["sha256"], "commit": c2},
        {"why": "a private tag's commit", "revision": "sha256:" + c5["sha256"], "commit": c5},
        {"why": "a header the default's commit cites", "revision": h("h2"), "commit": c4, "variant": "model.bf16.cozytensors"},
        {"why": "a header the default does not cite: the last-created tag's commit", "revision": h("h3"),
         "commit": c6, "variant": "model.fp8.cozytensors"},
        {"why": "a header one commit cites", "revision": h("h1"), "commit": c1, "variant": "model.bf16.cozytensors"},
        {"why": "a header no commit cites", "revision": h("h0")},
        {"why": "a tag name is exact, not a version match", "revision": "1.9.0"},
        {"why": "no such tag", "revision": "nope"},
    ]
    # The unversioned default alone, over tag lists: finals before pre-releases, then PEP 440 order.
    def default(why, tags, expect=None):
        out = {"why": why, "repo": repo(*(tag(name, **flags) for name, flags in tags))}
        return {**out, "default": expect} if expect else out
    public, yanked = {"public": True}, {"public": True, "yanked": True}
    defaults = [
        default("a final beats a newer pre-release", [("1.0", public), ("2.0rc1", public)], "1.0"),
        default("a final beats a newer dev release", [("1.0", public), ("1.1.dev1", public)], "1.0"),
        default("a post-release is final", [("1.0", public), ("1.0.post1", public)], "1.0.post1"),
        default("a post-release's dev release is a pre-release", [("1.0", public), ("1.0.post1.dev1", public)], "1.0"),
        default("only pre-releases: the highest", [("1.0a1", public), ("1.0rc1", public), ("1.0b2", public),
                                                    ("0.9.dev3", public)], "1.0rc1"),
        default("equal versions: the later created", [("1.0", public), ("1.0.0", public)], "1.0.0"),
        default("equal versions, the other way round", [("1.0.0", public), ("1.0", public)], "1.0"),
        default("a yanked final falls back to an older final, not a newer pre-release",
                [("1.0", public), ("2.0", yanked), ("2.1rc1", public)], "1.0"),
        default("a local version is final and orders after its public one", [("1.0+local.1", public), ("1.0", public)],
                "1.0+local.1"),
        default("an epoch outranks a larger release", [("2!0.1", public), ("10.0", public)], "2!0.1"),
        default("private tags are not candidates", [("1.0", public), ("2.0", {})], "1.0"),
        default("a public name that is not a version is never the default", [("1.0.0", public), ("1.0.0-audit.1", public)],
                "1.0.0"),
        default("no public, non-yanked tag: none", [("main", {}), ("1.0", yanked)]),
    ]
    return {"repo": doc, "trees": [json.loads(t) for t in trees], "cases": cases, "defaults": defaults}


def main():
    green, red = HERE / "vectors" / "green", HERE / "vectors" / "red"
    for path in [*green.glob("repo-*"), *red.glob("repo-*")]:
        path.unlink()
    for verdict, docs in ((green, REPOS_GREEN), (red, REPOS_RED)):
        for name, doc in docs.items():
            (verdict / f"repo-{name}.json").write_bytes(canonical(doc))
    (HERE / "../spec/examples/repo-basic.json").write_bytes(canonical(REPO_BASIC))
    doc = resolution()
    rows = lambda key: ",\n".join(canonical(c).decode() for c in doc[key])  # noqa: E731
    (HERE / "vectors" / "resolve.json").write_text(
        '{"repo":' + canonical(doc["repo"]).decode() + ',\n"trees":[\n' + rows("trees") + '],\n"cases":[\n'
        + rows("cases") + '],\n"defaults":[\n' + rows("defaults") + "\n]}\n")
    print(f"{len(REPOS_GREEN)} green and {len(REPOS_RED)} red repo vectors, {len(doc['cases'])} resolutions, "
          f"{len(doc['defaults'])} defaults")
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
