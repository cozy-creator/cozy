#!/usr/bin/env python3
"""An independent cozytensors/1 reader and header encoder, written from the spec: Python, numpy and cbor2.

    python reader.py <vectors-dir>     run the conformance vectors (exit 1 on any failure)

As a library: `Store(root).checkpoint("org/name", lane="x")` -> (manifest, header),
`read(store, header, component, key)` -> the tensor's logical bytes, `parse_header(bytes)` and
`encode(projection)`: a header's diagnostic projection (the CBOR tree as JSON, byte strings as
{"hex": ...}) to its canonical bytes.
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import sys
from pathlib import Path

import cbor2
import numpy as np

HEADER, MANIFEST, REPO = "cozytensors/1", "tensorfs.manifest/1", "tensorfs.repo/1"
BITS = {
    "BOOL": 8, "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6, "U8": 8, "I8": 8, "F8_E5M2": 8, "F8_E4M3": 8,
    "F8_E8M0": 8, "F8_E4M3FNUZ": 8, "F8_E5M2FNUZ": 8, "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
    "I32": 32, "U32": 32, "F32": 32, "C64": 64, "F64": 64, "I64": 64, "U64": 64,
}
PRE_CUT = "pre-cut cozytensors header (old layout); re-import or migrate"


class Refused(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


def invalid(detail: str) -> Refused:
    return Refused("invalid", detail)


def need(ok: bool, detail: str):
    if not ok:
        raise invalid(detail)


def check_format(tag, expected: str):
    need(isinstance(tag, str), "no format tag")
    name, _, version = tag.partition("/")
    major, dot, minor = version.partition(".")
    if f"{name}/{major}" != expected or (dot and not (minor.isascii() and minor.isdigit())):
        raise Refused("format", f"{tag!r} is not {expected}")


# ---------------------------------------------------------------- JSON documents


def load(data: bytes, expected: str) -> dict:
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise invalid(f"duplicate key {k!r}")
            out[k] = v
        return out

    def no_float(text):
        raise invalid(f"{text} is not an integer")

    def integer(text):
        v = int(text)
        if not -(2**63) <= v < 2**64:
            raise invalid(f"{text} is outside i64/u64")
        return v

    try:
        doc = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_float=no_float,
                         parse_int=integer, parse_constant=no_float)
    except (ValueError, UnicodeDecodeError) as e:
        raise invalid(str(e)) from None
    need(isinstance(doc, dict), "the document is not an object")
    check_format(doc.get("format"), expected)
    no_null(doc)
    return doc


def no_null(v):
    if v is None:
        raise invalid("null")
    for x in v.values() if isinstance(v, dict) else v if isinstance(v, list) else ():
        no_null(x)


def uint(v) -> bool:
    return type(v) is int and 0 <= v < 2**64


def field(doc: dict, name: str, kind, optional=False):
    if name not in doc:
        need(optional, f"missing {name!r}")
        return kind()
    v = doc[name]
    need(uint(v) if kind is int else isinstance(v, kind), f"{name!r} is not a {kind.__name__}")
    return v


def segment(s) -> bool:
    return isinstance(s, str) and s not in ("", ".", "..") and not any(c in s for c in "/\\\0")


def object_ref(v) -> dict:
    need(isinstance(v, dict), "an object ref is not an object")
    sha, length = field(v, "sha256", str), field(v, "length", int)
    need(len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), f"bad sha256 {sha!r}")
    return {"sha256": sha, "length": length}


def byte_len(dtype, shape) -> int | None:
    """Bytes of a known dtype and shape; None for an unknown dtype (its length stands)."""
    need(type(dtype) is str and dtype != "", f"bad dtype {dtype!r}")
    need(type(shape) is list and all(uint(x) for x in shape), f"bad shape {shape!r}")
    if dtype not in BITS:
        return None
    bits = BITS[dtype]
    for x in shape:
        bits *= x
        need(bits < 2**64, f"{dtype} {shape} overflows")
    need(bits % 8 == 0, f"{dtype} {shape} is not whole bytes")
    return bits // 8


def parse_manifest(data: bytes) -> dict:
    doc = load(data, MANIFEST)
    files = field(doc, "files", dict)
    for path in files:
        need(all(segment(s) for s in path.split("/")), f"bad path {path!r}")
    meta = field(doc, "metadata", dict, optional=True)
    return {"files": {p: object_ref(r) for p, r in files.items()}, "metadata": meta}


def parse_repo(data: bytes) -> dict:
    doc = load(data, REPO)
    name = field(doc, "name", str)
    org, _, rest = name.partition("/")
    need(segment(org) and segment(rest), f"bad repo name {name!r}")
    releases, seen = [], set()
    for r in field(doc, "releases", list):
        need(isinstance(r, dict), "a release is not an object")
        version = field(r, "version", str)
        need(version != "" and version not in seen, f"empty or repeated version {version!r}")
        seen.add(version)
        lanes = field(r, "lanes", dict)
        need("" not in lanes, "empty lane name")
        releases.append({"version": version, "lanes": {k: object_ref(v) for k, v in lanes.items()},
                         "yanked": field(r, "yanked", bool, optional=True),
                         "revision": field(r, "revision", int, optional=True)})
    checkpoints = [object_ref(c) for c in field(doc, "checkpoints", list, optional=True)]
    return {"name": name, "releases": releases, "checkpoints": checkpoints}


def release(repo: dict, version: str | None) -> dict:
    """`version`, yanked or not; with None, the newest release that is not yanked."""
    if version is None:
        return next(r for r in reversed(repo["releases"]) if not r["yanked"])
    return next(r for r in repo["releases"] if r["version"] == version)


# ---------------------------------------------------------------- the CBOR header


def strict(data: bytes):
    """The CBOR item of a header, refusing anything but its one deterministic encoding (spec §9):
    the canonical re-encoding must reproduce the bytes, which rules out non-preferred heads,
    indefinite lengths, unordered or repeated map keys and trailing bytes; then only integers in
    [-2^63, 2^64), byte and text strings, arrays, maps with text keys, true and false remain."""
    try:
        item = cbor2.loads(data)
        same = cbor2.dumps(item, canonical=True) == data
    except Exception as e:  # cbor2 raises several kinds on malformed bytes
        raise invalid(f"not CBOR: {e}") from None
    need(same, "not the deterministic encoding (RFC 8949 §4.2.1)")
    stack = [item]
    while stack:
        v = stack.pop()
        t = type(v)
        if t is list:
            stack.extend(v)
        elif t is dict:
            need(all(type(k) is str for k in v), "a map key that is not text")
            stack.extend(v.values())
        elif t is int:
            need(-(2**63) <= v < 2**64, f"{v} is outside [-2^63, 2^64)")
        else:
            need(t in (str, bytes, bool), f"a {t.__name__} item")
    return item


def record(v, n: int, what: str) -> list:
    """An array of at least `n` fields; fields after them are a later version's and are skipped."""
    need(type(v) is list and len(v) >= n, f"{what} is not an array of at least {n}")
    return v


def location(v, dtype, shape, chunks, rooms, what) -> dict:
    chunk, offset, length = record(v, 3, f"{what} location")[:3]
    need(uint(chunk) and uint(offset) and uint(length), f"{what}: location is not three uints")
    size = byte_len(dtype, shape)
    need(size in (None, length), f"{what}: length disagrees with dtype and shape")
    if length:
        need(chunk < len(chunks), f"{what}: chunk {chunk} of {len(chunks)}")
        need(offset % 4096 == 0, f"{what}: offset {offset} is not 4 KiB aligned")
        need(offset < chunks[chunk]["length"] and offset + length <= rooms[chunk],
             f"{what}: range runs past the chunks")
    return {"chunk": chunk, "offset": offset, "length": length}


def parse_header(data: bytes) -> dict:
    """Components, tensors and parts as dicts, which keep the header's (construction) order."""
    if data[:1] == b"{":
        raise Refused("pre_cut", PRE_CUT)
    root = strict(data)
    try:
        return header(root)
    except Refused:
        # The first CBOR layout: exactly five fields, the fifth the components array.
        if type(root) is list and len(root) == 5 and type(root[4]) is list:
            raise Refused("pre_cut", PRE_CUT) from None
        raise


def header(root) -> dict:
    tag, chunk_rows, encoding_rows, component_rows, meta = record(root, 5, "the header")[:5]
    check_format(tag, HEADER)
    need(type(chunk_rows) is list, "chunks is not an array")
    chunks = []
    for c in chunk_rows:
        sha, length = record(c, 2, "a chunk")[:2]
        need(type(sha) is bytes and len(sha) == 32 and uint(length), "a chunk is not [bstr .size 32, uint]")
        chunks.append({"sha256": sha.hex(), "length": length})
    rooms = list(itertools.accumulate((c["length"] for c in reversed(chunks))))[::-1]  # bytes of chunks[i:]
    need(type(encoding_rows) is list, "encodings is not an array")
    encodings, seen = [], set()
    for e in encoding_rows:
        scheme, params = record(e, 2, "an encoding")[:2]
        need(type(scheme) is str and scheme != "" and type(params) is dict, "an encoding is not [scheme, params]")
        identity = cbor2.dumps([scheme, params], canonical=True)
        need(identity not in seen, f"encoding {scheme} is listed twice")
        seen.add(identity)
        encodings.append({"scheme": scheme, "params": params})
    need(type(component_rows) is list, "components is not an array")
    components, cited = {}, 0
    for c in component_rows:
        name, tensor_rows = record(c, 2, "a component")[:2]
        need(segment(name) and name not in components, f"bad or repeated component name {name!r}")
        need(type(tensor_rows) is list, f"{name}: tensors is not an array")
        tensors = {}
        for t in tensor_rows:
            key, dtype, shape, index, body = record(t, 5, f"a {name} tensor")[:5]
            need(type(key) is str and key != "" and key not in tensors, f"{name}: bad or repeated key {key!r}")
            byte_len(dtype, shape)
            what = f"{name}/{key}"
            need(uint(index), f"{what}: the encoding index is not a uint")
            out = {"dtype": dtype, "shape": shape}
            if index == 0:
                out["data"] = location(body, dtype, shape, chunks, rooms, what)
            else:
                need(index <= len(encodings) and index - 1 <= cited, f"{what}: encoding {index} out of range or order")
                cited = max(cited, index)
                need(type(body) is list and body, f"{what}: no parts")
                out["encoding"], out["parts"] = encodings[index - 1], {}
                for p in body:
                    role, pd, ps, loc = record(p, 4, f"{what} part")[:4]
                    need(type(role) is str and role != "" and role not in out["parts"], f"{what}: bad or repeated role")
                    byte_len(pd, ps)
                    loc = location(loc, pd, ps, chunks, rooms, f"{what} part {role}")
                    out["parts"][role] = {"dtype": pd, "shape": ps, **loc}
            tensors[key] = out
        components[name] = tensors
    need(cited == len(encodings), f"{cited} of {len(encodings)} encodings are cited")
    need(type(meta) is dict, "metadata is not a map")
    for v in [meta, *(e["params"] for e in encodings)]:
        no_bytes(v)
    return {"chunks": chunks, "components": components, "metadata": meta}


def no_bytes(v):
    """Metadata and params hold values: no byte strings (only later fields may)."""
    need(type(v) is not bytes, "a byte string value")
    for x in v.values() if type(v) is dict else v if type(v) is list else ():
        no_bytes(x)


def project(item):
    """The diagnostic projection: the CBOR tree as JSON, byte strings as {"hex": ...}."""
    if type(item) is bytes:
        return {"hex": item.hex()}
    if type(item) is list:
        return [project(x) for x in item]
    if type(item) is dict:
        return {k: project(v) for k, v in item.items()}
    return item


def unproject(item):
    if type(item) is dict and list(item) == ["hex"]:
        return bytes.fromhex(item["hex"])
    if type(item) is list:
        return [unproject(x) for x in item]
    if type(item) is dict:
        return {k: unproject(v) for k, v in item.items()}
    return item


def encode(projection) -> bytes:
    """A header's canonical bytes from its projection."""
    return cbor2.dumps(unproject(projection), canonical=True)


# ------------------------------------------------------------------- store


def repo_key(name: str) -> str:
    """refs/<org>/<name>.json with ASCII A-Z lowered: repo names are case-insensitive."""
    return "refs/" + "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in name) + ".json"


class Store:
    """A store or static origin on disk: blobs/<aa>/<bb>/<sha256>, refs/<org>/<name>.json."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def blob(self, ref: dict) -> bytes:
        sha = ref["sha256"]
        data = (self.root / "blobs" / sha[:2] / sha[2:4] / sha).read_bytes()
        if len(data) != ref["length"] or hashlib.sha256(data).hexdigest() != sha:
            raise invalid(f"object {sha} does not match its ref")
        return data

    def checkpoint(self, name: str, version: str | None = None, lane: str = ""):
        repo = parse_repo((self.root / repo_key(name)).read_bytes())
        manifest = parse_manifest(self.blob(release(repo, version)["lanes"][lane]))
        return manifest, parse_header(self.blob(manifest["files"]["model.cozytensors"]))


def gather(store: Store, header: dict, loc: dict) -> bytes:
    out, chunk, start, left = b"", loc["chunk"], loc["offset"], loc["length"]
    while left:
        data = store.blob(header["chunks"][chunk])
        piece = data[start:start + left]
        out, chunk, start, left = out + piece, chunk + 1, 0, left - len(piece)
    return out


def read(store: Store, header: dict, component: str, key: str) -> bytes:
    t = header["components"][component][key]
    if "data" in t:
        return gather(store, header, t["data"])
    parts = {r: (p["dtype"], p["shape"], gather(store, header, p)) for r, p in t["parts"].items()}
    return decode(t["dtype"], t["shape"], t["encoding"], parts)


# ------------------------------------------------------------------ decode


def _e4m3() -> np.ndarray:
    out = np.empty(256, np.float32)
    for b in range(256):
        e, m = (b >> 3) & 15, b & 7
        v = np.nan if (e, m) == (15, 7) else (m * 2.0**-9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
        out[b] = -v if b & 0x80 else v
    return out


E4M3 = _e4m3()
E5M2 = (np.arange(256, dtype=np.uint16) << 8).view(np.float16).astype(np.float32)
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


E8M0 = np.array([2.0 ** (b - 127) if b < 255 else np.nan for b in range(256)], np.float32)


def values(dtype: str, data: bytes) -> np.ndarray:
    raw = np.frombuffer(data, np.uint8)
    if dtype == "F32":
        return np.frombuffer(data, "<f4").astype(np.float32)
    if dtype == "F4":
        return E2M1[np.stack([raw & 15, raw >> 4], -1).reshape(-1)]
    if dtype == "BF16":
        return (np.frombuffer(data, "<u2").astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(data, "<f2").astype(np.float32)
    return {"F8_E4M3": E4M3, "F8_E5M2": E5M2, "F8_E8M0": E8M0}[dtype][raw]


def to_dtype(dtype: str, x: np.ndarray) -> bytes:
    x = x.astype(np.float32).reshape(-1)
    nan = np.isnan(x)
    if dtype == "F32":
        bits = x.view(np.uint32).copy()
        bits[nan] = 0x7FC00000
        return bits.astype("<u4").tobytes()
    if dtype == "BF16":
        u = x.view(np.uint32).astype(np.uint64)
        bits = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        bits[nan] = 0x7FC0
        return bits.astype("<u2").tobytes()
    with np.errstate(over="ignore"):
        bits = x.astype(np.float16).view(np.uint16).copy()
    bits[nan] = 0x7E00
    return bits.astype("<u2").tobytes()


def hadamard(x: np.ndarray, k: int) -> np.ndarray:
    """Rows of x (int64, last axis 4**k) times H4 (x) ... (x) H4, unnormalized."""
    shape = x.shape
    x = x.reshape(-1, *([4] * k))
    for axis in range(1, k + 1):
        a, b, c, d = (np.take(x, i, axis=axis) for i in range(4))
        x = np.stack([a + b + c - d, a + b - c + d, a - b + c + d, b + c + d - a], axis=axis)
    return x.reshape(shape)


SCHEMES = {  # scheme: (required parts, optional parts, params)
    "fp8.tensorwise": ({"weight", "scale"}, {"input_scale"}, {"scale_dtype"}),
    "fp8.rowwise": ({"weight", "scale"}, set(), {"axis", "scale_dtype"}),
    "int8.tensorwise": ({"weight", "scale"}, {"input_scale"}, {"scale_dtype"}),
    "int8.rowwise": ({"weight", "scale"}, set(), {"axis", "scale_dtype"}),
    "int8.convrot": ({"weight", "scale"}, set(), {"rotation", "rotation_size", "scale_dtype"}),
    "mxfp8": ({"weight", "scale"}, set(), {"block_size", "scale_layout"}),
    "nvfp4": ({"weight", "scale", "scale_2"}, {"input_scale"},
              {"block_size", "global_scale", "nibble_order", "scale_layout"}),
}
FP8 = ("F8_E4M3", "F8_E5M2")


def blocked(rows: int, blocks: int) -> np.ndarray:
    """Where cuBLAS's 128x4-tiled layout stores block scale (i, b), for every i and b."""
    i, b = np.arange(rows)[:, None], np.arange(blocks)[None, :]
    tiles = -(-blocks // 4)
    return ((i // 128) * tiles + b // 4) * 512 + (i % 32) * 16 + (i % 128 // 32) * 4 + b % 4


def decode(dtype: str, shape: list, encoding: dict, parts: dict) -> bytes:
    with np.errstate(all="ignore"):  # NaN and infinity are values here, not errors
        return _decode(dtype, shape, encoding, parts)


def _decode(dtype: str, shape: list, encoding: dict, parts: dict) -> bytes:
    scheme, params = encoding["scheme"], encoding["params"]
    if scheme not in SCHEMES:
        raise Refused("unknown_scheme", scheme)

    def fail(detail):
        return Refused("decode", f"{scheme}: {detail}")

    required, optional, names = SCHEMES[scheme]
    if not required <= parts.keys() <= required | optional:
        raise fail(f"parts {sorted(parts)}")
    if not params.keys() <= names:  # params drive the decode: an unknown one is refused
        raise fail(f"params {sorted(params)}")

    def param(name, allowed):
        if params.get(name) not in allowed or type(params.get(name)) is bool:
            raise fail(f"param {name}={params.get(name)!r}")
        return params[name]

    if dtype not in ("F32", "BF16", "F16"):
        raise fail(f"logical dtype {dtype}")
    numel = int(np.prod(shape, dtype=np.int64))
    if scheme.endswith(".tensorwise"):
        rows, cols = 1, numel
    elif len(shape) >= 2:  # the last axis, and every leading axis as rows
        cols = shape[-1]
        rows = numel // cols if cols else 0
    else:
        raise fail(f"shape {shape}")

    def part(role, dtypes, want=None, count=None):
        pd, ps, data = parts[role]
        if pd not in dtypes or (want is not None and ps != list(want)) or (
                count is not None and int(np.prod(ps, dtype=np.int64)) != count):
            raise fail(f"part {role} is {pd} {ps}")
        return np.frombuffer(data, np.int8).astype(np.int64) if pd == "I8" else values(pd, data)

    def scale_dtype():  # a free-dtype scale as stored: read exactly, never rounded first
        return param("scale_dtype", ("F32", "BF16", "F16"))

    def scales(blocks, dtype_):
        """Block scale (i, b) for every row i and block b, from either layout."""
        if param("scale_layout", ("row_major", "cublas_128x4")) == "row_major":
            return part("scale", (dtype_,), want=[*shape[:-1], blocks]).reshape(rows, blocks)
        stored = [128 * -(-rows // 128), 4 * -(-blocks // 4)]
        return part("scale", (dtype_,), want=stored)[blocked(rows, blocks)]

    if scheme.endswith(".tensorwise"):
        w = part("weight", FP8 if scheme == "fp8.tensorwise" else ("I8",), want=shape)
        s = part("scale", (scale_dtype(),), count=1)[0]
        if "input_scale" in parts:
            part("input_scale", ("F32",), count=1)
        out = w.astype(np.float32) * s
    elif scheme.endswith(".rowwise"):
        w = part("weight", FP8 if scheme == "fp8.rowwise" else ("I8",), want=shape)
        axis = param("axis", (0, 1))
        s = part("scale", (scale_dtype(),), count=(rows, cols)[axis])
        w = w.astype(np.float32).reshape(rows, cols)
        out = w * (s.reshape(rows, 1) if axis == 0 else s.reshape(1, cols))
    elif scheme == "int8.convrot":
        param("rotation", ("hadamard",))
        b = param("rotation_size", [4**k for k in range(1, 9)])
        if cols % b:
            raise fail(f"{cols} columns in blocks of {b}")
        k = b.bit_length() // 2
        w = part("weight", ("I8",), want=shape).reshape(rows, cols // b, b)
        s = part("scale", (scale_dtype(),), count=rows)
        h = hadamard(w, k).astype(np.float32).reshape(rows, cols)
        out = (h * s.reshape(rows, 1)) * np.float32(2.0**-k)
    elif scheme == "mxfp8":
        param("block_size", (32,))
        n = -(-cols // 32)
        s = scales(n, "F8_E8M0")
        w = part("weight", FP8, want=shape).reshape(rows, cols)
        out = w * np.repeat(s, 32, axis=1)[:, :cols]
    else:  # nvfp4
        param("block_size", (16,))
        if cols % 16:
            raise fail(f"{cols} columns in blocks of 16")
        pd, ps, data = parts["weight"]
        if param("nibble_order", ("low_first", "high_first")) == "high_first":
            raw = np.frombuffer(data, np.uint8)
            parts = {**parts, "weight": (pd, ps, ((raw << 4) | (raw >> 4)).astype(np.uint8).tobytes())}
        w = part("weight", ("F4",), want=shape).reshape(rows, cols // 16, 16)
        s = scales(cols // 16, "F8_E4M3").reshape(rows, cols // 16, 1)
        g = part("scale_2", ("F32",), count=1)[0]
        if "input_scale" in parts:
            part("input_scale", ("F32",), count=1)
        if param("global_scale", ("multiply", "divide")) == "multiply":
            out = (w * s) * g
        else:
            out = w * (s / g)
    return to_dtype(dtype, out)


# ------------------------------------------------------------- conformance


def check(vectors: Path) -> int:
    failures, passed = [], 0
    cases = load_cases(vectors / "cases.json")
    store = Store(vectors / "store")
    for lane in ("all", "spans"):
        _, header = store.checkpoint("Conformance/Schemes", lane=lane)
        for comp, tensors in header["components"].items():
            for key in tensors:
                case = cases.get((lane, comp, key))
                try:
                    got = read(store, header, comp, key)
                    ok = case is not None and "expect" in case and got == base64.b64decode(case["expect"])
                except Refused as e:
                    ok = case is not None and case.get("refuse") == e.code
                passed += ok
                if not ok:
                    failures.append(f"{lane}:{comp}/{key}")
        missing = [k for k in cases if k[0] == lane and k[2] not in header["components"].get(k[1], {})]
        failures += [f"{lane}:{c}/{k} missing from the store" for _, c, k in missing]
        # Order is data: components by first appearance, tensors and parts as cases.json lists them.
        want = [(c, k, [p["role"] for p in case.get("parts", [])]) for (ln, c, k), case in cases.items() if ln == lane]
        want = [w for comp in dict.fromkeys(c for c, _, _ in want) for w in want if w[0] == comp]
        got = [(c, k, list(t.get("parts", {}))) for c, ts in header["components"].items() for k, t in ts.items()]
        ok = got == want
        passed += ok
        if not ok:
            failures.append(f"{lane}: header order differs from cases.json")
    # Release 2 is yanked and swaps the lanes: only an explicit version reaches it.
    _, yanked = store.checkpoint("Conformance/Schemes", version="2", lane="all")
    ok = list(yanked["components"]) == ["spans"]
    passed += ok
    if not ok:
        failures.append("an explicit yanked version")
    parsers = {"header": parse_header, "manifest": parse_manifest, "repo": parse_repo}
    for verdict in ("green", "red"):
        for path in sorted((vectors / verdict).iterdir()):
            kind = path.name.split("-")[0]
            if kind == "header" and path.suffix == ".json":
                continue  # a green header's projection, checked with its bytes
            data = path.read_bytes()
            try:
                parsers[kind](data)
                ok = verdict == "green"
            except Refused as e:
                ok = verdict == "red" and (e.code == "pre_cut") == path.name.startswith("header-pre-cut")
            if ok and verdict == "green" and kind == "header":
                projection = json.loads(path.with_suffix(".json").read_text())
                # Decode to the projection; encode the projection to the same bytes (or, for a
                # later version's fields, decode only: this writer drops them).
                ok = project(strict(data)) == projection and encode(projection) == data
            passed += ok
            if not ok:
                failures.append(f"{verdict}/{path.name}")
    for f in failures:
        print("FAIL", f)
    print(f"cozytensors/1 conformance: {passed} passed, {len(failures)} failed")
    return 1 if failures else 0


def load_cases(path: Path) -> dict:
    doc = json.loads(path.read_text())
    return {(c.get("lane", "all"), c["component"], c["key"]): c for c in doc["cases"]}


if __name__ == "__main__":
    sys.exit(check(Path(sys.argv[1])))
