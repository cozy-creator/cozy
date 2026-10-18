#!/usr/bin/env python3
"""An independent cozytensors/1 reader: Python stdlib and numpy only, written from the spec.

    python reader.py <vectors-dir>     run the conformance vectors (exit 1 on any failure)

As a library: `Store(root).checkpoint("org/name", lane="x")` -> (manifest, header), and
`read(store, header, component, key)` -> the tensor's logical bytes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HEADER, MANIFEST, REPO = "cozytensors/1", "tensorfs.manifest/1", "tensorfs.repo/1"
BITS = {
    "BOOL": 8, "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6, "U8": 8, "I8": 8, "F8_E5M2": 8, "F8_E4M3": 8,
    "F8_E8M0": 8, "F8_E4M3FNUZ": 8, "F8_E5M2FNUZ": 8, "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
    "I32": 32, "U32": 32, "F32": 32, "C64": 64, "F64": 64, "I64": 64, "U64": 64,
}


class Refused(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


def invalid(detail: str) -> Refused:
    return Refused("invalid", detail)


# ---------------------------------------------------------------- documents


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
    if not isinstance(doc, dict):
        raise invalid("the document is not an object")
    tag = doc.get("format")
    if not isinstance(tag, str):
        raise invalid("no format tag")
    name, _, version = tag.partition("/")
    major, dot, minor = version.partition(".")
    if f"{name}/{major}" != expected or (dot and not (minor.isascii() and minor.isdigit())):
        raise Refused("format", f"{tag!r} is not {expected}")
    return doc


def uint(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 2**64


def need(ok: bool, detail: str):
    if not ok:
        raise invalid(detail)


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
    need(isinstance(dtype, str) and dtype != "", f"bad dtype {dtype!r}")
    need(isinstance(shape, list) and all(uint(x) for x in shape), f"bad shape {shape!r}")
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


def parse_location(v: dict, dtype, shape, chunks, what) -> dict:
    loc = {k: field(v, k, int) for k in ("chunk", "offset", "length")}
    size = byte_len(dtype, shape)
    need(size in (None, loc["length"]), f"{what}: length disagrees with dtype and shape")
    if loc["length"]:
        c = loc["chunk"]
        need(c < len(chunks), f"{what}: chunk {c} of {len(chunks)}")
        room = sum(x["length"] for x in chunks[c:])
        need(loc["offset"] < chunks[c]["length"] and loc["offset"] + loc["length"] <= room,
             f"{what}: range runs past the chunks")
    return loc


def named(items, field_name: str, what: str) -> list:
    """An array of objects, each named by `field_name`, with no name repeated."""
    need(isinstance(items, list) and all(isinstance(x, dict) for x in items), f"{what} is not an array of objects")
    names = [field(x, field_name, str) for x in items]
    need(len(set(names)) == len(names), f"a repeated {field_name} in {what}")
    return items


def parse_header(data: bytes) -> dict:
    """Components, tensors and parts as dicts, which keep the header's (construction) order."""
    doc = load(data, HEADER)
    chunks = [object_ref(c) for c in field(doc, "chunks", list)]
    components = {}
    for comp in named(field(doc, "components", list), "name", "components"):
        name = comp["name"]
        need(segment(name), f"bad component name {name!r}")
        tensors = {}
        for t in named(field(comp, "tensors", list), "key", f"{name} tensors"):
            key = t["key"]
            need(key != "", f"{name}: an empty tensor key")
            dtype, shape = field(t, "dtype", str), field(t, "shape", list)
            byte_len(dtype, shape)
            data_, enc, parts = t.get("data"), t.get("encoding"), t.get("parts")
            out = {"dtype": dtype, "shape": shape}
            if data_ is not None and enc is None and parts is None:
                need(isinstance(data_, dict), "data is not an object")
                out["data"] = parse_location(data_, dtype, shape, chunks, f"{name}/{key}")
            elif data_ is None and isinstance(enc, dict) and isinstance(parts, list) and parts:
                scheme = field(enc, "scheme", str)
                need(scheme != "", "empty scheme")
                out["encoding"] = {"scheme": scheme, "params": field(enc, "params", dict, optional=True)}
                out["parts"] = {}
                for p in named(parts, "role", f"{name}/{key} parts"):
                    role = p["role"]
                    need(role != "", f"{name}/{key}: an empty role")
                    pd, ps = field(p, "dtype", str), field(p, "shape", list)
                    loc = parse_location(p, pd, ps, chunks, f"{name}/{key} part {role}")
                    out["parts"][role] = {"dtype": pd, "shape": ps, **loc}
            else:
                raise invalid(f"{name}/{key}: not data xor encoding+parts")
            tensors[key] = out
        components[name] = tensors
    meta = field(doc, "metadata", dict, optional=True)
    return {"chunks": chunks, "components": components, "metadata": meta}


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


def values(dtype: str, data: bytes) -> np.ndarray:
    raw = np.frombuffer(data, np.uint8)
    if dtype == "F32":
        return np.frombuffer(data, "<f4").astype(np.float32)
    if dtype == "F4":
        return E2M1[np.stack([raw & 15, raw >> 4], -1).reshape(-1)]
    return {"F8_E4M3": E4M3, "F8_E5M2": E5M2}[dtype][raw]


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


def narrow(dtype: str, x: np.float32) -> np.float32:
    if dtype == "F32":
        return np.float32(x)
    bits = np.frombuffer(to_dtype(dtype, np.array([x], np.float32)), "<u2")
    if dtype == "BF16":
        return (bits.astype(np.uint32) << 16).view(np.float32)[0]
    return bits.view(np.float16).astype(np.float32)[0]


def hadamard(x: np.ndarray, k: int) -> np.ndarray:
    """Rows of x (int64, last axis 4**k) times H4 (x) ... (x) H4, unnormalized."""
    shape = x.shape
    x = x.reshape(-1, *([4] * k))
    for axis in range(1, k + 1):
        a, b, c, d = (np.take(x, i, axis=axis) for i in range(4))
        x = np.stack([a + b + c - d, a + b - c + d, a - b + c + d, b + c + d - a], axis=axis)
    return x.reshape(shape)


SCHEMES = {  # scheme: (required parts, optional parts, params)
    "fp8-rowwise": ({"weight", "scale"}, set(), {"axis"}),
    "fp8-tensorwise": ({"weight", "scale"}, {"input_scale"}, set()),
    "int8-convrot": ({"weight", "scale"}, set(), {"block", "rotation"}),
    "nvfp4": ({"weight", "scale", "scale_2"}, {"input_scale"}, {"block"}),
}


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
    if not params.keys() <= names:
        raise fail(f"params {sorted(params)}")
    if dtype not in ("F32", "BF16", "F16"):
        raise fail(f"logical dtype {dtype}")
    if scheme != "fp8-tensorwise" and len(shape) != 2:
        raise fail(f"shape {shape}")
    rows, cols = shape if len(shape) == 2 else (1, int(np.prod(shape, dtype=np.int64)))

    def part(role, dtypes, want=None, count=None):
        pd, ps, data = parts[role]
        if pd not in dtypes or (want is not None and ps != list(want)) or (
                count is not None and int(np.prod(ps, dtype=np.int64)) != count):
            raise fail(f"part {role} is {pd} {ps}")
        return np.frombuffer(data, np.int8).astype(np.int64) if pd == "I8" else values(pd, data)

    def param(name):
        v = params.get(name)
        return v if uint(v) else None

    fp8 = ("F8_E4M3", "F8_E5M2")
    if scheme == "fp8-rowwise":
        axis = param("axis")
        if axis not in (0, 1):
            raise fail("axis")
        w = part("weight", fp8, want=shape).reshape(rows, cols)
        s = part("scale", ("F32",), count=shape[axis])
        out = w * (s.reshape(rows, 1) if axis == 0 else s.reshape(1, cols))
    elif scheme == "fp8-tensorwise":
        w = part("weight", fp8, want=shape)
        s = narrow(dtype, part("scale", ("F32",), count=1)[0])
        if "input_scale" in parts:
            part("input_scale", ("F32",), count=1)
        out = w * s
    elif scheme == "int8-convrot":
        block = param("block") or 0
        k = next((k for k in range(1, 9) if 4**k == block), None)
        if k is None or params.get("rotation") != "hadamard" or cols % block:
            raise fail(f"block {block}, rotation {params.get('rotation')!r}, {cols} columns")
        w = part("weight", ("I8",), want=shape).reshape(rows, cols // block, block)
        s = part("scale", ("F32",), count=rows)
        h = hadamard(w, k).astype(np.float32).reshape(rows, cols)
        out = (h * s.reshape(rows, 1)) * np.float32(2.0**-k)
    else:  # nvfp4
        if param("block") != 16 or cols % 16:
            raise fail("block")
        w = part("weight", ("F4",), want=shape).reshape(rows, cols // 16, 16)
        s = part("scale", ("F8_E4M3",), want=[rows, cols // 16]).reshape(rows, cols // 16, 1)
        s2 = part("scale_2", ("F32",), count=1)[0]
        if "input_scale" in parts:
            part("input_scale", ("F32",), count=1)
        out = (w * s) * s2
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
        for path in sorted((vectors / verdict).glob("*.json")):
            try:
                parsers[path.name.split("-")[0]](path.read_bytes())
                ok = verdict == "green"
            except Refused:
                ok = verdict == "red"
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
