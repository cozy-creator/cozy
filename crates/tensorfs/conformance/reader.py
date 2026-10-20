#!/usr/bin/env python3
"""An independent cozytensors/1 reader and header encoder, written from the spec: Python, numpy and cbor2.
Scheme decoders are in schemes/, one module per family.

    python reader.py <vectors-dir>     run the conformance vectors (exit 1 on any failure)

As a library: `Store(root).checkpoint("org/name", variant="x")` -> (tree, header),
`read(store, header, component, key)` -> the tensor's logical bytes, `parse_header(bytes)` and
`encode(projection)`: a header's diagnostic projection (the CBOR tree as JSON, byte strings as
{"hex": ...}) to its canonical bytes.
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import cbor2

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # schemes/, also under python -I
from schemes import affine, blocks, decode, grid, table  # noqa: E402
from schemes.base import Refused  # noqa: E402

HEADER, TREE, REPO = "cozytensors/1", "cozytensors.tree/1", "cozytensors.repo/1"
BITS = {
    "BOOL": 8, "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6, "U8": 8, "I8": 8, "F8_E5M2": 8, "F8_E4M3": 8,
    "F8_E8M0": 8, "F8_E4M3FNUZ": 8, "F8_E5M2FNUZ": 8, "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
    "I32": 32, "U32": 32, "F32": 32, "C64": 64, "F64": 64, "I64": 64, "U64": 64,
}
PRE_CUT = "pre-cut cozytensors header (old layout); re-import or migrate"


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


def parse_tree(data: bytes) -> dict:
    """A repo revision's files; `model.<variant>.cozytensors` is a variant's header."""
    doc = load(data, TREE)
    files = field(doc, "files", dict)
    for p in files:
        need(path(p) and p.split("/")[0] != "chunks", f"bad repo file path {p!r}")
    return {"files": {p: object_ref(r) for p, r in files.items()}}


def variant_path(variant: str) -> str:
    return f"model.{variant}.cozytensors" if variant else "model.cozytensors"


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
        releases.append({"version": version, "tree": object_ref(field(r, "tree", dict)),
                         "yanked": field(r, "yanked", bool, optional=True)})
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
    indefinite lengths, unordered or repeated map keys, floats not in their shortest form and
    trailing bytes; then only integers in [-2^63, 2^64), byte and text strings, arrays, maps with
    text keys, true, false, floats and null remain (header() keeps the last two to configs)."""
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
            need(t in (str, bytes, bool, float, type(None)), f"a {t.__name__} item")
    return item


def record(v, n: int, what: str) -> list:
    """An array of at least `n` fields; fields after them are a later version's and are skipped."""
    need(type(v) is list and len(v) >= n, f"{what} is not an array of at least {n}")
    return v


def path(s) -> bool:
    return isinstance(s, str) and all(segment(x) for x in s.split("/"))


def obj(v, what: str) -> dict:
    sha, length = record(v, 2, what)[:2]
    need(type(sha) is bytes and len(sha) == 32 and uint(length), f"{what} is not [bstr .size 32, uint]")
    return {"sha256": sha.hex(), "length": length}


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
    """Pipeline, components, tensors and parts as dicts, which keep the header's (construction)
    order."""
    if data[:1] == b"{":
        raise Refused("pre_cut", PRE_CUT)
    root = strict(data)
    try:
        return header(root)
    except Refused:
        # A layout before the pipeline: a cozytensors/1 root of exactly five fields.
        if type(root) is list and len(root) == 5:
            check_format(root[0], HEADER)
            raise Refused("pre_cut", PRE_CUT) from None
        raise


def header(root) -> dict:
    need(type(root) is list and root, "the header is not an array")
    check_format(root[0], HEADER)
    chunk_rows, encoding_rows, pipeline_row, component_rows, meta = record(root, 6, "the header")[1:6]
    need(type(chunk_rows) is list, "chunks is not an array")
    chunks = [obj(c, "a chunk") for c in chunk_rows]
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
    need(type(pipeline_row) is list and len(pipeline_row) != 1, "the pipeline is not [] or [class, config]")
    pipeline, configs = None, []
    if pipeline_row:
        cls, cfg = pipeline_row[:2]
        need(type(cls) is str and type(cfg) is dict, "the pipeline is not [class, config]")
        pipeline = {"class": cls, "config": cfg}
        configs.append(cfg)
    need(type(component_rows) is list, "components is not an array")
    components, cited = {}, 0
    for c in component_rows:
        name, cls, cfg, tensor_rows, asset_rows = record(c, 5, "a component")[:5]
        need(segment(name) and name not in components, f"bad or repeated component name {name!r}")
        need(type(cls) is str and type(cfg) is dict, f"{name}: class or config")
        configs.append(cfg)
        config_file = c[5] if len(c) > 5 else None
        if len(c) > 6 and config_file == "config.json":
            config_file = None  # later fields follow it: the default, written out
        need(config_file is None or (path(config_file) and config_file != "config.json"),
             f"{name}: config_file {config_file!r}")
        need(type(asset_rows) is dict, f"{name}: assets is not a map")
        assets = {}
        for k, v in asset_rows.items():
            need(path(k) and k != (config_file or "config.json"), f"{name}: asset {k!r}")
            assets[k] = obj(v, f"{name} asset {k}")
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
        components[name] = {"class": cls, "config": cfg, "tensors": tensors, "assets": assets,
                            "config_file": config_file}
    need(cited == len(encodings), f"{cited} of {len(encodings)} encodings are cited")
    need(type(meta) is dict, "metadata is not a map")
    for v in [meta, *(e["params"] for e in encodings), *configs]:
        no_bytes(v)
    only_in(root, {id(c) for c in configs})
    return {"chunks": chunks, "pipeline": pipeline, "components": components, "metadata": meta}


def no_bytes(v):
    """Metadata, params and configs hold values: no byte strings (only later fields may), and
    floats (configs only) are finite."""
    need(type(v) is not bytes and (type(v) is not float or math.isfinite(v)), f"a value {v!r}")
    for x in v.values() if type(v) is dict else v if type(v) is list else ():
        no_bytes(x)


def only_in(v, configs: set):
    """Floats and null appear only inside configs."""
    if id(v) in configs:
        return
    need(type(v) not in (float, type(None)), "a float or null outside a config")
    for x in v.values() if type(v) is dict else v if type(v) is list else ():
        only_in(x, configs)


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

    def checkpoint(self, name: str, version: str | None = None, variant: str = ""):
        repo = parse_repo((self.root / repo_key(name)).read_bytes())
        tree = parse_tree(self.blob(release(repo, version)["tree"]))
        return tree, parse_header(self.blob(tree["files"][variant_path(variant)]))


def gather(store: Store, header: dict, loc: dict) -> bytes:
    out, chunk, start, left = b"", loc["chunk"], loc["offset"], loc["length"]
    while left:
        data = store.blob(header["chunks"][chunk])
        piece = data[start:start + left]
        out, chunk, start, left = out + piece, chunk + 1, 0, left - len(piece)
    return out


def read(store: Store, header: dict, component: str, key: str) -> bytes:
    t = header["components"][component]["tensors"][key]
    if "data" in t:
        return gather(store, header, t["data"])
    parts = {r: (p["dtype"], p["shape"], gather(store, header, p)) for r, p in t["parts"].items()}
    return decode(t["dtype"], t["shape"], t["encoding"], parts)


# ------------------------------------------------------------- conformance


def check(vectors: Path) -> int:
    failures, passed = [], 0
    store = Store(vectors / "store")
    for path in sorted((vectors / "cases").glob("*.json")):
        family = path.stem
        cases = load_cases(path)
        repo = f"Conformance/{family}"
        for lane in dict.fromkeys(ln for ln, _, _ in cases):
            _, header = store.checkpoint(repo, variant=lane)
            for comp, c in header["components"].items():
                for key in c["tensors"]:
                    case = cases.get((lane, comp, key))
                    try:
                        got = read(store, header, comp, key)
                        ok = case is not None and "expect" in case and got == base64.b64decode(case["expect"])
                    except Refused as e:
                        ok = case is not None and case.get("refuse") == e.code
                    passed += ok
                    if not ok:
                        failures.append(f"{family}:{lane}:{comp}/{key}")
            missing = [k for k in cases
                       if k[0] == lane and k[2] not in header["components"].get(k[1], {"tensors": {}})["tensors"]]
            failures += [f"{family}:{lane}:{c}/{k} missing from the store" for _, c, k in missing]
            # Order is data: components by first appearance, tensors and parts as the file lists them.
            want = [(c, k, [p["role"] for p in case.get("parts", [])]) for (ln, c, k), case in cases.items() if ln == lane]
            want = [w for comp in dict.fromkeys(c for c, _, _ in want) for w in want if w[0] == comp]
            got = [(c, k, list(t.get("parts", {})))
                   for c, comp in header["components"].items() for k, t in comp["tensors"].items()]
            ok = got == want
            passed += ok
            if not ok:
                failures.append(f"{family}:{lane}: header order differs from {path.name}")
    # Core's release 2 is yanked and swaps its variants: only an explicit version reaches it.
    _, yanked = store.checkpoint("Conformance/core", version="2", variant="all")
    ok = list(yanked["components"]) == ["spans"]
    passed += ok
    if not ok:
        failures.append("an explicit yanked version")
    core_failures = check_cores(vectors / "cores.json")
    failures += core_failures
    passed += len(json.loads((vectors / "cores.json").read_text())["cases"]) - len(core_failures)
    parsers = {"header": parse_header, "tree": parse_tree, "repo": parse_repo}
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


def core_case(case: dict) -> np.ndarray:
    """A cores.json case's values (spec §6, building blocks)."""
    rows, cols = case["rows"], case["cols"]
    data = lambda k: base64.b64decode(case[k])  # noqa: E731
    f32 = lambda k: np.frombuffer(data(k), "<f4")  # noqa: E731
    if case["core"] == "blocks":
        # gguf's q4_0 {f16 d; u8 qs[16]}: element j in qs[j]'s low nibble, j + 16 in its high one,
        # (q - 8) x d; q8_0 {f16 d; i8 qs[32]}: q x d.
        s = blocks.structs(data("blocks"), case["size"])
        d = blocks.f16(s, 0)
        if case["type"] == "q4_0":
            q = affine.Packing(4, 8, fill="planar").lanes(s[:, 2:], case["elements"])
            return (q.astype(np.float32) - np.float32(8)) * d
        return s[:, 2:].view(np.int8).astype(np.float32) * d
    p = case["packing"]
    order = tuple(p["order"]) if isinstance(p.get("order"), list) else p.get("order", "low_first")
    packing = affine.Packing(**{**p, "order": order})
    g = case["grid"]
    index = np.array(g["index"]) if "index" in g else None
    g = grid.Grid(g["kind"], g.get("size", 0), g.get("layout", "row_major"), index)
    q = packing.unpack(data("words"), rows, cols)
    if case["core"] == "table":
        return table.decode(q, f32("table"), f32("scale"), g, rows, cols)
    zero = f32("zero") if "zero" in case else None
    bias = f32("bias") if "bias" in case else None
    return affine.dequant(q, f32("scale"), g, zero=zero, bias=bias)


def check_cores(path: Path) -> list:
    failures = []
    for case in json.loads(path.read_text())["cases"]:
        got = np.asarray(core_case(case), np.float32).reshape(-1)
        if got.astype("<f4").tobytes() != base64.b64decode(case["expect"]):
            failures.append(f"cores: {case['name']}")
    return failures


def load_cases(path: Path) -> dict:
    doc = json.loads(path.read_text())
    return {(c.get("lane", "all"), c["component"], c["key"]): c for c in doc["cases"]}


if __name__ == "__main__":
    sys.exit(check(Path(sys.argv[1])))
