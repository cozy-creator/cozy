#!/usr/bin/env python3
"""Independent Python conformance codec for the closed CozyTensors CBOR header."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Never

import cbor2

FORMAT = "cozytensors/1"
DTYPES = {
    "f64": 0,
    "f32": 1,
    "f16": 2,
    "bf16": 3,
    "f8_e4m3fn": 4,
    "f8_e5m2": 5,
    "i64": 6,
    "i32": 7,
    "i16": 8,
    "i8": 9,
    "u8": 10,
    "bool": 11,
}
DTYPE_NAMES = tuple(DTYPES)
WIDTHS = (8, 4, 2, 2, 1, 1, 8, 4, 2, 1, 1, 1)

DOC_MAX = 64 << 20
MAX_COMPONENTS = 256
MAX_TENSORS = 100_000
MAX_PARTS = 16
MAX_RANK = 32
MAX_KEY = 4096
MAX_NAME = 128
MAX_SEGMENTS = 65_536
MAX_TOTAL_REFS = 65_536
MAX_TOTAL_BYTES = 1 << 50
INLINE_MAX = 256
U64_MAX = (1 << 64) - 1

class Refusal(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def refuse(code: str, detail: str) -> Never:
    raise Refusal(code, detail)


def canonical_config(raw: bytes, where: str) -> bytes:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                refuse("DUPLICATE_KEY", f"{where} repeats key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique,
            parse_constant=lambda token: refuse(
                "MALFORMED_JSON", f"{where} contains {token}"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        refuse("MALFORMED_JSON", f"{where}: {error}")
    if type(value) is not dict:
        refuse("WRONG_TYPE", f"{where} is not a JSON object")
    return raw


def array(value: Any, arity: int | None, where: str) -> list[Any]:
    if type(value) is not list:
        refuse("WRONG_TYPE", f"{where} must be an array")
    if arity is not None and len(value) != arity:
        refuse("WRONG_ARITY", f"{where} has {len(value)} fields, expected {arity}")
    return value


def text(value: Any, where: str, limit: int, *, empty: bool = False) -> str:
    if type(value) is not str:
        refuse("WRONG_TYPE", f"{where} must be text")
    raw = value.encode("utf-8")
    if (
        (not empty and not raw)
        or len(raw) > limit
        or any(c < 0x20 or c > 0x7E for c in raw)
    ):
        refuse("TEXT_GRAMMAR", f"{where} is outside bounded printable ASCII")
    return value


def name(value: Any, where: str, limit: int) -> str:
    value = text(value, where, limit)
    if not all(c.isascii() and (c.isalnum() or c in "_-./") for c in value):
        refuse("TEXT_GRAMMAR", f"{where} leaves the closed name grammar")
    return value


def asset_name(value: Any, where: str) -> str:
    value = text(value, where, MAX_KEY)
    if value.startswith("/") or value.endswith("/") or "//" in value:
        refuse("TEXT_GRAMMAR", f"{where} has an empty path component")
    for component in value.split("/"):
        if component in {".", ".."}:
            refuse("TEXT_GRAMMAR", f"{where} has a dot component")
        if component.startswith(" ") or component.endswith((" ", ".")):
            refuse("TEXT_GRAMMAR", f"{where} has a reserved edge")
        if any(c in '\\:*?"<>|' for c in component):
            refuse("TEXT_GRAMMAR", f"{where} has a reserved character")
    return value


def uint(value: Any, where: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < 0 or value > U64_MAX:
        refuse("WRONG_TYPE", f"{where} must be a uint64")
    if positive and value == 0:
        refuse("ZERO_ELEMENT", f"{where} must be positive")
    return value


def digest(value: Any, where: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        refuse("MALFORMED_DIGEST", f"{where} must be exactly 32 bytes")
    return value


def diagnostic_digest(value: Any, where: str) -> bytes:
    value = text(value, where, 71)
    if not value.startswith("sha256:") or len(value) != 71:
        refuse("MALFORMED_DIGEST", f"{where} must be sha256:<64 lowercase hex>")
    try:
        raw = bytes.fromhex(value[7:])
    except ValueError:
        refuse("MALFORMED_DIGEST", f"{where} must be sha256:<64 lowercase hex>")
    if value != "sha256:" + raw.hex():
        refuse("MALFORMED_DIGEST", f"{where} must be sha256:<64 lowercase hex>")
    return raw


def dtype_code(value: Any, where: str, *, diagnostic: bool) -> int:
    if diagnostic:
        if type(value) is not str or value not in DTYPES:
            refuse("DTYPE_MISMATCH", f"{where} is not a closed dtype name")
        return DTYPES[value]
    value = uint(value, where)
    if value >= len(DTYPE_NAMES):
        refuse("DTYPE_MISMATCH", f"{where} is outside the closed dtype table")
    return value


def shape(value: Any, where: str) -> list[int]:
    out = array(value, None, where)
    if len(out) > MAX_RANK:
        refuse("RANK_CAP", f"{where} exceeds rank {MAX_RANK}")
    return [uint(v, f"{where}[{i}]", positive=True) for i, v in enumerate(out)]


def nbytes(dtype: int, dims: list[int], where: str) -> int:
    total = WIDTHS[dtype]
    for dim in dims:
        total *= dim
        if total > MAX_TOTAL_BYTES:
            refuse("TOTAL_BYTES_CAP", f"{where} exceeds the byte cap")
    return total


def object_ref(value: Any, where: str, *, diagnostic: bool) -> list[Any]:
    row = array(value, 2, where)
    dg = (
        diagnostic_digest(row[0], f"{where}.digest")
        if diagnostic
        else digest(row[0], f"{where}.digest")
    )
    return [dg, uint(row[1], f"{where}.length", positive=True)]


def sorted_unique(rows: list[list[Any]], key: int, where: str) -> None:
    values = [row[key] for row in rows]
    if values != sorted(set(values)):
        refuse("SORT_ORDER", f"{where} must be sorted and unique")


def _optional(value: Any, where: str, read: Any) -> list[Any]:
    row = array(value, None, where)
    if len(row) > 1:
        refuse("WRONG_ARITY", f"{where} must be [] or [value]")
    return [] if not row else [read(row[0])]


def _dim(value: Any, where: str, *, diagnostic: bool) -> list[Any]:
    if diagnostic:
        if type(value) is not dict or "t" not in value:
            refuse("WRONG_TYPE", f"{where} must be a tagged dimension object")
        tag = value["t"]
        shapes = {
            "axis": ({"axis", "t"}, 0, ("axis",)),
            "div": ({"axis", "by", "rem", "t"}, 1, ("axis", "by")),
            "ceil_div": ({"axis", "by", "t"}, 2, ("axis", "by")),
            "ceil_block": ({"axis", "block", "mul", "t"}, 3, ("axis", "block", "mul")),
            "elements_div": ({"by", "rem", "t"}, 4, ("by",)),
            "lit": ({"t", "value"}, 5, ("value",)),
        }
        if tag not in shapes:
            refuse("UNKNOWN_RELATION", f"{where} has unknown dimension tag")
        fields, code, names = shapes[tag]
        if set(value) != fields or ("rem" in value and value["rem"] != "forbid"):
            refuse("WRONG_FIELDS", f"{where} has wrong dimension fields")
        return [code, *(uint(value[name], f"{where}.{name}", positive=name != "axis") for name in names)]
    row = array(value, None, where)
    if not row:
        refuse("WRONG_ARITY", f"{where} is empty")
    code = uint(row[0], f"{where}.tag")
    arities = {0: 2, 1: 3, 2: 3, 3: 4, 4: 2, 5: 2}
    if arities.get(code) != len(row):
        refuse("UNKNOWN_RELATION", f"{where} has unknown tag/arity")
    return [code, *(uint(v, f"{where}.value", positive=i > 0 or code != 0) for i, v in enumerate(row[1:]))]


def _relation(value: Any, where: str, *, diagnostic: bool) -> list[Any]:
    if diagnostic:
        if type(value) is not dict or value.get("t") not in {"same", "dims"}:
            refuse("UNKNOWN_RELATION", f"{where} has unknown relation")
        if value["t"] == "same":
            if set(value) != {"t"}:
                refuse("WRONG_FIELDS", f"{where} same relation has extra fields")
            return [0]
        if set(value) != {"dims", "t"}:
            refuse("WRONG_FIELDS", f"{where} dims relation has wrong fields")
        dims = [_dim(v, f"{where}.dims[{i}]", diagnostic=True) for i, v in enumerate(array(value["dims"], None, f"{where}.dims"))]
        if len(dims) > MAX_RANK:
            refuse("COUNT_CAP", f"{where}.dims exceeds rank cap")
        return [1, dims]
    row = array(value, None, where)
    if row == [0]:
        return row
    if len(row) != 2 or row[0] != 1:
        refuse("UNKNOWN_RELATION", f"{where} has unknown tag/arity")
    dims = [_dim(v, f"{where}.dims[{i}]", diagnostic=False) for i, v in enumerate(array(row[1], None, f"{where}.dims"))]
    if len(dims) > MAX_RANK:
        refuse("COUNT_CAP", f"{where}.dims exceeds rank cap")
    return [1, dims]


def _carrier(value: Any, where: str, *, diagnostic: bool) -> list[Any]:
    if diagnostic:
        if type(value) is not dict or value.get("t") not in {"same_as_logical", "set"}:
            refuse("UNKNOWN_RELATION", f"{where} has unknown carrier")
        if value["t"] == "same_as_logical":
            if set(value) != {"t"}:
                refuse("WRONG_FIELDS", f"{where} carrier has extra fields")
            return [0]
        if set(value) != {"dtypes", "t"}:
            refuse("WRONG_FIELDS", f"{where} carrier has wrong fields")
        dtypes = [dtype_code(v, f"{where}.dtypes[{i}]", diagnostic=True) for i, v in enumerate(array(value["dtypes"], None, f"{where}.dtypes"))]
        if not dtypes or dtypes != sorted(set(dtypes)):
            refuse("SORT_ORDER", f"{where}.dtypes must be sorted and unique")
        return [1, dtypes]
    row = array(value, None, where)
    if row == [0]:
        return row
    if len(row) != 2 or row[0] != 1:
        refuse("UNKNOWN_RELATION", f"{where} has unknown carrier tag/arity")
    dtypes = [dtype_code(v, f"{where}.dtypes[{i}]", diagnostic=False) for i, v in enumerate(array(row[1], None, f"{where}.dtypes"))]
    if not dtypes or dtypes != sorted(set(dtypes)):
        refuse("SORT_ORDER", f"{where}.dtypes must be sorted and unique")
    return [1, dtypes]


def encoding_spec(value: Any, where: str, *, diagnostic: bool) -> list[Any]:
    if diagnostic:
        if type(value) is not dict or not {"logical_dtypes", "roles"}.issubset(value) or not set(value) <= {"logical_dtypes", "logical_rank", "roles", "vectors"}:
            refuse("WRONG_FIELDS", f"{where} has wrong encoding fields")
        dtypes = [dtype_code(v, f"{where}.logical_dtypes[{i}]", diagnostic=True) for i, v in enumerate(array(value["logical_dtypes"], None, f"{where}.logical_dtypes"))]
        rank = [] if "logical_rank" not in value else [uint(value["logical_rank"], f"{where}.logical_rank", positive=True)]
        if type(value["roles"]) is not dict:
            refuse("WRONG_TYPE", f"{where}.roles must be an object")
        roles = []
        for role_name, role in value["roles"].items():
            if type(role) is not dict or set(role) != {"carrier", "shape"}:
                refuse("WRONG_FIELDS", f"{where}.roles.{role_name} has wrong fields")
            roles.append([name(role_name, f"{where}.role", MAX_NAME), _carrier(role["carrier"], f"{where}.{role_name}.carrier", diagnostic=True), _relation(role["shape"], f"{where}.{role_name}.shape", diagnostic=True)])
        if "vectors" not in value:
            vectors = []
        else:
            ref = value["vectors"]
            if type(ref) is not dict or set(ref) != {"length", "sha256"}:
                refuse("WRONG_FIELDS", f"{where}.vectors has wrong fields")
            vectors = [[diagnostic_digest(ref["sha256"], f"{where}.vectors.sha256"), uint(ref["length"], f"{where}.vectors.length", positive=True)]]
    else:
        row = array(value, 4, where)
        dtypes = [dtype_code(v, f"{where}.logical_dtypes[{i}]", diagnostic=False) for i, v in enumerate(array(row[0], None, f"{where}.logical_dtypes"))]
        rank = _optional(row[1], f"{where}.logical_rank", lambda v: uint(v, f"{where}.logical_rank.value", positive=True))
        roles = []
        for i, role in enumerate(array(row[2], None, f"{where}.roles")):
            role = array(role, 3, f"{where}.roles[{i}]")
            roles.append([name(role[0], f"{where}.roles[{i}].name", MAX_NAME), _carrier(role[1], f"{where}.roles[{i}].carrier", diagnostic=False), _relation(role[2], f"{where}.roles[{i}].shape", diagnostic=False)])
        vectors = _optional(row[3], f"{where}.vectors", lambda v: object_ref(v, f"{where}.vectors.value", diagnostic=False))
    if not dtypes or dtypes != sorted(set(dtypes)) or not roles or len(roles) > MAX_PARTS:
        refuse("SORT_ORDER", f"{where} dtypes/roles are empty, unsorted, duplicate, or over cap")
    sorted_unique(roles, 0, f"{where}.roles")
    return [dtypes, rank, roles, vectors]


def project_encoding(value: list[Any]) -> dict[str, Any]:
    dim_names = ("axis", "div", "ceil_div", "ceil_block", "elements_div", "lit")
    def dim(row: list[Any]) -> dict[str, Any]:
        tag = row[0]
        if tag == 0: return {"axis": row[1], "t": "axis"}
        if tag == 1: return {"axis": row[1], "by": row[2], "rem": "forbid", "t": "div"}
        if tag == 2: return {"axis": row[1], "by": row[2], "t": "ceil_div"}
        if tag == 3: return {"axis": row[1], "block": row[2], "mul": row[3], "t": "ceil_block"}
        if tag == 4: return {"by": row[1], "rem": "forbid", "t": "elements_div"}
        return {"t": dim_names[tag], "value": row[1]}
    def relation(row: list[Any]) -> dict[str, Any]:
        return {"t": "same"} if row[0] == 0 else {"dims": [dim(v) for v in row[1]], "t": "dims"}
    def carrier(row: list[Any]) -> dict[str, Any]:
        return {"t": "same_as_logical"} if row[0] == 0 else {"dtypes": [DTYPE_NAMES[v] for v in row[1]], "t": "set"}
    out: dict[str, Any] = {
        "logical_dtypes": [DTYPE_NAMES[v] for v in value[0]],
        "roles": {role[0]: {"carrier": carrier(role[1]), "shape": relation(role[2])} for role in value[2]},
    }
    if value[1]: out["logical_rank"] = value[1][0]
    if value[3]: out["vectors"] = {"length": value[3][0][1], "sha256": "sha256:" + value[3][0][0].hex()}
    return out


def normalize_diagnostic(value: Any) -> list[Any]:
    if type(value) is not dict or set(value) != {
        "format",
        "configs",
        "assets",
        "encodings",
        "components",
    }:
        refuse("WRONG_FIELDS", "diagnostic root has the wrong fields")
    root = [
        value["format"],
        value["configs"],
        value["assets"],
        value["encodings"],
        value["components"],
    ]
    return normalize_header(root, diagnostic=True)


def normalize_header(document: Any, *, diagnostic: bool = False) -> list[Any]:
    root = array(document, 5, "header")
    if root[0] != FORMAT:
        refuse("UNKNOWN_FORMAT", f"header format is not {FORMAT}")

    raw_configs = array(root[1], None, "configs")
    if len(raw_configs) > MAX_COMPONENTS:
        refuse("COUNT_CAP", "config count exceeds its cap")
    configs: list[list[Any]] = []
    for i, value in enumerate(raw_configs):
        row = array(value, 2, f"configs[{i}]")
        if diagnostic:
            if type(row[1]) is not str:
                refuse("WRONG_TYPE", f"configs[{i}].canonical_json must be text")
            config = row[1].encode("utf-8")
        else:
            if type(row[1]) is not bytes:
                refuse("WRONG_TYPE", f"configs[{i}].canonical_json must be bytes")
            config = row[1]
        if len(config) > DOC_MAX:
            refuse("SIZE_CAP", f"configs[{i}].canonical_json exceeds its cap")
        canonical_config(config, f"configs[{i}].canonical_json")
        configs.append(
            [
                name(row[0], f"configs[{i}].name", MAX_NAME),
                config,
            ]
        )
    sorted_unique(configs, 0, "configs")

    raw_assets = array(root[2], None, "assets")
    if len(raw_assets) > MAX_SEGMENTS:
        refuse("COUNT_CAP", "asset count exceeds its cap")
    assets: list[list[Any]] = []
    for i, value in enumerate(raw_assets):
        row = array(value, 5, f"assets[{i}]")
        segs = [
            object_ref(v, f"assets[{i}].segments[{j}]", diagnostic=diagnostic)
            for j, v in enumerate(array(row[4], None, f"assets[{i}].segments"))
        ]
        if not segs or len(segs) > MAX_SEGMENTS:
            refuse("COUNT_CAP", f"assets[{i}].segments count is out of bounds")
        logical = uint(row[2], f"assets[{i}].logical_length", positive=True)
        if sum(s[1] for s in segs) != logical:
            refuse(
                "BYTE_LENGTH_MISMATCH",
                f"assets[{i}] segment lengths do not equal logical length",
            )
        assets.append(
            [
                asset_name(row[0], f"assets[{i}].name"),
                diagnostic_digest(row[1], f"assets[{i}].digest")
                if diagnostic
                else digest(row[1], f"assets[{i}].digest"),
                logical,
                text(row[3], f"assets[{i}].media_type", MAX_NAME),
                segs,
            ]
        )
    sorted_unique(assets, 0, "assets")

    encodings = [
        encoding_spec(v, f"encodings[{i}]", diagnostic=diagnostic)
        for i, v in enumerate(array(root[3], None, "encodings"))
    ]
    if not encodings or len(encodings) > MAX_TOTAL_REFS:
        refuse("COUNT_CAP", "encoding count is out of bounds")
    encoding_ids = [hashlib.sha256(cbor2.dumps(value, canonical=True)).digest() for value in encodings]
    if encoding_ids != sorted(set(encoding_ids)):
        refuse("SORT_ORDER", "encodings must be sorted and unique by identity")

    components: list[list[Any]] = []
    component_names: set[str] = set()
    tensor_count = 0
    total_refs = sum(len(asset[4]) for asset in assets)
    total_bytes = sum(len(config[1]) for config in configs) + sum(
        asset[2] for asset in assets
    )
    if total_refs > MAX_TOTAL_REFS or total_bytes > MAX_TOTAL_BYTES:
        refuse("TOTAL_REFS_CAP", "header aggregate work exceeds its cap")
    cited: set[int] = set()
    raw_components = array(root[4], None, "components")
    if not raw_components or len(raw_components) > MAX_COMPONENTS:
        refuse("EMPTY_COMPONENTS", "components count is out of bounds")
    for ci, value in enumerate(raw_components):
        row = array(value, 2, f"components[{ci}]")
        component_name = name(row[0], f"components[{ci}].name", MAX_NAME)
        if component_name in component_names:
            refuse("DUPLICATE_KEY", f"component {component_name!r} appears twice")
        component_names.add(component_name)
        tensors: list[list[Any]] = []
        tensor_names: set[str] = set()
        raw_tensors = array(row[1], None, f"components[{ci}].tensors")
        if not raw_tensors:
            refuse("EMPTY_COMPONENTS", f"component {name!r} has no tensors")
        for ti, value in enumerate(raw_tensors):
            where = f"components[{ci}].tensors[{ti}]"
            tensor = array(value, 5, where)
            key = name(tensor[0], f"{where}.key", MAX_KEY)
            if key in tensor_names:
                refuse("DUPLICATE_KEY", f"{name}/{key} appears twice")
            tensor_names.add(key)
            logical_dtype = dtype_code(
                tensor[1], f"{where}.logical_dtype", diagnostic=diagnostic
            )
            logical_shape = shape(tensor[2], f"{where}.logical_shape")
            encoding_index = uint(tensor[3], f"{where}.encoding_index")
            if encoding_index >= len(encodings):
                refuse("ENCODING_MISMATCH", f"{where}.encoding_index is out of range")
            cited.add(encoding_index)
            parts: list[list[Any]] = []
            part_names: set[str] = set()
            raw_parts = array(tensor[4], None, f"{where}.parts")
            if not raw_parts or len(raw_parts) > MAX_PARTS:
                refuse("COUNT_CAP", f"{where}.parts count is out of bounds")
            for pi, value in enumerate(raw_parts):
                pw = f"{where}.parts[{pi}]"
                part = array(value, 4, pw)
                role = name(part[0], f"{pw}.role", MAX_NAME)
                if role in part_names:
                    refuse("DUPLICATE_KEY", f"{where} role {role!r} appears twice")
                part_names.add(role)
                carrier = dtype_code(
                    part[1], f"{pw}.carrier_dtype", diagnostic=diagnostic
                )
                stored_shape = shape(part[2], f"{pw}.storage_shape")
                wanted = nbytes(carrier, stored_shape, pw)
                if diagnostic:
                    if type(part[3]) is not dict or len(part[3]) != 1:
                        refuse(
                            "WRONG_TYPE",
                            f"{pw}.body must select inline_hex or segments",
                        )
                    if "inline_hex" in part[3]:
                        try:
                            body: Any = bytes.fromhex(
                                text(
                                    part[3]["inline_hex"],
                                    f"{pw}.inline_hex",
                                    INLINE_MAX * 2,
                                    empty=True,
                                )
                            )
                        except ValueError:
                            refuse(
                                "WRONG_TYPE", f"{pw}.inline_hex is not lowercase hex"
                            )
                        if part[3]["inline_hex"] != body.hex():
                            refuse(
                                "WRONG_TYPE", f"{pw}.inline_hex is not lowercase hex"
                            )
                    elif "segments" in part[3]:
                        body = [
                            object_ref(v, f"{pw}.segments[{j}]", diagnostic=True)
                            for j, v in enumerate(
                                array(part[3]["segments"], None, f"{pw}.segments")
                            )
                        ]
                    else:
                        refuse(
                            "WRONG_TYPE",
                            f"{pw}.body must select inline_hex or segments",
                        )
                else:
                    body = part[3]
                    if type(body) is bytes:
                        pass
                    elif type(body) is list:
                        body = [
                            object_ref(v, f"{pw}.segments[{j}]", diagnostic=False)
                            for j, v in enumerate(body)
                        ]
                    else:
                        refuse(
                            "WRONG_TYPE", f"{pw}.body must be bytes or a segment array"
                        )
                if type(body) is bytes:
                    if len(body) != wanted:
                        refuse(
                            "BYTE_LENGTH_MISMATCH",
                            f"{pw} inline length differs from shape",
                        )
                    if len(body) > INLINE_MAX:
                        refuse("INLINE_THRESHOLD", f"{pw} must be segmented")
                    if carrier == DTYPES["bool"] and any(v > 1 for v in body):
                        refuse("BOOL_BITS", f"{pw} contains a non-boolean byte")
                else:
                    if not body or len(body) > MAX_SEGMENTS:
                        refuse("COUNT_CAP", f"{pw} segment count is out of bounds")
                    if sum(s[1] for s in body) != wanted:
                        refuse(
                            "BYTE_LENGTH_MISMATCH",
                            f"{pw} segment lengths differ from shape",
                        )
                    if wanted <= INLINE_MAX:
                        refuse("INLINE_THRESHOLD", f"{pw} must be inline")
                    total_refs += len(body)
                total_bytes += wanted
                if total_refs > MAX_TOTAL_REFS or total_bytes > MAX_TOTAL_BYTES:
                    refuse("TOTAL_REFS_CAP", "header aggregate work exceeds its cap")
                parts.append([role, carrier, stored_shape, body])
            tensors.append([key, logical_dtype, logical_shape, encoding_index, parts])
            tensor_count += 1
            if tensor_count > MAX_TENSORS:
                refuse("COUNT_CAP", "tensor count exceeds its cap")
        components.append([component_name, tensors])
    if cited != set(range(len(encodings))):
        refuse("UNCITED_ENCODING", "every encoding must be cited")
    return [FORMAT, configs, assets, encodings, components]


def project(header: list[Any]) -> dict[str, Any]:
    def ref(row: list[Any]) -> list[Any]:
        return ["sha256:" + row[0].hex(), row[1]]

    return {
        "format": FORMAT,
        "configs": [[r[0], r[1].decode("utf-8")] for r in header[1]],
        "assets": [
            [r[0], "sha256:" + r[1].hex(), r[2], r[3], [ref(s) for s in r[4]]]
            for r in header[2]
        ],
        "encodings": [project_encoding(value) for value in header[3]],
        "components": [
            [
                c[0],
                [
                    [
                        t[0],
                        DTYPE_NAMES[t[1]],
                        t[2],
                        t[3],
                        [
                            [
                                p[0],
                                DTYPE_NAMES[p[1]],
                                p[2],
                                {"inline_hex": p[3].hex()}
                                if type(p[3]) is bytes
                                else {"segments": [ref(s) for s in p[3]]},
                            ]
                            for p in t[4]
                        ],
                    ]
                    for t in c[1]
                ],
            ]
            for c in header[4]
        ],
    }


def encode_diagnostic(raw: bytes) -> bytes:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        refuse("MALFORMED_DIAGNOSTIC", str(exc))
    return cbor2.dumps(normalize_diagnostic(value), canonical=True)


def decode_strict(raw: bytes) -> list[Any]:
    if len(raw) > DOC_MAX:
        refuse("SIZE_CAP", f"header is larger than {DOC_MAX} bytes")
    try:
        value = cbor2.loads(raw)
    except (cbor2.CBORDecodeError, UnicodeDecodeError, ValueError) as exc:
        refuse("MALFORMED_CBOR", str(exc))
    header = normalize_header(value)
    if cbor2.dumps(header, canonical=True) != raw:
        refuse(
            "NONCANONICAL_ENCODING",
            "decode plus deterministic re-encode changed the bytes",
        )
    return header


def write(path: str | None, raw: bytes) -> None:
    if path:
        Path(path).write_bytes(raw)
    else:
        sys.stdout.buffer.write(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("encode", "decode"):
        cmd = sub.add_parser(name)
        cmd.add_argument("input")
        cmd.add_argument("--out")
    check = sub.add_parser("check")
    check.add_argument("input")
    identify = sub.add_parser("id")
    identify.add_argument("input")
    args = parser.parse_args()
    try:
        raw = Path(args.input).read_bytes()
        if args.command == "encode":
            write(args.out, encode_diagnostic(raw))
        elif args.command == "decode":
            doc = project(decode_strict(raw))
            write(args.out, (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode())
        elif args.command == "check":
            decode_strict(raw)
        else:
            decode_strict(raw)
            print("sha256:" + hashlib.sha256(raw).hexdigest())
    except Refusal as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
