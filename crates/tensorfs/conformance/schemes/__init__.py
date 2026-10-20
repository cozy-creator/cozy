"""The scheme decoders (spec §6): one module per family, registered in FAMILIES, and the checks
every scheme shares. Each element is computed in float32 by the scheme's formula, then rounded
once to the logical dtype (round_L)."""

from __future__ import annotations

import numpy as np

from . import fp8
from . import int8
from . import mxfp8
from . import nvfp4

# fp8.blockwise, mxfp4
from . import fp8_blockwise
from . import mxfp4

from . import gguf

# gptq, awq, ct
from . import gptq
from . import awq
from . import ct

# bnb
from . import bnb

# comfy
from . import comfy

from .base import Ctx, Refused

FAMILIES = (
    fp8,
    int8,
    mxfp8,
    nvfp4,

    # fp8.blockwise, mxfp4
    fp8_blockwise,
    mxfp4,

    gguf,

    # gptq, awq, ct
    gptq,
    awq,
    ct,

    # bnb
    bnb,

    # comfy
    comfy,
)
SCHEMES = {s.name: s for family in FAMILIES for s in family.SCHEMES}


def allowed(rule, v) -> bool:
    if callable(rule):
        return rule(v)
    if isinstance(rule, range):
        return type(v) is int and v in rule
    return any(type(v) is type(a) and v == a for a in rule)


def decode(dtype: str, shape: list, encoding: dict, parts: dict) -> bytes:
    """The logical bytes; parts maps each role to (dtype, shape, bytes)."""
    with np.errstate(all="ignore"):  # NaN and infinity are values here, not errors
        return _decode(dtype, shape, encoding, parts)


def _decode(dtype, shape, encoding, parts) -> bytes:
    name, params = encoding["scheme"], encoding["params"]
    if name not in SCHEMES:
        raise Refused("unknown_scheme", name)
    scheme = SCHEMES[name]

    def fail(detail):
        return Refused("decode", f"{name}: {detail}")

    roles = {p.role for p in scheme.parts}
    if not {p.role for p in scheme.parts if p.required} <= parts.keys() <= roles:
        raise fail(f"parts {sorted(parts)}")
    if params.keys() != scheme.params.keys():  # all required; an unknown one is refused
        raise fail(f"params {sorted(params)}")
    for key, rule in scheme.params.items():
        if not allowed(rule, params[key]):
            raise fail(f"param {key}={params[key]!r}")
    if dtype not in ("F32", "BF16", "F16"):
        raise fail(f"logical dtype {dtype}")
    # transposed: true (spec §6, building blocks): the parts hold the logical tensor with its last two axes
    # swapped; the scheme reads that stored tensor, and its values are swapped back.
    transposed = params.get("transposed") is True and len(shape) >= 2
    if transposed:
        shape = [*shape[:-2], shape[-1], shape[-2]]
    numel = int(np.prod(shape, dtype=np.int64))
    if scheme.flat:
        rows, cols = 1, numel
    elif len(shape) >= 2:  # the last axis, and every leading axis as rows
        cols = shape[-1]
        rows = numel // cols if cols else 0
    else:
        raise fail(f"shape {shape}")
    c = Ctx(scheme, dtype, shape, rows, cols, params, parts)
    for rule in scheme.parts:
        if rule.role not in parts:
            continue
        pd, ps, _ = parts[rule.role]
        kind, want = rule.shape(c)
        fits = ps == list(want) if kind == "exact" else int(np.prod(ps, dtype=np.int64)) == want
        if pd not in rule.dtypes or not fits:
            raise fail(f"part {rule.role} is {pd} {ps}")
    out = np.asarray(scheme.decode(c), np.float32)
    if transposed:
        out = out.reshape(shape).swapaxes(-1, -2)
    return round_l(dtype, out)


def round_l(dtype: str, x: np.ndarray) -> bytes:
    """round_L: nearest even, NaN canonical."""
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
