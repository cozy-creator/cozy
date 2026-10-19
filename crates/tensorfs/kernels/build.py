#!/usr/bin/env python3
"""Compile TensorFS decode cubins without a GPU; Triton is a build dependency only.

    uv run --no-project --python 3.12 --with triton==3.8.0 python kernels/build.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

import kernels as k

FAMILIES = (80, 86, 89, 90, 100, 120)
DTYPES = ("F32", "BF16", "F16")
STRUCTS = {name: getattr(k, "_" + name) for name in (
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q1_0", "q2_0"
)}
STRUCTS.update(mxfp4=k._mxfp4_struct, nvfp4=k._nvfp4_struct)
STAGES = ("unpack", "hadamard", "rotated")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path(__file__).parents[2] / "tensorfs-core/kernels/aot")
    parser.add_argument("--family", type=int, nargs="+", choices=FAMILIES, default=FAMILIES)
    args = parser.parse_args()
    out = args.out.resolve()
    records = []
    for family in args.family:
        target = GPUTarget("cuda", family, 32)
        for dtype, output in zip(DTYPES, range(3), strict=True):
            signature = {"out": "*i32" if output == 0 else "*i16", "words": "*u8", "slots": "*i32",
                         "scale": "*fp32", "term": "*fp32", "table": "*fp32", "factor": "*fp32",
                         "post": "*fp32", "index": "*u32", "error": "*i32", "params": "*i64"}
            variants = [("codes", k.codes, {"BLOCK": 256})]
            variants += [(name, k.structs, {"DECODE": fn, "BLOCK": 256}, 0) for name, fn in STRUCTS.items()]
            variants += [(name, getattr(k, name), {"BLOCK": 256}) for name in STAGES]
            for kind, fn, constants in variants:
                stage_signature = {**signature, "out": "*i32"} if kind in ("unpack", "hadamard") else signature
                compiled = triton.compile(ASTSource(fn, stage_signature, {"OUT": output, **constants}), target=target,
                                          options={"enable_fp_fusion": False, "num_warps": 4})
                image = compiled.asm["cubin"]
                if not image.startswith(b"\x7fELF"):
                    raise ValueError("compiler did not emit a CUDA ELF binary")
                relative = pathlib.Path(f"sm{family}") / f"{kind}-{dtype}.cubin"
                path = out / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(image)
                meta = compiled.metadata
                records.append({"family": family, "kind": kind, "dtype": dtype,
                                "file": str(relative), "symbol": meta.name, "shared": meta.shared,
                                "warps": meta.num_warps, "sha256": hashlib.sha256(image).hexdigest()})
                print(f"sm{family} {kind} {dtype}: {len(image)} bytes", flush=True)
    catalog = ["pub static BINARIES: &[Binary] = &["]
    for r in records:
        catalog.append('    Binary { family: %(family)d, kind: "%(kind)s", dtype: "%(dtype)s", '
                       'symbol: "%(symbol)s", shared: %(shared)d, warps: %(warps)d, '
                       'image: include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/aot/%(file)s")) },' % r)
    catalog.append("];\n")
    out.joinpath("catalog.rs").write_text("\n".join(catalog))
    out.joinpath("manifest.json").write_text(json.dumps({"triton": triton.__version__, "binaries": records}, indent=2) + "\n")
    for name in ("kernels.py", "build.py"):
        out.joinpath("source-" + name).write_bytes(pathlib.Path(__file__).with_name(name).read_bytes())
    assert len(records) == len(args.family) * len(DTYPES) * (1 + len(STRUCTS) + len(STAGES))
    print(f"{len(records)} prebuilt CUDA binaries; no driver or GPU used")


if __name__ == "__main__":
    main()
