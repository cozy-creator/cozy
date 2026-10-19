#!/usr/bin/env python3
"""Writes vectors/cases/gguf.json: gguf.<type> cases (spec §6 `gguf.*`) whose parts and expected values are ggml's
own C, llama.cpp ff5a233's ggml-quants.c (`quantize_row_<type>_ref`, `dequantize_row_<type>`), fetched at that commit,
checked against the SHA-256 below, and built here with gcc for baseline x86-64 with -ffp-contract=off.

- Each type: a tensor of real-looking weights through ggml's reference quantizer (`source` holds the input, so a
  closed-form encoder must give the parts exactly), decoded by ggml to F32, BF16 and F16; structs of random bytes
  (every scale pattern: NaN, infinity, subnormal); and refusals.
- Shapes: an expert stack, a rank-1 tensor, ComfyUI-GGUF's reshaped `[N/n, S]` blocks; gguf.nvfp4 with its scale.

    uv run --no-project --python 3.12 --with numpy python -I conformance/mine/gguf.py
"""

import base64
import ctypes
import hashlib
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "vectors" / "cases" / "gguf.json"
COMMIT = "ff5a23376725e5a889fa0bfb987b7541dc304da0"
SOURCES = {
    "ggml/include/ggml-alloc.h": "d114067f9099c4a0e2bbc0625c676d4894d9cf0ea27168ee072c7de0e71bb51d",
    "ggml/include/ggml-backend.h": "0e5e8e1f16d11ebbf501e38bc6705d79eee819b5f0b17476533f2100ed9674e2",
    "ggml/include/ggml-cpu.h": "316279e004cdeb8e6ef78599acb602bf79a8abdf897fed9fd1914808c1518c6e",
    "ggml/include/ggml.h": "475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3",
    "ggml/include/gguf.h": "e56714aab702e5ce62ee587a409643c08f7e93e8fbb77f48ef7cc85075f96fa4",
    "ggml/src/ggml-common.h": "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c",
    "ggml/src/ggml-cpu/ggml-cpu-impl.h": "fd3e46b3e95c8aadda96f17ed13a4a2e83d122d7108c5b56951c80cc29224950",
    "ggml/src/ggml-impl.h": "43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3",
    "ggml/src/ggml-quants.c": "5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab",
    "ggml/src/ggml-quants.h": "28ae5fca1f3be636b36cd6c4fa2ca74fd42d229bfbd5352eaf66f3727bb8a6da",
}
# Symbols ggml-quants.c references that the functions used here never call.
STUBS = """#include <stdlib.h>
#include <stddef.h>
void ggml_abort(const char * f, int l, const char * fmt, ...) { (void) f; (void) l; (void) fmt; abort(); }
size_t ggml_row_size(int t, long long n) { (void) t; (void) n; abort(); }
const char * ggml_type_name(int t) { (void) t; abort(); }
size_t ggml_type_size(int t) { (void) t; abort(); }
"""
TYPES = {  # name: (elements, bytes) per struct; ggml's symbol suffix
    "q4_0": (32, 18), "q4_1": (32, 20), "q5_0": (32, 22), "q5_1": (32, 24), "q8_0": (32, 34),
    "q2_k": (256, 84), "q3_k": (256, 110), "q4_k": (256, 144), "q5_k": (256, 176), "q6_k": (256, 210),
    "mxfp4": (32, 17), "nvfp4": (64, 36), "q1_0": (128, 18), "q2_0": (64, 18),
}
rng = np.random.default_rng(4949)


def ggml(tmp: Path):
    for path, sha in SOURCES.items():
        url = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{COMMIT}/{path}"
        data = urllib.request.urlopen(url).read()
        assert hashlib.sha256(data).hexdigest() == sha, path
        (tmp / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp / path).write_bytes(data)
    (tmp / "stubs.c").write_text(STUBS)
    lib = tmp / "libggml-quants.so"
    subprocess.run(["gcc", "-O2", "-std=gnu11", "-ffp-contract=off", "-fno-fast-math", "-DNDEBUG", "-fPIC", "-shared",
                    f"-I{tmp}/ggml/include", f"-I{tmp}/ggml/src", str(tmp / "ggml/src/ggml-quants.c"),
                    str(tmp / "stubs.c"), "-o", str(lib)], check=True)
    return ctypes.CDLL(str(lib))


def symbol(name: str) -> str:
    return name.replace("_k", "_K")


def quantize(lib, name, x: np.ndarray) -> np.ndarray:
    n, size = TYPES[name]
    x = np.ascontiguousarray(x, np.float32).reshape(-1)
    out = np.zeros(x.size // n * size, np.uint8)
    getattr(lib, f"quantize_row_{symbol(name)}_ref")(x.ctypes.data_as(ctypes.c_void_p),
                                                       out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(x.size))
    return out


def dequantize(lib, name, blocks: np.ndarray) -> np.ndarray:
    n, size = TYPES[name]
    out = np.zeros(blocks.size // size * n, np.float32)
    getattr(lib, f"dequantize_row_{symbol(name)}")(np.ascontiguousarray(blocks).ctypes.data_as(ctypes.c_void_p),
                                                   out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(out.size))
    return out


def round_l(dtype: str, x: np.ndarray) -> bytes:
    """round_L (spec §6): nearest even, canonical NaN."""
    x = x.astype(np.float32)
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


def b64(a) -> str:
    return base64.b64encode(a if isinstance(a, bytes) else np.ascontiguousarray(a).tobytes()).decode()


def blocks_shape(name, shape):
    n, size = TYPES[name]
    if shape[-1] % n == 0:
        return [*shape[:-1], shape[-1] // n, size]
    return [int(np.prod(shape)) // n, size]


def case(name, key, dtype, shape, blocks, expect=None, refuse=None, source=None, extra=(), params=None, parts=None):
    c = {"component": f"gguf.{name}", "key": key, "dtype": dtype, "shape": list(shape),
         "encoding": {"scheme": f"gguf.{name}", "params": params if params is not None else {"block_size": TYPES[name][0]}},
         "parts": parts if parts is not None else
         [{"role": "blocks", "dtype": "U8", "shape": blocks_shape(name, shape), "bits": b64(blocks)}, *extra]}
    if refuse:
        c["refuse"] = refuse
    else:
        c["expect"] = b64(expect)
    if source is not None:
        c["source"] = b64(source.astype("<f4"))
    return c


def weights(rows, cols):
    """Real-looking weights: per-row scales, a few outliers, a run of zeros, a constant block."""
    x = rng.standard_normal((rows, cols)).astype(np.float32) * np.exp(rng.uniform(-6, 2, (rows, 1))).astype(np.float32)
    x[0, :8] = 0
    x[-1, -16:] = x[-1, -1]
    flat = x.reshape(-1)
    flat[rng.integers(0, flat.size, 3)] *= 40
    return x


def main():
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        lib = ggml(Path(tmp))
        for name, (n, size) in TYPES.items():
            cols = 2 * n if n < 256 else 512
            x = weights(3, cols)
            blocks = quantize(lib, name, x)
            v = dequantize(lib, name, blocks)
            shape = [3, cols]
            cases.append(case(name, f"{3}x{cols}-f32", "F32", shape, blocks, round_l("F32", v), source=x))
            cases.append(case(name, f"{3}x{cols}-bf16", "BF16", shape, blocks, round_l("BF16", v)))
            cases.append(case(name, f"{3}x{cols}-f16", "F16", shape, blocks, round_l("F16", v)))
            raw = rng.integers(0, 256, 4 * size, dtype=np.uint8)
            if name == "mxfp4":
                raw[::size] = [0, 1, 254, 255]
            elif name == "nvfp4":
                raw[: 4 * size: size] = [0x00, 0x7F, 0x80, 0xFF]
            else:  # an F16 d of every kind: NaN, -infinity, subnormal
                d = {"q2_k": 80, "q3_k": 108, "q6_k": 208}.get(name, 0)
                for k, h in enumerate([0x7E01, 0xFC00, 0x0007]):
                    raw[k * size + d:k * size + d + 2] = [h & 0xFF, h >> 8]
            cases.append(case(name, f"bytes-{4 * n}", "F32", [4 * n], raw, round_l("F32", dequantize(lib, name, raw))))
        # Shapes: an expert stack, ComfyUI-GGUF's reshaped blocks, rank 1 (above), and nvfp4's per-tensor scale.
        x = weights(4, 256)
        b = quantize(lib, "q4_k", x)
        cases.append(case("q4_k", "experts-2x2x256-f32", "F32", [2, 2, 256], b, round_l("F32", dequantize(lib, "q4_k", b))))
        x = weights(16, 32).reshape(8, 4, 4, 4)
        b = quantize(lib, "q8_0", x)
        cases.append(case("q8_0", "reshaped-8x4x4x4-f32", "F32", [8, 4, 4, 4], b,
                          round_l("F32", dequantize(lib, "q8_0", b)), source=x))
        x = weights(2, 128)
        b = quantize(lib, "nvfp4", x)
        scale, inputs = np.float32(0.0078125 * 3), np.float32(5.5)
        extra = [{"role": "scale", "dtype": "F32", "shape": [1], "bits": b64(np.array([scale], "<f4"))},
                 {"role": "input_scale", "dtype": "F32", "shape": [1], "bits": b64(np.array([inputs], "<f4"))}]
        cases.append(case("nvfp4", "scaled-2x128-bf16", "BF16", [2, 128], b,
                          round_l("BF16", dequantize(lib, "nvfp4", b) * scale), extra=extra))
        # Refusals.
        q = quantize(lib, "q4_0", weights(1, 64))
        cases.append(case("q4_0", "block-size-64", "F32", [1, 64], q, refuse="decode", params={"block_size": 64}))
        cases.append(case("q4_0", "params-missing", "F32", [1, 64], q, refuse="decode", params={}))
        cases.append(case("q4_0", "param-unknown", "F32", [1, 64], q, refuse="decode", params={"block_size": 32, "axis": 0}))
        cases.append(case("q4_0", "dtype-i32", "I32", [1, 64], q, refuse="decode"))
        cases.append(case("q4_0", "elements-not-whole-blocks", "F32", [1, 63], q, refuse="decode",
                          parts=[{"role": "blocks", "dtype": "U8", "shape": [2, 18], "bits": b64(q)}]))
        cases.append(case("q4_0", "blocks-shape", "F32", [1, 64], q, refuse="decode",
                          parts=[{"role": "blocks", "dtype": "U8", "shape": [2, 18], "bits": b64(q)}]))
        cases.append(case("q4_0", "blocks-dtype", "F32", [1, 64], q, refuse="decode",
                          parts=[{"role": "blocks", "dtype": "I8", "shape": [1, 2, 18], "bits": b64(q)}]))
        cases.append(case("q4_0", "scale-not-a-part", "F32", [1, 64], q, refuse="decode", extra=extra[:1]))
        nv = quantize(lib, "nvfp4", weights(1, 64))
        two = {"role": "scale", "dtype": "F32", "shape": [2], "bits": b64(np.array([1, 2], "<f4"))}
        cases.append(case("nvfp4", "scale-two-elements", "F32", [1, 64], nv, refuse="decode", extra=[two]))
        cases.append(case("nvfp4", "blocks-missing", "F32", [1, 64], nv, refuse="decode", parts=extra[1:]))
    OUT.write_text(json.dumps({"cases": cases}))
    print(f"{len(cases)} gguf cases")


if __name__ == "__main__":
    main()
