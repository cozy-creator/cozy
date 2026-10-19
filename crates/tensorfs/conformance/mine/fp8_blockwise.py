#!/usr/bin/env python3
"""Writes vectors/cases/fp8_blockwise.json: `fp8.blockwise` decode cases (spec §6) from the real producers, on CPU.

- transformers FineGrainedFP8 (`Fp8Quantize`: DeepSeek's `weight_scale_inv`, `F32`, or a power of two under
  `scale_fmt: ue8m0`) quantizes; its `Fp8Dequantize` gives the expected bytes, and DeepSeek's own `weight_dequant`
  (`inference/kernel.py`, Triton's interpreter) reads the same parts to the same `F32` bytes.
- compressed-tensors' block strategy (llm-compressor's FP8_BLOCK: minmax qparams, padded edge tiles, `BF16`/`F16`
  scales) quantizes and decompresses.
- Real tiles by HTTP range at pinned revisions (DeepSeek-V3.1 `F32` with a short edge tile, DeepSeek-V4-Flash
  `F8_E8M0` under `.scale`, Qwen3-4B-FP8 `F16`, RedHatAI Qwen3-0.6B-FP8-BLOCK `BF16`), expected bytes from
  transformers' `Fp8Dequantize` or compressed-tensors' decompress.

Producer cases carry `source`, the `F32` values quantized: TensorFS's closed-form encoder reproduces their parts bit
for bit (tests/format_encode.rs).

    TRITON_INTERPRET=1 uv run --no-project --python 3.12 --with torch==2.14.1 --with transformers==5.20.1 \\
        --with compressed-tensors==0.19.0 --with triton==3.8.0 --with numpy python -I conformance/mine/fp8_blockwise.py
"""

import base64
import json
import struct
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import torch
from compressed_tensors.compressors.naive_quantized.base import FloatQuantizationCompressor
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.utils import calculate_qparams, maybe_pad_tensor_for_block_quant
from transformers import FineGrainedFP8Config
from transformers.integrations.finegrained_fp8 import Fp8Dequantize, Fp8Quantize

OUT = Path(__file__).resolve().parents[1] / "vectors" / "cases" / "fp8_blockwise.json"
HF = "https://huggingface.co/{}/resolve/{}/{}"
DEEPSEEK_KERNEL = ("deepseek-ai/DeepSeek-V3", "e815299b0bcbac849fa540c768ef21845365c9eb", "inference/kernel.py")
DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16", torch.float8_e8m0fnu: "F8_E8M0"}


def get(repo, rev, path, start=None, end=None) -> bytes:
    req = urllib.request.Request(HF.format(repo, rev, path))
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end - 1}")
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    assert start is None or len(data) == end - start, "short read"
    return data


# DeepSeek's reader, its own file at a pinned revision.
_dir = tempfile.mkdtemp()
Path(_dir, "deepseek_kernel.py").write_bytes(get(*DEEPSEEK_KERNEL))
sys.path.insert(0, _dir)
import deepseek_kernel  # noqa: E402


def b64(t: torch.Tensor) -> str:
    return base64.b64encode(t.contiguous().view(torch.uint8).numpy().tobytes()).decode()


def case(key, weight, scale, block, out_dtype, expect, source=None, producer=""):
    c = {
        "component": "fp8_blockwise", "key": key, "dtype": DTYPES[out_dtype], "shape": list(weight.shape),
        "encoding": {"scheme": "fp8.blockwise",
                     "params": {"block_size": list(block), "scale_dtype": DTYPES[scale.dtype]}},
        "parts": [{"role": "weight", "dtype": "F8_E4M3", "shape": list(weight.shape), "bits": b64(weight)},
                  {"role": "scale", "dtype": DTYPES[scale.dtype], "shape": list(scale.shape), "bits": b64(scale)}],
        "expect": b64(expect.to(out_dtype)), "producer": producer,
    }
    assert expect.dtype == out_dtype
    if source is not None:
        c["source"] = b64(source.to(torch.float32))
    return c


def values(shape, seed):
    """bf16-representable weights at a few magnitudes per tile, one all-zero tile, an outlier."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g) * torch.logspace(-3, -1, shape[-1])
    x[..., :16, :16] = 0
    x.view(-1)[7] = 0.75
    return x.to(torch.bfloat16).to(torch.float32)


def transformers_case(key, x, block, scale_fmt, outs):
    q = types.SimpleNamespace(quantization_config=FineGrainedFP8Config(weight_block_size=block, scale_fmt=scale_fmt))
    parts = Fp8Quantize(q).convert({"m.weight": x})
    w, s = parts["m.weight"], parts["m.weight_scale_inv"]
    deq = Fp8Dequantize(q)
    if x.ndim == 2 and s.dtype == torch.float32:
        # DeepSeek's weight_dequant: the same F32 bytes (Triton's interpreter stores F32 exactly; it truncates to
        # BF16, which the GPU rounds, so F32 is the comparison).
        assert torch.equal(deepseek_kernel.weight_dequant(w.contiguous(), s.contiguous(), block[0]),
                           deq._dequantize_one(w, s, torch.float32))
    producer = f"transformers {sys.modules['transformers'].__version__} FineGrainedFP8 scale_fmt={scale_fmt}"
    return [case(f"{key}-{DTYPES[o].lower()}", w, s, block, o, deq._dequantize_one(w, s, o), x if k == 0 else None,
                 producer) for k, o in enumerate(outs)]


def ct_case(key, x, block, dtype, outs):
    """llm-compressor's FP8_BLOCK: per-tile min/max of the padded weight, compressed-tensors' qparams and compressor."""
    args = QuantizationArgs(num_bits=8, type="float", strategy="block", block_structure=list(block), symmetric=True,
                            dynamic=False)
    scheme = QuantizationScheme(targets=["Linear"], weights=args)
    w = x.to(dtype)
    p = maybe_pad_tensor_for_block_quant(w, tuple(block))
    t = p.reshape(p.shape[0] // block[0], block[0], p.shape[1] // block[1], block[1])
    scale, zp = calculate_qparams(t.amin(dim=(1, 3)), t.amax(dim=(1, 3)), args)
    sd = FloatQuantizationCompressor.compress({"weight": w, "weight_scale": scale, "weight_zero_point": zp}, scheme)
    q, s = sd["weight"], sd["weight_scale"]
    back = FloatQuantizationCompressor.decompress(sd, scheme)["weight"]
    # DeepSeek's reader agrees in F32 on the short edge tiles.
    f32 = deepseek_kernel.weight_dequant(q.contiguous(), s.float().contiguous(), block[0])
    producer = f"compressed-tensors {sys.modules['compressed_tensors'].__version__} block {dtype}"
    return [case(f"{key}-{DTYPES[o].lower()}", q, s, block, o, back if o == dtype else f32.to(o),
                 w if k == 0 else None, producer) for k, o in enumerate(outs)]


def header(repo, rev, path):
    n = struct.unpack("<Q", get(repo, rev, path, 0, 8))[0]
    return json.loads(get(repo, rev, path, 8, 8 + n)), 8 + n


def rows_of(repo, rev, path, hdr, off, key, r0, r1):
    t = hdr[key]
    width = t["shape"][1] * {"F8_E4M3": 1, "F8_E8M0": 1, "F32": 4, "F16": 2, "BF16": 2}[t["dtype"]]
    a = off + t["data_offsets"][0]
    data = get(repo, rev, path, a + r0 * width, a + r1 * width)
    dt = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "F32": torch.float32,
          "F16": torch.float16, "BF16": torch.bfloat16}[t["dtype"]]
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(dt).reshape(r1 - r0, t["shape"][1])


def real_case(key, repo, rev, path, module, scale_suffix, rows, cols, reader):
    """Rows `rows` (tile-aligned, or the edge) × the first `cols` columns of a real weight, and its tiles' scales."""
    hdr, off = header(repo, rev, path)
    w = rows_of(repo, rev, path, hdr, off, f"{module}.weight", *rows)[:, :cols].contiguous()
    r0, r1 = rows[0] // 128, -(-rows[1] // 128)
    s = rows_of(repo, rev, path, hdr, off, f"{module}.{scale_suffix}", r0, r1)[:, :cols // 128].contiguous()
    expect = reader(w, s)
    return case(key, w, s, (128, 128), torch.bfloat16, expect,
                producer=f"{repo}@{rev} {path} {module} rows {rows[0]}..{rows[1]} cols 0..{cols}")


def transformers_reader(w, s):
    q = types.SimpleNamespace(quantization_config=FineGrainedFP8Config(weight_block_size=(128, 128)))
    return Fp8Dequantize(q)._dequantize_one(w, s, torch.bfloat16)


def ct_reader(w, s):
    args = QuantizationArgs(num_bits=8, type="float", strategy="block", block_structure=[128, 128], symmetric=True)
    scheme = QuantizationScheme(targets=["Linear"], weights=args)
    return FloatQuantizationCompressor.decompress({"weight": w, "weight_scale": s}, scheme)["weight"]


def refusals(base):
    """Spec refusals, as twins of a producer case."""
    def twin(key, **change):
        c = json.loads(json.dumps(base))
        c.pop("expect"), c.pop("source", None)
        c.update(key=key, component="refuse", refuse="decode", producer="")
        for path, v in change.items():
            *where, last = path.split("__")
            d = c
            for w in where:
                d = d[int(w)] if w.isdigit() else d[w]
            if v is None:
                del d[last]
            else:
                d[last] = v
        return c
    scale = base64.b64decode(base["parts"][1]["bits"])
    first = lambda n: base64.b64encode(scale[:n]).decode()  # noqa: E731
    return [
        twin("block-size-missing", encoding__params__block_size=None),
        twin("block-size-one-extent", encoding__params__block_size=[16]),
        twin("block-size-zero", encoding__params__block_size=[0, 16]),
        twin("scale-dtype-not-the-part", encoding__params__scale_dtype="F32"),
        twin("scale-dtype-u8", encoding__params__scale_dtype="U8", parts__1__dtype="U8", parts__1__bits=first(6)),
        twin("scale-whole-tiles-only", parts__1__shape=[1, 2], parts__1__bits=first(4)),
        twin("weight-e5m2", parts__0__dtype="F8_E5M2"),
        twin("logical-rank-1", shape=[960], parts__0__shape=[960]),
        twin("unknown-param", encoding__params__scale_fmt="ue8m0"),
    ]


def main():
    x = values((32, 32), 49)
    cases = transformers_case("transformers-32x32", x, (16, 16), "float", [torch.float32, torch.bfloat16, torch.float16])
    cases += transformers_case("transformers-ue8m0-32x32", x, (16, 16), "ue8m0", [torch.bfloat16])
    cases += transformers_case("transformers-experts-2x16x32", values((2, 16, 32), 50), (8, 16), "float",
                               [torch.bfloat16])
    edge = values((24, 40), 51)  # short edge tiles both ways
    ct = ct_case("ct-bf16-24x40", edge, (16, 16), torch.bfloat16, [torch.bfloat16, torch.float32])
    cases += ct + ct_case("ct-f16-24x40", edge, (16, 16), torch.float16, [torch.float16])
    cases += [
        real_case("deepseek-v3.1-kv_a_proj-edge", "deepseek-ai/DeepSeek-V3.1", "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db",
                  "model-00001-of-000163.safetensors", "model.layers.0.self_attn.kv_a_proj_with_mqa",
                  "weight_scale_inv", (512, 576), 128, transformers_reader),
        real_case("deepseek-v4-flash-wkv", "deepseek-ai/DeepSeek-V4-Flash", "60d8d70770c6776ff598c94bb586a859a38244f1",
                  "model-00003-of-00046.safetensors", "layers.1.attn.wkv", "scale", (0, 16), 256, transformers_reader),
        real_case("qwen3-4b-fp8-q_proj", "Qwen/Qwen3-4B-Instruct-2507-FP8", "8591804019c8b22094c3b5b4454e0edc05dffc98",
                  "model.safetensors", "model.layers.0.self_attn.q_proj", "weight_scale_inv", (0, 16), 256,
                  transformers_reader),
        real_case("redhat-qwen3-0.6b-fp8-block-q_proj", "RedHatAI/Qwen3-0.6B-FP8-BLOCK",
                  "5def00c475640c0527e3a7067b1a26b94decefb5", "model.safetensors", "model.layers.0.self_attn.q_proj",
                  "weight_scale", (0, 16), 256, ct_reader),
    ]
    cases += refusals(ct[0])
    OUT.write_text(json.dumps({"cases": cases}))
    print(f"{len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
