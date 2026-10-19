#!/usr/bin/env python3
"""Writes vectors/cases/mxfp4.json: `mxfp4` decode cases (spec §6) from the real producers, on CPU.

- gpt-oss: triton_kernels' `downcast_to_mxfp_torch` (the quantizer transformers runs, from the hub kernel it pins)
  quantizes; gpt-oss's own reader (`gpt_oss/torch/weights.py`, its `*_blocks [.., G, 16]` layout) and triton_kernels'
  `upcast_from_mxfp_torch` give the expected bytes. With `transposed: true` the logical tensor is transformers'
  `[E, K, N]` and the expected bytes are transformers' `convert_moe_packed_tensors` (gpt-oss's dequant, transposed).
- torchao `to_mx` (`FLOOR` scales, OCP's) and compressed-tensors' `mxfp4-pack-quantized` compressor quantize; each
  one's own dequantize gives the expected bytes.
- Real rows by HTTP range at pinned revisions: openai/gpt-oss-20b, AMD Quark (`pack_method: reorder`),
  mlx-community (`U32` words), DeepSeek-V4-Flash experts (`I8`, `expert_dtype: fp4`) and a compressed-tensors
  checkpoint; expected bytes from gpt-oss's reader, transformers' `Fp8Dequantize` or compressed-tensors' decompress.

Cases from triton_kernels carry `source`, the `F32` values quantized: TensorFS's closed-form encoder reproduces their
parts bit for bit (tests/format_encode.rs).

    uv venv --python 3.12 /tmp/mine && uv pip install --python /tmp/mine torch==2.14.1 transformers==5.20.1 \\
        kernels==0.17.2 compressed-tensors==0.19.0 torchao==0.18.0 safetensors numpy \\
        && uv pip install --python /tmp/mine --no-deps gpt-oss==0.0.9
    /tmp/mine/bin/python -I conformance/mine/mxfp4.py
"""

import base64
import json
import os
import struct
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import torch
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.utils import calculate_qparams
from gpt_oss.torch.weights import Checkpoint
from kernels import get_kernel
from safetensors.torch import save_file
from torchao.prototype.mx_formats.mx_tensor import to_dtype, to_mx
from transformers import FineGrainedFP8Config
from transformers.integrations.finegrained_fp8 import Fp8Dequantize
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

OUT = Path(__file__).resolve().parents[1] / "vectors" / "cases" / "mxfp4.json"
HF = "https://huggingface.co/{}/resolve/{}/{}"
# transformers' Mxfp4 quantizer loads exactly this kernel.
TRITON_KERNELS = get_kernel("kernels-community/gpt-oss-triton-kernels", version=1)
MXFP = TRITON_KERNELS.numerics_details.mxfp
DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16"}
CT_ARGS = QuantizationArgs(num_bits=4, type="float", strategy="group", group_size=32, symmetric=True,
                           scale_dtype=torch.uint8, dynamic=False)
CT_SCHEME = QuantizationScheme(targets=["Linear"], weights=CT_ARGS, format="mxfp4-pack-quantized")


def b64(t: torch.Tensor) -> str:
    return base64.b64encode(t.contiguous().view(torch.uint8).numpy().tobytes()).decode()


def case(key, weight, scale, out_dtype, expect, source=None, producer="", transposed=False):
    """`weight`: the stored F4 bytes as uint8 [.., C/2]; `scale`: E8M0 bytes [.., ⌈C/32⌉]; `transposed`: the logical
    tensor is the stored one with its last two axes swapped."""
    stored = [*weight.shape[:-1], weight.shape[-1] * 2]
    shape = [*stored[:-2], stored[-1], stored[-2]] if transposed else stored
    assert expect.dtype == out_dtype and list(expect.shape) == shape
    c = {
        "component": "mxfp4", "key": key, "dtype": DTYPES[out_dtype], "shape": shape,
        "encoding": {"scheme": "mxfp4", "params": {"block_size": 32, "transposed": transposed}},
        "parts": [{"role": "weight", "dtype": "F4", "shape": stored, "bits": b64(weight.view(torch.uint8))},
                  {"role": "scale", "dtype": "F8_E8M0", "shape": list(scale.shape), "bits": b64(scale)}],
        "expect": b64(expect), "producer": producer,
    }
    if source is not None:
        c["source"] = b64(source.to(torch.float32))
    return c


def gpt_oss_reader(weight, scale, dtype):
    """gpt-oss's own reader, on its `*_blocks [.., G, 16]` / `*_scales [.., G]` layout."""
    d = tempfile.mkdtemp()
    blocks = weight.reshape(*scale.shape, 16)
    save_file({"m.blocks": blocks.contiguous(), "m.scales": scale.contiguous()}, os.path.join(d, "m.safetensors"))
    return Checkpoint(d, torch.device("cpu"))._get_mxfp4_tensor("m.blocks", "m.scales", dtype=dtype)


def values(shape, seed):
    """bf16 weights at a few magnitudes, a zero block, and a block whose amax/6 is exactly a power of two."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g) * torch.logspace(-3, 0, shape[-1])
    x[..., 0, :32] = 0
    x[..., 1, :32] = x[..., 1, :32].clamp(-0.3, 0.3)
    x[..., 1, 5] = 0.375
    return x.to(torch.bfloat16).to(torch.float32)


def triton_cases(key, x, outs, reader):
    w, s = MXFP.downcast_to_mxfp_torch(x.to(torch.bfloat16), torch.uint8, axis=-1)
    producer = "gpt-oss triton_kernels downcast_to_mxfp_torch (ROUND_UP), kernels-community/gpt-oss-triton-kernels v1"
    out = []
    for k, o in enumerate(outs):
        expect = reader(w, s, o)
        # gpt-oss's reader and triton_kernels' agree wherever both apply.
        if x.shape[-1] % 32 == 0:
            assert torch.equal(expect, MXFP.upcast_from_mxfp_torch(w, s, o, axis=-1))
        out.append(case(f"{key}-{DTYPES[o].lower()}", w, s, o, expect, x if k == 0 else None, producer))
    return out


def transformers_cases(key, x, outs):
    """gpt-oss's [E, N, K] blocks as transformers' [E, K, N] tensor: `transposed: true`."""
    w, s = MXFP.downcast_to_mxfp_torch(x.to(torch.bfloat16), torch.uint8, axis=-1)
    blocks = w.reshape(*s.shape, 16)
    producer = "gpt-oss triton_kernels downcast_to_mxfp_torch; transformers convert_moe_packed_tensors"
    return [case(f"{key}-{DTYPES[o].lower()}", w, s, o, convert_moe_packed_tensors(blocks, s, dtype=o),
                 x.transpose(-1, -2) if k == 0 else None, producer, transposed=True) for k, o in enumerate(outs)]


def torchao_case(key, x):
    scale, data = to_mx(x, torch.float4_e2m1fn_x2, 32)
    data = data.view(torch.uint8)
    expect = to_dtype(data, scale, torch.float4_e2m1fn_x2, 32, torch.bfloat16)
    producer = f"torchao {sys.modules['torchao'].__version__} to_mx FLOOR"
    return case(key, data, scale.view(torch.uint8), torch.bfloat16, expect, producer=producer)


def ct_case(key, x):
    w = x.to(torch.bfloat16)
    g = w.reshape(*w.shape[:-1], -1, 32)
    scale, zp = calculate_qparams(g.amin(-1), g.amax(-1), CT_ARGS)
    sd = MXFP4PackedCompressor.compress({"weight": w, "weight_scale": scale, "weight_zero_point": zp}, CT_SCHEME)
    back = MXFP4PackedCompressor.decompress(sd, CT_SCHEME)["weight"].to(torch.bfloat16)
    producer = f"compressed-tensors {sys.modules['compressed_tensors'].__version__} mxfp4-pack-quantized"
    return case(key, sd["weight_packed"], sd["weight_scale"], torch.bfloat16, back, producer=producer)


def get(repo, rev, path, start, end) -> bytes:
    req = urllib.request.Request(HF.format(repo, rev, path), headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    assert len(data) == end - start, "short read"
    return data


def real(repo, rev, path, key, first, count):
    """`count` bytes from byte `first` of tensor `key`."""
    n = struct.unpack("<Q", get(repo, rev, path, 0, 8))[0]
    hdr = json.loads(get(repo, rev, path, 8, 8 + n))
    a = 8 + n + hdr[key]["data_offsets"][0] + first
    return torch.frombuffer(bytearray(get(repo, rev, path, a, a + count)), dtype=torch.uint8)


def real_case(key, repo, rev, path, weight_key, scale_key, row, cols, shape, reader, transposed=False):
    """Row `row` of a real weight of `cols` columns (counting from the stack's first matrix), stored as `shape`."""
    rows = int(torch.tensor(shape[:-1]).prod())
    w = real(repo, rev, path, weight_key, row * cols // 2, rows * cols // 2)
    s = real(repo, rev, path, scale_key, row * cols // 32, rows * cols // 32)
    w, s = w.reshape(*shape[:-1], cols // 2), s.reshape(*shape[:-1], cols // 32)
    return case(key, w, s, torch.bfloat16, reader(w, s), producer=f"{repo}@{rev} {path} {weight_key} row {row}",
                transposed=transposed)


def transformers_reader(w, s):
    q = types.SimpleNamespace(quantization_config=FineGrainedFP8Config(weight_block_size=(128, 128)))
    return Fp8Dequantize(q)._dequantize_one(w.view(torch.int8), s.view(torch.float8_e8m0fnu), torch.bfloat16)


def ct_reader(w, s):
    return MXFP4PackedCompressor.decompress({"weight_packed": w, "weight_scale": s}, CT_SCHEME)["weight"].to(
        torch.bfloat16)


def refusals(base):
    """Spec refusals, as twins of a producer case."""
    scale = base64.b64decode(base["parts"][1]["bits"])

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
    half = base64.b64encode(scale[: len(scale) // 2]).decode()
    return [
        twin("block-size-16", encoding__params__block_size=16),
        twin("block-size-missing", encoding__params__block_size=None),
        twin("scale-layout-unknown", encoding__params__scale_layout="row_major"),
        twin("transposed-missing", encoding__params__transposed=None),
        twin("transposed-not-a-boolean", encoding__params__transposed=1),
        twin("transposed-parts-not-swapped", encoding__params__transposed=True),
        twin("scale-u8", parts__1__dtype="U8"),
        twin("scale-short", parts__1__shape=[base["shape"][0], 1], parts__1__bits=half),
        twin("weight-u8", parts__0__dtype="U8", parts__0__shape=[base["shape"][0], base["shape"][1] // 2]),
        twin("logical-rank-1", shape=[base["shape"][0] * base["shape"][1]],
             parts__0__shape=[base["shape"][0] * base["shape"][1]]),
    ]


def main():
    gpt_oss = lambda w, s, o: gpt_oss_reader(w, s, o)  # noqa: E731
    upcast = lambda w, s, o: MXFP.upcast_from_mxfp_torch(w, s, o, axis=-1)  # noqa: E731
    cases = triton_cases("gpt-oss-8x64", values((8, 64), 49), [torch.float32, torch.bfloat16], gpt_oss)
    cases += triton_cases("gpt-oss-8x64-upcast", values((8, 64), 49), [torch.float16], upcast)[:1]
    cases += triton_cases("gpt-oss-experts-2x4x64", values((2, 4, 64), 50), [torch.bfloat16], gpt_oss)
    cases += triton_cases("gpt-oss-ragged-4x48", values((4, 48), 51), [torch.bfloat16], upcast)
    # transformers' gpt-oss tensors: [E, K, N] from [E, N, K] blocks; a square one would read either way.
    cases += transformers_cases("transformers-gpt-oss-2x64x8", values((2, 8, 64), 54), [torch.float32, torch.bfloat16])
    cases += transformers_cases("transformers-gpt-oss-square-1x32x32", values((1, 32, 32), 55), [torch.bfloat16])
    cases.append(torchao_case("torchao-floor-8x64", values((8, 64), 52)))
    cases.append(ct_case("ct-mxfp4-8x64", values((8, 64), 53)))
    real_reader = lambda w, s: gpt_oss_reader(w, s, torch.bfloat16)  # noqa: E731
    cases += [
        real_case("openai-gpt-oss-20b-down_proj", "openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee",
                  "model-00000-of-00002.safetensors", "model.layers.0.mlp.experts.down_proj_blocks",
                  "model.layers.0.mlp.experts.down_proj_scales", 0, 2880, [1, 2, 2880],
                  lambda w, s: convert_moe_packed_tensors(w.reshape(*s.shape, 16), s, dtype=torch.bfloat16),
                  transposed=True),
        real_case("amd-quark-gpt-oss-20b-down_proj", "amd/gpt-oss-20b-w-mxfp4-a-bf16",
                  "34f332d130203b37d3c7715874254cd7b5447f51", "model.safetensors",
                  "model.layers.0.mlp.experts.0.down_proj.weight", "model.layers.0.mlp.experts.0.down_proj.weight_scale",
                  0, 2880, [1, 2880], real_reader),
        real_case("mlx-gpt-oss-20b-down_proj", "mlx-community/gpt-oss-20b-MXFP4-Q8",
                  "773a7da77e569019bb0fd17a554b263738d669a3", "model-00001-of-00003.safetensors",
                  "model.layers.0.mlp.experts.down_proj.weight", "model.layers.0.mlp.experts.down_proj.scales",
                  0, 2880, [1, 2880], real_reader),
        real_case("deepseek-v4-flash-expert-w2", "deepseek-ai/DeepSeek-V4-Flash",
                  "60d8d70770c6776ff598c94bb586a859a38244f1", "model-00003-of-00046.safetensors",
                  "layers.1.ffn.experts.0.w2.weight", "layers.1.ffn.experts.0.w2.scale", 0, 2048, [1, 2048],
                  transformers_reader),
        real_case("ct-tinyllama-k_proj", "nm-testing/TinyLlama-1.1B-Chat-v1.0-MXFP4",
                  "4399e1c0764beafc23e7a5465391132c1a56dc30", "model.safetensors",
                  "model.layers.0.self_attn.k_proj.weight_packed", "model.layers.0.self_attn.k_proj.weight_scale",
                  0, 2048, [1, 2048], ct_reader),
    ]
    cases += refusals(cases[1])
    OUT.write_text(json.dumps({"cases": cases}))
    print(f"{len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
