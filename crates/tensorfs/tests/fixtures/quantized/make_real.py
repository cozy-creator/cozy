#!/usr/bin/env python3
"""Writes the real-* fixtures: tiles and rows of real fp8.blockwise and mxfp4 checkpoints, read by HTTP range at
pinned revisions, each in its producer's container and declaration. `real-<case>.expect.safetensors` holds
`dequant.<key>` (the producer's own reader: transformers' Fp8Dequantize, compressed-tensors' decompress, gpt-oss's
weights.py, transformers' convert_moe_packed_tensors) and, where a higher-precision original exists, `source.<key>`
(its BF16 weights, same rows).

    <venv with torch, transformers, compressed-tensors, safetensors, gpt-oss (--no-deps)>/bin/python -I make_real.py
"""

import json
import os
import struct
import tempfile
import types
import urllib.request
from pathlib import Path

import torch
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
from compressed_tensors.compressors.naive_quantized.base import FloatQuantizationCompressor
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from gpt_oss.torch.weights import Checkpoint
from safetensors.torch import save_file
from transformers import FineGrainedFP8Config
from transformers.integrations.finegrained_fp8 import Fp8Dequantize
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

HERE = Path(__file__).resolve().parent
HF = "https://huggingface.co/{}/resolve/{}/{}"
UNSLOTH_LAYER0 = "model-00001-of-00009.safetensors"
DT = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "F32": torch.float32, "F16": torch.float16,
      "BF16": torch.bfloat16, "U8": torch.uint8, "I8": torch.int8}


def get(repo, rev, path, start=None, end=None) -> bytes:
    req = urllib.request.Request(HF.format(repo, rev, path))
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end - 1}")
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


class Shard:
    def __init__(self, repo, rev, path):
        self.at = (repo, rev, path)
        n = struct.unpack("<Q", get(*self.at, 0, 8))[0]
        self.hdr, self.off = json.loads(get(*self.at, 8, 8 + n)), 8 + n

    def span(self, key, first, count):
        """`count` elements of `key` from element `first`, flattened."""
        t = self.hdr[key]
        size = DT[t["dtype"]].itemsize
        a = self.off + t["data_offsets"][0] + first * size
        data = bytearray(get(*self.at, a, a + count * size))
        return torch.frombuffer(data, dtype=torch.uint8).view(DT[t["dtype"]])

    def rows(self, key, r0, r1):
        """Rows r0..r1 of `key` along its first axis."""
        inner = self.hdr[key]["shape"][1:]
        n = int(torch.tensor(inner).prod())
        return self.span(key, r0 * n, (r1 - r0) * n).reshape(r1 - r0, *inner)


def fp8_reader(w, s):
    q = types.SimpleNamespace(quantization_config=FineGrainedFP8Config(weight_block_size=(128, 128)))
    return Fp8Dequantize(q)._dequantize_one(w, s, torch.bfloat16)


def ct_fp8_reader(w, s):
    args = QuantizationArgs(num_bits=8, type="float", strategy="block", block_structure=[128, 128], symmetric=True)
    return FloatQuantizationCompressor.decompress({"weight": w, "weight_scale": s},
                                                  QuantizationScheme(targets=["Linear"], weights=args))["weight"]


def ct_mxfp4_reader(w, s):
    args = QuantizationArgs(num_bits=4, type="float", strategy="group", group_size=32, symmetric=True,
                            scale_dtype=torch.uint8)
    scheme = QuantizationScheme(targets=["Linear"], weights=args, format="mxfp4-pack-quantized")
    return MXFP4PackedCompressor.decompress({"weight_packed": w, "weight_scale": s}, scheme)["weight"].to(torch.bfloat16)


def gpt_oss_reader(blocks, scales):
    d = tempfile.mkdtemp()
    save_file({"m.blocks": blocks.contiguous(), "m.scales": scales.contiguous()}, os.path.join(d, "m.safetensors"))
    return Checkpoint(d, torch.device("cpu"))._get_mxfp4_tensor("m.blocks", "m.scales", dtype=torch.bfloat16)


def write(case, config, tensors, expect):
    d = HERE / case
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(d / "model.safetensors"))
    save_file({k: v.contiguous() for k, v in expect.items()}, str(HERE / f"{case}.expect.safetensors"))
    print(case, {k: (str(v.dtype), list(v.shape)) for k, v in tensors.items()})


def qwen3_fp8():
    fp8 = Shard("Qwen/Qwen3-4B-Instruct-2507-FP8", "8591804019c8b22094c3b5b4454e0edc05dffc98", "model.safetensors")
    bf16 = Shard("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554",
                 "model-00001-of-00003.safetensors")
    t, e = {}, {}
    # q_proj: a whole tile; down_proj: 64 rows, a short tile.
    for m, rows in (("model.layers.0.self_attn.q_proj", 128), ("model.layers.0.mlp.down_proj", 64)):
        w = fp8.rows(f"{m}.weight", 0, rows)[:, :128]
        s = fp8.rows(f"{m}.weight_scale_inv", 0, 1)[:, :1]
        t[f"{m}.weight"], t[f"{m}.weight_scale_inv"] = w, s
        e[f"dequant.{m}.weight"] = fp8_reader(w, s)
        e[f"source.{m}.weight"] = bf16.rows(f"{m}.weight", 0, rows)[:, :128]
    t["model.layers.0.input_layernorm.weight"] = fp8.rows("model.layers.0.input_layernorm.weight", 0, 2560)
    config = {"quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                      "weight_block_size": [128, 128]}, "torch_dtype": "bfloat16"}
    write("real-qwen3-fp8", config, t, e)


def deepseek():
    v31 = Shard("deepseek-ai/DeepSeek-V3.1", "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db",
                "model-00001-of-000163.safetensors")
    m = "model.layers.0.self_attn.kv_a_proj_with_mqa"
    w, s = v31.rows(f"{m}.weight", 512, 576)[:, :128], v31.rows(f"{m}.weight_scale_inv", 4, 5)[:, :1]
    config = {"quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                      "weight_block_size": [128, 128], "scale_fmt": "ue8m0"}, "torch_dtype": "bfloat16"}
    write("real-deepseek-v3.1", config, {f"{m}.weight": w, f"{m}.weight_scale_inv": s},
          {f"dequant.{m}.weight": fp8_reader(w, s)})
    v4 = Shard("deepseek-ai/DeepSeek-V4-Flash", "60d8d70770c6776ff598c94bb586a859a38244f1",
               "model-00003-of-00046.safetensors")
    a, x = "layers.1.attn.wkv", "layers.1.ffn.experts.0.w2"
    wa, sa = v4.rows(f"{a}.weight", 0, 16)[:, :256], v4.rows(f"{a}.scale", 0, 1)[:, :2]
    wx, sx = v4.rows(f"{x}.weight", 0, 2), v4.rows(f"{x}.scale", 0, 2)
    config = {"quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                      "scale_fmt": "ue8m0", "weight_block_size": [128, 128]},
              "expert_dtype": "fp4", "torch_dtype": "bfloat16"}
    t = {f"{a}.weight": wa, f"{a}.scale": sa, f"{x}.weight": wx, f"{x}.scale": sx,
         "layers.1.attn.kv_norm.weight": v4.rows("layers.1.attn.kv_norm.weight", 0, 512)}
    write("real-deepseek-v4", config, t, {f"dequant.{a}.weight": fp8_reader(wa, sa),
                                          f"dequant.{x}.weight": fp8_reader(wx, sx)})


def ct():
    rev = "5def00c475640c0527e3a7067b1a26b94decefb5"
    block = Shard("RedHatAI/Qwen3-0.6B-FP8-BLOCK", rev, "model.safetensors")
    bf16 = Shard("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", "model.safetensors")
    m = "model.layers.0.self_attn.q_proj"
    w, s = block.rows(f"{m}.weight", 0, 128)[:, :128], block.rows(f"{m}.weight_scale", 0, 1)[:, :1]
    config = json.loads(get("RedHatAI/Qwen3-0.6B-FP8-BLOCK", rev, "config.json"))
    config = {"quantization_config": config["quantization_config"], "torch_dtype": "bfloat16"}
    write("real-ct-fp8-block", config, {f"{m}.weight": w, f"{m}.weight_scale": s},
          {f"dequant.{m}.weight": ct_fp8_reader(w, s), f"source.{m}.weight": bf16.rows(f"{m}.weight", 0, 128)[:, :128]})
    rev = "4399e1c0764beafc23e7a5465391132c1a56dc30"
    mx = Shard("nm-testing/TinyLlama-1.1B-Chat-v1.0-MXFP4", rev, "model.safetensors")
    orig = Shard("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "fe8a4ea1ffedaf415f4da2f062534de366a451e6", "model.safetensors")
    m = "model.layers.0.self_attn.k_proj"
    w, s = mx.rows(f"{m}.weight_packed", 0, 4), mx.rows(f"{m}.weight_scale", 0, 4)
    config = json.loads(get("nm-testing/TinyLlama-1.1B-Chat-v1.0-MXFP4", rev, "config.json"))
    config = {"quantization_config": config["quantization_config"], "torch_dtype": "bfloat16"}
    write("real-ct-mxfp4", config, {f"{m}.weight_packed": w, f"{m}.weight_scale": s},
          {f"dequant.{m}.weight": ct_mxfp4_reader(w, s), f"source.{m}.weight": orig.rows(f"{m}.weight", 0, 4)})


def gpt_oss():
    """gpt-oss's [E, N, K] blocks read as transformers' [E, K, N] tensors (`transposed: true`): experts 0 and 1, the
    first 64 of N and of K, so `down_proj` is square, the case a transpose would pass silently. The source is
    unsloth/gpt-oss-20b-BF16, transformers' own BF16 export of the same weights."""
    oa = Shard("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee", "model-00000-of-00002.safetensors")
    un = Shard("unsloth/gpt-oss-20b-BF16", "5a6fe4c0c1a86147cb9dc1bb99486e61d12a3c9e", UNSLOTH_LAYER0)
    t, e = {}, {}
    for proj, n in (("down_proj", 2880), ("gate_up_proj", 5760)):
        m, k = f"model.layers.0.mlp.experts.{proj}", 2880
        blocks = torch.stack([oa.span(f"{m}_blocks", x * n * k // 2, 64 * k // 2).reshape(64, k // 32, 16)[:, :2]
                              for x in (0, 1)])
        scales = torch.stack([oa.span(f"{m}_scales", x * n * k // 32, 64 * k // 32).reshape(64, k // 32)[:, :2]
                              for x in (0, 1)])
        t[f"{m}_blocks"], t[f"{m}_scales"] = blocks, scales
        t[f"{m}_bias"] = oa.rows(f"{m}_bias", 0, 2)[:, :64]
        e[f"dequant.{m}"] = convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16)
        e[f"source.{m}"] = torch.stack([un.span(m, x * k * n, 64 * n).reshape(64, n)[:, :64] for x in (0, 1)])
    config = {"quantization_config": {"modules_to_not_convert": ["model.layers.*.self_attn", "model.layers.*.mlp.router",
                                                                 "model.embed_tokens", "lm_head"],
                                      "quant_method": "mxfp4"}}
    write("real-gpt-oss", config, t, e)
    # AMD Quark's re-quantization of expert 0 (per-expert nn.Linear weights, [out, in]): OpenAI's values, exactly.
    q = Shard("amd/gpt-oss-20b-w-mxfp4-a-bf16", "34f332d130203b37d3c7715874254cd7b5447f51", "model.safetensors")
    k = "model.layers.0.mlp.experts.0.down_proj"
    w, s = q.rows(f"{k}.weight", 0, 4), q.rows(f"{k}.weight_scale", 0, 4)
    ob = oa.span("model.layers.0.mlp.experts.down_proj_blocks", 0, 4 * 1440).reshape(4, 90, 16)
    os_ = oa.span("model.layers.0.mlp.experts.down_proj_scales", 0, 4 * 90).reshape(4, 90)
    config = json.loads(get("amd/gpt-oss-20b-w-mxfp4-a-bf16", "34f332d130203b37d3c7715874254cd7b5447f51",
                            "config.json"))["quantization_config"]
    config = {"quantization_config": {**config, "exclude": []}, "torch_dtype": "bfloat16"}
    write("real-quark-mxfp4", config, {f"{k}.weight": w, f"{k}.weight_scale": s},
          {f"dequant.{k}.weight": gpt_oss_reader(w.reshape(4, 90, 16), s), f"source.{k}.weight": gpt_oss_reader(ob, os_)})


if __name__ == "__main__":
    qwen3_fp8()
    deepseek()
    ct()
    gpt_oss()
