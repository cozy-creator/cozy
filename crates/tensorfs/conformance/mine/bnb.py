#!/usr/bin/env python3
"""Writes vectors/cases/bnb.json: `bnb.nf4`, `bnb.fp4` and `bnb.int8` decode cases (spec §6) from bitsandbytes'
own CPU backend.

- Linear4bit and Linear8bitLt quantize on CPU and their state dicts are the parts (the bytes transformers saves);
  bitsandbytes' `dequantize_4bit` (the nested absmax through `dequantize_blockwise`) and `int8_vectorwise_dequant`
  give the expected bytes. These cases carry `source`, the `F32` values quantized: TensorFS's closed-form encoders
  reproduce their parts bit for bit (tests/format_bnb.rs).
- Real rows by HTTP range at pinned revisions: unsloth's double-quantized nf4 and lllyasviel's Forge nf4 (no
  config, its own quant_state only), a nested block's first rows; the expected bytes are bitsandbytes' dequantize of
  those bytes, the quant_state's `shape` set to the rows taken.

    uv venv --python 3.12 /tmp/mine && uv pip install --python /tmp/mine torch==2.14.1 bitsandbytes==0.50.2 numpy
    /tmp/mine/bin/python -I conformance/mine/bnb.py
"""

import base64
import json
import struct
import urllib.request
from pathlib import Path

import bitsandbytes as bnb
import bitsandbytes.functional as F
import numpy as np
import torch

OUT = Path(__file__).resolve().parents[1] / "vectors" / "cases" / "bnb.json"
HF = "https://huggingface.co/{}/resolve/{}/{}"
DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16"}
ST = {"F32": torch.float32, "U8": torch.uint8}
torch._dynamo.config.disable = True  # bitsandbytes' torch.compile'd fallbacks run eagerly
gen = torch.Generator().manual_seed(49)


def b64(t: torch.Tensor) -> str:
    return base64.b64encode(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).decode()


def part(role, t, dtype=None):
    return {"role": role, "dtype": dtype or {torch.uint8: "U8", torch.int8: "I8", torch.float32: "F32"}[t.dtype],
            "shape": list(t.shape), "bits": b64(t)}


def four_bit(sd):
    """A Linear4bit state dict as the scheme's parts and params."""
    blob = next(k for k in sd if ".quant_state.bitsandbytes__" in k)
    qs = json.loads(bytes(sd[blob].numpy()))
    dq = "nested_offset" in qs
    roles = ["weight", "absmax", "quant_map"] + (["nested_absmax", "nested_quant_map"] if dq else [])
    parts = [part(r, sd["weight" if r == "weight" else "weight." + r]) for r in roles]
    if dq:
        assert qs["nested_blocksize"] == 256 and qs["nested_dtype"] == "float32"
        parts.append(part("nested_offset", torch.tensor([qs["nested_offset"]], dtype=torch.float32).reshape(())))
    params = {"block_size": qs["blocksize"], "double_quant": dq}
    return "bnb." + qs["quant_type"], params, parts, qs


def dequant4(sd, dtype):
    w = sd["weight"]
    qs = F.QuantState.from_dict({k[len("weight."):]: v for k, v in sd.items() if k.startswith("weight.")}, "cpu")
    assert qs.dtype == dtype
    return F.dequantize_4bit(w, qs)


def linear4(x, quant_type, block, dq):
    lin = bnb.nn.Linear4bit(x.shape[1], x.shape[0], bias=False, compress_statistics=dq, quant_type=quant_type)
    lin.weight = bnb.nn.Params4bit(x, requires_grad=False, compress_statistics=dq, quant_type=quant_type,
                                   blocksize=block)
    return lin.to("cpu").state_dict()


def synthetic4(key, x, quant_type, block, dq):
    sd = linear4(x, quant_type, block, dq)
    scheme, params, parts, qs = four_bit(sd)
    assert params == {"block_size": block, "double_quant": dq} and qs["shape"] == list(x.shape)
    expect = dequant4(sd, x.dtype)
    return {"component": scheme, "key": key, "dtype": DTYPES[x.dtype], "shape": list(x.shape),
            "encoding": {"scheme": scheme, "params": params}, "parts": parts, "expect": b64(expect),
            "source": b64(x.float()), "producer": f"bitsandbytes {bnb.__version__} Linear4bit (CPU)"}


def synthetic8(key, x):
    lin = bnb.nn.Linear8bitLt(x.shape[1], x.shape[0], bias=False, has_fp16_weights=False)
    lin.weight = bnb.nn.Int8Params(x, requires_grad=False, has_fp16_weights=False)
    sd = lin.to("cpu").state_dict()
    assert sorted(sd) == ["SCB", "weight", "weight_format"] and sd["weight_format"].item() == 0
    expect = F.int8_vectorwise_dequant(sd["weight"], sd["SCB"]).to(x.dtype)
    return {"component": "bnb.int8", "key": key, "dtype": DTYPES[x.dtype], "shape": list(x.shape),
            "encoding": {"scheme": "bnb.int8", "params": {}},
            "parts": [part("weight", sd["weight"]), part("SCB", sd["SCB"]), part("weight_format", sd["weight_format"])],
            "expect": b64(expect), "source": b64(x.float()),
            "producer": f"bitsandbytes {bnb.__version__} Linear8bitLt (CPU)"}


def weights(shape, dtype, scale=0.02):
    return (torch.randn(shape, generator=gen) * scale).to(dtype)


def get(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(req) as r:
        data = r.read()
    assert len(data) == end - start
    return data


def real4(repo, rev, file, module, shape):
    """The first `shape` elements of a real 4-bit weight (a nested block's first), read by HTTP range."""
    url = HF.format(repo, rev, file)
    n = struct.unpack("<Q", get(url, 0, 8))[0]
    header = json.loads(get(url, 8, 8 + n))

    def read(key, count=None):
        e = header[key]
        a, b = e["data_offsets"]
        width = {"U8": 1, "F32": 4}[e["dtype"]]
        data = get(url, 8 + n + a, 8 + n + (b if count is None else a + count * width))
        return torch.frombuffer(bytearray(data), dtype=ST[e["dtype"]]).clone()

    blob_key = next(k for k in header if k.startswith(module + ".weight.quant_state.bitsandbytes__"))
    qs = json.loads(bytes(read(blob_key).numpy()))
    count, block = shape[0] * shape[1], qs["blocksize"]
    assert count % block == 0 and ("nested_offset" not in qs or count <= block * 256)
    sd = {"weight": read(module + ".weight", count // 2).reshape(-1, 1),
          "weight.absmax": read(module + ".weight.absmax", count // block),
          "weight.quant_map": read(module + ".weight.quant_map")}
    if "nested_offset" in qs:
        sd["weight.nested_absmax"] = read(module + ".weight.nested_absmax", 1)
        sd["weight.nested_quant_map"] = read(module + ".weight.nested_quant_map")
    full, qs["shape"] = qs["shape"], shape
    sd["weight.quant_state.bitsandbytes__" + qs["quant_type"]] = torch.tensor(list(json.dumps(qs).encode()),
                                                                             dtype=torch.uint8)
    dtype = getattr(torch, qs["dtype"])
    scheme, params, parts, _ = four_bit(sd)
    return {"component": scheme, "key": f"{repo}@{rev[:9]} {module} {full} first {count}", "dtype": DTYPES[dtype],
            "shape": shape, "encoding": {"scheme": scheme, "params": params}, "parts": parts,
            "expect": b64(dequant4(sd, dtype)), "producer": f"{repo}@{rev} {file}"}


def refuse(c, key, edit):
    c = json.loads(json.dumps(c))
    for k in ("expect", "source", "producer"):
        c.pop(k, None)
    edit(c)
    return {**c, "component": "refuse", "key": key, "refuse": "decode"}


def set_part(role, **kv):
    return lambda c: next(p for p in c["parts"] if p["role"] == role).update(kv)


def drop_part(role):
    return lambda c: c.__setitem__("parts", [p for p in c["parts"] if p["role"] != role])


def main():
    cases = []
    for quant_type in ("nf4", "fp4"):
        for dq in (False, True):
            tag = f"{quant_type}{'-dq' if dq else ''}"
            cases.append(synthetic4(f"{tag}-64-8x96-bf16", weights((8, 96), torch.bfloat16), quant_type, 64, dq))
            cases.append(synthetic4(f"{tag}-128-6x80-f16", weights((6, 80), torch.float16), quant_type, 128, dq))
            # 259 elements: an odd count (a padded last byte) and a short last block.
            cases.append(synthetic4(f"{tag}-64-7x37-f32", weights((7, 37), torch.float32), quant_type, 64, dq))
            x = weights((6, 100), torch.bfloat16)
            x[1, 28:] = 0  # a block of zeros, and one of tiny values
            x[3, 20:] = 1e-30
            cases.append(synthetic4(f"{tag}-128-6x100-zero-blocks-bf16", x, quant_type, 128, dq))
    # 261 blocks: two nested blocks, the second short; the absmax spread exercises the nested map.
    x = weights((17, 490), torch.bfloat16) * torch.logspace(-1, 1, 17).reshape(-1, 1).to(torch.bfloat16)
    cases.append(synthetic4("nf4-dq-32-17x490-bf16", x, "nf4", 32, True))
    cases.append(synthetic4("fp4-dq-256-2x400-bf16", weights((2, 400), torch.bfloat16), "fp4", 256, True))
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = weights((16, 64), dtype)
        x[5] = 0  # a zero row quantizes to zeros
        x[9, 3] = 0.5  # an outlier
        cases.append(synthetic8(f"16x64-{DTYPES[dtype].lower()}", x))
    unsloth = ("unsloth/Llama-3.2-1B-Instruct-bnb-4bit", "fcecd050a91025151bbf518e41a5577c356efdce", "model.safetensors")
    cases.append(real4(*unsloth, "model.layers.0.self_attn.k_proj", [2, 2048]))
    cases.append(real4(*unsloth, "model.layers.15.mlp.down_proj", [1, 4096]))
    cases.append(real4("lllyasviel/flux1-dev-bnb-nf4", "32d13814357ca76c52638d4644d0f7384208dc96",
                       "flux1-dev-bnb-nf4-v2.safetensors", "model.diffusion_model.double_blocks.0.img_attn.proj",
                       [2, 3072]))

    by_key = {c["key"]: c for c in cases}
    nf4, nf4_dq, int8 = by_key["nf4-64-8x96-bf16"], by_key["nf4-dq-64-8x96-bf16"], by_key["16x64-f16"]
    fp4_map = next(p for p in by_key["fp4-64-8x96-bf16"]["parts"] if p["role"] == "quant_map")["bits"]
    nf4_map = np.frombuffer(base64.b64decode(next(p for p in nf4["parts"] if p["role"] == "quant_map")["bits"]),
                            np.uint32).copy()
    nf4_map[3] ^= 1
    cases += [
        refuse(nf4, "nf4-with-fp4-map", set_part("quant_map", bits=fp4_map)),
        refuse(nf4, "nf4-map-one-ulp-off", set_part("quant_map", bits=base64.b64encode(nf4_map.tobytes()).decode())),
        refuse(nf4, "block-size-48", lambda c: c["encoding"]["params"].update(block_size=48)),
        refuse(nf4, "double-quant-missing", lambda c: c["encoding"]["params"].pop("double_quant")),
        refuse(nf4, "weight-flat", set_part("weight", shape=[384])),
        refuse(nf4, "absmax-codes-without-double-quant", lambda c: c["encoding"]["params"].update(double_quant=True)),
        refuse(nf4_dq, "double-quant-without-nested-offset", drop_part("nested_offset")),
        refuse(nf4_dq, "nested-parts-without-double-quant", lambda c: c["encoding"]["params"].update(
            double_quant=False)),
        refuse(nf4, "nf4-int8-dtype", lambda c: c.update(dtype="I8")),
        refuse(int8, "int8-weight-format-col", set_part("weight_format", bits=base64.b64encode(b"\x01").decode())),
        refuse(int8, "int8-scb-per-column", set_part("SCB", shape=[64], bits=base64.b64encode(bytes(256)).decode())),
        refuse(int8, "int8-unknown-param", lambda c: c["encoding"]["params"].update(threshold=6)),
    ]
    OUT.write_text(json.dumps({"cases": cases}))
    print(f"{len(cases)} cases, bitsandbytes {bnb.__version__}, torch {torch.__version__}")


if __name__ == "__main__":
    main()
