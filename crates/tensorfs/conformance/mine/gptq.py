#!/usr/bin/env python3
"""Writes vectors/cases/gptq.json: `gptq.int{2,3,4,8}` decode cases (spec §6) from the real producers, on CPU.

- GPTQModel's TorchLinear packs (`pack_original`) codes, zeros, fp16 scales and a g_idx (column order, or an
  activation order); `checkpoint_format: gptq` (v1) is its own v2 → v1 export
  (`convert_gptq_v2_to_v1_format_module`). Its offline dequantizer (`convert_gptq_file`, which reads a safetensors
  file the way a GPTQ checkpoint is read) gives the expected bytes, and agrees with the module's
  `dequantize_weight` on every case without a zero of 0.
- AutoGPTQ 0.7.1's `qlinear_cuda_old.QuantLinear` (its own file from the PyPI sdist) packs v1, and its CPU forward
  on an identity input gives the expected F16 bytes.
- A real 8-row band of Qwen2.5-0.5B-Instruct-GPTQ-Int4 by HTTP range, read by GPTQModel's dequantizer.

A zero of 0 is GPTQ v1's corner: v1 stores `z − 1`, and the spec reads each slot as `(r + 1) mod 2^b`, as
AutoGPTQ's and GPTQModel's offline dequantizers do. GPTQModel's v1 export subtracts `0x11111111` from whole words, so
the slot after a zero of 0 borrows one; `v1-zero-borrow` pins how those bytes decode.

Each producer case carries `pack`, the codes and zeros the producer packed (`I16`, `[R, C]` and `[W, N]`):
TensorFS's `pack` reproduces the parts from them bit for bit (tests/format_pack.rs).

    CUDA_VISIBLE_DEVICES= uv run --no-project --python 3.12 --with torch==2.14.1 --with gptqmodel==7.6.0 \\
        python -I conformance/mine/gptq.py
"""

import base64
import importlib.util
import io
import json
import struct
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import gptqmodel
import numpy as np
import torch
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.model import convert_gptq_v2_to_v1_format_module
from gptqmodel.utils.model_dequant import convert_gptq_file
from safetensors.torch import save_file

OUT = Path(__file__).resolve().parents[1] / "vectors" / "cases" / "gptq.json"
AUTOGPTQ = ("https://files.pythonhosted.org/packages/source/a/auto_gptq/auto_gptq-0.7.1.tar.gz",
            "auto_gptq-0.7.1/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py")
REAL = ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", "c34a4a91629f09f73a285f32dbd26106b033c654", "model.safetensors")
DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16"}
rng = np.random.default_rng(549)


def get(url, start=None, end=None) -> bytes:
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end - 1}")
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    assert start is None or len(data) == end - start, "short read"
    return data


def autogptq():
    """AutoGPTQ's own qlinear_cuda_old.py, from its sdist."""
    with tarfile.open(fileobj=io.BytesIO(get(AUTOGPTQ[0]))) as tar:
        source = tar.extractfile(AUTOGPTQ[1]).read()
    path = Path(tempfile.mkdtemp()) / "qlinear_cuda_old.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("qlinear_cuda_old", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def b64(t) -> str:
    t = torch.as_tensor(t)
    return base64.b64encode(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).decode()


def i16(a) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a, np.int64)).to(torch.int16)


def codes(n, k, g, bits, sym, act_order, zero_floor=1):
    """Codes, zeros and fp16 scales per group ([N, W]), and g_idx: GPTQ's activation order groups the columns in a
    permuted order (g_idx[perm[i]] = i // g)."""
    w = -(-k // g)
    maxq = (1 << bits) - 1
    q = rng.integers(0, maxq + 1, (n, k))
    z = np.full((n, w), 1 << (bits - 1)) if sym else rng.integers(zero_floor, maxq + 1, (n, w))
    s = torch.tensor(np.abs(rng.standard_normal((n, w))) * 0.02 + 0.002, dtype=torch.float16).float()
    g_idx = np.arange(k) // g
    if act_order:
        g_idx = g_idx[np.argsort(rng.permutation(k))]
    return q, z, s, g_idx


def gptqmodel_parts(q, z, s, g_idx, bits, g, sym, act_order, v1):
    """GPTQModel's TorchLinear packs W = (q − z)·s, whose round((W + z·s)/s) gives back q exactly."""
    n, k = q.shape
    gi = torch.as_tensor(g_idx, dtype=torch.int32)
    weight = (torch.as_tensor(q, dtype=torch.float32) - torch.as_tensor(z, dtype=torch.float32)[:, g_idx]) * s[:, g_idx]
    linear = torch.nn.Linear(k, n, bias=False)
    linear.weight.data = weight
    m = TorchLinear(bits=bits, group_size=g, sym=sym, desc_act=act_order, in_features=k, out_features=n)
    m.pack_original(linear, s, torch.as_tensor(z, dtype=torch.float32), gi)
    m.post_init()
    module_view = m.dequantize_weight().T  # v2 zeros, fp16 scales: [N, K]
    with torch.no_grad():
        m.scales = m.scales.float()
        exact = m.dequantize_weight().T.float()
        m.scales = m.scales.half()
    assert torch.equal(module_view.float(), exact.half().float())
    if v1:
        convert_gptq_v2_to_v1_format_module(m, QuantizeConfig(bits=bits))
    return {"qweight": m.qweight, "qzeros": m.qzeros, "scales": m.scales, "g_idx": m.g_idx.to(torch.int32)}, exact


def offline(parts, bits, g, sym, act_order, v1):
    """GPTQModel's offline dequantizer on a safetensors file holding the parts, in F32."""
    path = Path(tempfile.mkdtemp()) / "m.safetensors"
    save_file({f"m.{k}": v.contiguous() for k, v in parts.items()}, str(path))
    config = {"bits": bits, "group_size": g, "sym": sym, "desc_act": act_order,
              "checkpoint_format": "gptq" if v1 else "gptq_v2"}
    return convert_gptq_file(path, torch.float32, config, "cpu")["m.weight"]


def case(key, parts, bits, g, sym, act_order, v1, expect, dtype, producer, pack=None):
    n, k = parts["scales"].shape[1], parts["g_idx"].shape[0]
    c = {
        "component": "gptq", "key": key, "dtype": DTYPES[dtype], "shape": [n, k],
        "encoding": {"scheme": f"gptq.int{bits}",
                     "params": {"group_size": g, "symmetric": sym, "act_order": act_order, "zero_offset": int(v1)}},
        "parts": [{"role": r, "dtype": DTYPES.get(t.dtype, "I32"), "shape": list(t.shape), "bits": b64(t)}
                  for r, t in parts.items()],
        "expect": b64(expect.to(dtype)), "producer": producer,
    }
    if pack is not None:
        q, z = pack
        c["pack"] = {"q": b64(i16(q)), "zeros": b64(i16(np.asarray(z).T))}
    return c


def gptqmodel_case(key, bits, n, k, g, sym, act_order, v1, dtype):
    q, z, s, g_idx = codes(n, k, k if g == -1 else g, bits, sym, act_order)
    parts, exact = gptqmodel_parts(q, z, s, g_idx, bits, g, sym, act_order, v1)
    assert torch.equal(offline(parts, bits, g, sym, act_order, v1), exact)
    producer = f"GPTQModel {gptqmodel.__version__} TorchLinear {'v1 export' if v1 else 'gptq_v2'}"
    return case(key, parts, bits, g, sym, act_order, v1, exact, dtype, producer, (q, z))


def zero_borrow_case():
    """A v1 export with zeros of 0: the word subtraction borrows from the next slot."""
    bits, n, k, g = 4, 16, 64, 32
    q, z, s, g_idx = codes(n, k, g, bits, False, False, zero_floor=0)
    z[:, 0] = 0
    parts, exact = gptqmodel_parts(q, z, s, g_idx, bits, g, False, False, True)
    expect = offline(parts, bits, g, False, False, True)
    assert not torch.equal(expect, exact), "a borrow changes the next zero"
    producer = f"GPTQModel {gptqmodel.__version__} v1 export, read by its offline dequantizer"
    return case("v1-zero-borrow-int4-16x64", parts, bits, g, False, False, True, expect, torch.float32, producer)


def autogptq_case(cuda_old, key, bits, n, k, g, sym, zero_floor=1):
    """AutoGPTQ's packer (v1, column order) and its CPU dequantizing forward."""
    q, z, s, g_idx = codes(n, k, g, bits, sym, False, zero_floor)
    if zero_floor == 0:
        z[:, -1] = 0  # a whole group's zeros: AutoGPTQ's z − 1 as uint32 sets every slot of their words
    weight = (torch.as_tensor(q, dtype=torch.float32) - torch.as_tensor(z, dtype=torch.float32)[:, g_idx]) * s[:, g_idx]
    linear = torch.nn.Linear(k, n, bias=False)
    linear.weight.data = weight
    m = cuda_old.QuantLinear(bits, g, k, n, bias=False, use_cuda_fp16=False)
    m.pack(linear, s, torch.as_tensor(z, dtype=torch.int32), torch.as_tensor(g_idx, dtype=torch.int32))
    m.scales = m.scales.half()
    expect = m(torch.eye(k, dtype=torch.float16)).T
    parts = {"qweight": m.qweight, "qzeros": m.qzeros, "scales": m.scales,
             "g_idx": torch.as_tensor(g_idx, dtype=torch.int32)}
    return case(key, parts, bits, g, sym, False, True, expect, torch.float16, "AutoGPTQ 0.7.1 qlinear_cuda_old", (q, z))


def real_case():
    """Rows 0..8 of layer 0's q_proj: its qweight columns, a qzeros word, scales and the whole g_idx."""
    url = "https://huggingface.co/{}/resolve/{}/{}".format(*REAL)
    n = struct.unpack("<Q", get(url, 0, 8))[0]
    hdr, off = json.loads(get(url, 8, 8 + n)), 8 + n

    def cols(name, c0, c1):
        t = hdr[f"model.layers.0.self_attn.q_proj.{name}"]
        rows, width = t["shape"]
        size = {"I32": 4, "F16": 2}[t["dtype"]]
        a, _ = t["data_offsets"]
        data = get(url, off + a, off + a + rows * width * size)
        dt = {"I32": torch.int32, "F16": torch.float16}[t["dtype"]]
        return torch.frombuffer(bytearray(data), dtype=dt).reshape(rows, width)[:, c0:c1].contiguous()

    t = hdr["model.layers.0.self_attn.q_proj.g_idx"]
    a, b = t["data_offsets"]
    g_idx = torch.frombuffer(bytearray(get(url, off + a, off + b)), dtype=torch.int32)
    parts = {"qweight": cols("qweight", 0, 8), "qzeros": cols("qzeros", 0, 1), "scales": cols("scales", 0, 8),
             "g_idx": g_idx}
    expect = offline(parts, 4, 128, True, False, True)
    producer = f"{REAL[0]}@{REAL[1]} model.layers.0.self_attn.q_proj rows 0..8, by GPTQModel {gptqmodel.__version__}"
    return case("qwen2.5-0.5b-gptq-int4-q_proj-8x896", parts, 4, 128, True, False, True, expect, torch.float16,
                producer)


def doubled(bits: str) -> str:
    """A part's bytes twice: the length of the same shape in a dtype twice as wide."""
    return base64.b64encode(base64.b64decode(bits) * 2).decode()


def refusals(base):
    def twin(key, **change):
        c = json.loads(json.dumps(base))
        for k in ("expect", "pack"):
            c.pop(k, None)
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
    g_idx = np.frombuffer(base64.b64decode(base["parts"][3]["bits"]), "<i4")
    swapped = g_idx.copy()
    swapped[[0, -1]] = swapped[[-1, 0]]
    return [
        twin("gptq-symmetric-zeros-differ", encoding__params__symmetric=True),
        twin("gptq-act-order-false-permuted", parts__3__bits=b64(torch.as_tensor(swapped))),
        twin("gptq-zero-offset-2", encoding__params__zero_offset=2),
        twin("gptq-group-size-0", encoding__params__group_size=0),
        twin("gptq-g-idx-missing", parts=base["parts"][:3]),
        twin("gptq-scales-f32", parts__2__dtype="F32", parts__2__bits=doubled(base["parts"][2]["bits"])),
        twin("gptq-act-order-missing", encoding__params__act_order=None),
        twin("gptq-unknown-param", encoding__params__desc_act=False),
    ]


def main():
    torch.manual_seed(49)
    cuda_old = autogptq()
    cases = [
        gptqmodel_case("v1-sym-int4-32x64-f16", 4, 32, 64, 32, True, False, True, torch.float16),
        gptqmodel_case("v1-act-order-int4-32x64-bf16", 4, 32, 64, 32, False, True, True, torch.bfloat16),
        gptqmodel_case("v2-int4-32x64-f32", 4, 32, 64, 32, False, False, False, torch.float32),
        gptqmodel_case("v1-act-order-int3-32x64-bf16", 3, 32, 64, 32, False, True, True, torch.bfloat16),
        gptqmodel_case("v2-sym-int3-32x64-f16", 3, 32, 64, 16, True, False, False, torch.float16),
        gptqmodel_case("v2-whole-row-int2-32x64-bf16", 2, 32, 64, -1, False, False, False, torch.bfloat16),
        gptqmodel_case("v1-sym-int8-8x64-f16", 8, 8, 64, 32, True, False, True, torch.float16),
        gptqmodel_case("v2-act-order-int8-8x64-bf16", 8, 8, 64, 16, False, True, False, torch.bfloat16),
        zero_borrow_case(),
        autogptq_case(cuda_old, "autogptq-int4-32x64-f16", 4, 32, 64, 32, False),
        autogptq_case(cuda_old, "autogptq-int3-32x64-f16", 3, 32, 64, 32, False),
        autogptq_case(cuda_old, "autogptq-zero-0-int4-32x64-f16", 4, 32, 64, 32, False, zero_floor=0),
        real_case(),
    ]
    cases += refusals(cases[2])
    OUT.write_text(json.dumps({"cases": cases}))
    print(f"{len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
