#!/usr/bin/env python3
"""Writes vectors/cases/awq.json: `awq.int4` decode cases (spec §6) from the real producers, on CPU.

- AutoAWQ 0.2.9 (archived; its own files from the PyPI sdist, loaded without its model registry): `WQLinear_GEMM.
  from_linear` packs codes, zeros and fp16 scales; its `dequantize_gemm` gives the expected bytes.
- GPTQModel's AWQ GEMM packer (`AwqTorchLinear.pack`) writes the same bytes from the same inputs, and keeps `BF16`
  scales when given them.
- A real 8-row band of Qwen2.5-0.5B-Instruct-AWQ by HTTP range, read by `dequantize_gemm`.

Each producer case carries `pack`, the codes and zeros the producer packed (`I16`, `[R, C]` and `[W, N]`):
TensorFS's `pack` reproduces the parts from them bit for bit (tests/format_pack.rs).

    CUDA_VISIBLE_DEVICES= uv run --no-project --python 3.12 --with torch==2.14.1 --with gptqmodel==7.6.0 \\
        python -I conformance/mine/awq.py
"""

import base64
import importlib.util
import io
import json
import struct
import sys
import tarfile
import tempfile
import types
import urllib.request
from pathlib import Path

import gptqmodel
import numpy as np
import torch
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

OUT = Path(__file__).resolve().parents[1] / "vectors" / "cases" / "awq.json"
AUTOAWQ = "https://files.pythonhosted.org/packages/source/a/autoawq/autoawq-0.2.9.tar.gz"
REAL = ("Qwen/Qwen2.5-0.5B-Instruct-AWQ", "db09cd27ead7fee40cdee309693cf83601b9c899", "model.safetensors")
DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16"}
rng = np.random.default_rng(4049)


def get(url, start=None, end=None) -> bytes:
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end - 1}")
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    assert start is None or len(data) == end - start, "short read"
    return data


def autoawq():
    """AutoAWQ's GEMM module and packing utilities: its files, under package shells that skip `awq/__init__.py`
    (which imports every model class)."""
    root = Path(tempfile.mkdtemp())
    with tarfile.open(fileobj=io.BytesIO(get(AUTOAWQ))) as tar:
        for m in tar.getmembers():
            if m.name.startswith("autoawq-0.2.9/awq/") and m.isfile():
                path = root / m.name.split("/", 1)[1]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(tar.extractfile(m).read())
    for package in ("awq", "awq.utils", "awq.modules", "awq.modules.linear"):
        shell = types.ModuleType(package)
        shell.__path__ = [str(root.joinpath(*package.split(".")))]
        sys.modules[package] = shell

    def load(name):
        spec = importlib.util.spec_from_file_location(name, root.joinpath(*name.split(".")).with_suffix(".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    for name in ("awq.utils.module", "awq.utils.utils", "awq.utils.packing_utils"):
        load(name)
    return load("awq.modules.linear.gemm"), sys.modules["awq.utils.packing_utils"]


def b64(t) -> str:
    t = torch.as_tensor(t)
    return base64.b64encode(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).decode()


def i16(a) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a, np.int64)).to(torch.int16)


def codes(n, k, g):
    """Codes [N, K], zeros and fp16-exact scales per group ([W, N])."""
    w = k // g
    q = rng.integers(0, 16, (n, k))
    z = rng.integers(0, 16, (w, n))
    s = torch.tensor(np.abs(rng.standard_normal((w, n))) * 0.02 + 0.002, dtype=torch.float16).float()
    return q, z, s


def linear_of(q, z, s, g):
    """W = (q − z)·s, from which each packer's round((W + z·s)/s) gives back q exactly."""
    n, k = q.shape
    zf = torch.as_tensor(z, dtype=torch.float32).T.repeat_interleave(g, dim=1)
    weight = (torch.as_tensor(q, dtype=torch.float32) - zf) * s.T.repeat_interleave(g, dim=1)
    linear = torch.nn.Linear(k, n, bias=False)
    linear.weight.data = weight
    return linear


def case(key, parts, g, expect, dtype, producer, pack=None):
    n, k = parts["scales"].shape[1], parts["qweight"].shape[0]
    c = {
        "component": "awq", "key": key, "dtype": DTYPES[dtype], "shape": [n, k],
        "encoding": {"scheme": "awq.int4", "params": {"group_size": g}},
        "parts": [{"role": r, "dtype": DTYPES.get(t.dtype, "I32"), "shape": list(t.shape), "bits": b64(t)}
                  for r, t in parts.items()],
        "expect": b64(expect.to(dtype)), "producer": producer,
    }
    if pack is not None:
        c["pack"] = {"q": b64(i16(pack[0])), "zeros": b64(i16(pack[1]))}
    return c


def autoawq_case(gemm, utils, key, n, k, g, dtype):
    q, z, s = codes(n, k, g)
    linear = linear_of(q, z, s, g)
    m = gemm.WQLinear_GEMM.from_linear(linear, 4, g, scales=s, zeros=torch.as_tensor(z, dtype=torch.float32))
    parts = {"qweight": m.qweight, "qzeros": m.qzeros, "scales": m.scales}
    exact = utils.dequantize_gemm(m.qweight, m.qzeros, m.scales.float(), 4, g).T
    assert torch.equal(utils.dequantize_gemm(m.qweight, m.qzeros, m.scales, 4, g).T, exact.half())
    # GPTQModel's packer: the same bytes from the same inputs.
    p = AwqTorchLinear(bits=4, group_size=g, sym=False, desc_act=False, in_features=k, out_features=n)
    p.pack(linear, s.T.half(), torch.as_tensor(z, dtype=torch.float32).T)
    assert torch.equal(p.qweight, m.qweight) and torch.equal(p.qzeros, m.qzeros)
    return case(key, parts, g, exact, dtype, "AutoAWQ 0.2.9 WQLinear_GEMM (= GPTQModel AwqTorchLinear)", (q, z))


def gptqmodel_bf16_case(utils, key, n, k, g):
    """GPTQModel keeps BF16 scales; AutoAWQ's dequantizer reads them."""
    q, z, s = codes(n, k, g)
    s = s.to(torch.bfloat16).float()
    p = AwqTorchLinear(bits=4, group_size=g, sym=False, desc_act=False, in_features=k, out_features=n)
    p.pack(linear_of(q, z, s, g), s.T.to(torch.bfloat16), torch.as_tensor(z, dtype=torch.float32).T)
    assert p.scales.dtype == torch.bfloat16
    parts = {"qweight": p.qweight, "qzeros": p.qzeros, "scales": p.scales}
    exact = utils.dequantize_gemm(p.qweight, p.qzeros, p.scales.float(), 4, g).T
    return case(key, parts, g, exact, torch.bfloat16, f"GPTQModel {gptqmodel.__version__} AwqTorchLinear", (q, z))


def real_case(utils):
    """Rows 0..8 of layer 0's q_proj: one word column of qweight and qzeros, and 8 scale columns."""
    url = "https://huggingface.co/{}/resolve/{}/{}".format(*REAL)
    n = struct.unpack("<Q", get(url, 0, 8))[0]
    hdr, off = json.loads(get(url, 8, 8 + n)), 8 + n

    def cols(name, c0, c1):
        t = hdr[f"model.layers.0.self_attn.q_proj.{name}"]
        rows, width = t["shape"]
        dt, size = {"I32": (torch.int32, 4), "F16": (torch.float16, 2)}[t["dtype"]]
        a, _ = t["data_offsets"]
        data = get(url, off + a, off + a + rows * width * size)
        return torch.frombuffer(bytearray(data), dtype=dt).reshape(rows, width)[:, c0:c1].contiguous()

    parts = {"qweight": cols("qweight", 0, 1), "qzeros": cols("qzeros", 0, 1), "scales": cols("scales", 0, 8)}
    expect = utils.dequantize_gemm(parts["qweight"], parts["qzeros"], parts["scales"], 4, 128).T
    producer = f"{REAL[0]}@{REAL[1]} model.layers.0.self_attn.q_proj rows 0..8, by AutoAWQ 0.2.9 dequantize_gemm"
    return case("qwen2.5-0.5b-awq-q_proj-8x896", parts, 128, expect, torch.float16, producer)


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
    return [
        twin("awq-group-size-0", encoding__params__group_size=0),
        twin("awq-group-size-missing", encoding__params__group_size=None),
        twin("awq-zero-point-param", encoding__params__zero_point=True),
        twin("awq-qzeros-missing", parts=base["parts"][::2]),
        twin("awq-scales-f32", parts__2__dtype="F32", parts__2__bits=doubled(base["parts"][2]["bits"])),
        twin("awq-qweight-transposed", parts__0__shape=list(reversed(base["parts"][0]["shape"]))),
    ]


def main():
    gemm, utils = autoawq()
    cases = [
        autoawq_case(gemm, utils, "autoawq-32x64-g32-f16", 32, 64, 32, torch.float16),
        autoawq_case(gemm, utils, "autoawq-32x128-g64-bf16", 32, 128, 64, torch.bfloat16),
        autoawq_case(gemm, utils, "autoawq-32x64-g16-f32", 32, 64, 16, torch.float32),
        gptqmodel_bf16_case(utils, "gptqmodel-bf16-scales-16x64-g32", 16, 64, 32),
        real_case(utils),
    ]
    cases += refusals(cases[0])
    OUT.write_text(json.dumps({"cases": cases}))
    print(f"{len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
