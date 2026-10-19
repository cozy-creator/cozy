"""bitsandbytes checkpoints for `tests/quantized_import.rs`, laid out like `make.py`'s cases: `<case>/` and
`<case>.expect.safetensors` holding `source.<key>` (the BF16 weight before quantization) and `dequant.<key>`
(bitsandbytes' own dequantization, on CPU).

- `real-bnb-unsloth`: unsloth/Llama-3.2-1B-Instruct-bnb-4bit (nf4, double quant, `config.json` declaring
  bitsandbytes) against unsloth/Llama-3.2-1B-Instruct.
- `real-bnb-forge`: lllyasviel/flux1-dev-bnb-nf4 `flux1-dev-bnb-nf4-v2.safetensors` (nf4, no config: its
  quant_state documents declare it) against Comfy-Org/flux1-dev's BF16 `flux1-dev.safetensors`.
  Each takes a weight's first nested block (or rows) by HTTP range, its quant_state `shape` set to the rows taken.
- `bnb-fp4-dq`, `bnb-int8`: Linear4bit (fp4, double quant) and Linear8bitLt state dicts, with Transformers'
  `quantization_config`.

    uv venv --python 3.12 /tmp/mine && uv pip install --python /tmp/mine torch==2.14.1 bitsandbytes==0.50.2 safetensors
    /tmp/mine/bin/python -I make_bnb.py <this directory>
"""

import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

import bitsandbytes as bnb
import bitsandbytes.functional as F
import torch
from safetensors.torch import save_file

HF = "https://huggingface.co/{}/resolve/{}/{}"
torch._dynamo.config.disable = True  # bitsandbytes' torch.compile'd fallbacks run eagerly
OUT = Path(sys.argv[1])
gen = torch.Generator().manual_seed(6)
DTYPES = {"U8": torch.uint8, "F32": torch.float32, "BF16": torch.bfloat16}


class Remote:
    def __init__(self, repo, rev, file):
        self.url, self.name = HF.format(repo, rev, file), f"{repo}@{rev}/{file}"
        n = struct.unpack("<Q", self.get(0, 8))[0]
        raw = self.get(8, 8 + n)
        self.header_sha256, self.base, self.header = hashlib.sha256(raw).hexdigest(), 8 + n, json.loads(raw)

    def get(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        assert len(data) == end - start
        return data

    def tensor(self, key, count=None):
        """`key`'s first `count` elements (all of them by default)."""
        e = self.header[key]
        a, b = e["data_offsets"]
        dtype = DTYPES[e["dtype"]]
        if count is not None:
            b = a + count * dtype.itemsize
        return torch.frombuffer(bytearray(self.get(self.base + a, self.base + b)), dtype=dtype).clone()


def bf16(t):
    return t.to(torch.bfloat16).contiguous()


def dequant4(sd, w):
    qs = F.QuantState.from_dict({k[len(w) + 1:]: v for k, v in sd.items() if k.startswith(w + ".")}, "cpu")
    return F.dequantize_4bit(sd[w], qs)


def real(case, remote, reference, prefix, modules, plain, config=None):
    """`modules`: (module, rows) taken from the start of each weight, which starts a nested block."""
    sd, expect, sources = {}, {}, {}
    for module, rows in modules:
        w = f"{module}.weight"
        blob_key = next(k for k in remote.header if k.startswith(w + ".quant_state.bitsandbytes__"))
        qs = json.loads(bytes(remote.tensor(blob_key).numpy()))
        cols = qs["shape"][1]
        n, block = rows * cols, qs["blocksize"]
        assert n % block == 0 and ("nested_offset" not in qs or n % (block * 256) == 0)
        sd[w] = remote.tensor(w, n // 2).reshape(-1, 1)
        sd[w + ".absmax"] = remote.tensor(w + ".absmax", n // block)
        sd[w + ".quant_map"] = remote.tensor(w + ".quant_map")
        if "nested_offset" in qs:
            sd[w + ".nested_absmax"] = remote.tensor(w + ".nested_absmax", n // block // 256)
            sd[w + ".nested_quant_map"] = remote.tensor(w + ".nested_quant_map")
        qs["shape"] = [rows, cols]
        sd[blob_key] = torch.tensor(list(json.dumps(qs).encode()), dtype=torch.uint8)
        expect["dequant." + w] = dequant4(sd, w)
        ref = reference.tensor(w[len(prefix):], n).reshape(rows, cols)
        expect["source." + w] = ref
        err = ((expect["dequant." + w].float() - ref.float()).norm() / ref.float().norm()).item()
        sources[w] = {"rows": [0, rows], "rel_l2": round(err, 4)}
    for key in plain:
        sd[key] = remote.tensor(key)
    write(case, sd, expect, config)
    return {"case": case, "source": remote.name, "source_header_sha256": remote.header_sha256,
            "reference": reference.name, "reference_header_sha256": reference.header_sha256, "weights": sources,
            "sha256": {k: hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                       for k, v in sd.items()}}


def write(case, sd, expect, config):
    (OUT / case).mkdir(exist_ok=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, OUT / case / "model.safetensors")
    save_file({k: v.contiguous() for k, v in expect.items()}, OUT / f"{case}.expect.safetensors")
    if config is not None:
        (OUT / case / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def config(**quant):
    return {"architectures": ["Toy"], "torch_dtype": "bfloat16",
            "quantization_config": {"quant_method": "bitsandbytes", **quant}}


def synthetic():
    x = bf16(torch.randn(16, 64, generator=gen) * 0.02)
    lin = bnb.nn.Linear4bit(64, 16, bias=False, compress_statistics=True, quant_type="fp4")
    lin.weight = bnb.nn.Params4bit(x, requires_grad=False, compress_statistics=True, quant_type="fp4", blocksize=64)
    sd = {"proj." + k: v for k, v in lin.to("cpu").state_dict().items()}
    sd["norm.weight"] = bf16(torch.ones(64))
    expect = {"source.proj.weight": x, "dequant.proj.weight": dequant4(sd, "proj.weight")}
    write("bnb-fp4-dq", sd, expect, config(load_in_4bit=True, bnb_4bit_quant_type="fp4",
                                           bnb_4bit_use_double_quant=True, bnb_4bit_quant_storage="uint8"))

    x = bf16(torch.randn(16, 64, generator=gen) * 0.02)
    lin = bnb.nn.Linear8bitLt(64, 16, bias=True, has_fp16_weights=False)
    lin.weight = bnb.nn.Int8Params(x, requires_grad=False, has_fp16_weights=False)
    lin.bias = torch.nn.Parameter(bf16(torch.zeros(16)), requires_grad=False)
    sd = {"proj." + k: v for k, v in lin.to("cpu").state_dict().items()}
    dq = F.int8_vectorwise_dequant(sd["proj.weight"], sd["proj.SCB"]).to(torch.bfloat16)
    write("bnb-int8", sd, {"source.proj.weight": x, "dequant.proj.weight": dq},
          config(load_in_8bit=True, llm_int8_threshold=6.0))


def main():
    unsloth = Remote("unsloth/Llama-3.2-1B-Instruct-bnb-4bit", "fcecd050a91025151bbf518e41a5577c356efdce",
                     "model.safetensors")
    with urllib.request.urlopen(unsloth.url.replace("model.safetensors", "config.json")) as r:
        quant = json.load(r)["quantization_config"]
    provenance = [
        real("real-bnb-unsloth", unsloth,
             Remote("unsloth/Llama-3.2-1B-Instruct", "5a8abab4a5d6f164389b1079fb721cfab8d7126c", "model.safetensors"),
             "", [("model.layers.0.self_attn.k_proj", 8), ("model.layers.15.mlp.down_proj", 2)],
             ["model.layers.0.input_layernorm.weight"],
             {"architectures": ["Toy"], "torch_dtype": "bfloat16", "quantization_config": quant}),
        real("real-bnb-forge",
             Remote("lllyasviel/flux1-dev-bnb-nf4", "32d13814357ca76c52638d4644d0f7384208dc96",
                    "flux1-dev-bnb-nf4-v2.safetensors"),
             Remote("Comfy-Org/flux1-dev", "83c446ef27a6ac1e9e36ecf13257283aa12cf22a", "flux1-dev.safetensors"),
             "model.diffusion_model.", [("model.diffusion_model.double_blocks.0.img_attn.proj", 8)],
             ["model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale"]),
    ]
    synthetic()
    (OUT / "real-bnb-provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(json.dumps([p["weights"] for p in provenance]), f"bitsandbytes {bnb.__version__}")


if __name__ == "__main__":
    main()
