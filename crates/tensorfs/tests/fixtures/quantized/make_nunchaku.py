"""Nunchaku "v1 single-file" svdq-fp4 checkpoints for `tests/quantized_import.rs`, packed by v1's own inverse packers
(python-gen-worker 7ff17491 `gen_worker/models/svdq_layout.py`: `pack_qweight`, `pack_wscales`, `pack_vector`,
`pack_lowrank`, `quantize_lowrank`, deepcompressor's layouts) from random natural-layout parts.

`<case>/model.safetensors` is the Nunchaku file (its `quantization_config`, `model_class` and `config` metadata);
`<case>.expect.safetensors` holds, for each projection `<p>`, `part.<p>.weight.<role>`: the `svdq.nvfp4` part the
import must write, byte for byte (a fused `to_qkv` as `to_q`, `to_k`, `to_v`); `<p>.bias` unswizzled; and
`law.<p>.weight`, the logical weight in F32 by v1's own decode (`decode_linear`, `split_decoded`, then
`dequantize_decoded / smooth + up @ downᵀ`, as `fold_to_dense`).

    uv run --no-project --python 3.12 --with torch --with numpy --with safetensors --with msgspec \
      python -I make_nunchaku.py <this directory> <python-gen-worker>/src
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(1, sys.argv[2])
from gen_worker.models import svdq_layout as v1  # noqa: E402

HERE = Path(sys.argv[1])
torch.manual_seed(66)
E2M1 = torch.tensor(v1.E2M1_LUT)
QC = {"method": "svdquant", "weight": {"dtype": "fp4_e2m1_all", "scale_dtype": [None, "fp8_e4m3_nan"], "group_size": 16},
      "activation": {"dtype": "fp4_e2m1_all", "scale_dtype": "fp8_e4m3_nan", "group_size": 16}, "rank": 32}


def linear(n, k, r, *, channel=False, lowrank_quant=None):
    """Natural parts and their Nunchaku tensors."""
    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8)
    scales = torch.randint(0x28, 0x50, (n, k // 16), dtype=torch.uint8).view(torch.float8_e4m3fn)
    smooth = (torch.rand(k) * 3.75 + 0.25).to(torch.bfloat16)
    down = (torch.randn(r, k) * 0.05).to(torch.bfloat16)  # [R, K], the smoothing folded in
    up = (torch.randn(n, r) * 0.05).to(torch.bfloat16)
    bias = (torch.randn(n) * 0.1).to(torch.bfloat16)
    nat = {"codes": codes, "scales": scales, "smooth": smooth, "down": down, "up": up, "bias": bias}
    t = {"qweight": v1.pack_qweight(codes), "wscales": v1.pack_wscales(scales, n, k),
         "smooth_factor": v1.pack_vector(smooth, k), "smooth_factor_orig": torch.ones(k, dtype=torch.bfloat16),
         "bias": v1.pack_vector(bias, n)}
    if channel:
        nat["second"] = (torch.rand(n) * 1.5 + 0.5).to(torch.bfloat16)
        t["wcscales"] = v1.pack_vector(nat["second"], n)
    else:
        nat["second"] = torch.tensor([0.0123], dtype=torch.float32)
        t["wtscale"] = nat["second"].clone()
    if lowrank_quant:
        t["proj_down"], t["proj_down_scale"] = v1.quantize_lowrank(down.t().contiguous(), scheme=lowrank_quant, block_dim=0)
        t["proj_up"], t["proj_up_scale"] = v1.quantize_lowrank(up, scheme=lowrank_quant, block_dim=1)
        nat["down"] = (t["proj_down"].float() * t["proj_down_scale"].repeat_interleave(32, 0)).t().contiguous()
        nat["up"] = t["proj_up"].float() * t["proj_up_scale"].repeat_interleave(32, 1)
    else:
        t["proj_down"], t["proj_up"] = v1.pack_lowrank(down, down=True), v1.pack_lowrank(up, down=False)
    return nat, t


def law(dec):
    """v1's logical weight (`fold_to_dense` in f32)."""
    w = v1.dequantize_decoded(dec) / dec.smooth_factor.float().reshape(1, -1)
    return w + dec.proj_up.float() @ dec.proj_down.float().t()


def parts(nat, rows):
    """The svdq.nvfp4 parts of `nat`'s rows."""
    c = nat["codes"][rows]
    out = {"weight": (c[:, 0::2] | c[:, 1::2] << 4).contiguous(), "scale": nat["scales"][rows].contiguous(),
           "scale_2": torch.tensor(1.0 if nat["second"].numel() > 1 else nat["second"].item()),
           "pre_quant_scale": 1 / nat["smooth"].float()}
    if nat["second"].numel() > 1:
        out["channel_scale"] = nat["second"][rows].float().contiguous()
    out["lowrank_down"], out["lowrank_up"] = nat["down"].contiguous(), nat["up"][rows].contiguous()
    return out


def write(case, modules, plain, lowrank_quant=None):
    tensors, expect = dict(plain), {}
    for m, (nat, t, splits) in modules.items():
        tensors.update({f"{m}.{leaf}": v for leaf, v in t.items()})
        n, k = nat["codes"].shape
        dec = v1.decode_linear(t, n, k, lowrank_quant=lowrank_quant)
        assert torch.equal(dec.codes, nat["codes"]) and torch.equal(dec.scales.view(torch.uint8), nat["scales"].view(torch.uint8))
        decs = v1.split_decoded(dec, [n // len(splits)] * len(splits)) if len(splits) > 1 else (dec,)
        parent = m.rsplit(".", 1)[0]
        for o, (name, d) in enumerate(zip(splits, decs)):
            p = f"{parent}.{name}" if len(splits) > 1 else m
            rows = slice(o * d.out_features, (o + 1) * d.out_features)
            for role, v in parts(nat, rows).items():
                expect[f"part.{p}.weight.{role}"] = v
            expect[f"{p}.bias"] = nat["bias"][rows].contiguous()
            expect[f"law.{p}.weight"] = law(d).contiguous()
    meta = {"model_class": "NunchakuFluxTransformer2DModel", "quantization_config": json.dumps(QC),
            "config": json.dumps({"_class_name": "FluxTransformer2DModel"})}
    if lowrank_quant:
        meta[v1.LOWRANK_QUANT_KEY] = lowrank_quant
    (HERE / case).mkdir(exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, HERE / case / "model.safetensors", metadata=meta)
    save_file({k: v.contiguous().clone() for k, v in expect.items()}, HERE / f"{case}.expect.safetensors")


plain = {"transformer_blocks.0.norm.weight": torch.randn(128).to(torch.bfloat16),
         "proj_out.weight": torch.randn(16, 128).to(torch.bfloat16)}
qkv = linear(384, 128, 16, channel=True)
ff = linear(128, 128, 32)
write("nunchaku-svdq-fp4", {"transformer_blocks.0.attn.to_qkv": (*qkv, ["to_q", "to_k", "to_v"]),
                            "transformer_blocks.0.ff.net.0.proj": (*ff, ["proj"])}, plain)
lr8 = linear(128, 128, 32, lowrank_quant="int8")
write("nunchaku-svdq-fp4-lr8", {"transformer_blocks.0.ff.net.0.proj": (*lr8, ["proj"])}, plain, "int8")
