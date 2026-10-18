#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "diffusers==0.40.0", "transformers==5.16.1", "torch>=2.11", "numpy",
#   "einops==0.8.2", "omegaconf==2.4.0", "rotary-embedding-torch==0.9.1",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Generate the converter tables (`crates/tensorfs-core/converters/*.json`) from upstream's converters.

    uv run scripts/converters.py           # regenerate every table and compare byte for byte
    uv run scripts/converters.py --write   # rewrite them

Each family runs its reference converter over stand-ins built from real headers in `vectors/`:
coordinate-tracking views for diffusers' single-file converters, row-index tensors for the
official MiniMax-H3 converter. Every non-rename row's ops, applied with numpy semantics (the
engine's), must reproduce the converter's own output on those stand-ins, and every table's
targets must be the constructor's census. Canonical names are upstream's on-disk names: SDXL
`text_encoder` keeps `text_model.`, H3 drops `rope.inv_freq` (diffusers' converter does) and its
transformers land as `transformer`/`transformer_ref`. Those are spec data below. Rows are in the
target's construction order (its `state_dict()`), never sorted (#624).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import json
import math
import re
import struct
import sys
import tempfile
import types
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.loaders import single_file_utils

PINS = {"diffusers": "0.40.0", "transformers": "5.16.1"}
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "crates" / "tensorfs-core" / "converters"
VECTORS = ROOT / "vectors"
HF = "https://huggingface.co"
H3_SCRIPT = (
    "https://raw.githubusercontent.com/huggingface/diffusers/c60830ee365d520ab52b110dda562dd26f7b4d7f/"
    "scripts/convert_minimax_h3_to_diffusers.py",
    "86f61f62934d1eccf2a76ec57b6d072e2c23767c6e49d96aa1cf24d40422b42c",
)
H3_REFERENCE = "hf://MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe"
SEEDVR = "https://raw.githubusercontent.com/ByteDance-Seed/SeedVR/e4de8c24441a67e1b7df56abea10645059bb1185/"


def fetch(url: str, sha256: str | None = None) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        body = response.read()
    if sha256 and hashlib.sha256(body).hexdigest() != sha256:
        sys.exit(f"{url}: sha256 differs from the pinned {sha256}")
    return body


def header(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    raw = path.read_bytes()
    (length,) = struct.unpack("<Q", raw[:8])
    doc = json.loads(raw[8 : 8 + length])
    return {k: (v["dtype"], tuple(v["shape"])) for k, v in doc.items() if k != "__metadata__"}


def apply(a: np.ndarray, ops: dict[str, Any]) -> np.ndarray:
    """A row's ops with numpy semantics, which the engine implements."""
    if "view" in ops:
        a = a.reshape(ops["view"])
    if "take" in ops:
        a = np.take(a, ops["take"][1], axis=ops["take"][0])
    if "permute" in ops:
        a = a.transpose(ops["permute"])
    if "shape" in ops:
        a = a.reshape(ops["shape"])
    return a


def row(target: str, source: str, ops: dict[str, Any] | None = None) -> list[Any]:
    if ops:
        return [target, source, ops]
    return [target] if target == source else [target, source]


# ------------------------------------------------------------------ diffusers single-file


def contiguous(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))


@dataclass(frozen=True)
class Materialized:
    op: str
    sources: tuple[str, ...]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: Materialized(f"{self.op}.{name}", self.sources)


@dataclass(frozen=True)
class Route:
    """One source tensor's coordinates: shape, element strides and element offset."""

    key: str
    shape: tuple[int, ...]
    offset: int
    strides: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def T(self) -> Route:
        assert self.ndim == 2
        return replace(self, shape=self.shape[::-1], strides=self.strides[::-1])

    def dim(self) -> int:
        return self.ndim

    def size(self, axis: int | None = None) -> Any:
        return self.shape if axis is None else self.shape[axis]

    def numel(self) -> int:
        return math.prod(self.shape)

    def contiguous(self) -> Route:
        return self

    clone = detach = contiguous

    def __getitem__(self, key: Any) -> Route:
        index = key if isinstance(key, tuple) else (key,)
        index = (*index, *([slice(None)] * (self.ndim - len(index))))
        shape, stride, offset = [], [], self.offset
        for size, step, item in zip(self.shape, self.strides, index, strict=True):
            if isinstance(item, int):
                offset += item * step
            else:
                start, stop, jump = item.indices(size)
                shape.append(len(range(start, stop, jump)))
                stride.append(step * jump)
                offset += start * step
        return replace(self, shape=tuple(shape), strides=tuple(stride), offset=offset)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: Materialized(name, (self.key,))

    @classmethod
    def __torch_function__(cls, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        sources: list[str] = []

        def collect(v: Any) -> None:
            if isinstance(v, Route):
                sources.append(v.key)
            elif isinstance(v, list | tuple):
                for item in v:
                    collect(item)

        collect(args)
        return Materialized(getattr(func, "__name__", str(func)), tuple(sources))


def route_ops(route: Route, source: tuple[int, ...]) -> dict[str, Any] | None:
    """The row ops reproducing `route`, checked element for element."""
    size = math.prod(route.shape)
    if route.strides == contiguous(route.shape) and route.offset % size == 0:
        if route.shape == source:
            ops = None
        elif size == math.prod(source):
            ops = {"shape": list(route.shape)}
        else:
            parts = math.prod(source) // size
            ops = {"view": [parts, -1], "take": [0, [route.offset // size]], "shape": list(route.shape)}
    elif route.ndim == 2 and route.offset == 0 and route.shape == source[::-1]:
        ops = {"permute": [1, 0]}
    else:
        raise SystemExit(f"{route.key}: a layout no row op expresses")
    flat = np.arange(math.prod(source))
    want = np.lib.stride_tricks.as_strided(flat[route.offset :], route.shape, [s * 8 for s in route.strides])
    got = flat.reshape(source) if ops is None else apply(flat.reshape(source), ops)
    assert np.array_equal(got, want), route.key
    return ops


def reference_config(reference: str, path: str) -> dict[str, Any]:
    repository, revision = reference.removeprefix("hf://").split("@")
    return json.loads(fetch(f"{HF}/{repository}/resolve/{revision}/{path}"))


def construct(model: str, config: dict[str, Any]) -> Any:
    library, name = model.split(":")
    kind = getattr(importlib.import_module(library), name)
    with torch.device("meta"):
        return kind.from_config(config) if library == "diffusers" else kind(kind.config_class(**config))


SINGLE_FILE = {
    "sdxl.single_file": {
        "reference": "hf://stabilityai/stable-diffusion-xl-base-1.0@462165984030d82259a11f4367a4eed129e94a7b",
        "header": "civitai-sdxl-headers/128078-92696.safetensors.header",
        "cast": "F16",
        "components": [
            # Transformers 5's CLIPTextModel drops `text_model.`; on disk it is there (D4).
            {"name": "text_encoder", "prefix": "conditioner.embedders.0.transformer.", "model": "transformers:CLIPTextModel",
             "config": "text_encoder/config.json", "convert": "convert_ldm_clip_checkpoint", "arguments": ["checkpoint"],
             "census": {"text_model.": ""}, "discard": ["text_model.embeddings.position_ids"],
             "optional": ["text_model.embeddings.position_ids"], "upstream": "sdxl-headers/text_encoder__model.fp16.safetensors.header"},
            {"name": "text_encoder_2", "prefix": "conditioner.embedders.1.model.", "model": "transformers:CLIPTextModelWithProjection",
             "config": "text_encoder_2/config.json", "convert": "convert_open_clip_checkpoint", "arguments": ["model", "checkpoint", "prefix"],
             "optional": ["transformer.text_model.embeddings.position_ids"], "upstream": "sdxl-headers/text_encoder_2__model.fp16.safetensors.header"},
            {"name": "unet", "prefix": "model.diffusion_model.", "model": "diffusers:UNet2DConditionModel", "config": "unet/config.json",
             "convert": "convert_ldm_unet_checkpoint", "arguments": ["checkpoint", "config"],
             "upstream": "sdxl-headers/unet__diffusion_pytorch_model.fp16.safetensors.header"},
            {"name": "vae", "prefix": "first_stage_model.", "model": "diffusers:AutoencoderKL", "config": "vae/config.json",
             "convert": "convert_ldm_vae_checkpoint", "arguments": ["checkpoint", "config"],
             "upstream": "sdxl-headers/vae__diffusion_pytorch_model.fp16.safetensors.header"},
        ],
    },
    "anima.single_file": {
        "reference": "hf://circlestone-labs/Anima-Base-v1.0-Diffusers@073c3a9db359c31ad0e8aa268d15775473c2176c",
        "cast": "BF16",
        "components": [
            {"name": "text_conditioner", "header": "anima-civitai-headers/2945208-2824391.safetensors.header", "prefix": "net.llm_adapter.",
             "model": "diffusers:AnimaTextConditioner", "config": "text_conditioner/config.json",
             "upstream": "anima-hf-headers/text_conditioner/diffusion_pytorch_model.safetensors.header"},
            {"name": "text_encoder", "header": "anima-civitai-headers/2945208-2824387.safetensors.header", "prefix": "model.",
             "model": "transformers:Qwen3Model", "config": "text_encoder/config.json",
             "upstream": "anima-hf-headers/text_encoder/model.safetensors.header"},
            {"name": "transformer", "header": "anima-civitai-headers/2945208-2824391.safetensors.header", "prefix": "net.",
             "model": "diffusers:CosmosTransformer3DModel", "config": "transformer/config.json",
             "convert": "convert_cosmos_transformer_checkpoint_to_diffusers", "arguments": ["checkpoint"],
             "upstream": "anima-hf-headers/transformer/diffusion_pytorch_model.safetensors.header"},
            {"name": "vae", "header": "anima-civitai-headers/2945208-2004692.safetensors.header", "prefix": "",
             "model": "diffusers:AutoencoderKLQwenImage", "config": "vae/config.json", "convert": "convert_wan_vae_to_diffusers",
             "arguments": ["checkpoint"], "upstream": "anima-hf-headers/vae/diffusion_pytorch_model.safetensors.header"},
        ],
    },
}


def single_file(name: str) -> tuple[dict[str, Any], dict[str, dict[str, tuple[int, ...]]]]:
    """The table, and each component's target shapes (for the LoRA tables)."""
    spec = SINGLE_FILE[name]
    prefixes = [c["prefix"] for c in spec["components"]]
    components, shapes = [], {}
    for c in spec["components"]:
        prefix = c["prefix"]
        source = header(VECTORS / c.get("header", spec.get("header")))
        checkpoint = {
            k: Route(k, shape, 0, contiguous(shape))
            for k, (_, shape) in source.items()
            if max((p for p in prefixes if k.startswith(p)), key=len, default=None) == prefix
        }
        config = reference_config(spec["reference"], c["config"])
        model = construct(c["model"], config)
        arguments = {"checkpoint": checkpoint, "config": config, "model": model, "prefix": prefix}
        if c.get("convert"):
            converted = getattr(single_file_utils, c["convert"])(*(arguments[a] for a in c["arguments"]))
        else:
            converted = {k.removeprefix(prefix): r for k, r in checkpoint.items()}
        census_of = c.get("census", {})

        def on_disk(census_key: str) -> str:
            for disk, census in census_of.items():
                if census_key.startswith(census):
                    return disk + census_key.removeprefix(census)
            return census_key

        rows, used, shapes[c["name"]] = [], set(), {}
        for key, value in model.state_dict().items():
            target = on_disk(key)
            route = converted.pop(target)
            assert isinstance(route, Route), (name, target, route)
            assert route.shape == tuple(value.shape), (name, target)
            local = route.key.removeprefix(prefix)
            used.add(local)
            rows.append(row(target, local, route_ops(route, source[route.key][1])))
            shapes[c["name"]][target] = route.shape
        assert sorted(converted) == sorted(c.get("discard", [])), (name, c["name"], sorted(converted))
        upstream = header(VECTORS / c["upstream"])
        assert {t: tuple(s) for t, s in shapes[c["name"]].items()} == {k: s for k, (_, s) in upstream.items()}, (
            name, c["name"], "targets are not upstream's on-disk names and shapes")
        present = {k.removeprefix(prefix) for k in checkpoint}
        optional = sorted(c.get("optional", []))
        components.append({"name": c["name"], "prefix": prefix, "cast": spec["cast"],
                           "drop": sorted(present - used - set(optional)), "optional": optional, "rows": rows})
    return {"name": name, "reference": spec["reference"], "components": components}, shapes


# ------------------------------------------------------------------ MiniMax-H3


def h3_converter() -> Any:
    body = fetch(*H3_SCRIPT)
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as f:
        f.write(body)
    spec = importlib.util.spec_from_file_location("convert_minimax_h3_to_diffusers", f.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Path(f.name).unlink()
    return module


def row_ops(rows: np.ndarray, source: tuple[int, ...]) -> dict[str, Any] | None:
    """The fewest-member view/take whose output rows are `rows` of a `source`-shaped tensor."""
    count, out = source[0], len(rows)
    rest = [-1] if len(source) > 1 else []
    if out == count and np.array_equal(rows, np.arange(count)):
        return None
    unit = next((i for i in range(1, out) if rows[i] != rows[i - 1] + 1), out)
    stand_in = np.arange(count * 2).reshape(count, 2) if rest else np.arange(count)
    blocks = count // unit
    for members in range(2, blocks + 1):
        if count % unit or blocks % members or out % (blocks // members * unit):
            continue
        groups = blocks // members
        picked = [int(b) % members for b in rows[::unit][: out // (groups * unit)] // unit]
        view = [members, -1] if groups == 1 else [groups, members, -1]
        ops = {"view": view, "take": [len(view) - 2, picked], "shape": [out, *rest]}
        if np.array_equal(apply(stand_in, ops), stand_in[rows]):
            return ops
    raise SystemExit(f"rows of {source}: no view/take expresses them")


def h3_index(rows: int) -> torch.Tensor:
    return torch.arange(rows).reshape(rows, 1)


def h3_native(h3: Any) -> tuple[dict[str, Any], dict[str, tuple[str, tuple[int, ...]]]]:
    """The native transformer table, and the native header (for the LoRA)."""
    config = h3.MINIMAX_H3_TRANSFORMER_CONFIG
    heads, head_dim = config["num_attention_heads"], config["attention_head_dim"]
    from diffusers import MiniMaxH3Transformer3DModel

    with torch.device("meta"):
        census = {k: tuple(v.shape) for k, v in MiniMaxH3Transformer3DModel(**config).state_dict().items()}
    variants = {}
    for variant in ("FL2VA", "Ref2VA"):
        merged: dict[str, tuple[str, tuple[int, ...]]] = {}
        for shard in sorted((VECTORS / "h3-headers").glob(f"{variant}__transformer__model-*.safetensors")):
            merged.update(header(shard))
        variants[variant] = merged
    source = variants["FL2VA"]
    assert {k: s for k, (_, s) in variants["Ref2VA"].items()} == {k: s for k, (_, s) in source.items()}
    mapped, dropped, expect = {}, [], {}
    for key, (_, shape) in source.items():
        stand_in = h3_index(shape[0]) if len(shape) > 1 else torch.arange(shape[0])
        if key.endswith(".attn.qkv_proj.weight"):
            stand_in = h3.reorder_interleaved_qkv(stand_in, heads, head_dim)
            norm = key.replace("qkv_proj.weight", "q_norm.weight")
            expect[norm] = [head_dim]
            assert source[norm][1] == (head_dim,), norm
        outputs = h3.convert_transformer_key(key, stand_in, config)
        if not outputs:
            dropped.append(key)
        for target, tensor in outputs:
            ops = row_ops(tensor.reshape(tensor.shape[0], -1)[:, 0].numpy(), shape)
            if ops:
                expect[key] = list(shape)
            mapped[target] = row(target, key, ops)
            got = apply(np.zeros(shape, np.int8), ops) if ops else np.zeros(shape, np.int8)
            assert got.shape == census[target], (target, got.shape, census[target])
    assert sorted(mapped) == sorted(census), "targets are not the transformer's census"
    component = {"name": "transformer", "folders": {"FL2VA/transformer": "transformer", "Ref2VA/transformer": "transformer_ref"},
                 "drop": sorted(dropped), "expect": dict(sorted(expect.items())), "rows": [mapped[k] for k in census]}
    return {"name": "h3.native", "reference": H3_REFERENCE, "components": [component]}, source


def h3_lora(h3: Any, source: dict[str, tuple[str, tuple[int, ...]]]) -> dict[str, Any]:
    """Trainer LoRA factors: A as is, B through diffusers' own transformer conversion given the
    trainer's contiguous Q/K/V thirds (docs/h3-lora-source-layout.md), so fc1 B halves swap."""
    config = h3.MINIMAX_H3_TRANSFORMER_CONFIG
    leaves = {"attn.qkv_proj": "attn_qkv_proj", "attn.out_proj": "attn_out_proj", "mlp.fc1": "mlp_fc1", "mlp.fc2": "mlp_fc2"}
    blocks = [("blocks", i, True) for i in range(config["num_layers"])]
    blocks += [("token_refiner.blocks", i, False) for i in range(config["num_refiner_layers"])]
    rows = []
    for family, i, kohya in blocks:
        for leaf, flat in leaves.items():
            native = f"{family}.{i}.{leaf}"
            stems = [f"diffusion_model.{native}"] + ([f"lora_unet_blocks_{i}_{flat}"] if kohya else [])
            fused, inp = source[f"{native}.weight"][1]
            for target, b in h3.convert_transformer_key(f"{native}.weight", h3_index(fused), config):
                ops: dict[str, Any] = {"io": [inp, fused]}
                if b_ops := row_ops(b[:, 0].numpy(), (fused, 1)):
                    ops["b"] = b_ops
                rows.extend([target.removesuffix(".weight"), stem, ops] for stem in stems)
    # Rows in the transformer's construction order (#624): its state_dict, not the trainer's.
    from diffusers import MiniMaxH3Transformer3DModel

    with torch.device("meta"):
        census = {k: i for i, k in enumerate(MiniMaxH3Transformer3DModel(**config).state_dict())}
    rows.sort(key=lambda r: census[r[0] + ".weight"])
    lora = {"a": [".lora_A.weight", ".lora_down.weight"], "b": [".lora_B.weight", ".lora_up.weight"], "alpha": [".alpha"]}
    return {"name": "h3.lora", "explicit": True, "lora": lora, "components": [{"name": "transformer", "rows": rows}]}


# ------------------------------------------------------------------ SDXL LoRA

SDXL_LORA = {
    "unet": ("lora_unet_", re.compile(
        r"(down_blocks\.[12]|mid_block|up_blocks\.[01])\.attentions\.\d+\.(proj_in|proj_out|transformer_blocks\.\d+\."
        r"(attn[12]\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2)))")),
    "text_encoder": ("lora_te1_", re.compile(r"text_model\.encoder\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|mlp\.fc[12])")),
    "text_encoder_2": ("lora_te2_", re.compile(r"text_model\.encoder\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|mlp\.fc[12])")),
}


def sdxl_lora(table: dict[str, Any], shapes: dict[str, dict[str, tuple[int, ...]]]) -> dict[str, Any]:
    """Kohya SDXL factors onto the single-file table's modules: UNet in its LDM or Diffusers
    spelling, both CLIP towers in theirs, linear layers only."""
    components = []
    for c in table["components"]:
        if c["name"] not in SDXL_LORA:
            continue
        prefix, reviewed = SDXL_LORA[c["name"]]
        rows = []
        for r in c["rows"]:
            target, source = r[0], r[1] if len(r) > 1 else r[0]
            module = target.removesuffix(".weight")
            if not target.endswith(".weight") or not reviewed.fullmatch(module):
                continue
            out, inp = shapes[c["name"]][target]
            spellings = [module] + ([source.removesuffix(".weight")] if c["name"] == "unet" else [])
            for spelling in dict.fromkeys(spellings):
                rows.append([module, spelling.replace(".", "_"), {"io": [inp, out]}])
        components.append({"name": c["name"], "prefix": prefix, "rows": rows})
    lora = {"a": [".lora_down.weight", ".lora_A.weight"], "b": [".lora_up.weight", ".lora_B.weight"], "alpha": [".alpha"]}
    return {"name": "sdxl.lora", "lora": lora, "components": components}


# ------------------------------------------------------------------ SeedVR2


# The seedvr2 package's NaDiT and VAE arguments (SeedVR's configs_7b).
SEEDVR_DIT = dict(
    vid_in_channels=33, vid_out_channels=16, vid_dim=3072, txt_in_dim=5120, txt_dim=3072, emb_dim=18432,
    heads=24, head_dim=128, expand_ratio=4, norm="rms", norm_eps=1e-5, ada="single", qk_bias=False,
    qk_norm="rms", qk_rope=True, patch_size=(1, 2, 2), num_layers=36, mlp_type="normal", shared_mlp=False,
    shared_qkv=False, block_type=["mmdit_sr"] * 36, window=[(4, 3, 3)] * 36,
    window_method=["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 18,
)
SEEDVR_VAE = dict(
    act_fn="silu", block_out_channels=(128, 256, 512, 512), down_block_types=("DownEncoderBlock3D",) * 4,
    up_block_types=("UpDecoderBlock3D",) * 4, in_channels=3, out_channels=3, latent_channels=16,
    layers_per_block=2, norm_num_groups=32, slicing_sample_min_size=4, temporal_scale_num=2, inflation_mode="pad",
    spatial_downsample_factor=8, temporal_downsample_factor=4, use_quant_conv=False, use_post_quant_conv=False,
    freeze_encoder=False,
)


def seedvr_census() -> dict[str, list[str]]:
    """NaDiT's and the VAE's `state_dict()` order, built on the meta device from SeedVR's code at
    the pinned commit. Its attention module imports flash_attn, which construction never calls."""
    commit = SEEDVR.rstrip("/").rsplit("/", 1)[1]
    body = fetch(f"https://codeload.github.com/ByteDance-Seed/SeedVR/zip/{commit}")
    root = Path(tempfile.mkdtemp())
    zipfile.ZipFile(io.BytesIO(body)).extractall(root)
    sys.path.insert(0, str(root / f"SeedVR-{commit}"))
    stub = types.ModuleType("flash_attn")
    stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    stub.flash_attn_varlen_func = None
    sys.modules.setdefault("flash_attn", stub)
    from models.dit.nadit import NaDiT
    from models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper

    with torch.device("meta"):
        return {"dit": list(NaDiT(**SEEDVR_DIT).state_dict()),
                "vae": list(VideoAutoencoderKLWrapper(**SEEDVR_VAE).state_dict())}


def seedvr2() -> dict[str, Any]:
    """The package's normalize composition: the two fp16 checkpoints as they are, and
    upstream's fixed text conditioning read straight from the pinned `.pt` files' storage
    (no unpickling). Rows follow NaDiT's/the VAE's construction order; the banked headers hold
    exactly their keys."""
    census = seedvr_census()
    components = []
    for name, file in (("dit", "seedvr2_ema_7b_fp16.safetensors"), ("vae", "ema_vae_fp16.safetensors")):
        keys = header(VECTORS / "seedvr2-headers" / f"{file}.header")
        assert sorted(keys) == sorted(census[name]), name
        components.append({"name": name, "cast": "F16", "rows": [[k] for k in census[name]]})
    rows = []
    for key, file, shape, digest in (
        ("positive", "pos_emb.pt", (58, 5120), "fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799"),
        ("negative", "neg_emb.pt", (64, 5120), "6a43e5800ef2354f1c156d27535834da055cbec8248298b8923492bba2076581"),
    ):
        body = fetch(SEEDVR + file, digest)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            f.write(body)
            f.flush()
            info = zipfile.ZipFile(f.name).getinfo(f"{file.removesuffix('.pt')}/data/0")
            tensor = torch.load(f.name, weights_only=True, map_location="cpu")
        (name_len, extra_len) = struct.unpack("<HH", body[info.header_offset + 26 : info.header_offset + 30])
        offset = info.header_offset + 30 + name_len + extra_len
        want = tensor.contiguous().view(torch.uint16).numpy().tobytes()
        assert info.compress_type == zipfile.ZIP_STORED and tuple(tensor.shape) == shape and tensor.dtype == torch.bfloat16
        assert body[offset : offset + len(want)] == want, file
        rows.append([key, {"sha256": digest, "length": len(body), "offset": offset, "dtype": "BF16", "shape": list(shape)}])
    components.append({"name": "conditioning", "rows": rows})
    return {"name": "seedvr2", "files": {"config.json": {"architecture": "SeedVR2-7B"}}, "components": components}


def generate() -> dict[str, dict[str, Any]]:
    sdxl, shapes = single_file("sdxl.single_file")
    anima, _ = single_file("anima.single_file")
    h3 = h3_converter()
    native, source = h3_native(h3)
    return {
        "sdxl.single_file": sdxl,
        "anima.single_file": anima,
        "h3.native": native,
        "h3.lora": h3_lora(h3, source),
        "sdxl.lora": sdxl_lora(sdxl, shapes),
        "seedvr2": seedvr2(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="rewrite the tables")
    args = parser.parse_args()
    for package, version in PINS.items():
        if importlib.metadata.version(package) != version:
            sys.exit(f"{package} {importlib.metadata.version(package)} is not the pinned {version}")
    stale = []
    for name, table in generate().items():
        path = OUT / f"{name}.json"
        body = (json.dumps(table, separators=(",", ":")) + "\n").encode()
        if args.write:
            path.write_bytes(body)
        elif not path.is_file() or path.read_bytes() != body:
            stale.append(path.name)
        print(f"{path.name}: {len(body)} bytes {'written' if args.write else 'checked'}")
    if stale:
        sys.exit(f"stale converter tables {stale}: run scripts/converters.py --write")


if __name__ == "__main__":
    main()
