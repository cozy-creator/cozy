#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "diffusers @ git+https://github.com/huggingface/diffusers@7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941",
#   "transformers==5.20.0",
#   "torch>=2.11",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Generate the constructor-order references (`crates/tensorfs-core/orders/<class>.json`).

    task -d crates/tensorfs orders        # rewrite them (uv run scripts/orders.py --write)
    uv run scripts/orders.py              # regenerate and compare byte for byte

A Diffusers or Transformers folder carries no tensor order (safetensors sorts its keys), so an
import orders it by the `state_dict()` order of the model class its config names (spec §10).
Each reference is that order for one class, built on the meta device from pinned configs, named
as the checkpoints store the keys, and kept as a tree: for each parent pattern (digit segments
`#`), its children in order, merged over every instance of that parent and every config (a down
block with attentions and one without agree on `attentions` < `resnets` < `downsamplers`).
`vectors/orders/*.json` holds full key orders of models the tests import; never references.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import diffusers
import torch
import transformers
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tensorfs-core" / "orders"
TESTS = ROOT / "tensorfs-core" / "vectors" / "orders"
DIFFUSERS = "0.41.0.dev0"  # git 7ecb1e1b: Flux3 is in Diffusers main only
TRANSFORMERS = "5.20.0"

SDXL = ("stabilityai/stable-diffusion-xl-base-1.0", "462165984030d82259a11f4367a4eed129e94a7b")
ANIMA = ("circlestone-labs/Anima-Base-v1.0-Diffusers", "073c3a9db359c31ad0e8aa268d15775473c2176c")
QWEN_IMAGE = ("Qwen/Qwen-Image", "75e0b4be04f60ec59a75f475837eced720f823b6")
QWEN_IMAGE_21 = ("Qwen/Qwen-Image-2.1", "790c92633540aa0cb11d9abf19eb46d861714758")
QWEN3_VL = ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203")
QWEN3_5 = ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
# FLUX.3 Klein's repository is gated; our released lanes carry its component configs inline in
# their v1 headers (read anonymously from Tensorhub), checked in as `orders/inputs/`:
# (lane, manifest, header, file, sha256).
LANE_CONFIGS = [
    ("fidika/flux-3-klein@0.1.0/bf16", "30615fedd4da5f7fa1a406320ff24783ae6ec2fe0e697e472c4a0c3d79e6647f",
     "eb206412f6bd896f11dca285fff593a9f587de6bc9456bc16920d0094c64bcc0", "flux-3-klein-0.1.0-bf16.transformer.json",
     "d4fe5c8c25ff4d8da4671c7255fde2137d6c0556f218d043e17872c9d9782ad5"),
    ("fidika/flux-3-klein@0.1.0/bf16", "30615fedd4da5f7fa1a406320ff24783ae6ec2fe0e697e472c4a0c3d79e6647f",
     "eb206412f6bd896f11dca285fff593a9f587de6bc9456bc16920d0094c64bcc0", "flux-3-klein-0.1.0-bf16.vae.json",
     "97b82ffab5caa6c770eba04f001e2618d18870eb81896573c243629474a911be"),
    ("fidika/flux-3-klein-base@0.1.0/bf16", "0116217d1469b88d7e639f1cf8d989ba258a58af4d755267bd0ae45dec008933",
     "f7c52c23e843dc3afa8a70ff1a2d5b88215d5c4ccd03095332b7a02b3a294647",
     "flux-3-klein-base-0.1.0-bf16.transformer.json",
     "23d930769e38557099693f9eefb77364621705e43f0f0da3da0f349876bfb692"),
]
# MiniMax-H3's transformer has no Diffusers config upstream; diffusers' converter defines it.
H3_SCRIPT = (
    "https://raw.githubusercontent.com/huggingface/diffusers/c60830ee365d520ab52b110dda562dd26f7b4d7f/"
    "scripts/convert_minimax_h3_to_diffusers.py",
    "86f61f62934d1eccf2a76ec57b6d072e2c23767c6e49d96aa1cf24d40422b42c",
)

# Component configs (repo, revision, folder) each class's reference is built from.
SOURCES = [
    (SDXL, "unet"), (SDXL, "vae"), (SDXL, "text_encoder"), (SDXL, "text_encoder_2"),
    (ANIMA, "transformer"), (ANIMA, "text_encoder"), (ANIMA, "text_conditioner"), (ANIMA, "vae"),
    (QWEN_IMAGE, "transformer"), (QWEN_IMAGE, "vae"), (QWEN_IMAGE, "text_encoder"),
    (QWEN_IMAGE_21, "transformer"), (QWEN_IMAGE_21, "vae"), (QWEN_IMAGE_21, "text_encoder"),
    (QWEN3_VL, ""), (QWEN3_5, ""),
]
# Models the tests import; their full orders are never references.
CHECKS = [("tiny-sdxl-pipe", ("hf-internal-testing/tiny-sdxl-pipe", "20594cbc343cfcfe447af5c87cdaf6c436b453f2"),
           ["unet", "vae", "text_encoder", "text_encoder_2"])]


def get(url: str, first: int | None = None, last: int | None = None) -> bytes:
    req = urllib.request.Request(url)
    if first is not None:
        req.add_header("Range", f"bytes={first}-{last}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def hub(repo: tuple[str, str], path: str) -> str:
    return f"https://huggingface.co/{repo[0]}/resolve/{repo[1]}/{path}"


def join(folder: str, name: str) -> str:
    return f"{folder}/{name}" if folder else name


def stored_keys(repo: tuple[str, str], folder: str) -> list[str]:
    """The tensor names the repository stores for a component (index or safetensors header)."""
    for name in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
        try:
            return list(json.loads(get(hub(repo, join(folder, name + ".index.json"))))["weight_map"])
        except urllib.error.HTTPError:
            pass
        try:
            n = struct.unpack("<Q", get(hub(repo, join(folder, name)), 0, 7))[0]
        except urllib.error.HTTPError:
            continue
        header = json.loads(get(hub(repo, join(folder, name)), 8, 7 + n))
        return [k for k in header if k != "__metadata__"]
    raise SystemExit(f"{repo[0]}: no weights for {folder!r}")


def stored_names(model, keys: list[str], stored: list[str]) -> list[str]:
    """`state_dict()` keys under the names checkpoints store them by: Transformers renames on
    load, and its reversed transforms give the saved names. Stored names the class lacks (an
    MTP head it does not build) are reported; they keep their source order at import."""
    candidates = [keys]
    if isinstance(model, transformers.PreTrainedModel):
        for legacy in (True, False):
            transforms = [t.reverse_transform() for t in get_model_conversion_mapping(model, add_legacy=legacy)[::-1]]
            renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
            converters = [t for t in transforms if isinstance(t, WeightConverter)]
            candidates.append([rename_source_key(k, renamings, converters, reverse=True)[0] for k in keys])
    named = max(candidates, key=lambda c: len(set(c) & set(stored)))
    lacking = sorted(set(stored) - set(named))
    if 2 * len(lacking) > len(stored):
        raise SystemExit(f"{type(model).__name__}: most stored names are unknown: {lacking[:5]}")
    if lacking:
        print(f"{type(model).__name__}: {len(lacking)} stored names the class lacks, e.g. {lacking[:3]}", file=sys.stderr)
    return named


def construct(config: dict):
    """The class name and model `config` builds, on the meta device."""
    if "_class_name" in config:
        name = config["_class_name"]
        with torch.device("meta"):
            return name, getattr(diffusers, name).from_config(config)
    name = config["architectures"][0]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text(json.dumps(config))
        conf = transformers.AutoConfig.from_pretrained(d)
    with torch.device("meta"):
        return name, getattr(transformers, name)(conf)


def h3_transformer() -> tuple[str, str, list[str]]:
    url, digest = H3_SCRIPT
    body = get(url)
    if hashlib.sha256(body).hexdigest() != digest:
        raise SystemExit(f"{url}: sha256 differs from the pinned {digest}")
    with tempfile.NamedTemporaryFile("wb", suffix=".py") as f:
        f.write(body)
        f.flush()
        spec = importlib.util.spec_from_file_location("convert_minimax_h3_to_diffusers", f.name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    with torch.device("meta"):
        model = diffusers.MiniMaxH3Transformer3DModel(**module.MINIMAX_H3_TRANSFORMER_CONFIG)
    return "MiniMaxH3Transformer3DModel", url, list(model.state_dict())


def pattern(key: str) -> str:
    return ".".join("#" if s.isdigit() else s for s in key.split("."))


def tree(orders: list[list[str]]) -> dict[str, list[str]]:
    """Parent pattern -> its children in order, merged over the parent's instances in every
    order: each instance's order is a constraint; ties go to the first appearance."""
    instances: dict[tuple[int, str], list[str]] = {}
    for n, keys in enumerate(orders):
        for k in keys:
            segs = k.split(".")
            for d in range(len(segs)):
                child = "#" if segs[d].isdigit() else segs[d]
                seen = instances.setdefault((n, ".".join(segs[:d])), [])
                if child not in seen:
                    seen.append(child)
    by_pattern: dict[str, list[list[str]]] = {}
    for (_, parent), children in instances.items():
        by_pattern.setdefault(pattern(parent), []).append(children)
    out = {}
    for parent, seqs in by_pattern.items():
        first: dict[str, int] = {}
        for seq in seqs:
            for c in seq:
                first.setdefault(c, len(first))
        after: dict[str, set[str]] = {c: set() for c in first}
        for seq in seqs:
            for a, b in zip(seq, seq[1:]):
                after[a].add(b)
        before = {c: sum(c in after[a] for a in first) for c in first}
        order: list[str] = []
        while len(order) < len(first):
            left = [c for c in first if c not in order]
            ready = [c for c in left if before[c] == 0] or left  # a cycle: first appearance wins
            c = min(ready, key=first.get)
            order.append(c)
            for b in after[c]:
                before[b] -= 1
        out[parent] = order
    return out


def dump(doc: dict) -> bytes:
    return (json.dumps(doc, indent=1) + "\n").encode()


def build() -> dict[Path, bytes]:
    for module, version in ((diffusers, DIFFUSERS), (transformers, TRANSFORMERS)):
        if module.__version__ != version:
            raise SystemExit(f"{module.__name__} {module.__version__} is not the pinned {version}")
    by_class: dict[str, tuple[list[str], list[list[str]]]] = {}
    for repo, folder in SOURCES:
        name, model = construct(json.loads(get(hub(repo, join(folder, "config.json")))))
        keys = stored_names(model, list(model.state_dict()), stored_keys(repo, folder))
        sources, orders = by_class.setdefault(name, ([], []))
        sources.append(f"hf://{repo[0]}@{repo[1]}/{folder}".rstrip("/"))
        orders.append(keys)
    for lane, manifest, header, file, digest in LANE_CONFIGS:
        body = (OUT / "inputs" / file).read_bytes()
        if hashlib.sha256(body).hexdigest() != digest:
            raise SystemExit(f"{file}: sha256 differs from the pinned {digest}")
        name, model = construct(json.loads(body))
        sources, orders = by_class.setdefault(name, ([], []))
        sources.append(f"tensorhub://{lane} manifest {manifest} header {header}")
        orders.append(list(model.state_dict()))
    name, source, keys = h3_transformer()
    by_class[name] = ([source], [keys])
    out = {}
    for name, (sources, orders) in sorted(by_class.items()):
        doc = {"class": name, "sources": sources, "generator": "scripts/orders.py",
               "diffusers": DIFFUSERS, "transformers": TRANSFORMERS, "children": tree(orders)}
        out[OUT / f"{name}.json"] = dump(doc)
    for test, repo, folders in CHECKS:
        orders = {}
        for folder in folders:
            _, model = construct(json.loads(get(hub(repo, join(folder, "config.json")))))
            orders[folder] = stored_names(model, list(model.state_dict()), stored_keys(repo, folder))
        out[TESTS / f"{test}.json"] = dump({"source": f"hf://{repo[0]}@{repo[1]}", "components": orders})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    write = ap.parse_args().write
    outputs = build()
    stale = [p for p, b in outputs.items() if not p.exists() or p.read_bytes() != b]
    extra = [p for p in OUT.glob("*.json") if p not in outputs]
    if write:
        for p in stale:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(outputs[p])
        for p in extra:
            p.unlink()
    for p in stale + extra:
        print(("rewrote " if write else "differs: ") + str(p.relative_to(ROOT)))
    if (stale or extra) and not write:
        sys.exit(1)


if __name__ == "__main__":
    main()
