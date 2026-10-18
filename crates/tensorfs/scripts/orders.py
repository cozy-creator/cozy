#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["diffusers==0.40.0", "transformers==5.16.1", "torch>=2.11"]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Generate the constructor-order references (`crates/tensorfs-core/converters/orders.json`).

    uv run scripts/orders.py           # regenerate and compare byte for byte
    uv run scripts/orders.py --write   # rewrite them

A Diffusers or Transformers folder carries no tensor order (safetensors sorts its keys), so an
import orders it by the `state_dict()` order of the constructed model (spec §10). Each reference
here is that order for one pinned component config, built on the meta device, named as the
repository stores the keys, and kept as a tree: for each parent pattern (digit segments written
`#`), its children in order, merged over every instance of that parent (a down block with
attentions and one without agree on `attentions` < `resnets` < `downsamplers`).
`vectors/orders/*.json` holds full key orders of other pinned models, which the import tests
compare against.
"""

from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "crates" / "tensorfs-core" / "converters" / "orders.json"
TESTS = ROOT / "crates" / "tensorfs-core" / "vectors" / "orders"
PINS = {"diffusers": "0.40.0", "transformers": "5.16.1"}

# (name, repo, revision, component folders): the references imports are ordered by.
REFERENCES = [
    ("sdxl", "stabilityai/stable-diffusion-xl-base-1.0", "462165984030d82259a11f4367a4eed129e94a7b",
     ["unet", "vae", "text_encoder", "text_encoder_2"]),
    ("anima", "circlestone-labs/Anima-Base-v1.0-Diffusers", "073c3a9db359c31ad0e8aa268d15775473c2176c",
     ["transformer", "text_encoder", "text_conditioner", "vae"]),
    ("qwen-image", "Qwen/Qwen-Image", "75e0b4be04f60ec59a75f475837eced720f823b6",
     ["transformer", "vae", "text_encoder"]),
]
# Models the tests import; their full orders are never references.
CHECKS = [
    ("tiny-sdxl-pipe", "hf-internal-testing/tiny-sdxl-pipe", "20594cbc343cfcfe447af5c87cdaf6c436b453f2",
     ["unet", "vae", "text_encoder", "text_encoder_2"]),
]


def get(repo: str, revision: str, path: str, first: int | None = None, last: int | None = None) -> bytes:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
    req = urllib.request.Request(url)
    if first is not None:
        req.add_header("Range", f"bytes={first}-{last}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch(repo: str, revision: str, path: str) -> dict:
    return json.loads(get(repo, revision, path))


def disk_keys(repo: str, revision: str, component: str) -> list[str]:
    """The tensor names the repository stores for `component` (index or safetensors header)."""
    for name in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
        try:
            return list(fetch(repo, revision, f"{component}/{name}.index.json")["weight_map"])
        except urllib.error.HTTPError:
            pass
        try:
            n = struct.unpack("<Q", get(repo, revision, f"{component}/{name}", 0, 7))[0]
        except urllib.error.HTTPError:
            continue
        header = json.loads(get(repo, revision, f"{component}/{name}", 8, 7 + n))
        return [k for k in header if k != "__metadata__"]
    raise SystemExit(f"{repo}: no weights for {component}")


def on_disk(keys: list[str], disk: list[str]) -> list[str]:
    """`state_dict()` keys renamed to the stored names, where the two differ by a prefix (a
    Transformers class whose checkpoint keeps `text_model.`). Every stored name must appear."""
    stored, keyset = set(disk), set(keys)
    added: dict[str, int] = {}
    for d in stored - keyset:
        segs = d.split(".")
        for i in range(1, len(segs)):
            if ".".join(segs[i:]) in keyset:
                prefix = ".".join(segs[:i]) + "."
                added[prefix] = added.get(prefix, 0) + 1
                break

    def name(k: str) -> str:
        if k in stored:
            return k
        for prefix in sorted(added, key=added.get, reverse=True):
            if prefix + k in stored:
                return prefix + k
        segs = k.split(".")
        for i in range(1, len(segs)):
            if ".".join(segs[i:]) in stored:
                return ".".join(segs[i:])
        return k  # a buffer or tied weight the checkpoint does not store

    named = [name(k) for k in keys]
    missing = stored - set(named)
    if missing:
        raise SystemExit(f"stored names the constructor lacks: {sorted(missing)[:5]}")
    return named


def state_dict_order(config: dict, stored: list[str]) -> tuple[str, list[str]]:
    """The class and the `state_dict()` keys of the model `config` builds, on the meta device,
    under the names the checkpoint stores them by."""
    if "_class_name" in config:
        cls = getattr(diffusers, config["_class_name"])
        with torch.device("meta"):
            model = cls.from_config(config)
        return config["_class_name"], on_disk(list(model.state_dict()), stored)
    name = config["architectures"][0]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text(json.dumps(config))
        conf = transformers.AutoConfig.from_pretrained(d)
    with torch.device("meta"):
        model = getattr(transformers, name)(conf)
    keys = list(model.state_dict())
    # Transformers renames checkpoint keys on load; its reversed transforms give the saved names.
    for legacy in (True, False):
        transforms = [t.reverse_transform() for t in get_model_conversion_mapping(model, add_legacy=legacy)[::-1]]
        renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
        converters = [t for t in transforms if isinstance(t, WeightConverter)]
        named = [rename_source_key(k, renamings, converters, reverse=True)[0] for k in keys]
        if set(stored) <= set(named):
            return name, named
    return name, on_disk(keys, stored)


def pattern(key: str) -> str:
    return ".".join("#" if s.isdigit() else s for s in key.split("."))


def tree(keys: list[str]) -> dict[str, list[str]]:
    """Parent pattern -> its children in order, merged over the parent's instances: each
    instance's order is a constraint; ties go to the first appearance."""
    instances: dict[str, list[str]] = {}
    for k in keys:
        segs = k.split(".")
        for d in range(len(segs)):
            child = "#" if segs[d].isdigit() else segs[d]
            seen = instances.setdefault(".".join(segs[:d]), [])
            if child not in seen:
                seen.append(child)
    by_pattern: dict[str, list[list[str]]] = {}
    for parent, children in instances.items():
        by_pattern.setdefault(pattern(parent), []).append(children)
    out = {}
    for parent, seqs in by_pattern.items():
        first: dict[str, int] = {}
        for seq in seqs:
            for c in seq:
                first.setdefault(c, len(first))
        after = {c: set() for c in first}
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


def build() -> tuple[bytes, dict[str, bytes]]:
    for package, version in PINS.items():
        have = sys.modules[package].__version__
        if have != version:
            raise SystemExit(f"{package} {have} is not the pinned {version}")
    references = []
    for name, repo, revision, components in REFERENCES:
        for component in components:
            stored = disk_keys(repo, revision, component)
            cls, keys = state_dict_order(fetch(repo, revision, f"{component}/config.json"), stored)
            references.append({"name": f"{name}/{component}", "class": cls,
                               "source": f"hf://{repo}@{revision}/{component}", "children": tree(keys)})
    doc = {"generator": "scripts/orders.py", **PINS, "references": references}
    checks = {}
    for name, repo, revision, components in CHECKS:
        orders = {c: state_dict_order(fetch(repo, revision, f"{c}/config.json"),
                                      disk_keys(repo, revision, c))[1] for c in components}
        checks[f"{name}.json"] = (json.dumps({"source": f"hf://{repo}@{revision}", "components": orders},
                                             indent=1) + "\n").encode()
    return (json.dumps(doc, indent=1) + "\n").encode(), checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    write = ap.parse_args().write
    table, checks = build()
    outputs = {OUT: table, **{TESTS / n: b for n, b in checks.items()}}
    stale = [p for p, b in outputs.items() if not p.exists() or p.read_bytes() != b]
    if write:
        for p in stale:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(outputs[p])
    for p in stale:
        print(("wrote " if write else "differs: ") + str(p.relative_to(ROOT)))
    if stale and not write:
        sys.exit(1)


if __name__ == "__main__":
    main()
