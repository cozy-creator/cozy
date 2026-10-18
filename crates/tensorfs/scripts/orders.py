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
here is that order for one pinned component config, built on the meta device, kept as patterns:
digit segments written `#`, each pattern once, in first appearance. `vectors/orders/*.json` holds
full key orders of other pinned models, which the import tests compare against.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import diffusers
import torch
import transformers

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


def fetch(repo: str, revision: str, path: str) -> dict:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def state_dict_order(config: dict) -> tuple[str, list[str]]:
    """The class and the `state_dict()` keys of the model `config` builds, on the meta device."""
    if "_class_name" in config:
        cls = getattr(diffusers, config["_class_name"])
        with torch.device("meta"):
            model = cls.from_config(config)
        return config["_class_name"], list(model.state_dict())
    name = config["architectures"][0]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text(json.dumps(config))
        conf = transformers.AutoConfig.from_pretrained(d)
    with torch.device("meta"):
        model = getattr(transformers, name)(conf)
    return name, list(model.state_dict())


def pattern(key: str) -> str:
    return ".".join("#" if s.isdigit() else s for s in key.split("."))


def build() -> tuple[bytes, dict[str, bytes]]:
    for package, version in PINS.items():
        have = sys.modules[package].__version__
        if have != version:
            raise SystemExit(f"{package} {have} is not the pinned {version}")
    references = []
    for name, repo, revision, components in REFERENCES:
        for component in components:
            cls, keys = state_dict_order(fetch(repo, revision, f"{component}/config.json"))
            patterns = list(dict.fromkeys(pattern(k) for k in keys))
            references.append({"name": f"{name}/{component}", "class": cls,
                               "source": f"hf://{repo}@{revision}/{component}", "patterns": patterns})
    doc = {"generator": "scripts/orders.py", **PINS, "references": references}
    checks = {}
    for name, repo, revision, components in CHECKS:
        orders = {c: state_dict_order(fetch(repo, revision, f"{c}/config.json"))[1] for c in components}
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
