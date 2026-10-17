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
"""Derive single-file converter route tables from diffusers' own single-file converters.

    uv run scripts/single-file-routes.py           # regenerate every family and compare
    uv run scripts/single-file-routes.py --write   # rewrite the tables

A family is one spec, `crates/tensorfs-core/src/ingest/routes/<family>.spec.json`: the
reviewed carrier header(s), the pinned reference pipeline whose configs build the target
census, and per component the source prefix, target class and the diffusers
`single_file_utils` conversion (`null` when the upstream mapping only strips the prefix). A
component sees the keys of its header whose longest matching family prefix is its own, the
view the converter projects. The conversion runs over coordinate-tracking stand-ins, never
tensor values, and the target models are constructed on the meta device. Every census key
becomes one route: a rekey, squeeze, row-block split or transpose of one source run. Any
other operation is recorded as an explicit `materialize` route, which the converter refuses
until it implements that op; nothing is silently dropped. The converter id carries this
generator's version and the diffusers/transformers pins, so a changed table is a new id.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import struct
import sys
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from diffusers.loaders import single_file_utils

GENERATOR = 1
PINS = {"diffusers": "0.40.0", "transformers": "5.16.1"}
ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "crates/tensorfs-core/src/ingest/routes"


def contiguous(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(math.prod(shape[index + 1 :]) for index in range(len(shape)))


@dataclass(frozen=True)
class Materialized:
    """A value no single source run spells: recorded, never guessed."""

    op: str
    sources: tuple[str, ...]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: Materialized(f"{self.op}.{name}", self.sources)


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

    def clone(self) -> Route:
        return self

    def detach(self) -> Route:
        return self

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> Route:
        index = key if isinstance(key, tuple) else (key,)
        index = (*index, *([slice(None)] * (self.ndim - len(index))))
        shape, stride, offset = [], [], self.offset
        for size, step, item in zip(self.shape, self.strides, index, strict=True):
            if isinstance(item, int):
                assert 0 <= item < size
                offset += item * step
            else:
                start, stop, jump = item.indices(size)
                assert jump > 0
                shape.append(len(range(start, stop, jump)))
                stride.append(step * jump)
                offset += start * step
        return replace(self, shape=tuple(shape), strides=tuple(stride), offset=offset)

    def __getattr__(self, name: str) -> Any:
        # reshape/view/to/float/... and anything else change bytes or layout.
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: Materialized(name, (self.key,))

    @classmethod
    def __torch_function__(cls, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        sources: list[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, Route):
                sources.append(value.key)
            elif isinstance(value, list | tuple):
                for item in value:
                    collect(item)

        collect(args)
        return Materialized(getattr(func, "__name__", str(func)), tuple(sources))


def classify(route: Route | Materialized, source: tuple[int, ...]) -> list[Any]:
    """The route's relational kind against its source's shape; `materialize` otherwise."""
    if isinstance(route, Materialized):
        return ["materialize", route.op, list(route.sources)]
    if route.strides == contiguous(route.shape):
        size = math.prod(route.shape)
        squeezed = source
        while squeezed and squeezed[-1] == 1:
            squeezed = squeezed[:-1]
        if route.offset == 0 and route.shape == source:
            return []
        if route.offset == 0 and route.shape == squeezed:
            return ["squeeze"]
        parts, rest = divmod(math.prod(source), size) if size else (0, 1)
        if (
            not rest
            and route.offset % size == 0
            and route.shape == (source[0] // parts, *source[1:])
            and source[0] % parts == 0
        ):
            return ["split", route.offset // size, parts]
    if (
        route.ndim == 2
        and route.offset == 0
        and route.shape == source[::-1]
        and route.strides == (1, source[1])
    ):
        return ["transpose"]
    return ["materialize", "view", [route.key]]


def header(path: Path) -> dict[str, tuple[int, ...]]:
    raw = path.read_bytes()
    (length,) = struct.unpack("<Q", raw[:8])
    document = json.loads(raw[8 : 8 + length])
    return {key: tuple(row["shape"]) for key, row in document.items() if key != "__metadata__"}


def reference_file(base: str, reference: str, path: str) -> dict[str, Any]:
    repository, revision = reference.removeprefix("hf://").split("@")
    assert len(revision) == 40, "a reference pins one exact commit"
    url = f"{base}/{repository}/resolve/{revision}/{path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())


def target(spec: dict[str, Any], config: dict[str, Any]) -> Any:
    library, name = spec["model"].split(":")
    kind = getattr(importlib.import_module(library), name)
    with torch.device("meta"):
        if library == "diffusers":
            return kind.from_config(config)
        return kind(kind.config_class(**config))


def generate(spec_path: Path, base: str) -> bytes:
    spec = json.loads(spec_path.read_text())
    prefixes = [component["prefix"] for component in spec["components"]]
    components = []
    for component in spec["components"]:
        name, prefix = component["component"], component["prefix"]
        shapes = header(ROOT / component.get("header", spec.get("header")))
        checkpoint = {
            key: Route(key, shape, 0, contiguous(shape))
            for key, shape in shapes.items()
            if max((p for p in prefixes if key.startswith(p)), key=len, default=None) == prefix
        }
        config = reference_file(base, spec["reference"], component["config"])
        model = target(component, config)
        census = [(key, tuple(value.shape)) for key, value in model.state_dict().items()]
        arguments = {"checkpoint": checkpoint, "config": config, "model": model, "prefix": prefix}
        if component["convert"] is None:
            converted = {key.removeprefix(prefix): route for key, route in checkpoint.items()}
        else:
            convert = getattr(single_file_utils, component["convert"])
            converted = convert(*(arguments[a] for a in component["arguments"]))
        mapped: dict[str, Any] = {}
        for key, value in converted.items():
            for old, new in component["rename"].items():
                key = new + key.removeprefix(old) if key.startswith(old) else key
            mapped[key] = value
        routes, used = [], set()
        for key, shape in census:
            route = mapped.pop(key)
            sources = [route.key] if isinstance(route, Route) else list(route.sources)
            assert all(s.startswith(prefix) for s in sources), (name, key, sources)
            local = [s.removeprefix(prefix) for s in sources]
            used.update(local)
            if isinstance(route, Route):
                assert route.shape == shape, (name, key, route.shape, shape)
                routes.append([key, local[0], *classify(route, shapes[route.key])])
            else:
                routes.append([key, local, *classify(route, ())])
        assert sorted(mapped) == sorted(component["discard"]), (name, sorted(mapped))
        present = {key.removeprefix(prefix) for key in checkpoint}
        optional = sorted(component["optional"])
        assert not used & set(optional), (name, "an optional source is routed")
        components.append(
            {
                "component": name,
                "prefix": prefix,
                "drop": sorted(present - used - set(optional)),
                "optional": optional,
                "routes": routes,
            }
        )
    document = {
        "converter": f"{spec['name']}/{GENERATOR}-diffusers{PINS['diffusers']}"
        f"-transformers{PINS['transformers']}",
        "reference": spec["reference"],
        "components": components,
    }
    return (json.dumps(document, separators=(",", ":")) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="rewrite the tables")
    parser.add_argument("--huggingface", default="https://huggingface.co")
    args = parser.parse_args()
    for package, version in PINS.items():
        installed = importlib.metadata.version(package)
        if installed != version:
            sys.exit(f"{package} {installed} is not the pinned {version}")
    stale = []
    for spec in sorted(ROUTES.glob("*.spec.json")):
        table = spec.with_name(spec.name.removesuffix(".spec.json") + ".routes.json")
        generated = generate(spec, args.huggingface)
        if args.write:
            table.write_bytes(generated)
        elif not table.is_file() or table.read_bytes() != generated:
            stale.append(table.name)
        print(f"{table.name}: {'written' if args.write else 'checked'}")
    if stale:
        sys.exit(f"stale route tables {stale}: run scripts/single-file-routes.py --write")


if __name__ == "__main__":
    main()
