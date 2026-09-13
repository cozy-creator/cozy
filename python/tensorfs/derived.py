"""Typed derivation inputs and an execution-owner supplied output capability.

These classes marshal declarations into the existing native parser. They do not
compute geometry, hashes or custody. An output opener is supplied by the trusted
embedding owner; creating a Python object grants no access to a Store.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ._derived_channel import DerivedTransaction, serve_derived


@dataclass(frozen=True, slots=True)
class Source:
    manifest: str
    length: int


@dataclass(frozen=True, slots=True)
class PartSource:
    source: str
    component: str
    tensor: str
    role: str


@dataclass(frozen=True, slots=True)
class Part:
    dtype: str
    shape: Sequence[int]
    source: PartSource | None = None


@dataclass(frozen=True, slots=True)
class Tensor:
    logical_dtype: str
    shape: Sequence[int]
    encoding: str
    parts: Mapping[str, Part]


@dataclass(frozen=True, slots=True)
class Target:
    source: str = ""
    source_component: str = ""
    drop: Sequence[str] = ()
    add: Mapping[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Config:
    kind: Literal["copy", "derive", "add"]
    source: str = ""
    source_config: str = ""


@dataclass(frozen=True, slots=True)
class Derivation:
    sources: Mapping[str, Source]
    targets: Mapping[str, Target]
    configs: Mapping[str, Config]
    order: Sequence[tuple[str, str]]

    def native_arguments(
        self,
        max_new_bytes: int,
    ) -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any], list[tuple[str, str]], int
    ]:
        """Marshal into Store.derived_declaration/estimate/begin, never validate twice."""
        targets: dict[str, Any] = {}
        for name, target in self.targets.items():
            additions: dict[str, Any] = {}
            for key, tensor in target.add.items():
                parts: dict[str, Any] = {}
                for role, part in tensor.parts.items():
                    value: dict[str, Any] = {
                        "dtype": part.dtype,
                        "shape": list(part.shape),
                    }
                    if part.source is not None:
                        selected = part.source
                        value["source"] = {
                            "source": selected.source,
                            "component": selected.component,
                            "tensor": selected.tensor,
                            "role": selected.role,
                        }
                    parts[role] = value
                additions[key] = {
                    "logical_dtype": tensor.logical_dtype,
                    "shape": list(tensor.shape),
                    "encoding": tensor.encoding,
                    "parts": parts,
                }
            value = {"drop": list(target.drop), "add": additions}
            if target.source or target.source_component:
                value.update(
                    source=target.source, source_component=target.source_component
                )
            targets[name] = value
        configs: dict[str, Any] = {}
        for name, config in self.configs.items():
            value = {"kind": config.kind}
            if config.source or config.source_config:
                value.update(source=config.source, source_config=config.source_config)
            configs[name] = value
        return (
            {
                name: (source.manifest, source.length)
                for name, source in self.sources.items()
            },
            targets,
            configs,
            list(self.order),
            max_new_bytes,
        )


class OutputCapability:
    """One declared output's owner-supplied opener; no storage authority is minted here."""

    def __init__(self, open_output: Callable[[Derivation], DerivedTransaction]) -> None:
        self._open = open_output
        self._used = False

    def open(self, declaration: Derivation) -> DerivedTransaction:
        if self._used:
            raise ValueError("output capability was already opened")
        self._used = True
        return self._open(declaration)


def derive(output: OutputCapability, declaration: Derivation) -> DerivedTransaction:
    """Bind through the execution owner, then use TensorFS's scoped native writer."""
    return output.open(declaration)


__all__ = [
    "Source",
    "PartSource",
    "Part",
    "Tensor",
    "Target",
    "Config",
    "Derivation",
    "OutputCapability",
    "DerivedTransaction",
    "derive",
    "serve_derived",
]
