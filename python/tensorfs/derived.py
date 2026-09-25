"""Typed derivation inputs and an execution-owner supplied output capability.

These classes marshal declarations into the existing native parser. They do not
compute geometry, hashes or custody. An output opener is supplied by the trusted
embedding owner; creating a Python object grants no access to a Store.
"""

from __future__ import annotations

import base64

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ._derived_channel import DerivedTransaction, serve_derived, serve_derived_replay


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
class SourceInspection:
    """Selected native source geometry and canonical configs; no payload locations."""

    source: Source
    components: Mapping[str, Mapping[str, Tensor]]
    configs: Mapping[str, bytes]
    identities: Mapping[tuple[str, str], str] = field(default_factory=dict)

    @classmethod
    def from_native(cls, value: Mapping[str, Any]) -> SourceInspection:
        """Project Store.inspect_derived_source facts without validating them again."""
        source = value["source"]
        return cls(
            Source(source["manifest_id"], source["manifest_length"]),
            {
                component["name"]: {
                    tensor["key"]: Tensor(
                        tensor["logical_dtype"],
                        tuple(tensor["shape"]),
                        tensor["encoding"],
                        {
                            part["role"]: Part(part["dtype"], tuple(part["shape"]))
                            for part in tensor["parts"]
                        },
                    )
                    for tensor in component["tensors"]
                }
                for component in value["components"]
            },
            {config["name"]: config["data"] for config in value["configs"]},
            {
                (component["name"], tensor["key"]): tensor["identity"]
                for component in value["components"]
                for tensor in component["tensors"]
            },
        )


@dataclass(frozen=True, slots=True)
class SourceCapability(Source):
    """An exact source whose trusted owner admits metadata inspection before derive.

    The owner binds the callable to this source and the accepted input scope.
    Constructing this data object never grants Store access or broader I/O.
    """

    _inspect: Callable[[Sequence[str], Sequence[str]], SourceInspection] = field(
        repr=False, compare=False
    )
    _close: Callable[[], None] | None = field(default=None, repr=False, compare=False)
    _read_part: Callable[[str, str, str, int, object], None] | None = field(
        default=None, repr=False, compare=False
    )
    _check: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def inspect(
        self, *, components: Sequence[str] = (), configs: Sequence[str] = ()
    ) -> SourceInspection:
        return self._inspect(tuple(components), tuple(configs))

    @classmethod
    def from_fd(cls, manifest: str, length: int, fd: int) -> SourceCapability:
        """Own a preconnected inspector for one exact already-admitted source."""
        from ._source_channel import SourceClient

        client = SourceClient(fd, Source(manifest, length))
        return cls(manifest, length, client.inspect, client.close, client.read_part_into, client.check)

    def read_part_into(
        self, component: str, key: str, role: str, offset: int, into: object
    ) -> None:
        """Fill a bounded writable buffer from one semantic part under this source lease."""
        if self._read_part is None:
            raise ValueError("source capability has no part reader")
        self._read_part(component, key, role, offset, into)

    def check(self) -> None:
        """Recheck the native lease and accepted execution lifetime."""
        if self._check is not None:
            self._check()

    def close(self) -> None:
        if self._close is not None:
            self._close()

    def __enter__(self) -> SourceCapability:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


@dataclass(frozen=True, slots=True)
class Derivation:
    sources: Mapping[str, Source]
    targets: Mapping[str, Target]
    configs: Mapping[str, Config]
    order: Sequence[tuple[str, str]]
    files: Mapping[str, bytes] = field(default_factory=dict)

    def native_arguments(
        self,
        max_new_bytes: int,
    ) -> (
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[tuple[str, str]], int]
        | tuple[
            dict[str, Any], dict[str, Any], dict[str, Any], list[tuple[str, str]], int,
            dict[str, str],
        ]
    ):
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
        arguments = (
            {
                name: (source.manifest, source.length)
                for name, source in self.sources.items()
            },
            targets,
            configs,
            list(self.order),
            max_new_bytes,
        )
        if not self.files:
            return arguments
        return (
            *arguments,
            {name: base64.b64encode(data).decode("ascii") for name, data in self.files.items()},
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


from ._source_channel import serve_source  # noqa: E402


__all__ = [
    "Source",
    "PartSource",
    "Part",
    "Tensor",
    "Target",
    "Config",
    "SourceInspection",
    "SourceCapability",
    "Derivation",
    "OutputCapability",
    "DerivedTransaction",
    "derive",
    "serve_derived",
    "serve_derived_replay",
    "serve_source",
]
