"""Typed derivation inputs and an execution-owner supplied output capability.

These classes marshal declarations into the existing native parser. They do not
compute geometry, hashes or custody. An output opener is supplied by the trusted
embedding owner; creating a Python object grants no access to a Store.
"""

from __future__ import annotations

import base64
import threading

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from . import _ext
from ._derived_channel import (
    ChannelRefusal, DerivedTransaction, _Channel, _METADATA, _READ, serve_derived, serve_derived_replay,
)
from .records import (
    CheckpointReceipt, DerivedReceipt, ObjectRef, PartReceipt, InspectedSource,
    DeclarationTargets, DeclarationConfigs, DeclarationFiles, DerivationArguments, PartDeclaration,
    TensorDeclaration, TargetDeclaration, ConfigDeclaration,
)


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
    def from_native(cls, value: InspectedSource) -> SourceInspection:
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
    """Access to one exact artifact's tensor metadata, configs and bounded part reads.

    The owner binds these operations to this source and the accepted input scope; the
    capability supplies usable model data through its inspection and read operations.
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
    ) -> DerivationArguments:
        """Marshal into Store.derived_declaration/estimate/begin, never validate twice."""
        targets: DeclarationTargets = {}
        for name, target in self.targets.items():
            additions: dict[str, TensorDeclaration] = {}
            for key, tensor in target.add.items():
                parts: dict[str, PartDeclaration] = {}
                for role, part in tensor.parts.items():
                    value: PartDeclaration = {
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
            target_value: TargetDeclaration = {"drop": list(target.drop), "add": additions}
            if target.source or target.source_component:
                target_value["source"] = target.source
                target_value["source_component"] = target.source_component
            targets[name] = target_value
        configs: DeclarationConfigs = {}
        for name, config in self.configs.items():
            config_value: ConfigDeclaration = {"kind": config.kind}
            if config.source or config.source_config:
                config_value["source"] = config.source
                config_value["source_config"] = config.source_config
            configs[name] = config_value
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
        files: DeclarationFiles = {
            name: base64.b64encode(data).decode("ascii") for name, data in self.files.items()
        }
        return (*arguments, files)


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


class SourceClient:
    def __init__(self, fd: int, source: Source) -> None:
        self.channel = _Channel(fd)
        self.source = source
        self.lock = threading.Lock()

    def close(self) -> None:
        self.channel.close()

    def _result(self, request: dict[str, Any]) -> Any:
        self.channel.send(request)
        answer = self.channel.reply()
        if "result" not in answer:
            raise ChannelRefusal("MISSING_FIELD", "source answer has no result")
        return answer["result"]

    def _receive_into(self, view: memoryview) -> None:
        while view:
            block = self.channel.receive()
            if not isinstance(block, bytes) or not block or len(block) > len(view):
                raise ChannelRefusal("LENGTH_MISMATCH", "invalid source byte stream")
            view[:len(block)] = block
            view = view[len(block):]

    def inspect(self, components: Sequence[str], configs: Sequence[str]) -> SourceInspection:
        with self.lock:
            try:
                value = self._result({"op": "inspect", "components": list(components), "configs": list(configs)})
                if not isinstance(value, dict):
                    raise ChannelRefusal("WRONG_TYPE", "invalid source inspection response")
                total = 0
                for config in value["configs"]:
                    length = config.pop("length")
                    if type(length) is not int or not 0 <= length <= _METADATA - total:
                        raise ChannelRefusal("SIZE_CAP", "source configurations exceed their bound")
                    total += length
                    data = bytearray(length)
                    self._receive_into(memoryview(data))
                    config["data"] = bytes(data)
                result = SourceInspection.from_native(cast(InspectedSource, value))
                if result.source != self.source:
                    raise ChannelRefusal("TRANSACTION_CONFLICT", "source inspection changed identity")
                return result
            except BaseException:
                self.channel.close()
                raise

    def check(self) -> None:
        with self.lock:
            try:
                if self._result({"op": "check", "manifest": self.source.manifest, "manifest_length": self.source.length}) is not None:
                    raise ChannelRefusal("WRONG_TYPE", "invalid source check response")
            except BaseException:
                self.channel.close()
                raise

    def read_part_into(self, component: str, key: str, role: str, offset: int, into: object) -> None:
        view = memoryview(into)  # type: ignore[arg-type]
        if view.readonly or not view.c_contiguous or view.nbytes > _READ:
            raise ChannelRefusal("SIZE_CAP", "source read needs a bounded writable contiguous buffer")
        view = view.cast("B")
        with self.lock:
            try:
                length = self._result({"op": "read", "manifest": self.source.manifest, "manifest_length": self.source.length, "component": component, "key": key,
                                       "role": role, "offset": offset, "length": len(view)})
                if type(length) is not int or length != len(view):
                    raise ChannelRefusal("LENGTH_MISMATCH", "source read changed its length")
                self._receive_into(view)
            except BaseException:
                self.channel.close()
                raise


if _ext.STORE:
    #: The owner's half of a source channel, for tests and tools: ``serve_source(store, fd,
    #: manifest, length, *, check_current=None)`` serves until the peer closes it.
    serve_source = _ext.serve_source


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
    "ObjectRef",
    "DerivedReceipt",
    "CheckpointReceipt",
    "PartReceipt",
    "derive",
    "serve_derived",
    "serve_derived_replay",
]
if _ext.STORE:
    __all__.append("serve_source")
