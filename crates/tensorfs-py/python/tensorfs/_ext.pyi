"""`tensorfs._ext`: the compiled extension's types."""

from collections.abc import Callable, ItemsView, Iterator, KeysView, Mapping, Sequence, ValuesView
from os import PathLike
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol, TypeVar, overload
from typing_extensions import Buffer

from .api import CheckpointFile

__version__: str
STORE_FORMAT: int

def _kernel_binary(family: int, core: str, dtype: str) -> tuple[bytes, str, int, int]:
    """Internal engine binary payload, entry symbol, shared bytes and warp count."""

#: `tensorfs._ext.varena`, typed in full by `varena/__init__.pyi` (the public package).
varena: ModuleType

#: Physical format inspection for storage integrations.
raw: ModuleType

StrPath = str | PathLike[str]
_T = TypeVar("_T")

class Error(Exception):
    """A TensorFS error. `.code` names its family; `.detail` the cause."""

    code: str
    detail: str

class InvalidError(Error):
    """A document, argument or tensor breaks the format or a grammar (`invalid`)."""

class FormatError(Error):
    """Another document kind or major version (`format`)."""

class NotFoundError(Error):
    """No such ref, release, variant, component, tensor, asset or object (`not_found`)."""

class IntegrityError(Error):
    """Bytes that do not match their digest or length (`integrity`)."""

class UnknownSchemeError(Error):
    """The tensor's scheme has no decoder here; its parts still read (`unknown_scheme`)."""

class DecodeError(Error):
    """A known scheme whose parts or params break its definition (`decode`)."""

class StorageError(Error):
    """The filesystem failed (`io`)."""

class DiskFullError(Error):
    """The disk or a quota is full (`full`)."""

class NetworkError(Error):
    """A transfer failed or stalled (`network`)."""

class DeniedError(Error):
    """Refused: egress policy, credentials, TLS trust or the server (`denied`)."""

class ConflictError(Error):
    """A revision, conditional write or server policy conflicts (`conflict`)."""

class UnsupportedError(Error):
    """A source or target this build cannot represent (`unsupported`)."""

class Ceiling:
    """What reads can reach: the volume's sequential GB/s (past the page cache when `disk_direct`)
    and, when a CUDA device was asked, pinned host-to-device GB/s over `streams` streams."""

    @property
    def disk_gbps(self) -> float: ...
    @property
    def disk_direct(self) -> bool: ...
    @property
    def h2d_gbps(self) -> float | None: ...
    @property
    def streams(self) -> int | None: ...

def ceiling(device: int | None = None, path: StrPath | None = None) -> Ceiling:
    """`path`'s volume (default: the TensorFS home) and, with `device`, pinned H2D to that GPU;
    measured unless kept this boot."""

class VerificationReport:
    """Local object references found missing or corrupt; verification never repairs them."""

    @property
    def missing(self) -> list[tuple[str, int]]: ...
    @property
    def corrupt(self) -> list[tuple[str, int]]: ...
    @property
    def ok(self) -> bool: ...

class Encoding:
    """A scheme and its complete params; str() is `scheme {key: value, ...}`."""

    @property
    def scheme(self) -> str: ...
    @property
    def params(self) -> dict[str, Any]: ...
    def __str__(self) -> str: ...

class TensorMetadata:
    """A tensor's logical dtype and shape, its encoding (None when plain)."""

    @property
    def dtype(self) -> str: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def encoding(self) -> Encoding | None: ...

class Array:
    """Owned CPU or CUDA bytes; DLPack consumers share them, and tobytes copies to the host."""

    @property
    def dtype(self) -> str: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def nbytes(self) -> int: ...
    def tobytes(self) -> bytes: ...
    def __dlpack__(
        self,
        *,
        stream: object | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

class Encoded:
    """An encoded tensor as stored (`decode=False`): its logical dtype and shape, scheme,
    params and part arrays by role."""

    @property
    def dtype(self) -> str: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def scheme(self) -> str: ...
    @property
    def params(self) -> dict[str, Any]: ...
    @property
    def parts(self) -> dict[str, Array]: ...

class Tensors(Mapping[str, TensorMetadata]):
    """Tensor metadata in construction order, with engine-backed host and CUDA reads."""

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
    def __getitem__(self, key: str) -> TensorMetadata: ...
    def __contains__(self, key: object) -> bool: ...
    def keys(self) -> KeysView[str]: ...
    def values(self) -> ValuesView[TensorMetadata]: ...
    def items(self) -> ItemsView[str, TensorMetadata]: ...
    @overload
    def get(self, key: str, default: None = None) -> TensorMetadata | None: ...
    @overload
    def get(self, key: str, default: TensorMetadata) -> TensorMetadata: ...
    @overload
    def get(self, key: str, default: _T) -> TensorMetadata | _T: ...
    @overload
    def read(self, keys: str, *, device: str = "cpu", decode: bool = False) -> Array | Encoded: ...
    @overload
    def read(self, keys: list[str] | tuple[str, ...] | None = None, *, device: str = "cpu", decode: bool = False) -> dict[str, Array | Encoded]: ...
    @overload
    def read(self, keys: Sequence[str], *, device: str = "cpu", decode: bool = False) -> Array | Encoded | dict[str, Array | Encoded]: ...
    def read_into(self, key: str, out: object, *, role: str | None = None, decode: bool = False,
                  offset: int = 0) -> None:
        """Fill a writable contiguous Buffer or DLPack array from `offset`: a plain tensor's data,
        one stored part (`role`), or with `decode=True` an encoded tensor's logical bytes. An
        encoded tensor with neither raises `InvalidError` naming its roles."""

class ClassRef:
    """A class as `(library, version, name)`, for any implementation and never an import path:
    `ClassRef("diffusers", "0.35.1", "SD3Transformer2DModel")`. `ClassRef("", "", "")` is an
    unknown model; `version` is `""` when nothing records it. A component's tensors are that
    class's `state_dict` at that version. Iterates as the 3-tuple; equal and hashable by value."""

    def __init__(self, library: str, version: str, name: str) -> None: ...
    @property
    def library(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def name(self) -> str: ...
    def __iter__(self) -> Iterator[str]: ...

class Pipeline:
    """What `model_index.json` held."""

    @property
    def cls(self) -> ClassRef: ...
    @property
    def config(self) -> dict[str, Any]: ...

class Component:
    """One component of a checkpoint: class, config, tensors and assets."""

    @property
    def name(self) -> str: ...
    @property
    def cls(self) -> ClassRef: ...
    @property
    def config(self) -> dict[str, Any]: ...
    @property
    def config_file(self) -> str:
        """The file name `config` exports as: `config.json` unless the writer named another."""
    @property
    def tensors(self) -> Tensors:
        """`key -> TensorMetadata`, in construction order."""
    @property
    def assets(self) -> dict[str, tuple[str, int]]:
        """`name -> (sha256, length)`: run data in the component's folder."""
    def asset(self, name: str) -> bytes:
        """An asset's bytes, fetched when absent."""
    def export_files(self, directory: StrPath | None = None) -> Path:
        """Config and assets only, never weights, written to `directory`, or by default to a
        folder the cache manages and evicts like its other entries."""

class Checkpoint:
    """One cozytensors read from a store."""

    def close(self) -> None:
        """Release this handle's access and cached components; existing child handles remain usable."""
    def __enter__(self) -> Checkpoint: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...
    def verify(self, components: Sequence[str] | None = None) -> VerificationReport:
        """Check the actual header and selected full chunks/assets, without fetching or repair."""
    @property
    def id(self) -> str:
        """`sha256:<hex>` of the header."""
    @property
    def tree(self) -> str | None:
        """`sha256:<hex>` of the repository tree, when known."""
    @property
    def variant(self) -> str | None: ...
    @property
    def repo_id(self) -> str | None: ...
    @property
    def revision(self) -> str | None:
        """The tag this checkpoint was resolved through, when known."""
    @property
    def pipeline(self) -> Pipeline | None: ...
    @property
    def components(self) -> dict[str, Component]:
        """In construction order."""
    @property
    def chunks(self) -> list[tuple[str, int]]:
        """The header's chunk objects in order: `(sha256, length)`."""
    @property
    def metadata(self) -> dict[str, Any]: ...
    def open_file(self) -> CheckpointFile:
        """The checkpoint as one read-only, seekable file: its header, then its chunks and assets
        in header order at 4 KiB-aligned offsets, assembled from the store on demand; nothing is
        written."""
    def missing(self, components: Sequence[str] | None = None) -> list[tuple[str, int]]:
        """The chunks and assets of `components` (all when None) not here."""
    def download(self, components: Sequence[str] | None = None, *, priority: int = 0,
              progress: Callable[[str, int, int], object] | None = None) -> None: ...

class _CheckpointFile:
    """The native half of `CheckpointFile`."""
    def readinto(self, buffer: Buffer) -> int: ...
    def seek(self, offset: int, whence: int = 0) -> int: ...
    def tell(self) -> int: ...
    def close(self) -> None: ...

class ComponentWriter:
    @property
    def name(self) -> str: ...
    def put(self, key: str, dtype: str, shape: Sequence[int], data: Buffer) -> None: ...
    def put_array(self, key: str, data: Any, *, dtype: str | None = None, shape: Sequence[int] | None = None) -> None: ...
    def copy_component(self, key: str, source: Component, as_key: str | None = None) -> None: ...
    def put_encoded(
        self,
        key: str,
        dtype: str,
        shape: Sequence[int],
        scheme: str,
        parts: Mapping[str, tuple[str, Sequence[int], Buffer]],
        params: Mapping[str, Any] | None = None,
    ) -> None: ...
    def asset(self, name: str, data: Buffer) -> None: ...
    def order(self, keys: Sequence[str]) -> None: ...
    def has(self, key: str) -> bool: ...

class Writer:
    def __init__(self, dest: StrPath | None = None, *, variant: str | None = None,
                 base: Checkpoint | None = None, resume: str | None = None,
                 cache_dir: StrPath | None = None, repo_id: str | None = None, revision: str | None = None,
                 endpoint: str | None = None, token: _Token | None = None, ca_der: bytes | None = None) -> None: ...
    def pipeline(self, cls: ClassRef | tuple[str, str, str] | Literal[""] | None, config: Mapping[str, Any]) -> None: ...
    def component(
        self, name: str, cls: ClassRef | tuple[str, str, str] | Literal[""] | None = None, config: Mapping[str, Any] | None = None,
        config_file: str | None = None,
    ) -> ComponentWriter:
        """Declares a component (again: class and config change, tensors stay)."""
    def set_metadata(self, key: str, value: Any) -> None: ...
    def has(self, component: str, key: str) -> bool: ...
    def commit(self) -> Checkpoint: ...
    def abandon(self) -> None: ...

class _Client:
    """Internal marshaling adapter over the public Rust session."""
    def __init__(self, root: StrPath | None = None, *, endpoint: str | None = None,
                 token: _Token | None = None, ca_der: bytes | None = None) -> None: ...
    def open(self, name_or_path: str, *, revision: str | None = None,
             variant: str | None = None, bits: int | None = None, gpus: Sequence[str] | None = None,
             local_files_only: bool = False, lazy: bool = True) -> Checkpoint: ...
    def materialize(self, identity: tuple[str, str | None, str | None], *,
                          components: Sequence[str] | None = None,
                          local_dir: StrPath | None = None, local_files_only: bool = False) -> str: ...
    def download_file(self, repo_id: str, filename: str, *, revision: str | None = None,
                     local_dir: StrPath | None = None, local_files_only: bool = False) -> str: ...
    def upload_folder(self, folder_path: StrPath, repo_id: str,
                      revision: str | None = None, delete_patterns: Sequence[str] = ...) -> str: ...
    def convert_to_cozytensors(self, source: str | Sequence[str], *, converter: str | None = None,
                source_variant: str | None = None,
                progress: Callable[[str, int, int], object] | None = None) -> Checkpoint: ...


class TokenProvider(Protocol):
    def headers(self, method: str, url: str, headers: Mapping[str, str], payload: str | None) -> Mapping[str, str]: ...
    def retry(self, method: str, url: str, headers: Mapping[str, str], payload: str | None,
              status: int, response_headers: Mapping[str, str]) -> bool: ...

class Endpoint:
    def __init__(self, url: str, token: _Token | None = None, *, ca_der: bytes | None = None,
                 cache_dir: StrPath | None = None) -> None: ...
    def repo(self, repo_id: str) -> tuple[dict[str, Any], str] | None: ...
    def commit(self, repo_id: str, revision: str) -> tuple[str, str | None]: ...
    def update_repo(self, repo_id: str, change: Any) -> dict[str, Any]: ...

_Token = str | TokenProvider | Callable[[str, str, Mapping[str, str], str | None], Mapping[str, str]]

def _materialize(checkpoint: Checkpoint, directory: StrPath, *, format: str = "cozytensors",
                 decode: bool = False) -> str: ...
