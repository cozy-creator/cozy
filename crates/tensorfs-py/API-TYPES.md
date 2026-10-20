# Python API declarations

The declarations of the `tensorfs` package. For usage, see the [Python guide](API.md).

`TensorMetadata` describes the logical tensor. Ordinary encoded reads expose named
`Array` values without physical locations; `tensorfs.raw` provides advanced physical
inspection and resident-part decoding for storage integrations.

The provider/DLPack protocols describe accepted inputs and are not additional package exports.

```python
from __future__ import annotations
import io
from collections.abc import Buffer, Callable, ItemsView, Iterator, KeysView, Mapping, Sequence, ValuesView
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, TypeVar, overload
StrPath = str | PathLike[str]
_T = TypeVar('_T')
Token = str | TokenProvider | Callable[[str, str, Mapping[str, str], str | None], Mapping[str, str]]
__version__: str

class TokenProvider(Protocol):

    def headers(self, method: str, url: str, headers: Mapping[str, str], payload: str | None) -> Mapping[str, str]:
        ...

class RetryingTokenProvider(TokenProvider, Protocol):

    def retry(self, method: str, url: str, headers: Mapping[str, str], payload: str | None, status: int, response_headers: Mapping[str, str]) -> bool:
        ...

class DLPack(Protocol):

    def __dlpack_device__(self) -> tuple[int, int]:
        ...

    def __dlpack__(self, stream: int | None=None) -> object:
        ...

def configure(*, endpoint: str | None=None, cache_dir: StrPath | None=None, token: Token | None=None, ca_der: bytes | None=None) -> None:
    ...

def convert_to_cozytensors(source: str | Sequence[str], *, converter: str | None=None, source_variant: str | None=None, cache_dir: StrPath | None=None, token: Token | None=None, progress: Callable[[str, int, int], object] | None=None) -> Checkpoint:
    ...

def download_file(repo_id: str, filename: str, *, revision: str | None=None, cache_dir: StrPath | None=None, local_dir: StrPath | None=None, local_files_only: bool=False, token: Token | None=None, endpoint: str | None=None) -> str:
    ...

def open(name_or_path: StrPath, *, revision: str | None=None, variant: str | None=None, endpoint: str | None=None, local_files_only: bool=False, cache_dir: StrPath | None=None, token: Token | None=None, lazy: bool=True) -> Checkpoint:
    ...

def materialize(source: str | Checkpoint, *, revision: str | None=None, variant: str | None=None, components: Sequence[str] | None=None, cache_dir: StrPath | None=None, local_dir: StrPath | None=None, local_files_only: bool=False, token: Token | None=None, endpoint: str | None=None, format: str='cozytensors', decode: bool=False) -> str:
    ...

def upload_folder(*, repo_id: str, folder_path: StrPath, revision: str='main', token: Token | None=None, endpoint: str | None=None, cache_dir: StrPath | None=None) -> str:
    ...

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

class Encoding:
    """A scheme and its complete params; str() is `scheme {key: value, ...}`."""

    @property
    def scheme(self) -> str:
        ...

    @property
    def params(self) -> dict[str, Any]:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

class TensorMetadata:
    """A tensor's logical dtype and shape, and its encoding (None when plain)."""

    @property
    def dtype(self) -> str:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def encoding(self) -> Encoding | None:
        ...

    def __repr__(self) -> str:
        ...

class Array:
    """Owned CPU or CUDA bytes; DLPack consumers share them, and tobytes copies to the host."""

    @property
    def dtype(self) -> str:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def nbytes(self) -> int:
        ...

    def tobytes(self) -> bytes:
        ...

    def __dlpack__(self, *, stream: object | None=None, max_version: tuple[int, int] | None=None, dl_device: tuple[int, int] | None=None, copy: bool | None=None) -> object:
        ...

    def __dlpack_device__(self) -> tuple[int, int]:
        ...

    def __repr__(self) -> str:
        ...

class Encoded:
    """An encoded tensor as stored (`decode=False`): logical dtype and shape, scheme, params
    and part arrays by role."""

    @property
    def dtype(self) -> str:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def scheme(self) -> str:
        ...

    @property
    def params(self) -> dict[str, Any]:
        ...

    @property
    def parts(self) -> dict[str, Array]:
        ...

class Tensors(Mapping[str, TensorMetadata]):
    """Tensor metadata in construction order, with engine-backed host and CUDA reads."""

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[str]:
        ...

    def __getitem__(self, key: str) -> TensorMetadata:
        ...

    def __contains__(self, key: object) -> bool:
        ...

    def keys(self) -> KeysView[str]:
        ...

    def values(self) -> ValuesView[TensorMetadata]:
        ...

    def items(self) -> ItemsView[str, TensorMetadata]:
        ...

    @overload
    def get(self, key: str, default: None=None) -> TensorMetadata | None:
        ...

    @overload
    def get(self, key: str, default: TensorMetadata) -> TensorMetadata:
        ...

    @overload
    def get(self, key: str, default: _T) -> TensorMetadata | _T:
        ...

    @overload
    def read(self, keys: str, *, device: str = "cpu", decode: bool = False) -> Array | Encoded:
        ...

    @overload
    def read(self, keys: Sequence[str] | None = None, *, device: str = "cpu", decode: bool = False) -> dict[str, Array | Encoded]:
        ...

    def read_into(self, key: str, out: object, *, role: str | None=None, decode: bool=False, offset: int=0) -> None:
        """Fill a writable contiguous Buffer or DLPack array from `offset`: a plain tensor's data,
        one stored part (`role`), or with `decode=True` an encoded tensor's logical bytes. An
        encoded tensor with neither raises `InvalidError` naming its roles."""

    def __repr__(self) -> str:
        ...

class ClassRef:
    """A class as `(library, version, name)`, for any implementation and never an import path:
    `ClassRef("diffusers", "0.35.1", "SD3Transformer2DModel")`. `ClassRef("", "", "")` is an
    unknown model; `version` is `""` when nothing records it. A component's tensors are that
    class's `state_dict` at that version. Frozen; iterates as the 3-tuple; equal and hashable by
    value. Writers take it or a `(library, version, name)` tuple."""

    def __init__(self, library: str, version: str, name: str) -> None: ...
    library: str
    version: str
    name: str

class Pipeline:
    """What `model_index.json` held."""

    @property
    def cls(self) -> ClassRef:
        ...

    @property
    def config(self) -> dict[str, Any]:
        ...

class Component:
    """One component of a checkpoint: class, config, tensors and assets."""

    @property
    def name(self) -> str:
        ...

    @property
    def cls(self) -> ClassRef:
        """`ClassRef(library, version, name)`; `ClassRef("", "", "")` when unknown."""

    @property
    def config(self) -> dict[str, Any]:
        ...

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

    def export_files(self, directory: str | PathLike[str] | None=None) -> Path:
        """Write configuration and supporting files, no tensor weights, to `directory`, or by
        default to a folder the cache manages and evicts like its other entries."""
        ...

    def __repr__(self) -> str:
        ...

class VerificationReport:
    @property
    def missing(self) -> list[tuple[str, int]]:
        """Absent local objects, as (sha256_hex, byte_length)."""

    @property
    def corrupt(self) -> list[tuple[str, int]]:
        """Local objects whose stored length or digest is wrong."""

    @property
    def ok(self) -> bool:
        """True when both collections are empty."""

class CheckpointFile(io.RawIOBase):
    """`Checkpoint.open_file()`: readable and seekable, not writable. It holds what it may read until
    closed, and fetches what it lacks when its checkpoint may fetch."""

    def readinto(self, buffer: Buffer) -> int: ...
    def seek(self, offset: int, whence: int = 0) -> int: ...
    def tell(self) -> int: ...
    def close(self) -> None: ...

class Checkpoint:
    """One cozytensors read from a store."""

    def close(self) -> None:
        """Release this handle's access and cached components; existing child handles remain usable."""

    def __enter__(self) -> Checkpoint:
        ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        ...

    def verify(self, components: Sequence[str] | None=None) -> VerificationReport:
        """Audit the header and selected local data; never fetch, repair or delete objects."""

    def open_file(self) -> CheckpointFile:
        """The checkpoint as one read-only, seekable file: its header, then its chunks and assets in header
        order at 4 KiB-aligned offsets, assembled from the store on demand; nothing is written."""

    @property
    def id(self) -> str:
        """`sha256:<hex>` of the header."""

    @property
    def tree(self) -> str | None:
        ...

    @property
    def variant(self) -> str | None:
        ...

    @property
    def repo_id(self) -> str | None:
        ...

    @property
    def revision(self) -> str | None:
        ...

    @property
    def pipeline(self) -> Pipeline | None:
        ...

    @property
    def components(self) -> dict[str, Component]:
        """In construction order."""

    @property
    def chunks(self) -> list[tuple[str, int]]:
        """The header's chunk objects in order: `(sha256, length)`."""

    @property
    def metadata(self) -> dict[str, Any]:
        ...

    def missing(self, components: Sequence[str] | None=None) -> list[tuple[str, int]]:
        """The chunks and assets of `components` (all when None) not here."""

    def download(self, components: Sequence[str] | None=None, *, priority: int=0, progress: Callable[[str, int, int], object] | None=None) -> None:
        ...

class Endpoint:

    def __init__(self, url: str, token: Token | None=None, *, ca_der: bytes | None=None, cache_dir: StrPath | None=None) -> None:
        ...

    def repo(self, repo_id: str) -> tuple[dict[str, object], str] | None:
        ...

    def update_repo(self, repo_id: str, change: Callable[[dict[str, object] | None], dict[str, object] | None]) -> dict[str, object]:
        ...

    def commit(self, repo_id: str, revision: str) -> tuple[str, str | None]:
        """A revision (tag, commit or variant header) to its commit id, and the tag it went through."""

class ComponentWriter:

    def put(self, key: str, data: Buffer | DLPack, *, dtype: str | None=None, shape: Sequence[int] | None=None) -> None:
        ...

    def put_encoded(self, key: str, *, dtype: str, shape: Sequence[int], scheme: str, params: Mapping[str, object], parts: Mapping[str, tuple[str, Sequence[int], Buffer]]) -> None:
        ...

    def copy(self, key: str, source: Component, *, as_key: str | None=None) -> None:
        ...

    def asset(self, name: str, data: Buffer) -> None:
        ...

    def order(self, keys: Sequence[str]) -> None:
        ...

    def has(self, key: str) -> bool:
        ...

class Writer:
    metadata: dict[str, object]
    local_checkpoint: Checkpoint | None

    def __init__(self, dest: StrPath | None=None, *, variant: str | None=None, base: Checkpoint | None=None, resume: str | None=None, repo_id: str | None=None, revision: str | None=None, endpoint: str | None=None, token: Token | None=None, cache_dir: StrPath | None=None) -> None:
        """Write an exact header-file path, or only the shared store when dest=None.

        The file is a handle into the shared local store that roots its data while it holds
        this header; `materialize` makes the portable copy. With `repo_id`, commit makes a commit
        and moves the private tag `revision` (default `main`) to it. `variant` names
        `model.<variant>.cozytensors` in a `repo_id` upload and is refused without `repo_id`.
        """

    def __enter__(self) -> Writer:
        ...

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None) -> None:
        ...

    def pipeline(self, cls: str, config: Mapping[str, object]) -> None:
        ...

    def component(self, name: str, *, cls: str, config: Mapping[str, object], config_file: str | None=None) -> ComponentWriter:
        ...

    def commit(self) -> Checkpoint:
        ...

    def abandon(self) -> None:
        ...

    def has(self, component: str, key: str) -> bool:
        ...
```

## `tensorfs.raw` — advanced physical inspection and resident decode

These declarations belong to `tensorfs.raw`, not the root package.
See [resident decoding](API.md#advanced-decode-resident-encoded-parts) for placement,
ownership and error behavior.

```python
from collections.abc import Buffer, Mapping, Sequence
from tensorfs import Checkpoint
# DLPack is the accepted input protocol described above, not a package export.

class Part:
    @property
    def dtype(self) -> str: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def chunk(self) -> int: ...
    @property
    def offset(self) -> int: ...
    @property
    def length(self) -> int: ...

def parts(checkpoint: Checkpoint, component: str, key: str) -> dict[str, Part]:
    """Inspect stored ranges without fetching bytes; plain tensors use role ''."""

def decode_into(
    scheme: str,
    parts: Mapping[str, tuple[str, Sequence[int], Buffer | DLPack]],
    out: Buffer | DLPack,
    *,
    dtype: str,
    shape: Sequence[int],
    params: Mapping[str, object] | None = None,
) -> None:
    """Synchronously decode already-resident parts into an exact logical destination.

    Part tuples describe format bytes; carrier dtype/shape may differ. The output
    must match logical dtype/shape, byte length and writable contiguous storage.
    Nonempty input/output overlap is refused; duplicate read-only inputs are allowed.
    All inputs/output are on Host, or on one CUDA device with a supported resident
    scheme. Metadata refusals precede writes; output is unspecified on value/runtime
    failure. Native owners remain held when CUDA completion cannot be proved.
    """
```
