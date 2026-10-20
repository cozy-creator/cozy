# Python API declarations for review

Temporary review copy of the planned TensorFS 0.13 declarations. Implementation status is
separate; this is not a claim that every declaration is available on current master.
For normal development, use the [Python usage guide](API.md).

`TensorMetadata` describes the logical tensor. Physical storage descriptors are internal;
encoded reads expose named `Array` values without chunk locations or byte offsets.

The provider/DLPack protocols describe accepted inputs and are not additional package exports.

```python
from __future__ import annotations
from collections.abc import Buffer, Callable, ItemsView, Iterator, KeysView, Mapping, Sequence, ValuesView
from os import PathLike
from types import TracebackType
from typing import Any, Protocol, TypeVar, overload
StrPath = str | PathLike[str]
_T = TypeVar('_T')
Token = str | DPoP | TokenProvider | Callable[[str, str, Mapping[str, str], str | None], Mapping[str, str]]
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

def configure(*, endpoint: str | None=None, cache_dir: StrPath | None=None, token: Token | None=None) -> None:
    ...

def convert(source: str | Sequence[str], *, converter: str | None=None, variant: str | None=None, cache_dir: StrPath | None=None, token: Token | None=None, progress: Callable[[str, int, int], object] | None=None) -> Checkpoint:
    ...

def export(src: Checkpoint, dir: StrPath, *, format: str='safetensors', dequantize: str | None=None) -> list[str]:
    ...

def hub_download(repo_id: str, filename: str, *, revision: str | None=None, cache_dir: StrPath | None=None, local_dir: StrPath | None=None, local_files_only: bool=False, token: Token | None=None, endpoint: str | None=None, ca_der: bytes | None=None) -> str:
    ...

def resolve_source(source: str, *, token: Token | None=None) -> str:
    ...

def open(name_or_path: StrPath, *, revision: str | None=None, variant: str | None=None, endpoint: str | None=None, local_files_only: bool=False, cache_dir: StrPath | None=None, token: Token | None=None, ca_der: bytes | None=None) -> Checkpoint:
    ...

def snapshot_download(repo_id: str, *, revision: str | None=None, variant: str | None=None, components: Sequence[str] | None=None, cache_dir: StrPath | None=None, local_dir: StrPath | None=None, local_files_only: bool=False, token: Token | None=None, endpoint: str | None=None, ca_der: bytes | None=None) -> str:
    ...

def upload_folder(*, repo_id: str, folder_path: StrPath, revision: str | None=None, token: Token | None=None, endpoint: str | None=None, cache_dir: StrPath | None=None, ca_der: bytes | None=None) -> str:
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
    """A tensor's stored scheme, params and raw part arrays (`decode=False`)."""

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
    def read(self, keys: str, *, device: str = "cpu", decode: bool = True) -> Array | Encoded:
        ...

    @overload
    def read(self, keys: Sequence[str] | None = None, *, device: str = "cpu", decode: bool = True) -> dict[str, Array | Encoded]:
        ...

    def read_into(self, key: str, out: object, *, role: str | None=None, offset: int=0) -> None:
        """Fill a writable contiguous Buffer or DLPack array; a role selects stored part bytes."""

    def __repr__(self) -> str:
        ...

class Pipeline:
    """What `model_index.json` held."""

    @property
    def cls(self) -> str:
        """`diffusers.<Class>`, `""` when unknown."""

    @property
    def config(self) -> dict[str, Any]:
        ...

class Component:
    """One component of a checkpoint: class, config, tensors and assets."""

    @property
    def name(self) -> str:
        ...

    @property
    def cls(self) -> str:
        """`<library>.<Class>`, `""` when unknown."""

    @property
    def config(self) -> dict[str, Any]:
        ...

    @property
    def tensors(self) -> Tensors:
        """`key -> TensorMetadata`, in construction order."""

    @property
    def assets(self) -> dict[str, tuple[str, int]]:
        """`name -> (sha256, length)`: run data in the component's folder."""

    def asset(self, name: str) -> bytes:
        """An asset's bytes; a parsed header raises `UnsupportedError`."""

    def export_files(self, directory: str | PathLike[str]) -> str:
        """Write configuration and supporting files for a file-based consumer; no tensor weights."""
        ...

    def __repr__(self) -> str:
        ...

class Checkpoint:
    """One cozytensors read from a store, or only its header (`parse`)."""

    def close(self) -> None:
        """Release this handle's access and cached components; existing child handles remain usable."""

    def __enter__(self) -> Checkpoint:
        ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        ...

    def verify(self) -> list[tuple[str, int]]:
        """Hash locally available header, chunks and assets; report missing or corrupt objects."""

    def open_chunk(self, index: int) -> int:
        """A new read-only OS descriptor owned and closed by the caller."""

    @staticmethod
    def parse(header: bytes) -> Checkpoint:
        """A `cozytensors/1` header's pipeline, components and metadata, with no store: component reads,
        `missing`, `chunk_paths` and `Component.asset` refuse (`UnsupportedError`)."""

    @property
    def id(self) -> str | None:
        """`sha256:<hex>` of the header; None for a parsed header."""

    @property
    def header_ref(self) -> tuple[str, int] | None:
        ...

    @property
    def tree_ref(self) -> tuple[str, int] | None:
        ...

    @property
    def tree(self) -> str | None:
        ...

    @property
    def name(self) -> str | None:
        ...

    @property
    def release(self) -> str | None:
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
    def chunk_paths(self) -> list[str]:
        """Each chunk's file in the store, in header order; a parsed header raises."""

    @property
    def metadata(self) -> dict[str, Any]:
        ...

    def missing(self, components: Sequence[str] | None=None) -> list[tuple[str, int]]:
        """The chunks and assets of `components` (all when None) not here."""

    def fetch(self, components: Sequence[str] | None=None, *, priority: int=0, progress: Callable[[str, int, int], object] | None=None) -> None:
        ...

class DPoP:

    def issued_token(self) -> str | None:
        ...

    @staticmethod
    def grant(key: str, *, token_endpoint: str, client_id: str, resource: str, claims: Mapping[str, Any], endpoint: str | None=None, ca_der: bytes | None=None) -> DPoP:
        ...

    @staticmethod
    def token(key: str, access_token: str, resource: str | None=None, *, endpoint: str | None=None) -> DPoP:
        ...

class Endpoint:

    def __init__(self, url: str, token: Token | None=None, *, ca_der: bytes | None=None, cache_dir: StrPath | None=None) -> None:
        ...

    def repo(self, repo_id: str) -> tuple[dict[str, object], str] | None:
        ...

    def update_repo(self, repo_id: str, change: Callable[[dict[str, object] | None], dict[str, object] | None]) -> dict[str, object]:
        ...

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

    def __init__(self, dest: StrPath | None=None, *, variant: str | None=None, base: Checkpoint | None=None, resume: str | None=None, repo_id: str | None=None, endpoint: str | None=None, token: Token | None=None, cache_dir: StrPath | None=None, ca_der: bytes | None=None) -> None:
        ...

    def __enter__(self) -> Writer:
        ...

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None) -> None:
        ...

    def pipeline(self, cls: str, config: Mapping[str, object]) -> None:
        ...

    def component(self, name: str, *, cls: str, config: Mapping[str, object]) -> ComponentWriter:
        ...

    def commit(self) -> Checkpoint:
        ...

    def abandon(self) -> None:
        ...

    def has(self, component: str, key: str) -> bool:
        ...
```
