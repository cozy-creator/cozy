"""TensorFS v2: `tensorfs._ext.v2`, checked against the built extension by `scripts/stub_conform.py`."""

from collections.abc import Buffer, Mapping, Sequence
from os import PathLike
from typing import Any

StrPath = str | PathLike[str]
#: `org/name[@release]`, `sha256:<hex>`, or a `Manifest`.
Ref = str | Manifest

class Error(Exception):
    """A TensorFS error. `.code` names its family; `.detail` the cause."""

    code: str
    detail: str

class InvalidError(Error):
    """A document, argument or tensor breaks the format or a grammar (`invalid`)."""

class FormatError(Error):
    """Another document kind or major version (`format`)."""

class NotFoundError(Error):
    """No such ref, release, lane, tensor, file or object (`not_found`)."""

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

class UnsupportedError(Error):
    """A source or target this build cannot represent (`unsupported`)."""

class Manifest:
    """A checkpoint's manifest object, and the ref, release and lane naming it when one does."""

    @property
    def sha256(self) -> str: ...
    @property
    def length(self) -> int: ...
    @property
    def name(self) -> str | None: ...
    @property
    def release(self) -> str | None: ...
    @property
    def lane(self) -> str | None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class Part:
    """One stored part: a plain tensor's data (role `""`) or an encoded tensor's part."""

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

class Encoding:
    @property
    def scheme(self) -> str: ...
    @property
    def params(self) -> dict[str, Any]: ...

class Info:
    """A tensor's logical dtype and shape, its encoding (None when plain), its parts."""

    @property
    def dtype(self) -> str: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def encoding(self) -> Encoding | None: ...
    @property
    def parts(self) -> dict[str, Part]: ...

class Array:
    """A tensor's bytes on the CPU; `torch.from_dlpack` and `numpy.from_dlpack` view them."""

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

class Checkpoint:
    @property
    def manifest(self) -> Manifest: ...
    @property
    def tensors(self) -> dict[str, dict[str, Info]]:
        """`tensors[component][key]`."""
    @property
    def files(self) -> list[str]: ...
    @property
    def metadata(self) -> dict[str, Any]: ...
    def file(self, path: str) -> bytes: ...
    def read(self, component: str, key: str) -> Array:
        """Logical bytes; an encoded tensor goes through the reference decode."""

class Writer:
    def put(self, component: str, key: str, dtype: str, shape: Sequence[int], data: Buffer) -> None: ...
    def put_encoded(
        self,
        component: str,
        key: str,
        dtype: str,
        shape: Sequence[int],
        scheme: str,
        parts: Mapping[str, tuple[str, Sequence[int], Buffer]],
        params: Mapping[str, Any] | None = None,
    ) -> None: ...
    def copy(self, component: str, key: str, source: Checkpoint) -> None: ...
    def file(self, path: str, data: Buffer) -> None: ...
    def has(self, component: str, key: str) -> bool: ...
    def commit(self) -> Manifest: ...
    def abandon(self) -> None: ...

class Store:
    def __init__(self, root: StrPath) -> None:
        """The store at `root` (`~` expands), created when absent."""
    @property
    def root(self) -> str: ...
    def import_(
        self,
        src: str | Sequence[str],
        name: str | None = None,
        converter: str | None = None,
        variant: str | None = None,
    ) -> Manifest:
        """A file, directory, `hf://`, `civitai://` or URL, refd as `name` (`local/<its name>`)."""
    def pull(self, origin: str, ref: str, lane: str | None = None) -> Manifest:
        """`org/name[@release]` from a static origin's URL or directory."""
    def checkpoint(self, ref: Ref, lane: str | None = None) -> Checkpoint: ...
    def writer(self, base: Ref | None = None, key: str | None = None) -> Writer: ...
    def export(
        self,
        ref: Ref,
        dir: StrPath,
        safetensors: bool = True,
        dequantize: str | None = None,
        *,
        origin: bool = False,
        lane: str | None = None,
    ) -> list[str]:
        """A Diffusers/Transformers folder, or with `origin=True` the static origin layout."""
    def gc(self, older_than: float | None = None) -> dict[str, Any]: ...

def regions(ck: Checkpoint, max_bytes: int = ...) -> list[list[str]]:
    """`<component>/<key>` regions: chunk-sharing tensors together, up to `max_bytes` each."""
