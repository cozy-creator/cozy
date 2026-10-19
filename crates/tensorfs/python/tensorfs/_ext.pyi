"""`tensorfs._ext`, checked against the built extension by `scripts/stub_conform.py`."""

from collections.abc import Buffer, Mapping, Sequence
from os import PathLike
from types import ModuleType
from typing import Any, Protocol

from .records import Header as HeaderRecord
from .records import ReadItem

__version__: str
STORE_FORMAT: int
CLAIM_PROTOCOL: int

#: `tensorfs._ext.varena`, typed in full by `varena.pyi` (the `tensorfs.varena` facade).
varena: ModuleType

StrPath = str | PathLike[str]
#: `org/name[@release]`, `sha256:<hex>` of a tree or a header, or a stored `Checkpoint`.
Ref = str | Checkpoint

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
    """A scheme and its complete params; str() is `scheme {key: value, ...}`."""

    @property
    def scheme(self) -> str: ...
    @property
    def params(self) -> dict[str, Any]: ...
    def __str__(self) -> str: ...

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

class Pipeline:
    """What `model_index.json` held."""

    @property
    def cls(self) -> str:
        """`diffusers.<Class>`, `""` when unknown."""
    @property
    def config(self) -> dict[str, Any]: ...

class Component:
    """One component of a checkpoint: class, config, tensors and assets."""

    @property
    def name(self) -> str: ...
    @property
    def cls(self) -> str:
        """`<library>.<Class>`, `""` when unknown."""
    @property
    def config(self) -> dict[str, Any]: ...
    @property
    def config_file(self) -> str:
        """The rendered config's file name (`config.json` by default)."""
    @property
    def tensors(self) -> dict[str, Info]:
        """`key -> Info`, in construction order."""
    @property
    def assets(self) -> dict[str, tuple[str, int]]:
        """`name -> (sha256, length)`: run data in the component's folder."""
    def asset(self, name: str) -> bytes:
        """An asset's bytes; a parsed header raises `UnsupportedError`."""
    @property
    def path(self) -> str: ...

class Checkpoint:
    """One cozytensors read from a store, or only its header (`parse`)."""

    @staticmethod
    def parse(header: bytes) -> Checkpoint:
        """A `cozytensors/1` header's pipeline, components and metadata, with no store: `read`,
        `missing`, `chunk_paths` and `Component.asset` refuse (`UnsupportedError`)."""
    @property
    def id(self) -> str | None:
        """`sha256:<hex>` of the header; None for a parsed header."""
    @property
    def header_ref(self) -> tuple[str, int] | None: ...
    @property
    def tree_ref(self) -> tuple[str, int] | None: ...
    @property
    def tree(self) -> str | None: ...
    @property
    def name(self) -> str | None: ...
    @property
    def release(self) -> str | None: ...
    @property
    def variant(self) -> str | None: ...
    @property
    def repo_id(self) -> str | None: ...
    @property
    def revision(self) -> str | None: ...
    @property
    def path(self) -> str: ...
    @property
    def pipeline(self) -> Pipeline | None: ...
    @property
    def components(self) -> dict[str, Component]:
        """In construction order."""
    @property
    def chunks(self) -> list[tuple[str, int]]:
        """The header's chunk objects in order: `(sha256, length)` (`Part.chunk` indexes them)."""
    @property
    def chunk_paths(self) -> list[str]:
        """Each chunk's file in the store, in header order; a parsed header raises."""
    @property
    def tensors(self) -> dict[str, dict[str, Info]]:
        """`tensors[component][key]`."""
    @property
    def metadata(self) -> dict[str, Any]: ...
    def missing(self, components: Sequence[str] | None = None) -> list[tuple[str, int]]:
        """The chunks and assets of `components` (all when None) not here."""
    def fetch(self, components: Sequence[str] | None = None, *, priority: int = 0, progress: Any = None) -> None: ...
    def read(self, component: str, key: str) -> Array:
        """Logical bytes; an encoded tensor goes through the reference decode."""

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
    def copy(
        self, key: str, source: Checkpoint, from_component: str | None = None, from_key: str | None = None
    ) -> None:
        """`from_component/from_key` (default: this component, `key`) of a stored checkpoint."""
    def asset(self, name: str, data: Buffer) -> None: ...
    def order(self, keys: Sequence[str]) -> None: ...
    def has(self, key: str) -> bool: ...

class Writer:
    def __init__(self, dest: StrPath | None = None, *, variant: str | None = None,
                 base: Checkpoint | None = None, resume: str | None = None,
                 cache_dir: StrPath | None = None, repo_id: str | None = None,
                 endpoint: str | None = None, token: Any = None, ca_der: bytes | None = None) -> None: ...
    def pipeline(self, cls: str, config: Mapping[str, Any]) -> None: ...
    def component(
        self, name: str, cls: str = "", config: Mapping[str, Any] | None = None, config_file: str | None = None
    ) -> ComponentWriter:
        """Declares a component (again: class and config change, tensors stay)."""
    def repo_file(self, path: str, data: Buffer) -> None: ...
    def set_metadata(self, key: str, value: Any) -> None: ...
    def has(self, component: str, key: str) -> bool: ...
    def commit(self) -> Checkpoint: ...
    def abandon(self) -> None: ...

class Store:
    def __init__(self, root: StrPath) -> None:
        """The store at `root` (`~` expands), created when absent."""
    @property
    def root(self) -> str: ...
    def open_path(self, path: StrPath, variant: str | None = None) -> Checkpoint: ...
    def open_remote(self, repo_id: str, revision: str | None = None, variant: str | None = None,
                    endpoint: str = "https://tensorhub.com", token: Any = None, ca_der: bytes | None = None) -> Checkpoint: ...
    def upload_folder(self, folder_path: StrPath, repo_id: str, revision: str | None = None,
                      endpoint: str = "https://tensorhub.com", token: Any = None, ca_der: bytes | None = None) -> str: ...
    def upload_local(self, folder_path: StrPath, repo_id: str, revision: str | None = None) -> str: ...
    def snapshot(self, ck: Checkpoint, dest: StrPath, variants: Sequence[str] | None = None) -> str: ...
    def snapshot_download(self, repo_id: str, *, revision: str | None = None,
                          variant: str | None = None, components: Sequence[str] | None = None,
                          local_dir: StrPath | None = None, local_files_only: bool = False,
                          endpoint: str = "https://tensorhub.com", token: Any = None, ca_der: bytes | None = None) -> str: ...
    def hub_download(self, repo_id: str, filename: str, *, revision: str | None = None,
                     local_dir: StrPath | None = None, local_files_only: bool = False,
                     endpoint: str = "https://tensorhub.com", token: Any = None, ca_der: bytes | None = None) -> str: ...
    def convert(self, source: str | Sequence[str], *, converter: str | None = None,
                variant: str | None = None, token: Any = None, progress: Any = None) -> Checkpoint: ...
    def resolve_source(self, source: str, token: Any = None) -> str: ...
    def import_(
        self,
        src: str | Sequence[str],
        name: str | None = None,
        converter: str | None = None,
        variant: str | None = None,
    ) -> Checkpoint:
        """A file, directory, `hf://`, `civitai://` or URL, named `name` (`local/<its name>`)."""
    def pull(
        self,
        ref: str,
        variant: str | None = None,
        origin: str | None = None,
        hub: str | None = None,
        token: str | None = None,
    ) -> Checkpoint:
        """`org/name[@release]` and its `variant` from Tensorhub (`hub`, else tensorhub.com;
        `token` reads private trees), or from a static `origin`'s URL or directory."""
    def checkpoint(self, ref: Ref, variant: str | None = None) -> Checkpoint: ...
    def name(self, repo: str, version: str, ck: Checkpoint, variant: str | None = None) -> Checkpoint:
        """`ck` as `variant` (its main dtype by default) of `repo`'s release `version` here."""
    def writer(self, base: Ref | None = None, key: str | None = None) -> Writer: ...
    def export(
        self,
        ref: Ref,
        dir: StrPath,
        safetensors: bool = True,
        dequantize: str | None = None,
        *,
        origin: bool = False,
        variant: str | None = None,
    ) -> list[str]:
        """A Diffusers/Transformers folder, or with `origin=True` the static origin layout."""
    def gc(self, older_than: float | None = None) -> dict[str, Any]: ...

class DPoP:
    def issued_token(self) -> str | None: ...
    @staticmethod
    def grant(key: str, *, token_endpoint: str, client_id: str, resource: str,
              claims: Mapping[str, Any], endpoint: str | None = None, ca_der: bytes | None = None) -> DPoP: ...
    @staticmethod
    def token(key: str, access_token: str, resource: str | None = None, *, endpoint: str | None = None) -> DPoP: ...

class TokenProvider(Protocol):
    def headers(self, method: str, url: str, headers: Mapping[str, str], payload: str | None) -> Mapping[str, str]: ...
    def retry(self, method: str, url: str, headers: Mapping[str, str], payload: str | None,
              status: int, response_headers: Mapping[str, str]) -> bool: ...

class Endpoint:
    def __init__(self, url: str, token: Any = None, *, ca_der: bytes | None = None) -> None: ...
    def repo(self, repo_id: str) -> tuple[dict[str, Any], str] | None: ...
    def update_repo(self, repo_id: str, change: Any) -> dict[str, Any]: ...

def regions(
    ck: Checkpoint,
    max_bytes: int = ...,
    boundaries: Sequence[str] | None = None,
    only: Sequence[str] | None = None,
) -> list[list[str]]:
    """`<component>/<key>` regions: chunk-sharing tensors together, up to `max_bytes` each, never
    across a `<component>/<key prefix>` boundary (a block) unless a shared chunk does. `only`
    (blocks or tensors, in the same form) limits the work and the result to those tensors and
    the ones sharing a chunk with them."""

# ------------------------------------------------------------------------- the CBOR header

class Closure:
    """Encoding specs a CBOR header's tensors may cite, by digest."""

    def __init__(self) -> None: ...
    def insert(self, data: bytes) -> str: ...
    def digests(self) -> list[str]: ...
    def __len__(self) -> int: ...

def parse_header(data: bytes, closure: Closure | None = None) -> HeaderRecord:
    """A CBOR header's plain-data projection; failures are `tensorfs.Error` families."""

class ReadPlan:
    """A CBOR header's read plan, what `tensorfs.varena` registers from until it
    registers checkpoints."""

    def select(self, whats: Sequence[str]) -> ReadPlan: ...
    @property
    def bytes(self) -> int: ...
    @property
    def order(self) -> str: ...
    def items(self) -> list[ReadItem]: ...
    def __len__(self) -> int: ...

def plan(
    header: bytes,
    traversal: Sequence[tuple[str, str]],
    components: Sequence[str] | None = None,
    window: int = ...,
) -> ReadPlan: ...
