# TensorFS developer API reference

This document lists the **planned public Rust and Python API for TensorFS 0.13.0**.
The implementation is under integration. It is separate from the technical
[cozytensors format specification](spec/cozytensors.md).
Start with the [usage guide](README.md) for examples.

- [Python: functions, objects, properties and errors](#python)
- [Rust: client, checkpoints, writers and endpoints](#rust)
- [Behavior and ownership](#behavior-and-ownership)
- [Rust details still being finalized](#rust-details-still-being-finalized)

## Python

Install one package and use `import tensorfs`. The eight top-level functions below are
followed by every public object, its attributes/properties, methods and error types.
Signatures include the context-manager, mapping and DLPack protocols where supported.
`...` denotes an implementation body omitted from this reference.

`TokenProvider` and `DLPack` describe accepted protocols; they do not require inheritance.
`Buffer` means a Python buffer-protocol value. DLPack inputs support framework arrays
without importing that framework into TensorFS.

```python
from __future__ import annotations
from collections.abc import Buffer, Callable, ItemsView, Iterator, KeysView, Mapping, Sequence, ValuesView
from os import PathLike
from types import TracebackType
from typing import Any, Protocol, TypeVar, overload

StrPath = str | PathLike[str]
_T = TypeVar("_T")
Token = str | DPoP | TokenProvider | Callable[[str, str, Mapping[str, str], str | None], Mapping[str, str]]
__version__: str


class TokenProvider(Protocol):

    def headers(self, method: str, url: str, headers: Mapping[str, str], payload: str | None) -> Mapping[str, str]:
        ...

    def retry(self, method: str, url: str, headers: Mapping[str, str], payload: str | None, status: int, response_headers: Mapping[str, str]) -> bool:
        ...

class DLPack(Protocol):

    def __dlpack__(self) -> object:
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

class Part:
    """One stored part: a plain tensor's data (role `""`) or an encoded tensor's part."""

    @property
    def dtype(self) -> str:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def chunk(self) -> int:
        ...

    @property
    def offset(self) -> int:
        ...

    @property
    def length(self) -> int:
        ...

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

class Info:
    """A tensor's logical dtype and shape, its encoding (None when plain), its parts."""

    @property
    def dtype(self) -> str:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def encoding(self) -> Encoding | None:
        ...

    @property
    def parts(self) -> dict[str, Part]:
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

class Tensors(Mapping[str, Info]):
    """Tensor metadata in construction order, with engine-backed host and CUDA reads."""

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[str]:
        ...

    def __getitem__(self, key: str) -> Info:
        ...

    def __contains__(self, key: object) -> bool:
        ...

    def keys(self) -> KeysView[str]:
        ...

    def values(self) -> ValuesView[Info]:
        ...

    def items(self) -> ItemsView[str, Info]:
        ...

    @overload
    def get(self, key: str, default: None=None) -> Info | None:
        ...

    @overload
    def get(self, key: str, default: Info) -> Info:
        ...

    @overload
    def get(self, key: str, default: _T) -> Info | _T:
        ...

    def read(self, key: str, *, decode: bool=True) -> Array | Encoded:
        ...

    def read_all(self, keys: Sequence[str] | None=None, *, device: str='cpu', decode: bool=True) -> dict[str, Array | Encoded]:
        ...

    def read_into(self, key: str, out: object, *, role: str | None=None, offset: int=0) -> None:
        """Fill a writable contiguous Buffer or DLPack array; a role selects stored part bytes."""

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
    def config_file(self) -> str:
        """The rendered config's file name (`config.json` by default)."""

    @property
    def tensors(self) -> Tensors:
        """`key -> Info`, in construction order."""

    @property
    def assets(self) -> dict[str, tuple[str, int]]:
        """`name -> (sha256, length)`: run data in the component's folder."""

    def asset(self, name: str) -> bytes:
        """An asset's bytes; a parsed header raises `UnsupportedError`."""

    @property
    def path(self) -> str:
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
    def path(self) -> str:
        ...

    @property
    def pipeline(self) -> Pipeline | None:
        ...

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

    def component(self, name: str, *, cls: str, config: Mapping[str, object], config_file: str='config.json') -> ComponentWriter:
        ...

    def commit(self) -> Checkpoint:
        ...

    def abandon(self) -> None:
        ...

    def has(self, component: str, key: str) -> bool:
        ...
```

The companion `tensorfs.varena` namespace is a placement/memory-management interface;
it is not needed for ordinary TensorFS use or Diffusers/ComfyUI loading. Its implementation
and type declarations are separate from the checkpoint library surface above.

## Rust

Use the Rust crate `tensorfs`. `Client` owns configuration and a session; checkpoints,
components and arrays retain the access and byte lifetimes they need. Network operations
are async on the caller's runtime. Rust reads explicitly follow `Checkpoint::fetch`;
Python performs that fetch behind its synchronous read methods.

The following is the approved planned interface, with the request-aware provider contract
already implemented by the transport. Public `Client` integration is still in progress.
The remaining option/writer details are listed explicitly below rather than implied to be
implemented by a local-only wrapper.

```rust
pub struct Ref { pub repo_id: String, pub revision: Option<String>, pub variant: Option<String>, pub endpoint: Option<String> }
impl FromStr for Ref;                                        // the shorthand
pub struct Config {                                          // locations and credentials only
    pub endpoint: String, pub cache_dir: PathBuf, pub mirror: Option<PathBuf>,   // mirror: a datacenter volume
    pub token: Option<Arc<dyn TokenProvider>>,               // a String converts into a bearer provider
}
impl Config { pub fn from_env() -> Config }
pub struct Request<'a> {
    pub method: &'a str,
    pub url: &'a str,
    pub headers: &'a [(String, String)],
    pub payload: Option<&'a str>, // SHA-256 of the request body, when known
}
#[async_trait] pub trait TokenProvider: Send + Sync {
    async fn headers(&self, request: &Request<'_>) -> Result<Vec<(String, String)>>;
    fn retry(&self, sent: &Request<'_>, status: u16, headers: &HeaderMap) -> Result<bool> { Ok(false) }
}
pub struct Client;                                           // a session (as hf-hub's Api), not a cache handle
impl Client {
    pub fn new(config: Config) -> Result<Client>;
    pub async fn open(&self, name_or_path: &str, opts: &Open) -> Result<Checkpoint>;   // Open { revision, variant, endpoint, local_files_only }
    pub async fn snapshot_download(&self, r: &Ref, want: &Fetch<'_>) -> Result<PathBuf>;
    pub async fn hub_download(&self, r: &Ref, filename: &str) -> Result<PathBuf>;
    pub async fn upload_folder(&self, r: &Ref, folder: &Path, progress: Option<&Progress>) -> Result<ObjectRef>;
    pub async fn convert(&self, sources: &[String], opts: &Convert<'_>) -> Result<Checkpoint>;
    pub async fn resolve_source(&self, source: &str) -> Result<String>;
    pub fn writer(&self, dest: Option<&Path>, base: Option<&Checkpoint>, resume: Option<&str>) -> Result<Writer>;
}
pub fn export(ck: &Checkpoint, dir: &Path, format: Format, dequantize: Option<Dtype>) -> Result<Vec<String>>;
impl Checkpoint {   // holds its bytes against eviction until dropped
    pub fn id(&self) -> &ObjectRef;  pub fn named(&self) -> Option<&Ref>;
    pub fn pipeline(&self) -> Option<&Pipeline>;
    pub fn components(&self) -> &[Component];   pub fn component(&self, name: &str) -> Result<&Component>;
    pub fn chunk(&self, index: usize) -> Result<File>;
    pub fn missing(&self, components: Option<&[String]>) -> Vec<ObjectRef>;
    pub async fn fetch(&self, want: &Fetch<'_>) -> Result<Fetched>;             // Fetch { components, priority, progress }
    pub fn folder(&self) -> Result<PathBuf>;
}
impl Component {    // name, class, config (format::Value), tensors, assets
    pub fn read(&self, key: &str) -> Result<Vec<u8>>;                           // decoded
    pub fn read_part_into(&self, key: &str, role: &str, offset: u64, buf: &mut [u8]) -> Result<()>;
    pub fn asset(&self, name: &str) -> Result<Vec<u8>>;
    pub fn folder(&self) -> Result<PathBuf>;
}
impl Writer {   // folder(dest, variant), pipeline(cls, config), component(name, cls, config) -> &mut ComponentWriter,
                // set_metadata, commit(self) -> Checkpoint, abandon; ComponentWriter: put/Data, copy_as, asset, set_order, has
}
pub struct Endpoint;   // the xs-054 client, public for tools such as the CLI's release verbs
impl Endpoint {
    pub fn new(url: &str, token: Option<Arc<dyn TokenProvider>>) -> Result<Endpoint>;
    pub async fn repo(&self, repo_id: &str) -> Result<Option<(Repo, String)>>;  // (doc, ETag)
    pub async fn update_repo(&self, repo_id: &str, f: impl FnMut(Option<Repo>) -> Result<Option<Repo>>) -> Result<Repo>;
}   // Endpoint::new("local", …) is this machine's local/ repos; a None document removes one
```

### Supporting Rust types

| Type | Planned contents / role |
|---|---|
| `Open` | Optional revision, variant and endpoint; `local_files_only`. Default options use configured endpoint/cache and normal online resolution. |
| `Fetch<'a>` | Optional component selection, priority and progress callback. |
| `Convert<'a>` | Optional converter, variant, credentials and progress callback. |
| `Progress` | Progress reporting for source resolution, fetching, conversion or publishing. |
| `ObjectRef` | Immutable SHA-256 plus byte length. A header and a repository tree have distinct references. |
| `Pipeline` | Class and configuration of the whole pipeline. |
| `Component` | Name, class, configuration, tensors and assets; holds its checkpoint's access. |
| `ComponentWriter` | Tensor/encoded-part writes, component reuse, assets, tensor order and presence checks. |
| `Dtype`, `Encoding`, `Value`, `Format` | Tensor dtype, encoding description, structured configuration/metadata, and export format. |
| `Result<T>` / error | Typed storage, format, integrity, encoding, network, authorization and conflict failures. |

### Rust details still being finalized

The method inventory above is the approved plan. These signatures must be completed by
the public-client integration before this document can claim the Rust interface is final:

- Exact `Open`, `Fetch`, `Convert` and progress/error type declarations and defaults.
- Public writer/component-writer argument and result types, including variant/configuration,
  plain and encoded input buffers, reuse, metadata and assets.
- Streaming writer finalization: local commit, async remote finish, the exact published tree,
  cancellation and retained local output after upload failure.
- Advanced device/read-into and encoded-part Rust interfaces over the existing I/O engine.

These are tracked API-design items, not additional public functions silently omitted here.
The Rust owner is resolving them against the same engine and behaviors used by Python.

## Behavior and ownership

- `open` reads metadata first. Python tensor reads fetch only required objects; explicit
  `fetch` can prefetch chosen components. Offline mode performs no network request.
- `decode=True` returns logical values. `decode=False` returns the declared encoding and
  stored parts. Device selection and `read_into` use the same TensorFS byte engine.
- Checkpoints and child handles keep their required bytes alive. Closing a parent does
  not invalidate an already returned child/array/consumer. Raw descriptors returned by
  `open_chunk` are owned by the caller, who closes them.
- A writer context commits on normal exit. On failure it abandons unless a resume key
  requests retained state. `local_checkpoint` preserves an already committed output when
  publishing fails; that error remains an error.
- A successful publish identifies the exact committed tree. `revision=None` appends an
  unreleased checkpoint; it does not create a public release.
- Tokens may be supplied directly or by provider. Public reads need no account. Providers
  are consulted per request; credentials do not follow a redirect to another origin.
- The cache root and endpoint can be configured. Ordinary callers do not manage cache
  records, chunk indexes, SQLite tables or a TensorFS daemon.

The [distribution specification](spec/distribution.md) describes endpoint wire operations;
[cozytensors.md](spec/cozytensors.md) describes stored bytes. Neither is the developer API.
