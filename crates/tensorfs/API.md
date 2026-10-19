# TensorFS developer API reference

This document lists the **planned public Rust and Python API for TensorFS 0.13.0**.
The implementation is under integration. It is separate from the technical
[cozytensors format specification](spec/cozytensors.md).
Start with the [usage guide](README.md) for examples.

- [Python: functions, objects, properties and errors](#python)
- [Rust: client, checkpoints, writers and endpoints](#rust)
- [Behavior and ownership](#behavior-and-ownership)

## Python

Install one package and use `import tensorfs`. The eight top-level functions below are
followed by every public object, its attributes/properties, methods and error types.
Signatures include the context-manager, mapping and DLPack protocols where supported.
`...` denotes an implementation body omitted from this reference.

`TokenProvider`, `RetryingTokenProvider` and `DLPack` below are protocol sketches for
accepted objects, not additional package exports. They do not require inheritance.
Only `headers` is required on a token provider; `retry` is optional. A CUDA DLPack
producer must support the `stream=` argument; a CPU producer may expose no-argument
`__dlpack__`. Ordinary inherited `object`/`Exception` methods are not repeated.
`Buffer` means a Python buffer-protocol value. DLPack inputs support framework arrays
without importing that framework into TensorFS.

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

    def __repr__(self) -> str:
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

The following is the concrete planned public interface. Its implementation is being integrated;
these declarations are an API contract, not a claim that the current master has implemented them.
Use the `tensorfs` Rust crate. Network operations are async on your runtime, followed by synchronous
reads of fetched data. There is no hidden global Tokio runtime in the core client.

### Client, selection and checkpoint objects

Names from the standard library (`Path`, `PathBuf`, `File`, `Read`, `Range`, `Arc`,
`BTreeMap`, `FromStr`) retain their normal Rust meanings. `HeaderMap` is the HTTP response-header map.
`Metadata` is `BTreeMap<String, Value>`; `format::Config` is structured JSON configuration.

```rust
pub type Result<T> = std::result::Result<T, Failure>;
pub struct Failure { pub kind: Kind, pub detail: String }
pub enum Kind {
    Invalid, Format, NotFound, Integrity, UnknownScheme, Decode,
    Io, Full, Network, Denied, Unsupported, Conflict,
}
impl Kind { pub fn code(self) -> &'static str; }

#[derive(Clone)]
pub struct Config {
    pub endpoint: String,
    pub cache_dir: PathBuf,
    pub mirror: Option<PathBuf>,
    pub token: Option<Arc<dyn TokenProvider>>,
    pub ca_der: Option<Vec<u8>>, // DER CA bytes, not base64 on the Rust API
}
impl Config { pub fn from_env() -> Config; }
// Debug must redact provider/CA contents.

pub struct Ref {
    pub repo_id: String,
    pub revision: Option<String>,
    pub variant: Option<String>,
    pub endpoint: Option<String>,
}
impl FromStr for Ref {
    type Err = Failure;
    fn from_str(value: &str) -> Result<Ref>;
}
// <org>/<name>[@revision[/variant]], and sha256:<64 lowercase hex>.
// Client.open checks an existing folder/.cozytensors path before parsing a Ref.

#[derive(Default)]
pub struct Open {
    pub revision: Option<String>,
    pub variant: Option<String>,
    pub endpoint: Option<String>,
    pub local_files_only: bool,
}
pub type Progress<'a> = dyn Fn(u64, u64) -> Result<()> + Sync + 'a;
pub type ConvertProgress<'a> = dyn Fn(&str, u64, u64) -> Result<()> + Sync + 'a;

#[derive(Clone, Debug, Default)]
pub enum Selection {
    #[default] All,
    Components(Vec<String>),
    Tensors { component: String, keys: Vec<String> },
    Part { component: String, key: String, role: Option<String>, range: Option<Range<u64>> },
    Assets { component: String, names: Option<Vec<String>> },
}
#[derive(Default)]
pub struct Fetch<'a> {
    pub selection: Selection,
    pub priority: u64,
    pub progress: Option<&'a Progress<'a>>,
    // Callback arguments: held bytes, total bytes. Error cancels this fetch.
}
pub struct Fetched { pub local: u64, pub volume: u64, pub origin: u64 }

#[derive(Clone)]
pub struct Client { /* private store/config */ }
impl Client {
    pub fn new(config: Config) -> Result<Client>;
    pub async fn open(&self, name_or_path: &str, options: &Open) -> Result<Checkpoint>;
    pub async fn snapshot_download(
        &self, reference: &Ref, want: &Fetch<'_>,
        local_dir: Option<&Path>, local_files_only: bool,
    ) -> Result<PathBuf>;
    pub async fn hub_download(
        &self, reference: &Ref, filename: &str,
        local_dir: Option<&Path>, local_files_only: bool,
    ) -> Result<PathBuf>;
    pub async fn upload_folder(&self, reference: &Ref, folder: &Path,
        progress: Option<&Progress<'_>>) -> Result<ObjectRef>;
}
// Python components= maps to Selection::Components. Asset selection does not fetch tensor chunks.

#[derive(Clone)]
pub struct Checkpoint { /* owned source, holds, metadata; no public Store */ }
impl Checkpoint {
    pub fn id(&self) -> &ObjectRef;
    pub fn header(&self) -> &format::Header;
    pub fn metadata(&self) -> &Metadata;
    pub fn chunks(&self) -> &[ObjectRef];
    pub fn tree(&self) -> Option<&ObjectRef>;
    pub fn verify(&self) -> Result<Vec<ObjectRef>>;
    pub fn view(&self) -> io::view::View;
    pub fn named(&self) -> Option<&Ref>;
    pub fn pipeline(&self) -> Option<&format::Pipeline>;
    pub fn components(&self) -> &[Component];
    pub fn component(&self, name: &str) -> Result<&Component>;
    pub fn chunk(&self, index: usize) -> Result<File>;
    pub fn missing(&self, selection: &Selection) -> Result<Vec<ObjectRef>>;
    pub async fn fetch(&self, want: &Fetch<'_>) -> Result<Fetched>;
    pub fn folder(&self) -> Result<PathBuf>;
}
// Synchronous chunk/read/asset/folder operations use bytes made available by fetch().
// view() retains its source and availability state for the byte engine.

#[derive(Clone)]
pub struct Component { /* owns shared checkpoint source; immutable metadata view */ }
impl Component {
    pub fn name(&self) -> &str;
    pub fn class(&self) -> &str;
    pub fn config(&self) -> &format::Config;
    pub fn config_file(&self) -> &str;
    pub fn tensors(&self) -> &[format::Tensor];
    pub fn assets(&self) -> &BTreeMap<String, ObjectRef>;
    pub fn read(&self, key: &str) -> Result<Vec<u8>>;
    pub fn read_part_into(&self, key: &str, role: &str, offset: u64, buf: &mut [u8]) -> Result<()>;
    pub fn asset(&self, name: &str) -> Result<Vec<u8>>;
    pub fn folder(&self) -> Result<PathBuf>;
}
// Tensors selects every part of each named key. Part selects the chunks overlapping
// its stored byte range. Invalid names, roles and ranges return Failure.
// A selected snapshot contains only the requested objects, even when others are cached.

#[derive(Clone)]
pub struct Endpoint { /* unchanged native endpoint/transport internally */ }
impl Endpoint {
    pub fn new(url: &str, token: Option<Arc<dyn TokenProvider>>) -> Result<Endpoint>;
    pub fn with_config(url: &str, config: &Config) -> Result<Endpoint>;
    pub async fn repo(&self, repo_id: &str) -> Result<Option<(Repo, String)>>;
    pub async fn update_repo(
        &self, repo_id: &str,
        change: impl FnMut(Option<Repo>) -> Result<Option<Repo>>,
    ) -> Result<Repo>;
}
// Endpoint("local") resolves default cache root; with_config supplies explicit
// cache_dir/CA/provider. No callback changes a different repo or bypasses native CAS.
```

### Credentials and errors

The public provider returns the same twelve-family `Failure` as other public operations.
A private adapter preserves the existing transport implementation. `retry` is optional
and defaults to `Ok(false)`; it permits a retry after a response such as a refreshed token
or challenge. A provider receives the known body digest, not the body contents.

```rust
pub struct Request<'a> {
    pub method: &'a str,
    pub url: &'a str,
    pub headers: &'a [(String, String)],
    pub payload: Option<&'a str>, // known hex SHA256 of request body
}
#[async_trait]
pub trait TokenProvider: Send + Sync {
    async fn headers(&self, request: &Request<'_>) -> Result<Vec<(String, String)>>;
    fn retry(&self, sent: &Request<'_>, status: u16, headers: &HeaderMap)
        -> Result<bool> { Ok(false) }
}
pub fn bearer(token: impl Into<String>) -> Arc<dyn TokenProvider>;
```

### Conversion, writing and export

`Writer::commit` is async for both local and remote writers. On a remote upload failure,
`committed()` retains the strong local checkpoint. Successful commit returns the exact
published tree through the checkpoint; it does not resolve a mutable reference again.

```rust
// Returning an error from the callback cancels conversion.
#[derive(Default)]
pub struct Convert<'a> {
    pub converter: Option<&'a str>,
    pub variant: Option<&'a str>,
    pub progress: Option<&'a ConvertProgress<'a>>,
}
impl Client {
    pub async fn convert(&self, sources: &[String], options: &Convert<'_>) -> Result<Checkpoint>;
    pub async fn resolve_source(&self, source: &str) -> Result<String>;
    pub fn writer(&self, options: WriterOptions<'_>) -> Result<Writer>;
}
#[derive(Default)]
pub struct WriterOptions<'a> {
    pub dest: Option<&'a Path>, pub variant: Option<&'a str>,
    pub base: Option<&'a Checkpoint>, pub resume: Option<&'a str>,
    pub repo_id: Option<&'a str>, pub endpoint: Option<&'a str>,
    pub token: Option<Arc<dyn TokenProvider>>, pub ca_der: Option<&'a [u8]>,
}
pub struct Writer { /* owned local and optional remote write state */ }
impl Writer {
    pub fn pipeline(&mut self, class: &str, config: format::Config) -> Result<()>;
    pub fn component(&mut self, name: &str, class: &str,
        config: format::Config, config_file: Option<&str>) -> Result<ComponentWriter<'_>>;
    pub fn set_metadata(&mut self, key: &str, value: Value) -> Result<()>;
    pub fn has(&self, component: &str, key: &str) -> Result<bool>;
    pub async fn commit(&mut self) -> Result<Checkpoint>;
    pub fn committed(&self) -> Option<Checkpoint>; // local commit survives remote failure
    pub fn abandon(self) -> Result<()>;
}
pub struct PartData<'a> {
    pub role: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data: &'a mut dyn Read,
}
pub struct ComponentWriter<'a> { /* borrows the writer */ }
impl ComponentWriter<'_> {
    pub fn put(&mut self, key: &str, dtype: Dtype, shape: &[u64], data: &mut dyn Read) -> Result<()>;
    pub fn put_encoded(&mut self, key: &str, dtype: Dtype, shape: &[u64],
        encoding: Encoding, parts: Vec<PartData<'_>>) -> Result<()>;
    pub fn copy(&mut self, key: &str, source: &Component, as_key: Option<&str>) -> Result<()>;
    pub fn asset(&mut self, name: &str, data: &[u8]) -> Result<()>;
    pub fn order(&mut self, keys: &[String]) -> Result<()>;
    pub fn has(&self, key: &str) -> Result<bool>;
}
pub enum Format { Safetensors }
pub fn export(checkpoint: &Checkpoint, dir: &Path, format: Format,
    dequantize: Option<Dtype>) -> Result<Vec<String>>;
```

### Metadata and integration types

| Type | Public meaning |
|---|---|
| `ObjectRef` | `{ sha256: String, length: u64 }`; `of(bytes)` hashes content, `id()` returns `sha256:…`, `key()` returns its blob key. |
| `format::Header` | One variant's chunks, pipeline, components and loading metadata, as defined by the format spec. |
| `format::Pipeline` | `{ class: String, config: format::Config }`. |
| `format::Tensor` | Tensor key, logical dtype and shape, and its plain location or encoding/parts. |
| `Encoding` | Scheme name plus complete parameters. |
| `Dtype` | A supported dtype or a preserved unknown dtype name; byte interpretation follows the format spec. |
| `Value` | The bounded-integer, boolean, string, list and string-map metadata domain below. |
| `Repo` | Mutable repository document containing releases and unreleased trees; its schema is in the format spec. |
| `io::view::View` | An owned source/availability handle passed to the existing byte engine and placement integration. |

The low-level byte engine and codec helpers used by in-tree integrations are separate from
this application-facing session interface. They do not introduce another model cache,
transport or copy loop.

## Behavior and ownership

### Values and identities

- Metadata and encoding parameters accept recursive booleans, integers in the signed-64
  through unsigned-64 range, strings, lists and string-keyed maps. Floats, `None` and bytes
  are refused there. Pipeline/component configuration is ordinary JSON and has a separate
  value domain.
- `header_ref` and `tree_ref` are `(bare_sha256_hex, byte_length)` pairs. `id` and `tree`
  are `sha256:…` strings. `repo_id` aliases `name`; `revision` aliases `release`.
- `Checkpoint.path` fetches/materializes the checkpoint. `Component.path` fetches that
  component's assets and renders its configuration; it does not fetch its tensor weights.

### Reads and writes

- `read` returns CPU data. `read_all` accepts `cpu`, `cuda` and `cuda:N`. A plain tensor
  still returns `Array` when `decode=False`; an encoded tensor returns `Encoded`.
- `read_into` offsets and destination lengths are bytes. `role` selects stored part bytes;
  otherwise encoded tensors decode. There is no conversion to the destination's dtype.
- Missing mapping keys raise `KeyError`; `get` returns its supplied default. `Tensors`
  supplies its mapping operations explicitly and is registered with `Mapping`.
- `put` requires dtype and shape for a raw Buffer; DLPack inputs supply them and reject
  contradictory explicit values. `copy(key, source, as_key=...)` reads source key `key`
  and can rename the destination. `has` sees current/resumed writes, not untouched base tensors.
- Python `Writer.commit` caches a successful result and re-raises its first commit failure.
  A native failure after local commit carries `error.checkpoint`, also available as
  `Writer.local_checkpoint`.

### Endpoints and callbacks

- `Endpoint.repo` returns `(document, ETag)` or `None`. `update_repo` can invoke its change
  callback again after a conflict, so the callback must tolerate retries. Returning `None`
  empties/removes the repository reference.
- Progress callbacks receive `(stage, completed, total)`: fetch/download counts are bytes;
  conversion counts are tensors. Ordinary Python callback exceptions can propagate.
- In the currently frozen Python wrapper, `configure` changes only non-`None` arguments.
  Its configured token is consulted by open/download/upload/Writer; convert, resolve_source
  and explicitly constructed Endpoint use their own token/default-provider selection.
  Uniform configuration and endpoint scoping are being reviewed during the session move;
  do not assume a configured token applies to every external provider.

### Lifetimes and publication

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
