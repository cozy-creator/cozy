# Rust API declarations for review

Temporary review copy of the concrete planned TensorFS 0.13 interface. Implementation is
in progress; declarations omit bodies and are not a standalone compilable module.
For normal development, use the [Rust usage guide](API.md).

## Client, selection and returned objects

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
}
// Synchronous chunk/read/asset/folder operations use bytes made available by fetch().
// view() retains its source and availability state for the byte engine.

#[derive(Clone)]
pub struct Component { /* owns shared checkpoint source; immutable metadata view */ }
impl Component {
    pub fn name(&self) -> &str;
    pub fn class(&self) -> &str;
    pub fn config(&self) -> &format::Config;
    pub fn tensors(&self) -> &[format::Tensor];
    pub fn assets(&self) -> &BTreeMap<String, ObjectRef>;
    pub fn read(&self, key: &str) -> Result<Vec<u8>>;
    pub fn read_part_into(&self, key: &str, role: &str, offset: u64, buf: &mut [u8]) -> Result<()>;
    pub fn asset(&self, name: &str) -> Result<Vec<u8>>;
    pub fn export_files(&self, directory: &Path) -> Result<PathBuf>;
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

## Credentials

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

## Conversion, writers and export

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
    // Exact header-file path, or None for a checkpoint in the shared store.
    // A file destination and a repository variant label cannot be combined.
    pub dest: Option<&'a Path>, pub variant: Option<&'a str>,
    pub base: Option<&'a Checkpoint>, pub resume: Option<&'a str>,
    pub repo_id: Option<&'a str>, pub endpoint: Option<&'a str>,
    pub token: Option<Arc<dyn TokenProvider>>, pub ca_der: Option<&'a [u8]>,
}
pub struct Writer { /* owned local and optional remote write state */ }
impl Writer {
    pub fn pipeline(&mut self, class: &str, config: format::Config) -> Result<()>;
    pub fn component(&mut self, name: &str, class: &str,
        config: format::Config) -> Result<ComponentWriter<'_>>;
    pub fn set_metadata(&mut self, key: &str, value: Value) -> Result<()>;
    pub fn has(&self, component: &str, key: &str) -> Result<bool>;
    pub async fn commit(&mut self) -> Result<Checkpoint>;
    pub fn committed(&self) -> Option<Checkpoint>; // local commit survives remote failure
    pub fn abandon(self) -> Result<()>;
}
pub struct ComponentWriter<'a> { /* borrows the writer */ }
impl ComponentWriter<'_> {
    pub fn put(&mut self, key: &str, dtype: Dtype, shape: &[u64], data: &mut dyn Read) -> Result<()>;
    pub fn put_encoded(&mut self, key: &str, dtype: Dtype, shape: &[u64],
        encoding: Encoding, parts: Vec<raw::PartData<'_>>) -> Result<()>;
    pub fn copy(&mut self, key: &str, source: &Component, as_key: Option<&str>) -> Result<()>;
    pub fn asset(&mut self, name: &str, data: &[u8]) -> Result<()>;
    pub fn order(&mut self, keys: &[String]) -> Result<()>;
    pub fn has(&self, key: &str) -> Result<bool>;
}
pub enum Format { Safetensors, Cozytensors }
pub fn export(checkpoint: &Checkpoint, dir: &Path, format: Format,
    dequantize: Option<Dtype>) -> Result<Vec<String>>;
```

## Low-level streaming input

This is for raw encoded-stream producers. Ordinary readers and model consumers do not
need storage descriptors or this module.

```rust
pub mod raw {
    pub struct PartData<'a> {
        pub role: String,
        pub dtype: Dtype,
        pub shape: Vec<u64>,
        pub data: &'a mut dyn Read,
    }
}
```
