use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::failure::{fail, Kind, Result};
use crate::format::{self, Body, ObjectRef, PartBytes};
use crate::io::{Dst, ReadOpts, Target};
use crate::store::retention::Hold;
use crate::store::{Checkpoint as Stored, Store};

use super::{remote::Remote, Download, Fetched, Format, Ref, Selection};

struct LocalRepo {
    root: PathBuf,
    files: BTreeMap<String, PathBuf>,
}

struct Data {
    raw: Stored,
    header_bytes: Arc<[u8]>,
    named: Option<Ref>,
    tree: Option<ObjectRef>,
    remote: Option<Remote>,
    store: Store,
    local_repo: Option<LocalRepo>,
    variant: Option<String>,
    _extra: Option<Arc<Hold>>,
}

/// An owned source. A cloned component keeps this data alive after its parent drops.
#[derive(Clone)]
pub struct Checkpoint {
    data: Arc<Data>,
    components: Arc<[Component]>,
}

impl fmt::Debug for Checkpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Checkpoint")
            .field("id", self.id())
            .field("named", &self.data.named)
            .finish()
    }
}

impl Checkpoint {
    pub(super) fn local(raw: Stored, source: Remote, path: &Path) -> Result<Self> {
        let (header_bytes, variant) = local_header_bytes(&raw, path)?;
        let local_repo = if path.is_dir() {
            let mut repo = LocalRepo {
                root: path.to_path_buf(),
                files: BTreeMap::new(),
            };
            let mut files = BTreeMap::new();
            local_repo_files(&repo, path, &mut files)?;
            repo.files = files;
            Some(repo)
        } else {
            None
        };
        let store = source.store.clone();
        let data = Data {
            raw,
            header_bytes: header_bytes.into(),
            named: None,
            tree: None,
            remote: Some(source),
            _extra: None,
            store,
            local_repo,
            variant,
        };
        Ok(Self::from_data(data))
    }

    pub(super) fn managed(
        raw: Stored,
        named: Option<Ref>,
        tree: Option<ObjectRef>,
        remote: Option<Remote>,
        extra: Vec<ObjectRef>,
        store: Store,
        variant: Option<String>,
    ) -> Result<Self> {
        let extra = Some(store.hold(&extra)?);
        let header_bytes = store.read(raw.id())?;
        Ok(Self::from_data(Data {
            raw,
            header_bytes: header_bytes.into(),
            named,
            tree,
            remote,
            _extra: extra,
            variant,
            store,
            local_repo: None,
        }))
    }

    fn from_data(data: Data) -> Self {
        let data = Arc::new(data);
        let components = data
            .raw
            .components()
            .iter()
            .map(|c| Component {
                data: Arc::clone(&data),
                name: c.name.clone(),
            })
            .collect::<Vec<_>>()
            .into();
        Self { data, components }
    }

    pub(super) fn stored(&self) -> &Stored {
        &self.data.raw
    }

    pub(super) fn header_bytes(&self) -> Arc<[u8]> {
        self.data.header_bytes.clone()
    }

    pub fn id(&self) -> &ObjectRef {
        self.data.raw.id()
    }
    pub fn header(&self) -> &format::Header {
        self.data.raw.header()
    }
    pub fn metadata(&self) -> &format::Metadata {
        &self.header().metadata
    }
    pub fn chunks(&self) -> &[ObjectRef] {
        &self.header().chunks
    }
    pub fn variant(&self) -> Option<&str> {
        self.data
            .variant
            .as_deref()
            .or_else(|| self.data.named.as_ref().and_then(|r| r.variant.as_deref()))
    }
    pub fn named(&self) -> Option<&Ref> {
        self.data.named.as_ref()
    }
    pub fn pipeline(&self) -> Option<&format::Pipeline> {
        self.data.raw.pipeline()
    }
    pub fn components(&self) -> &[Component] {
        &self.components
    }

    pub fn component(&self, name: &str) -> Result<&Component> {
        self.components
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| crate::failure::Failure {
                kind: Kind::NotFound,
                detail: format!("no component {name:?}"),
            })
    }

    pub fn missing(&self, selection: &Selection) -> Result<Vec<ObjectRef>> {
        self.wanted(selection)?
            .into_iter()
            .filter_map(|object| match self.data.raw.holds(&object) {
                Ok(true) => None,
                Ok(false) => Some(Ok(object)),
                Err(error) => Some(Err(error.into())),
            })
            .collect()
    }

    pub async fn download(&self, want: &Download<'_>) -> Result<Fetched> {
        let objects = self.wanted(&want.selection)?;
        self.fetch_list(objects, want).await
    }

    async fn fetch_list(&self, objects: Vec<ObjectRef>, want: &Download<'_>) -> Result<Fetched> {
        let total = objects.iter().map(|o| o.length).sum();
        let mut local = 0;
        let mut missing = Vec::new();
        for object in objects {
            if object == *self.id() || self.data.raw.holds(&object)? {
                local += object.length;
            } else {
                missing.push(object);
            }
        }
        if missing.is_empty() {
            if let Some(report) = want.progress {
                report(total, total)?;
            }
            return Ok(Fetched {
                local,
                ..Fetched::default()
            });
        }
        match &self.data.remote {
            Some(remote) => {
                let progress = |held, _| match want.progress {
                    Some(report) => report(local + held, total),
                    None => Ok(()),
                };
                let selected = Download {
                    priority: want.priority,
                    progress: want.progress.map(|_| &progress as &super::Progress<'_>),
                    ..Download::default()
                };
                let mut landed = remote.fetch(&missing, &selected).await?;
                landed.local += local;
                Ok(landed)
            }
            None => fail(
                Kind::NotFound,
                format!("{} is not present", missing[0].id()),
            ),
        }
    }

    fn wanted(&self, selection: &Selection) -> Result<Vec<ObjectRef>> {
        super::selection::objects(&self.data.raw, selection)
    }

    /// Download the checkpoint and known repository files, then write an explicit directory.
    /// `decode` writes each encoded tensor in its logical dtype (safetensors only); otherwise
    /// the stored encoding is kept.
    pub async fn materialize(&self, dir: &Path, format: Format, decode: bool) -> Result<PathBuf> {
        if matches!(format, Format::Cozytensors) && decode {
            return fail(Kind::Invalid, "decode requires the safetensors format");
        }
        let runtime =
            tokio::runtime::Handle::try_current().map_err(|error| crate::failure::Failure {
                kind: Kind::Unsupported,
                detail: format!("materialize requires a Tokio runtime: {error}"),
            })?;
        match fs::metadata(dir) {
            Ok(metadata) if !metadata.is_dir() => {
                return Err(super::io_failure(
                    dir.display(),
                    std::io::Error::from(std::io::ErrorKind::AlreadyExists),
                ));
            }
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                return Err(super::io_failure(dir.display(), error));
            }
            _ => {}
        }
        self.prepare_export().await?;
        let (checkpoint, dir) = (self.clone(), dir.to_path_buf());
        let worker = runtime.spawn_blocking(move || -> Result<PathBuf> {
            fs::create_dir_all(&dir).map_err(|error| super::io_failure(dir.display(), error))?;
            match format {
                Format::Safetensors => checkpoint.export_to(&dir, decode),
                Format::Cozytensors => checkpoint.export_cozytensors(&dir),
            }?;
            Ok(dir)
        });
        match worker.await {
            Ok(result) => result,
            Err(error) => fail(Kind::Io, format!("export worker failed: {error}")),
        }
    }

    async fn prepare_export(&self) -> Result<()> {
        let mut objects = self.wanted(&Selection::All)?;
        if let Some(tree) = &self.data.tree {
            objects.extend(
                self.data
                    .store
                    .tree(tree)?
                    .repo_files()
                    .map(|(_, object)| object.clone()),
            );
        }
        let objects = objects
            .into_iter()
            .map(|object| (object.sha256.clone(), object))
            .collect::<BTreeMap<_, _>>();
        self.fetch_list(objects.into_values().collect(), &Download::default())
            .await?;
        Ok(())
    }

    fn export_paths(&self) -> Result<BTreeMap<String, PathBuf>> {
        // Serialization follows the explicit export operation's async preparation.
        if let Some(missing) = self.missing(&Selection::All)?.first() {
            return fail(
                Kind::NotFound,
                format!(
                    "{} is not present; fetch it asynchronously first",
                    missing.id()
                ),
            );
        }
        let mut repo = BTreeMap::new();
        if let Some(tree) = &self.data.tree {
            for (path, object) in self.data.store.tree(tree)?.repo_files() {
                repo.insert(path.to_owned(), self.data.store.path(object)?);
            }
        } else if let Some(folder) = &self.data.local_repo {
            repo = folder.files.clone();
        }
        for path in repo.values() {
            File::open(path).map_err(|e| super::io_failure(path.display(), e))?;
        }
        Ok(repo)
    }

    pub(super) fn export_to(&self, dir: &Path, decode: bool) -> Result<Vec<String>> {
        let repo = self.export_paths()?;
        crate::export::safetensors(&self.data.raw, &repo, dir, decode)
    }

    pub(crate) async fn fetch_objects_exact(
        &self,
        objects: &[ObjectRef],
        want: &Download<'_>,
    ) -> Result<Fetched> {
        let mut allowed = self.wanted(&Selection::All)?;
        allowed.push(self.id().clone());
        allowed.extend(self.data.tree.iter().cloned());
        let mut allowed: std::collections::BTreeSet<_> = allowed.into_iter().collect();
        // Explicit context objects are independent of checkpoint-only All. A request for
        // the tree itself must also work when that metadata needs to be downloaded again.
        if objects.iter().any(|object| !allowed.contains(object)) {
            if let Some(tree) = &self.data.tree {
                allowed.extend(
                    self.data
                        .store
                        .tree(tree)?
                        .repo_files()
                        .map(|(_, object)| object.clone()),
                );
            }
        }
        let mut selected = BTreeMap::new();
        for object in objects {
            if !allowed.contains(object) {
                return fail(
                    Kind::Invalid,
                    format!("{} is not an exact object of this checkpoint", object.id()),
                );
            }
            selected.insert(object.sha256.clone(), object.clone());
        }
        self.fetch_list(selected.into_values().collect(), want)
            .await
    }

    /// # Safety
    /// Targets must satisfy the existing engine's exclusive destination/lifetime contract.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn read_into_raw(
        &self,
        targets: &[Target<'_>],
        opts: &ReadOpts,
    ) -> Result<crate::io::ReadStats> {
        Ok(unsafe { self.data.raw.read_into(targets, opts) }?)
    }

    pub(super) fn export_cozytensors(&self, dir: &Path) -> Result<Vec<String>> {
        let repo = self.export_paths()?;
        let repo_objects = match &self.data.tree {
            Some(tree) => self
                .data
                .store
                .tree(tree)?
                .repo_files()
                .map(|(p, r)| (p.to_owned(), r.clone()))
                .collect(),
            None => BTreeMap::new(),
        };
        crate::export::cozytensors(
            &self.data.raw,
            &repo,
            &repo_objects,
            dir,
            self.variant().unwrap_or(""),
            &self.data.header_bytes,
        )
    }

    pub fn tree(&self) -> Option<&ObjectRef> {
        self.data.tree.as_ref()
    }

    /// Inspect the header and selected full chunk/asset objects without fetching or
    /// changing data. Repository trees/files and other variants are outside its scope.
    pub fn verify(&self, selection: &Selection) -> Result<crate::VerificationReport> {
        let objects = super::selection::objects(&self.data.raw, selection)?;
        crate::verification::verify(&self.data.raw, objects)
    }

    /// The stored spans of the parts `regions` names (`component/key`, or `component/key#role`
    /// for one part), in header order: what placement copies.
    pub fn spans(&self, regions: &[Vec<String>]) -> Result<Vec<crate::io::Span>> {
        Ok(crate::io::view::plan(self.header(), regions)?.0)
    }

    /// The stored dtype and shape of the part `component/key#role` (`#value`: plain data).
    pub fn stored_part(&self, name: &str) -> Option<crate::io::StoredPart> {
        crate::io::view::part(self.header(), name)
    }

    /// This checkpoint's chunks as a read-engine source. A read fetches the chunks it needs
    /// and lacks on the process runtime, so call it from outside an async context.
    #[cfg(unix)]
    pub fn source(&self, direct_io: bool) -> crate::io::source::Source {
        crate::io::source::Source::checkpoint(&self.view(), direct_io)
    }

    #[cfg(unix)]
    fn view(&self) -> crate::io::view::View {
        let checkpoint = self.clone();
        crate::io::view::View::from(&self.data.raw).with_fetch(move |objects| {
            let want = Download::default();
            crate::raw::runtime()?.block_on(checkpoint.fetch_objects_exact(objects, &want)).map(drop)
        })
    }
}

/// A tensor as stored: a plain tensor's data, or an encoded tensor's logical dtype and shape,
/// encoding, and parts by role.
#[derive(Clone, Debug, PartialEq)]
pub enum Read {
    Plain(Vec<u8>),
    Encoded {
        dtype: format::Dtype,
        shape: Vec<u64>,
        encoding: format::Encoding,
        parts: BTreeMap<String, Vec<u8>>,
    },
}

/// What [`Component::read_into`] reads, as Python's `read_into` keywords.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReadInto<'a> {
    /// One stored part of an encoded tensor; `""` is a plain tensor's data.
    pub role: Option<&'a str>,
    /// An encoded tensor's logical bytes, through its scheme's decode.
    pub decode: bool,
    /// The first byte read.
    pub offset: u64,
}

#[derive(Clone)]
pub struct Component {
    data: Arc<Data>,
    name: String,
}

impl fmt::Debug for Component {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Component")
            .field("name", &self.name)
            .field("checkpoint", self.data.raw.id())
            .finish()
    }
}

impl Component {
    fn metadata(&self) -> &format::Component {
        self.data
            .raw
            .component(&self.name)
            .expect("component validated against an immutable header")
    }

    pub(super) fn stored(&self) -> &Stored {
        &self.data.raw
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn class(&self) -> &str {
        &self.metadata().class
    }
    pub fn config(&self) -> &format::Config {
        &self.metadata().config
    }
    /// The file name the config exports as: `config.json` unless the writer named another.
    pub fn config_file(&self) -> &str {
        self.metadata().config_name()
    }
    pub fn tensors(&self) -> &[format::Tensor] {
        &self.metadata().tensors
    }
    pub fn assets(&self) -> &BTreeMap<String, ObjectRef> {
        &self.metadata().assets
    }

    /// A tensor's stored bytes, as Python's `read(decode=False)`: a plain tensor's data, or an
    /// encoded tensor's parts by role with its encoding. Download the payload first.
    pub fn read(&self, key: &str) -> Result<Read> {
        let tensor = self.data.raw.tensor(&self.name, key)?;
        let part = |role: &str, length: u64| {
            let mut bytes = vec![0; length as usize];
            self.read_part(key, role, 0, &mut bytes).map(|()| bytes)
        };
        Ok(match &tensor.body {
            Body::Data(at) => Read::Plain(part("", at.length)?),
            Body::Encoded { encoding, parts } => Read::Encoded {
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
                encoding: encoding.clone(),
                parts: parts
                    .iter()
                    .map(|p| Ok((p.role.clone(), part(&p.role, p.at().length)?)))
                    .collect::<Result<_>>()?,
            },
        })
    }

    /// A tensor's logical bytes in its dtype, as Python's `read(decode=True)`: an encoded
    /// tensor goes through its scheme's reference decode.
    pub fn read_decoded(&self, key: &str) -> Result<Vec<u8>> {
        let tensor = self.data.raw.tensor(&self.name, key)?;
        let (encoding, parts) = match (self.read(key)?, &tensor.body) {
            (Read::Plain(bytes), _) => return Ok(bytes),
            (
                Read::Encoded {
                    encoding, parts, ..
                },
                Body::Encoded { parts: stored, .. },
            ) => (encoding, (parts, stored)),
            _ => unreachable!("a read follows the tensor's body"),
        };
        let (bytes, stored) = parts;
        let views = stored
            .iter()
            .map(|p| {
                let view = PartBytes { dtype: &p.dtype, shape: &p.shape, bytes: &bytes[&p.role] };
                (p.role.as_str(), view)
            })
            .collect();
        Ok(format::decode(&tensor.dtype, &tensor.shape, &encoding, &views)?)
    }

    /// `out.len()` bytes of `key` from `offset`, as Python's `read_into`: a plain tensor's data,
    /// one stored part (`role`), or with `decode` an encoded tensor's logical bytes. An encoded
    /// tensor with neither is `Invalid`, naming its roles. Download the payload first.
    pub fn read_into(&self, key: &str, out: &mut [u8], options: &ReadInto<'_>) -> Result<()> {
        let tensor = self.data.raw.tensor(&self.name, key)?;
        let role = match (options.role, &tensor.body) {
            (Some(_), _) if options.decode => {
                return fail(
                    Kind::Invalid,
                    "role and decode exclude each other: a role reads one stored part",
                );
            }
            (Some(role), _) => role,
            (None, Body::Data(_)) => "",
            (None, Body::Encoded { parts, .. }) if !options.decode => {
                let roles: Vec<_> = parts.iter().map(|p| p.role.as_str()).collect();
                let roles = roles.join(", ");
                let detail = format!("{key} is encoded: pass decode or a role ({roles})");
                return fail(Kind::Invalid, detail);
            }
            (None, Body::Encoded { .. }) => {
                let bytes = self.read_decoded(key)?;
                let start = usize::try_from(options.offset).unwrap_or(usize::MAX);
                let end = start.saturating_add(out.len());
                let Some(window) = bytes.get(start..end) else {
                    let (len, total) = (out.len(), bytes.len());
                    let detail = format!("{key}: {len} bytes from {start} of {total}");
                    return fail(Kind::Invalid, detail);
                };
                out.copy_from_slice(window);
                return Ok(());
            }
        };
        self.read_part(key, role, options.offset, out)
    }

    /// This safe borrow is held until the existing synchronous engine returns.
    #[allow(unsafe_code)]
    fn read_part(&self, key: &str, role: &str, offset: u64, buf: &mut [u8]) -> Result<()> {
        let target = Target {
            component: &self.name,
            key,
            role,
            offset,
            dst: Dst::Host {
                ptr: buf.as_mut_ptr(),
                len: buf.len(),
            },
        };
        // SAFETY: buf is exclusively borrowed and remains alive until read_into
        // completes. The engine validates the part/range; the owned source keeps
        // every file/retention owner alive. No new chunk copy path is introduced.
        unsafe { self.data.raw.read_into(&[target], &ReadOpts::default()) }?;
        Ok(())
    }

    pub fn asset(&self, name: &str) -> Result<Vec<u8>> {
        Ok(self.data.raw.asset(&self.name, name)?)
    }
    /// Write config and support files, never weights, into `dir`; with `None`, into a folder
    /// the cache manages and evicts like its other entries. Download Selection::Assets first;
    /// this synchronous operation never downloads.
    pub fn export_files(&self, dir: Option<&Path>) -> Result<PathBuf> {
        let store = &self.data.store;
        if let Some(dir) = dir {
            return Ok(self.data.raw.export_component_files(&self.name, dir)?);
        }
        let identity = format!("{}:files:{}", self.data.raw.id().sha256, self.name);
        let name = crate::sha256::hex_digest(identity.as_bytes());
        store.used(&name, [])?;
        let dir = store.root().join("snapshots").join(name);
        self.data.raw.export_component_files(&self.name, &dir)?;
        store.maintain(0)?;
        Ok(dir)
    }
}

// A plain D10 folder stores repository files outside chunks/ and variant headers. Keep paths
// until export, rather than admitting every payload during metadata-only Client::open.
fn local_repo_files(
    source: &LocalRepo,
    folder: &Path,
    out: &mut BTreeMap<String, PathBuf>,
) -> Result<()> {
    for entry in fs::read_dir(folder).map_err(|e| super::io_failure(folder.display(), e))? {
        let entry = entry.map_err(|e| super::io_failure(folder.display(), e))?;
        if folder == source.root && entry.file_name() == format::CHUNKS_DIR {
            continue;
        }
        let path = entry.path();
        let kind = entry
            .file_type()
            .map_err(|e| super::io_failure(path.display(), e))?;
        if kind.is_dir() {
            local_repo_files(source, &path, out)?;
        } else if kind.is_file() {
            let name = path
                .strip_prefix(&source.root)
                .expect("walk remains under its root")
                .iter()
                .map(|part| {
                    part.to_str().ok_or_else(|| crate::failure::Failure {
                        kind: Kind::Invalid,
                        detail: "repository paths must be UTF-8".into(),
                    })
                })
                .collect::<Result<Vec<_>>>()?
                .join("/");
            format::check_file(&name)?;
            if format::variant_of(&name).is_none() {
                out.insert(name, path);
            }
        }
    }
    Ok(())
}

fn local_header_bytes(raw: &Stored, path: &Path) -> Result<(Vec<u8>, Option<String>)> {
    if path.is_file() {
        let bytes = fs::read(path).map_err(|e| super::io_failure(path.display(), e))?;
        if &ObjectRef::of(&bytes) != raw.id() {
            return fail(Kind::Integrity, "header changed during open");
        }
        return Ok((bytes, None));
    }
    for entry in fs::read_dir(path).map_err(|e| super::io_failure(path.display(), e))? {
        let entry = entry.map_err(|e| super::io_failure(path.display(), e))?;
        let name = entry.file_name();
        let Some(variant) = name.to_str().and_then(format::variant_of) else {
            continue;
        };
        let bytes =
            fs::read(entry.path()).map_err(|e| super::io_failure(entry.path().display(), e))?;
        if &ObjectRef::of(&bytes) == raw.id() {
            return Ok((bytes, Some(variant.to_owned())));
        }
    }
    fail(Kind::Integrity, "the selected header changed during open")
}
