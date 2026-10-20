use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::failure::{fail, Kind, Result};
use crate::format::{self, Body, ObjectRef, PartBytes};
use crate::io::{Dst, ReadOpts, Target};
use crate::store::{CacheGuard, Checkpoint as Stored, Store};

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
    _extra: CacheGuard,
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
            _extra: store.cache_hold(&[])?,
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
        let extra = store.cache_hold(&extra)?;
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

    /// Newly owned descriptor, valid after every checkpoint/component handle drops.
    pub fn chunk(&self, index: usize) -> Result<File> {
        if index >= self.chunks().len() {
            return fail(Kind::Invalid, format!("no chunk {index}"));
        }
        Ok(self.data.raw.chunk(index)?)
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
    pub async fn materialize(
        &self,
        dir: &Path,
        format: Format,
        dequantize: Option<format::Dtype>,
    ) -> Result<PathBuf> {
        if matches!(format, Format::Cozytensors) && dequantize.is_some() {
            return fail(
                Kind::Invalid,
                "native cozytensors export preserves encoded bytes; dequantize is unsupported",
            );
        }
        if let (Format::Safetensors, Some(dtype)) = (format, dequantize.as_ref()) {
            if !matches!(
                dtype,
                format::Dtype::F32 | format::Dtype::BF16 | format::Dtype::F16
            ) {
                return fail(
                    Kind::Invalid,
                    format!("dequantize to {dtype}: one of F32, BF16, F16"),
                );
            }
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
                Format::Safetensors => checkpoint.export_to(&dir, dequantize.as_ref()),
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

    pub(super) fn export_to(
        &self,
        dir: &Path,
        dequantize: Option<&format::Dtype>,
    ) -> Result<Vec<String>> {
        let repo = self.export_paths()?;
        self.data.store.cache_preserve_snapshot(dir)?;
        crate::export::safetensors(&self.data.raw, &repo, dir, dequantize)
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
        self.data.store.cache_preserve_snapshot(dir)?;
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

    /// The same placement/byte-engine view, retaining this source without a cycle.
    /// Its callback checks availability only; Rust callers explicitly await fetch.
    pub fn view(&self) -> crate::io::view::View {
        let data = Arc::clone(&self.data);
        crate::io::view::View::from(&data.raw).with_fetch(move |objects| {
            for object in objects {
                if !data.raw.holds(object)? {
                    return fail(
                        Kind::NotFound,
                        format!(
                            "{} is not present; fetch it asynchronously first",
                            object.id()
                        ),
                    );
                }
            }
            Ok(())
        })
    }
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
    pub fn tensors(&self) -> &[format::Tensor] {
        &self.metadata().tensors
    }
    pub fn assets(&self) -> &BTreeMap<String, ObjectRef> {
        &self.metadata().assets
    }

    /// Payload bytes must first be made available by async Checkpoint::fetch.
    pub fn read(&self, key: &str) -> Result<Vec<u8>> {
        let tensor = self.data.raw.tensor(&self.name, key)?;
        match &tensor.body {
            Body::Data(at) => {
                let mut bytes = vec![0; at.length as usize];
                self.read_part_into(key, "", 0, &mut bytes)?;
                Ok(bytes)
            }
            Body::Encoded { encoding, parts } => {
                let mut buffers = Vec::with_capacity(parts.len());
                for part in parts {
                    let mut bytes = vec![0; part.at().length as usize];
                    self.read_part_into(key, &part.role, 0, &mut bytes)?;
                    buffers.push(bytes);
                }
                let views = parts
                    .iter()
                    .zip(&buffers)
                    .map(|(part, bytes)| {
                        (
                            part.role.as_str(),
                            PartBytes {
                                dtype: &part.dtype,
                                shape: &part.shape,
                                bytes,
                            },
                        )
                    })
                    .collect();
                Ok(format::decode(
                    &tensor.dtype,
                    &tensor.shape,
                    encoding,
                    &views,
                )?)
            }
        }
    }

    /// This safe borrow is held until the existing synchronous engine returns.
    #[allow(unsafe_code)]
    pub fn read_part_into(&self, key: &str, role: &str, offset: u64, buf: &mut [u8]) -> Result<()> {
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
    /// Write config and support files into the caller's directory, never model weights.
    /// Download Selection::Assets first; this synchronous operation never downloads.
    pub fn export_files(&self, dir: &Path) -> Result<PathBuf> {
        Ok(self.data.raw.export_component_files(&self.name, dir)?)
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
