use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::failure::{fail, Kind, Result};
use crate::format::{self, Body, ObjectRef, PartBytes};
use crate::io::{Dst, ReadOpts, Target};
use crate::store2::{CacheGuard, Checkpoint as Stored, Store};

use super::{remote::Remote, Fetch, Fetched, Ref, Selection};

struct LocalRepo {
    root: PathBuf,
    header: Option<PathBuf>,
}

struct Data {
    raw: Stored,
    named: Option<Ref>,
    tree: Option<ObjectRef>,
    remote: Option<Remote>,
    store: Store,
    local_repo: Option<LocalRepo>,
    variant: Option<String>,
    _folder: Option<CacheGuard>,
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
        let local_repo = LocalRepo {
            root: if path.is_file() {
                path.parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf()
            } else {
                path.to_path_buf()
            },
            header: path.is_file().then(|| path.to_path_buf()),
        };
        let store = source.store.clone();
        let data = Data {
            raw,
            named: None,
            tree: None,
            remote: Some(source),
            _extra: store.cache_hold(&[])?,
            store,
            local_repo: Some(local_repo),
            variant: None,
            _folder: None,
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
        // Immutable owned directory fencing is acquired before this tree-key path exists.
        let folder = tree
            .as_ref()
            .map(|tree| store.cache_directory(&store.root().join("snapshots").join(&tree.sha256)))
            .transpose()?;
        Ok(Self::from_data(Data {
            raw,
            named,
            tree,
            remote,
            _extra: extra,
            _folder: folder,
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

    pub async fn fetch(&self, want: &Fetch<'_>) -> Result<Fetched> {
        let objects = self.wanted(&want.selection)?;
        let total = objects.iter().map(|o| o.length).sum();
        let mut local = 0;
        let mut missing = Vec::new();
        for object in objects {
            if self.data.raw.holds(&object)? {
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
                let selected = Fetch {
                    priority: want.priority,
                    progress: want.progress.map(|_| &progress as &super::Progress<'_>),
                    ..Fetch::default()
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
        let mut objects = super::selection::objects(&self.data.raw, selection)?;
        if matches!(selection, Selection::All) {
            if let Some(tree) = &self.data.tree {
                objects.extend(
                    self.data
                        .store
                        .tree(tree)?
                        .repo_files()
                        .map(|(_, object)| object.clone()),
                );
            }
        }
        Ok(objects
            .into_iter()
            .map(|o| (o.sha256.clone(), o))
            .collect::<BTreeMap<_, _>>()
            .into_values()
            .collect())
    }

    pub(super) fn export_to(
        &self,
        dir: &Path,
        dequantize: Option<&format::Dtype>,
    ) -> Result<Vec<String>> {
        // This synchronous door never starts a runtime or hidden download. Fetch All first.
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
            local_repo_files(folder, &folder.root, &mut repo)?;
        }
        self.data.store.cache_preserve_snapshot(dir)?;
        crate::export::safetensors(&self.data.raw, &repo, dir, dequantize)
    }

    pub fn folder(&self) -> Result<PathBuf> {
        if let Some(local) = &self.data.local_repo {
            return Ok(local.root.clone());
        }
        if let Some(tree) = &self.data.tree {
            if let Some(missing) = self.missing(&Selection::All)?.first() {
                return fail(
                    Kind::NotFound,
                    format!(
                        "{} is not present; fetch it asynchronously first",
                        missing.id()
                    ),
                );
            }
            let parsed = self.data.store.tree(tree)?;
            let variant = match self.data.variant.as_deref() {
                Some(variant) => variant.to_owned(),
                None => parsed
                    .variants()
                    .find(|(_, header)| *header == self.id())
                    .map(|(variant, _)| variant.to_owned())
                    .ok_or_else(|| crate::failure::Failure {
                        kind: Kind::Integrity,
                        detail: "checkpoint header is not a variant of its source tree".into(),
                    })?,
            };
            let dir = self.data.store.root().join("snapshots").join(&tree.sha256);
            self.data.store.snapshot(tree, &dir, Some(&[variant]))?;
            self.data.store.cache_snapshot(&dir)?;
            return Ok(dir);
        }
        Ok(self
            .data
            .raw
            .path_for(self.data.variant.as_deref().unwrap_or(""))?)
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
    /// Fetch Selection::Assets first; this synchronous operation never downloads.
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
        if source.header.as_ref() == Some(&path) {
            continue;
        }
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
