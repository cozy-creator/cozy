//! The public session: async object movement followed by synchronous engine reads.
//!
//! The backing store and downloader stay implementation details. Checkpoints and
//! component children own their sources and retention, independently of the Client.

mod checkpoint;
mod convert;
mod endpoint;
mod file;
mod provider;
mod remote;
mod selection;
mod writer;

pub use crate::fetch::Landed as Fetched;
pub use checkpoint::{Checkpoint, Component, Read, ReadInto};
pub use convert::{Convert, ConvertProgress};
pub use endpoint::Endpoint;
pub use file::CheckpointFile;
pub use provider::{bearer, HeaderMap, Request, TokenProvider};
pub use writer::{ComponentWriter, PartData, Writer, WriterOptions};

use std::fmt;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use crate::failure::{fail, Failure, Kind, Result};
use crate::format::{is_digest, repo_name, ObjectRef, ROOT_ORG};
use crate::store::Store;

/// Locations and an ordinary supplied provider. Credentials are never formatted.
#[derive(Clone)]
pub struct Config {
    pub endpoint: String,
    pub cache_dir: PathBuf,
    pub mirror: Option<PathBuf>,
    pub token: Option<Arc<dyn TokenProvider>>,
    pub ca_der: Option<Vec<u8>>,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            endpoint: crate::fetch::token::default_endpoint(),
            // Validation belongs to Client::new. An absent location must never
            // turn into a store in the process's working directory.
            cache_dir: crate::home::resolve(None).unwrap_or_default(),
            mirror: std::env::var_os("TENSORFS_MIRROR").map(PathBuf::from),
            token: None,
            ca_der: None,
        }
    }
}

impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("endpoint", &self.endpoint)
            .field("cache_dir", &self.cache_dir)
            .field("mirror", &self.mirror)
            .field("token", &self.token.as_ref().map(|_| "configured"))
            .field("ca_der", &self.ca_der.as_ref().map(|_| "configured"))
            .finish()
    }
}

/// A model identity, separate from the location serving it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ref {
    pub repo_id: String,
    pub revision: Option<String>,
    pub variant: Option<String>,
    pub endpoint: Option<String>,
}

impl FromStr for Ref {
    type Err = Failure;

    fn from_str(text: &str) -> Result<Self> {
        let (name, selection) = text
            .split_once('@')
            .map_or((text, None), |(n, r)| (n, Some(r)));
        let (revision, variant) = match selection {
            Some(selection) => {
                let (revision, variant) = selection
                    .split_once('/')
                    .map_or((selection, None), |(r, v)| (r, Some(v)));
                if revision.is_empty() || variant.is_some_and(|v| v.is_empty()) {
                    return fail(
                        Kind::Invalid,
                        "a reference needs a nonempty revision and variant",
                    );
                }
                (Some(revision.to_owned()), variant.map(str::to_owned))
            }
            None => (None, None),
        };
        let digest = name.strip_prefix("sha256:");
        if digest.is_some_and(|d| !is_digest(d)) {
            return fail(
                Kind::Invalid,
                "a digest is sha256: followed by 64 lowercase hex digits",
            );
        }
        // A name with no org is first-party, the root namespace's: `sdxl` is `_/sdxl`.
        let name = match digest.is_some() || name.contains('/') {
            true => name.to_owned(),
            false => format!("{ROOT_ORG}/{name}"),
        };
        if digest.is_none() && !repo_name(&name) {
            return fail(Kind::Invalid, "a repository is [<org>/]<name>");
        }
        Ok(Self {
            repo_id: name,
            revision,
            variant,
            endpoint: None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Open {
    /// The precision lane asked, 16, 8 or 4 bits: only it answers (an engine reading only 16-bit
    /// weights asks 16); none prefers 8 and takes whatever works.
    pub lane: Option<u32>,
    /// Compute capabilities (e.g. "8.9"); none detects visible devices, empty selects CPU.
    pub gpus: Option<Vec<String>>,
    pub revision: Option<String>,
    pub variant: Option<String>,
    pub endpoint: Option<String>,
    pub local_files_only: bool,
    /// Load metadata only by default; false also waits for this checkpoint's bytes.
    pub lazy: bool,
}

impl Default for Open {
    fn default() -> Self {
        Self {
            lane: None,
            gpus: None,
            revision: None,
            variant: None,
            endpoint: None,
            local_files_only: false,
            lazy: true,
        }
    }
}

/// A failing progress callback cancels this fetch. Values are held and total bytes.
pub type Progress<'a> = dyn Fn(u64, u64) -> Result<()> + Sync + 'a;

#[derive(Clone, Debug, Default)]
pub enum Selection {
    #[default]
    All,
    Components(Vec<String>),
    Tensors {
        component: String,
        keys: Vec<String>,
    },
    Part {
        component: String,
        key: String,
        role: Option<String>,
        range: Option<Range<u64>>,
    },
    Assets {
        component: String,
        names: Option<Vec<String>>,
    },
}

#[derive(Default)]
pub struct Download<'a> {
    pub selection: Selection,
    pub priority: u64,
    pub progress: Option<&'a Progress<'a>>,
}

/// A session, not a public cache handle.
#[derive(Clone)]
pub struct Client {
    store: Store,
    config: Arc<Config>,
}

impl Client {
    pub fn new(config: Config) -> Result<Self> {
        if config.cache_dir.as_os_str().is_empty() {
            let why = crate::home::resolve(None).err().map_or_else(|| "it is empty".into(), |e| e.detail);
            return fail(Kind::Invalid, format!("name cache_dir: {why}"));
        }
        let store = Store::open(&config.cache_dir)?;
        Ok(Self {
            store,
            config: Arc::new(config),
        })
    }

    /// One path for a plain folder, local identity, digest, or remote repository.
    /// Lazy opens return metadata only; eager opens await this checkpoint's bytes.
    pub async fn open(&self, name_or_path: &str, options: &Open) -> Result<Checkpoint> {
        let path = Path::new(name_or_path);
        let constraints = crate::precision::Constraints {
            bits: options.lane,
            gpus: match &options.gpus {
                Some(gpus) => gpus.clone(),
                None if options.variant.is_some() || path.is_file() => vec![],
                None => platform::gpu::probe(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref())
                    .map_err(|e| Failure { kind: Kind::Invalid, detail: format!("query GPUs: {e}") })?
                    .into_iter().map(|g| g.compute_capability.map(|(a,b)| format!("{a}.{b}")).unwrap_or_default()).collect(),
            },
            ..Default::default()
        };
        constraints.check()?; // Even an explicit variant's.
        let checkpoint = if path.exists() {
            let path = path
                .canonicalize()
                .map_err(|e| io_failure(path.display(), e))?;
            let selected = if path.is_dir() && options.variant.is_none() {
                let mut profiles = Vec::new();
                for name in crate::store::folder_files(&path, "")? {
                    if crate::format::variant_of(&name).is_some() {
                        let bytes = std::fs::read(path.join(&name)).map_err(|e| io_failure(&name, e))?;
                        let header = crate::format::Header::from_cbor(&bytes)?;
                        profiles.push((name, crate::precision::Profile::of(&header)));
                    }
                }
                let candidates: Vec<_> = profiles.iter().map(|(n,p)| (n.as_str(),p)).collect();
                let (at, reasons) = constraints.select(name_or_path, &candidates)?;
                for reason in reasons { tracing::warn!("{reason}"); }
                Some(profiles[at].0.clone())
            } else { options.variant.clone() };
            let raw = crate::store::Checkpoint::folder(
                &path,
                selected.as_deref(),
                Some(&self.store),
            )?;
            // A header file is a handle into the cache, rooted where it is opened; the cache's
            // own files are not.
            if path.is_file() && !path.starts_with(self.store.root()) {
                self.store.root_file(&path, raw.id())?;
            }
            let source = remote::Remote {
                store: self.store.clone(),
                endpoint: Endpoint::with_config(
                    options.endpoint.as_deref().unwrap_or(&self.config.endpoint),
                    &self.config,
                )?,
                mirror: self.config.mirror.clone(),
                offline: options.local_files_only,
            };
            Checkpoint::local(raw, source, &path)?
        } else {
            let mut reference: Ref = name_or_path.parse()?;
            if let Some(revision) = &options.revision {
                reference.revision = Some(revision.clone());
            }
            if let Some(variant) = &options.variant {
                reference.variant = Some(variant.clone());
            }
            reference.endpoint = options.endpoint.clone();
            let source = self.source(&reference, options.local_files_only)?;
            source.open(reference, &constraints).await?
        };
        // Every open answers to the one rule: a variant named, or a header by its id, is its sole
        // candidate (spec §7). A GPU-less read takes any; asked bits or GPUs it cannot use refuse.
        let named = checkpoint.variant().unwrap_or(name_or_path).to_owned();
        constraints.select(name_or_path, &[(&named, &crate::precision::Profile::of(checkpoint.header()))])?;
        if !options.lazy {
            checkpoint.download(&Download::default()).await?;
        }
        Ok(checkpoint)
    }

    /// Holds every fetched object while building the view. A `local_dir` belongs to the caller.
    pub async fn materialize(
        &self,
        reference: &Ref,
        want: &Download<'_>,
        local_dir: Option<&Path>,
        local_files_only: bool,
    ) -> Result<PathBuf> {
        let source = self.source(reference, local_files_only)?;
        source.snapshot(reference, want, local_dir).await
    }

    pub async fn download_file(
        &self,
        reference: &Ref,
        filename: &str,
        local_dir: Option<&Path>,
        local_files_only: bool,
    ) -> Result<PathBuf> {
        let source = self.source(reference, local_files_only)?;
        source.file(reference, filename, local_dir).await
    }

    /// Commits the folder's files on top of tag `reference.revision` (none: [`fetch::MAIN`]) of
    /// the repository, as Hugging Face's `upload_folder`: the tag's tree keeps its other files,
    /// loses the paths a `delete_patterns` glob matches, and takes the folder's files, added or
    /// replaced. A new tag is private; an existing one keeps its flags. The folder may be thin
    /// (headers and repo files): a chunk or asset it lacks comes from the store, and a remote
    /// repository refuses only one the endpoint lacks too. The commit.
    pub async fn upload_folder(
        &self,
        reference: &Ref,
        folder: &Path,
        delete_patterns: &[&str],
        progress: Option<&Progress<'_>>,
    ) -> Result<ObjectRef> {
        let tag = reference.revision.as_deref().unwrap_or(crate::fetch::MAIN);
        if !crate::format::tag_name(tag) {
            return fail(Kind::Invalid, format!("{tag:?} is not a tag name"));
        }
        let (tree, absent) = self.store.admit_folder(folder)?;
        if let (Some(object), true) = (absent.first(), reference.repo_id.starts_with("local/")) {
            return fail(
                Kind::NotFound,
                format!("{}: in neither the folder nor the store", object.id()),
            );
        }
        let parsed = self.store.tree(&tree)?;
        let mut objects = vec![tree.clone()];
        objects.extend(parsed.files.values().cloned());
        for (_, header) in parsed.variants() {
            objects.extend(self.store.checkpoint(header)?.header().objects().cloned());
        }
        let _hold = self.store.hold(&objects)?;
        let total = objects.iter().map(|object| object.length).sum();
        if let Some(report) = progress {
            report(0, total)?;
        }
        // A local repository's endpoint is this store: one commit path for both.
        let source = self.source(reference, false)?;
        let control = crate::fetch::UploadControl { cancelled: &|| false, progress: &|_, _| {}, commit: &|| true };
        let uploaded = crate::fetch::upload_tree_with(
            &source.endpoint.inner,
            &reference.repo_id,
            &tree,
            &self.store,
            tag,
            delete_patterns,
            &control,
        )
        .await
        .map_err(provider::from_native)?;
        if let Some(report) = progress {
            report(total, total)?;
        }
        Ok(uploaded.commit)
    }

    fn source(&self, reference: &Ref, offline: bool) -> Result<remote::Remote> {
        let url = if reference.repo_id.starts_with("local/") {
            "local"
        } else {
            reference
                .endpoint
                .as_deref()
                .unwrap_or(&self.config.endpoint)
        };
        let endpoint = Endpoint::with_config(url, &self.config)?;
        Ok(remote::Remote {
            store: self.store.clone(),
            endpoint,
            mirror: self.config.mirror.clone(),
            offline: offline || reference.repo_id.starts_with("local/"),
        })
    }
}

fn io_failure(context: impl fmt::Display, error: std::io::Error) -> Failure {
    Failure {
        kind: if error.kind() == std::io::ErrorKind::NotFound {
            Kind::NotFound
        } else {
            Kind::Io
        },
        detail: format!("{context}: {error}"),
    }
}

/// Output supported by the existing serializer.
#[derive(Clone, Copy, Debug)]
pub enum Format {
    Safetensors,
    Cozytensors,
}
