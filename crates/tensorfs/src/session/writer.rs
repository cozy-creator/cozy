use std::collections::BTreeSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::failure::{fail, Failure, Kind, Result};
use crate::format::{Config, Dtype, Encoding, ObjectRef, Tree, Value};
use crate::store2::Store;
use crate::writer::{self as native, Data, Source};

use super::{Checkpoint, Client, Component, Endpoint, Ref, TokenProvider};

/// Input for an explicitly encoded tensor, exposed by the public raw namespace.
pub struct PartData<'a> {
    pub role: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data: &'a mut dyn Read,
}

#[derive(Default)]
pub struct WriterOptions<'a> {
    pub dest: Option<&'a Path>,
    pub variant: Option<&'a str>,
    pub base: Option<&'a Checkpoint>,
    pub resume: Option<&'a str>,
    pub repo_id: Option<&'a str>,
    pub endpoint: Option<&'a str>,
    pub token: Option<Arc<dyn TokenProvider>>,
    pub ca_der: Option<&'a [u8]>,
}

#[derive(Clone)]
struct Context {
    store: Store,
    dest: Option<PathBuf>,
    variant: String,
    repo_id: Option<String>,
    endpoint: Option<String>,
}

struct State {
    native: Option<native::Writer>,
    committing: bool,
    committed: Option<Checkpoint>,
    declared: BTreeSet<String>,
}

/// The native writer owns all progress roots. The finalizer retains them until its flush ends.
pub struct Writer {
    state: Arc<Mutex<State>>,
    context: Context,
}

pub struct ComponentWriter<'a> {
    writer: &'a mut Writer,
    name: String,
}

fn from_writer(error: crate::store2::Error) -> Failure {
    if let crate::store2::Error::Io { source, .. } = &error {
        if let Some(error) = source
            .get_ref()
            .and_then(|e| e.downcast_ref::<crate::err::Refusal>())
        {
            return super::provider::from_native(error.clone());
        }
    }
    error.into()
}

impl Client {
    pub fn writer(&self, options: WriterOptions<'_>) -> Result<Writer> {
        if options.dest.is_some() && options.repo_id.is_some() {
            return fail(
                Kind::Invalid,
                "a Writer names a header file or an endpoint repository",
            );
        }
        if options.dest.is_some() && options.variant.is_some() {
            return fail(Kind::Invalid, "a literal header file has no variant label");
        }
        let variant = options.variant.unwrap_or("").to_owned();
        crate::format::check_file(&crate::format::variant_path(&variant))?;
        let store = self.store.clone();
        let declared = options
            .base
            .map(|base| {
                base.components()
                    .iter()
                    .map(|c| c.name().to_owned())
                    .collect()
            })
            .unwrap_or_default();
        let base = options.base.map(Checkpoint::stored);
        let dest = options
            .dest
            .map(std::path::absolute)
            .transpose()
            .map_err(|e| super::io_failure("writer destination", e))?;
        let mut destination_endpoint = None;
        let raw = if let Some(repo_id) = options.repo_id {
            let reference: Ref = repo_id.parse()?;
            if reference.repo_id.starts_with("sha256:")
                || reference.revision.is_some()
                || reference.variant.is_some()
            {
                return fail(
                    Kind::Invalid,
                    "Writer repo_id names a repository, without a revision",
                );
            }
            let mut config = (*self.config).clone();
            if let Some(token) = options.token {
                config.token = Some(token);
            }
            if let Some(ca) = options.ca_der {
                config.ca_der = Some(ca.to_vec());
            }
            let url = if repo_id.starts_with("local/") {
                "local"
            } else {
                options.endpoint.unwrap_or(&config.endpoint)
            };
            let endpoint = Endpoint::with_config(url, &config)?;
            if !repo_id.starts_with("local/") {
                destination_endpoint = Some(endpoint.inner.url().to_owned());
            }
            let runtime = tokio::runtime::Handle::try_current().map_err(|_| Failure {
                kind: Kind::Unsupported,
                detail: "a streaming Writer needs the caller's active Tokio runtime".into(),
            })?;
            let stream = Arc::new(crate::fetch::Stream::new(endpoint.inner, runtime));
            store
                .streaming_writer_resumable(stream, repo_id, &variant, base, options.resume)
                .map_err(from_writer)?
        } else {
            store.writer(base, options.resume).map_err(from_writer)?
        };
        Ok(Writer {
            state: Arc::new(Mutex::new(State {
                native: Some(raw),
                committing: false,
                committed: None,
                declared,
            })),
            context: Context {
                store,
                dest,
                variant,
                repo_id: options.repo_id.map(str::to_owned),
                endpoint: destination_endpoint,
            },
        })
    }
}

impl Context {
    fn checkpoint(
        &self,
        header: &ObjectRef,
        files: &std::collections::BTreeMap<String, ObjectRef>,
        published: Option<&ObjectRef>,
    ) -> Result<Checkpoint> {
        // First acquire the native header owner, then build its repository context.
        let raw = self.store.checkpoint(header)?;
        if self.dest.is_some() {
            // A literal header contains run data only. Its returned handle has the same
            // metadata as reopening that file, without inventing a repository context.
            return Checkpoint::managed(
                raw,
                None,
                None,
                None,
                Vec::new(),
                self.store.clone(),
                None,
            );
        }
        let mut parsed = Tree {
            files: files.clone(),
        };
        parsed.set_variant(&self.variant, header.clone());
        let tree = self.store.put_tree(&parsed)?;
        if published.is_some_and(|id| id != &tree) {
            return fail(
                Kind::Integrity,
                "published tree differs from the locally committed tree",
            );
        }
        let named = published
            .and_then(|_| self.repo_id.as_ref())
            .map(|name| Ref {
                repo_id: name.clone(),
                revision: None,
                variant: Some(self.variant.clone()),
                endpoint: self.endpoint.clone(),
            });
        let extra = std::iter::once(tree.clone())
            .chain(files.values().cloned())
            .collect();
        let checkpoint = Checkpoint::managed(
            raw,
            named,
            Some(tree.clone()),
            None,
            extra,
            self.store.clone(),
            Some(self.variant.clone()),
        )?;
        // The generated public tree has its own cache lifetime, independent of the header
        // and the native owner's flat repository-file root.
        self.store.cache_touch(&tree)?;
        if self.repo_id.is_some() && published.is_none() {
            // A failed upload is a durable, retryable local result after every live handle
            // drops. Its normal tree pin retains header, payload and repository bytes.
            self.store
                .pin_roots(&tree.sha256, std::slice::from_ref(&tree))?;
        }
        Ok(checkpoint)
    }

    fn publish_file(&self, checkpoint: &Checkpoint) -> Result<()> {
        if let Some(dest) = &self.dest {
            let header = checkpoint.id();
            // The content-addressed normal pin is durable independently of a resume key,
            // the path's lifetime, and any future overwrite of this header file.
            self.store
                .pin_roots(&header.sha256, std::slice::from_ref(header))?;
            let bytes = self.store.read(header)?;
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| super::io_failure(parent.display(), e))?;
            }
            platform::durable::replace(dest, &bytes, 0o644)
                .map_err(|e| super::io_failure(dest.display(), e.into()))?;
        }
        Ok(())
    }
}

impl Writer {
    fn with<T>(&mut self, change: impl FnOnce(&mut native::Writer) -> Result<T>) -> Result<T> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match state.native.as_mut() {
            Some(raw) => change(raw),
            None => fail(
                if state.committing {
                    Kind::Conflict
                } else {
                    Kind::Invalid
                },
                "the Writer is committing or already closed",
            ),
        }
    }

    pub fn pipeline(&mut self, class: &str, config: Config) -> Result<()> {
        self.with(|raw| {
            raw.pipeline(class, config);
            Ok(())
        })
    }
    pub fn component(
        &mut self,
        name: &str,
        class: &str,
        config: Config,
    ) -> Result<ComponentWriter<'_>> {
        self.with(|raw| {
            raw.component(name, class, config, None)
                .map(drop)
                .map_err(from_writer)
        })?;
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .declared
            .insert(name.to_owned());
        Ok(ComponentWriter {
            writer: self,
            name: name.to_owned(),
        })
    }
    /// Reopen a declared logical component without changing its class or config.
    pub fn component_writer(&mut self, name: &str) -> Result<ComponentWriter<'_>> {
        {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if state.native.is_none() {
                return fail(Kind::Conflict, "the Writer is committing or already closed");
            }
            if !state.declared.contains(name) {
                return fail(Kind::NotFound, format!("no declared component {name:?}"));
            }
        }
        Ok(ComponentWriter {
            writer: self,
            name: name.to_owned(),
        })
    }
    pub fn set_metadata(&mut self, key: &str, value: Value) -> Result<()> {
        self.with(|raw| {
            raw.set_metadata(key, value);
            Ok(())
        })
    }
    pub fn has(&self, component: &str, key: &str) -> Result<bool> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match &state.native {
            Some(raw) => Ok(raw.has(component, key)),
            None => fail(
                if state.committing {
                    Kind::Conflict
                } else {
                    Kind::Invalid
                },
                "the Writer is committing or already closed",
            ),
        }
    }

    /// Sealing/final publication is a transaction. Its owned worker keeps the result available
    /// through committed(), even if a caller drops the awaiting future before it settles.
    pub async fn commit(&mut self) -> Result<Checkpoint> {
        let raw = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let Some(raw) = state.native.take() else {
                return fail(
                    if state.committing {
                        Kind::Conflict
                    } else {
                        Kind::Invalid
                    },
                    "the Writer is committing or already closed",
                );
            };
            state.committing = true;
            raw
        };
        let (state, context) = (Arc::clone(&self.state), self.context.clone());
        let result = tokio::task::spawn_blocking(move || {
            let mut raw = raw;
            let result = raw.commit().map_err(from_writer);
            let local = raw
                .committed()
                .map(|header| context.checkpoint(header, raw.repo_files(), raw.published()))
                .transpose();
            let publication = match (&result, &local) {
                (Ok(_), Ok(Some(checkpoint))) => context.publish_file(checkpoint),
                _ => Ok(()),
            };
            let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
            state.committing = false;
            if let Ok(Some(checkpoint)) = &local {
                state.committed = Some(checkpoint.clone());
            }
            if result.is_err() {
                state.native = Some(raw);
            }
            result?;
            publication?;
            local?.ok_or_else(|| Failure {
                kind: Kind::Integrity,
                detail: "Writer sealed no local checkpoint".into(),
            })
        })
        .await;
        match result {
            Ok(result) => result,
            Err(error) => {
                self.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .committing = false;
                fail(Kind::Io, format!("Writer finalizer failed: {error}"))
            }
        }
    }

    pub fn committed(&self) -> Option<Checkpoint> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .committed
            .clone()
    }

    pub fn abandon(self) -> Result<()> {
        let raw = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if state.committing {
                return fail(
                    Kind::Conflict,
                    "cannot abandon a commit that is already sealing",
                );
            }
            state.native.take()
        };
        // No Context owns a Stream: dropping this final native writer aborts its queued tasks.
        if let Some(raw) = raw {
            raw.abandon();
        }
        Ok(())
    }
}

impl ComponentWriter<'_> {
    pub fn put(
        &mut self,
        key: &str,
        dtype: Dtype,
        shape: &[u64],
        data: &mut dyn Read,
    ) -> Result<()> {
        self.writer.with(|raw| {
            raw.get(&self.name)
                .put(key, &dtype, shape, Data::Plain(Source::Read(data)))
                .map_err(from_writer)
        })
    }
    pub fn put_encoded(
        &mut self,
        key: &str,
        dtype: Dtype,
        shape: &[u64],
        encoding: Encoding,
        parts: Vec<PartData<'_>>,
    ) -> Result<()> {
        let parts = parts
            .into_iter()
            .map(|p| native::PartData {
                role: p.role,
                dtype: p.dtype,
                shape: p.shape,
                data: Source::Read(p.data),
            })
            .collect();
        self.writer.with(|raw| {
            raw.get(&self.name)
                .put(key, &dtype, shape, Data::Encoded { encoding, parts })
                .map_err(from_writer)
        })
    }
    pub fn copy(&mut self, key: &str, source: &Component, as_key: Option<&str>) -> Result<()> {
        self.writer.with(|raw| {
            raw.get(&self.name)
                .copy_as(as_key.unwrap_or(key), source.stored(), source.name(), key)
                .map_err(from_writer)
        })
    }
    pub fn asset(&mut self, name: &str, data: &[u8]) -> Result<()> {
        self.writer
            .with(|raw| raw.get(&self.name).asset(name, data).map_err(from_writer))
    }
    pub fn order(&mut self, keys: &[String]) -> Result<()> {
        let keys = keys.iter().map(String::as_str).collect::<Vec<_>>();
        self.writer.with(|raw| {
            raw.get(&self.name).set_order(&keys);
            Ok(())
        })
    }
    pub fn has(&self, key: &str) -> Result<bool> {
        self.writer.has(&self.name, key)
    }
}
