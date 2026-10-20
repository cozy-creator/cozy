use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use crate::failure::{fail, Failure, Kind, Result};
use crate::fetch::{self, Fetcher, ObjectStore, Origin, VolumeOrigin, Want};
use crate::format::{Header, ObjectRef, Tree};
use crate::store2::Store;

use super::provider::from_native;
use super::{io_failure, Checkpoint, Endpoint, Fetch, Fetched, Ref};

#[derive(Clone)]
pub(super) struct Remote {
    pub store: Store,
    pub endpoint: Endpoint,
    pub mirror: Option<PathBuf>,
    pub offline: bool,
}

impl Remote {
    pub async fn fetch(&self, objects: &[ObjectRef], want: &Fetch<'_>) -> Result<Fetched> {
        let unique: BTreeMap<_, _> = objects.iter().map(|o| (&o.sha256, o.clone())).collect();
        let objects: Vec<_> = unique.into_values().collect();
        let _hold = self.store.cache_hold(&objects)?;
        let total = objects.iter().map(|o| o.length).sum();
        if let Some(report) = want.progress {
            report(0, total)?;
        }
        if self.offline && self.mirror.is_none() {
            for object in &objects {
                if !self.store.reuse(object)? {
                    return fail(Kind::NotFound, format!("{} is not cached", object.id()));
                }
            }
            if let Some(report) = want.progress {
                report(total, total)?;
            }
            return Ok(Fetched {
                local: total,
                ..Fetched::default()
            });
        }
        let store: Arc<dyn ObjectStore> = Arc::new(self.store.clone());
        let mirror = self
            .mirror
            .as_ref()
            .map(|p| VolumeOrigin::new(p, Duration::from_secs(24 * 60 * 60)));
        let (origin, volume): (Arc<dyn Origin>, _) = if self.offline {
            // A configured mirror is a local-files source, not permission to
            // contact the remote endpoint. Use the existing verified downloader
            // with a file-only origin and no network fallback.
            (Arc::new(mirror.expect("offline source has a mirror")), None)
        } else {
            (self.endpoint.inner.clone(), mirror)
        };
        // Constructing/dropping this fetcher scopes cancellation to this operation.
        // All network tasks run on the caller's Tokio runtime, never a nested one.
        let fetcher = Fetcher::new(
            store,
            self.endpoint.transport.clone(),
            volume,
            fetch::Config::default(),
        );
        let progress = fetch::Progress::default();
        let selected = Want {
            priority: want.priority,
            progress: Some(&progress),
            ..Want::default()
        };
        let mut work = std::pin::pin!(fetcher.objects(&origin, &objects, &selected));
        let mut tick = tokio::time::interval(Duration::from_millis(250));
        let result = loop {
            tokio::select! {
                biased;
                result = &mut work => break result.map_err(from_native)?,
                _ = tick.tick(), if want.progress.is_some() => {
                    if let Some(report) = want.progress {
                        report(progress.held.load(Ordering::Relaxed), total)?;
                    }
                }
            }
        };
        fetcher.settled().await;
        if let Some(report) = want.progress {
            report(total, total)?;
        }
        Ok(result)
    }

    async fn root(&self, reference: &Ref) -> Result<(ObjectRef, Option<String>)> {
        let digest = reference
            .revision
            .as_deref()
            .filter(|r| r.starts_with("sha256:"))
            .or_else(|| {
                reference
                    .repo_id
                    .starts_with("sha256:")
                    .then_some(reference.repo_id.as_str())
            });
        if let Some(digest) = digest {
            let hex = digest.trim_start_matches("sha256:");
            if !super::is_digest(hex) {
                return fail(Kind::Invalid, "invalid SHA256 reference");
            }
            let path = self.store.path(&ObjectRef {
                sha256: hex.to_owned(),
                length: 0,
            })?;
            match std::fs::metadata(&path) {
                Ok(metadata) => {
                    return Ok((
                        ObjectRef {
                            sha256: hex.to_owned(),
                            length: metadata.len(),
                        },
                        None,
                    ))
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(io_failure(path.display(), error)),
            }
            if self.offline {
                if let Some(mirror) = &self.mirror {
                    let object = ObjectRef {
                        sha256: hex.to_owned(),
                        length: 0,
                    };
                    let path = mirror.join(object.key());
                    let metadata = std::fs::metadata(&path)
                        .map_err(|error| io_failure(path.display(), error))?;
                    return Ok((
                        ObjectRef {
                            length: metadata.len(),
                            ..object
                        },
                        None,
                    ));
                }
                return fail(Kind::NotFound, format!("{digest} is not cached"));
            }
            let object = self
                .endpoint
                .inner
                .object(hex)
                .await
                .map_err(from_native)?
                .ok_or_else(|| Failure {
                    kind: Kind::NotFound,
                    detail: format!("no object {digest}"),
                })?;
            return Ok((object, None));
        }
        let cached = || -> Result<Option<(ObjectRef, Option<String>)>> {
            let Some(repo) = self.store.get_ref(&reference.repo_id)? else {
                return Ok(None);
            };
            Ok(fetch::release_of(&repo, reference.revision.as_deref())
                .ok()
                .map(|(version, tree)| (tree, Some(version))))
        };
        if reference.revision.is_some() || self.offline {
            if let Some(found) = cached()? {
                return Ok(found);
            }
        }
        let local;
        let endpoint = if self.offline {
            let Some(mirror) = &self.mirror else {
                return fail(
                    Kind::NotFound,
                    format!("{} is not cached", reference.repo_id),
                );
            };
            // Keep release selection and parsing in the existing endpoint. An
            // absolute directory cannot be interpreted as an HTTP endpoint, and
            // carries no provider even if Config supplies one for the Hub.
            let path =
                std::path::absolute(mirror).map_err(|error| io_failure(mirror.display(), error))?;
            let path = path.to_str().ok_or_else(|| Failure {
                kind: Kind::Invalid,
                detail: "mirror path is not UTF-8".into(),
            })?;
            local = Endpoint {
                inner: Arc::new(
                    fetch::Endpoint::new(&self.endpoint.transport, path, None)
                        .map_err(from_native)?,
                ),
                transport: self.endpoint.transport.clone(),
            };
            &local
        } else {
            &self.endpoint
        };
        let repo = match endpoint.inner.repo_attempt(&reference.repo_id).await {
            Ok(Some((repo, _))) => repo,
            Ok(None) => return fail(Kind::NotFound, format!("no repo {}", reference.repo_id)),
            Err(error) => {
                let transport = matches!(&error, fetch::GetError::Exchange(_));
                let error = from_native(error.refusal());
                if transport && error.kind == Kind::Network {
                    if let Some(found) = cached()? {
                        if self.store.reuse(&found.0)? {
                            return Ok(found);
                        }
                    }
                }
                return Err(error);
            }
        };
        let (version, tree) =
            fetch::release_of(&repo, reference.revision.as_deref()).map_err(from_native)?;
        Ok((tree, Some(version)))
    }

    fn remember(&self, reference: &Ref, root: &ObjectRef, version: Option<&str>) -> Result<()> {
        if let Some(version) = version {
            if let Some(repo) = self.store.update_ref(&reference.repo_id, |old| {
                Some(fetch::with_tree(
                    old,
                    &reference.repo_id,
                    root,
                    Some(version),
                ))
            })? {
                self.store.cache_ref(&repo)?;
            }
        }
        Ok(())
    }

    pub async fn open(&self, mut reference: Ref) -> Result<Checkpoint> {
        let (root, version) = self.root(&reference).await?;
        let _root_hold = self.store.cache_hold(std::slice::from_ref(&root))?;
        self.fetch(std::slice::from_ref(&root), &Fetch::default())
            .await?;
        let bytes = self.store.read(&root)?;
        let (tree, header, variant, mut extra) = match Tree::from_json(&bytes) {
            Ok(tree) => {
                let (variant, header) = tree.variant(reference.variant.as_deref())?;
                let extra = tree.repo_files().map(|(_, r)| r.clone()).collect();
                (
                    Some(root.clone()),
                    header.clone(),
                    Some(variant.to_owned()),
                    extra,
                )
            }
            Err(_) => {
                Header::from_cbor(&bytes)?;
                (None, root.clone(), reference.variant.clone(), Vec::new())
            }
        };
        let _metadata_hold = self.store.cache_hold(std::slice::from_ref(&header))?;
        self.fetch(std::slice::from_ref(&header), &Fetch::default())
            .await?;
        let raw = match &tree {
            Some(tree) => self.store.variant(tree, variant.as_deref())?,
            None => self.store.checkpoint(&header)?,
        };
        if tree.is_some() {
            self.remember(&reference, &root, version.as_deref())?;
        }
        reference.revision = version.or(reference.revision);
        reference.variant = variant;
        if let Some(tree) = &tree {
            extra.push(tree.clone());
        }
        let variant = reference.variant.clone();
        let named = (!reference.repo_id.starts_with("sha256:")).then_some(reference);
        Checkpoint::managed(
            raw,
            named,
            tree,
            Some(self.clone()),
            extra,
            self.store.clone(),
            variant,
        )
    }

    pub async fn snapshot(
        &self,
        reference: &Ref,
        want: &Fetch<'_>,
        local_dir: Option<&Path>,
    ) -> Result<PathBuf> {
        let (root, version) = self.root(reference).await?;
        let _root_hold = self.store.cache_hold(std::slice::from_ref(&root))?;
        self.fetch(std::slice::from_ref(&root), &Fetch::default())
            .await?;
        let tree = self.store.tree(&root)?;
        let headers: Vec<_> = match reference.variant.as_deref() {
            Some(variant) => vec![tree.variant(Some(variant))?.1.clone()],
            None => tree.variants().map(|(_, h)| h.clone()).collect(),
        };
        let _header_hold = self.store.cache_hold(&headers)?;
        self.fetch(&headers, &Fetch::default()).await?;
        let mut payload = Vec::new();
        for header in &headers {
            payload.extend(super::selection::objects(
                &self.store.checkpoint(header)?,
                &want.selection,
            )?);
        }
        let mut objects: Vec<_> = tree.repo_files().map(|(_, r)| r.clone()).collect();
        objects.extend(payload.iter().cloned());
        let _object_hold = self.store.cache_hold(&objects)?;
        self.fetch(&objects, want).await?;
        self.remember(reference, &root, version.as_deref())?;
        let identity = match &want.selection {
            super::Selection::All => root.sha256.clone(),
            _ => crate::sha256::hex_digest(
                format!(
                    "{}:{:?}:{:?}",
                    root.sha256, reference.variant, want.selection
                )
                .as_bytes(),
            ),
        };
        let dest = local_dir
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.store.root().join("snapshots").join(identity));
        let _directory = if local_dir.is_none() {
            Some(self.store.cache_directory(&dest)?)
        } else {
            self.store.cache_preserve_snapshot(&dest)?;
            None
        };
        // Materialize the selected closure, not every payload that happens to be
        // present in the shared cache. Existing materialize owns all copying.
        std::fs::create_dir_all(dest.join(crate::format::CHUNKS_DIR))
            .map_err(|e| io_failure(dest.display(), e))?;
        for (name, object) in tree.repo_files() {
            self.store.materialize(object, &dest.join(name))?;
        }
        for (variant, header) in tree
            .variants()
            .filter(|(v, _)| reference.variant.as_deref().is_none_or(|want| want == *v))
        {
            self.store
                .materialize(header, &dest.join(crate::format::variant_path(variant)))?;
        }
        for object in &payload {
            self.store.materialize(
                object,
                &dest.join(crate::format::CHUNKS_DIR).join(&object.sha256),
            )?;
        }
        if local_dir.is_none() {
            self.store.cache_snapshot(&dest)?;
        }
        self.store.cache_touch(&root)?;
        let _ = self.store.cache_maintain(0)?;
        Ok(dest)
    }

    pub async fn file(
        &self,
        reference: &Ref,
        filename: &str,
        local_dir: Option<&Path>,
    ) -> Result<PathBuf> {
        let (root, version) = self.root(reference).await?;
        let _root_hold = self.store.cache_hold(std::slice::from_ref(&root))?;
        self.fetch(std::slice::from_ref(&root), &Fetch::default())
            .await?;
        let tree = self.store.tree(&root)?;
        let object = tree.files.get(filename).ok_or_else(|| Failure {
            kind: Kind::NotFound,
            detail: format!("no repo file {filename:?}"),
        })?;
        let _object_hold = self.store.cache_hold(std::slice::from_ref(object))?;
        self.fetch(std::slice::from_ref(object), &Fetch::default())
            .await?;
        self.remember(reference, &root, version.as_deref())?;
        let folder = local_dir
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.store.root().join("snapshots").join(&root.sha256));
        let _directory = if local_dir.is_none() {
            Some(self.store.cache_directory(&folder)?)
        } else {
            self.store.cache_preserve_snapshot(&folder)?;
            None
        };
        let dest = folder.join(filename);
        self.store.materialize(object, &dest)?;
        if local_dir.is_none() {
            self.store.cache_snapshot(&folder)?;
        }
        self.store.cache_touch(&root)?;
        let _ = self.store.cache_maintain(0)?;
        Ok(dest)
    }
}
