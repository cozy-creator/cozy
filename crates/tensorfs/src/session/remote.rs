use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use crate::failure::{fail, Failure, Kind, Result};
use crate::fetch::{self, Fetcher, ObjectStore, Origin, VolumeOrigin, Want};
use crate::format::{Header, ObjectRef, Repo, Tag, Tree};
use crate::store::Store;

use super::provider::from_native;
use super::{io_failure, Checkpoint, Endpoint, Download, Fetched, Ref};

/// What a reference names: the root object (a commit, or a bare header), the tag it resolved
/// through, and the variant a header digest names.
struct Root {
    object: ObjectRef,
    tag: Option<Tag>,
    variant: Option<String>,
}

#[derive(Clone)]
pub(super) struct Remote {
    pub store: Store,
    pub endpoint: Endpoint,
    pub mirror: Option<PathBuf>,
    pub offline: bool,
}

impl Remote {
    pub async fn fetch(&self, objects: &[ObjectRef], want: &Download<'_>) -> Result<Fetched> {
        let unique: BTreeMap<_, _> = objects.iter().map(|o| (&o.sha256, o.clone())).collect();
        let objects: Vec<_> = unique.into_values().collect();
        let _hold = self.store.hold(&objects)?;
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

    /// A commit's tree, fetched into the store.
    async fn tree(&self, commit: &ObjectRef) -> Result<Tree> {
        self.fetch(std::slice::from_ref(commit), &Download::default()).await?;
        Ok(self.store.tree(commit)?)
    }

    /// What `reference` names: a bare object (`sha256:` repo id), or its revision resolved once
    /// through the repo document (spec §7).
    async fn root(&self, reference: &Ref) -> Result<Root> {
        if let Some(hex) = reference.repo_id.strip_prefix("sha256:") {
            return Ok(Root { object: self.object(hex).await?, tag: None, variant: None });
        }
        let revision = reference.revision.as_deref();
        // The local record of tags this store resolved before; its trees are held.
        let cached = || -> Result<Option<Root>> {
            let Some(repo) = self.store.get_ref(&reference.repo_id)? else {
                return Ok(None);
            };
            let found = repo.resolve(revision).map(|r| r.find(|commit| self.store.tree(commit)));
            Ok(match found {
                Some(Ok(Some((tag, variant)))) => Some(Root { object: tag.commit.clone(), tag: Some(tag.clone()), variant }),
                _ => None,
            })
        };
        // A digest names immutable content; a tag moves, so only offline is it read from the record.
        if self.offline || revision.is_some_and(|r| r.starts_with("sha256:")) {
            if let Some(found) = cached()? {
                return Ok(found);
            }
        }
        let local;
        let endpoint = if self.offline {
            let Some(mirror) = &self.mirror else {
                return fail(Kind::NotFound, format!("{} is not cached", reference.repo_id));
            };
            // An absolute directory cannot be interpreted as an HTTP endpoint, and carries no
            // provider even if Config supplies one for the Hub.
            let path = std::path::absolute(mirror).map_err(|error| io_failure(mirror.display(), error))?;
            let path = path.to_str().ok_or_else(|| Failure { kind: Kind::Invalid, detail: "mirror path is not UTF-8".into() })?;
            local = Endpoint {
                inner: Arc::new(fetch::Endpoint::new(&self.endpoint.transport, path, None).map_err(from_native)?),
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
                        if self.store.reuse(&found.object)? {
                            return Ok(found);
                        }
                    }
                }
                return Err(error);
            }
        };
        match fetch::resolve(&repo, revision, async |commit: &ObjectRef| self.tree(commit).await).await {
            Ok((tag, variant)) => Ok(Root { object: tag.commit.clone(), tag: Some(tag), variant }),
            Err(error) if error.kind == Kind::NotFound && revision.is_some_and(|r| r.starts_with("sha256:")) => {
                // Tags do not enumerate retained history. An explicit immutable commit can
                // still be readable at this endpoint; never substitute the tagged default.
                let digest = revision.expect("digest revision");
                let Some(object) = endpoint.inner.object(digest).await.map_err(from_native)? else { return Err(error) };
                self.tree(&object).await?;
                Ok(Root { object, tag: None, variant: None })
            }
            Err(error) => Err(error),
        }
    }

    /// A bare object by digest: from the store, the offline mirror, or the endpoint.
    async fn object(&self, hex: &str) -> Result<ObjectRef> {
        let at = |length| ObjectRef { sha256: hex.to_owned(), length };
        let path = self.store.path(&at(0))?;
        match std::fs::metadata(&path) {
            Ok(metadata) => return Ok(at(metadata.len())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(io_failure(path.display(), error)),
        }
        if self.offline {
            let Some(mirror) = &self.mirror else {
                return fail(Kind::NotFound, format!("sha256:{hex} is not cached"));
            };
            let path = mirror.join(at(0).key());
            let metadata = std::fs::metadata(&path).map_err(|error| io_failure(path.display(), error))?;
            return Ok(at(metadata.len()));
        }
        let object = self.endpoint.inner.object(hex).await.map_err(from_native)?;
        object.ok_or_else(|| Failure { kind: Kind::NotFound, detail: format!("no object sha256:{hex}") })
    }

    /// Records the tag a remote resolution went through, so the store resolves it again offline.
    fn remember(&self, reference: &Ref, tag: Option<&Tag>) -> Result<()> {
        if let Some(tag) = tag.filter(|_| !reference.repo_id.starts_with("local/")) {
            self.store.update_ref(&reference.repo_id, |old| {
                let mut repo = old.unwrap_or_else(|| Repo::new(&reference.repo_id));
                match repo.tag_mut(&tag.name) {
                    Some(held) => *held = tag.clone(),
                    None => repo.tags.push(tag.clone()),
                }
                Some(repo)
            })?;
        }
        Ok(())
    }

    pub async fn open(&self, mut reference: Ref, constraints: &crate::precision::Constraints) -> Result<Checkpoint> {
        let Root { object: root, tag, variant } = self.root(&reference).await?;
        let wanted = fetch::pick_variant(reference.revision.as_deref(), variant, reference.variant.as_deref())?;
        let _root_hold = self.store.hold(std::slice::from_ref(&root))?;
        self.fetch(std::slice::from_ref(&root), &Download::default())
            .await?;
        let bytes = self.store.read(&root)?;
        let (tree, header, variant, mut extra) = match Tree::from_json(&bytes) {
            Ok(tree) => {
                let mut held_profiles = true;
                for (_, header) in tree.variants() { held_profiles &= self.store.reuse(header)?; }
                let indexed = if !held_profiles && !self.offline && wanted.is_none() && !reference.repo_id.starts_with("local/") && !reference.repo_id.starts_with("sha256:") {
                    self.endpoint.inner.select(&reference.repo_id, &root.id(), constraints).await.map_err(from_native)?
                } else { None };
                let selected;
                let wanted = if let Some(answer) = &indexed {
                    if answer.commit != root { return fail(Kind::Integrity, "selection names a different commit"); }
                    let Some(name) = &answer.variant else { return fail(Kind::NotFound, "no variant fits the requested GPUs and bits"); };
                    if tree.header(name) != answer.header.as_ref() { return fail(Kind::Integrity, "selection header does not belong to its variant"); }
                    for reason in &answer.reasons { tracing::warn!("{reason}"); }
                    Some(name.as_str())
                } else if wanted.is_none() {
                    let headers: Vec<_> = tree.variants().map(|(_,h)| h.clone()).collect();
                    let _headers_hold = self.store.hold(&headers)?;
                    self.fetch(&headers, &Download::default()).await?;
                    let profiles: Vec<_> = tree.variants().map(|(n,h)| {
                        Ok((n, crate::precision::Profile::of(&Header::from_cbor(&self.store.read(h)?)?)))
                    }).collect::<Result<_>>()?;
                    let candidates: Vec<_> = profiles.iter().map(|(n,p)| (*n,p)).collect();
                    let Some((at,reasons)) = constraints.select(&reference.repo_id, &candidates)? else {
                        return fail(Kind::NotFound, "no variant fits the requested GPUs and bits");
                    };
                    for reason in reasons { tracing::warn!("{reason}"); }
                    selected = profiles[at].0.to_owned();
                    Some(selected.as_str())
                } else { wanted.as_deref() };
                let (variant, header) = tree.variant(wanted).map_err(|e| absent(wanted.is_some(), e))?;
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
        let _metadata_hold = self.store.hold(std::slice::from_ref(&header))?;
        self.fetch(std::slice::from_ref(&header), &Download::default())
            .await?;
        let raw = match &tree {
            Some(tree) => self.store.variant(tree, variant.as_deref())?,
            None => self.store.checkpoint(&header)?,
        };
        if tree.is_some() {
            self.remember(&reference, tag.as_ref())?;
        }
        reference.revision = tag.map(|t| t.name).or(reference.revision);
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
        want: &Download<'_>,
        local_dir: Option<&Path>,
    ) -> Result<PathBuf> {
        let Root { object: root, tag, variant } = self.root(reference).await?;
        let wanted = fetch::pick_variant(reference.revision.as_deref(), variant, reference.variant.as_deref())?;
        let _root_hold = self.store.hold(std::slice::from_ref(&root))?;
        self.fetch(std::slice::from_ref(&root), &Download::default())
            .await?;
        let tree = self.store.tree(&root)?;
        let wanted = wanted.as_deref().map(|name| tree.variant(Some(name)).map(|(path,_)|path.to_owned()).map_err(|e| absent(true,e))).transpose()?;
        let headers: Vec<_> = match wanted.as_deref() {
            Some(variant) => vec![tree.variant(Some(variant)).map_err(|e| absent(true, e))?.1.clone()],
            None => tree.variants().map(|(_, h)| h.clone()).collect(),
        };
        let _header_hold = self.store.hold(&headers)?;
        self.fetch(&headers, &Download::default()).await?;
        let mut payload = Vec::new();
        for header in &headers {
            payload.extend(super::selection::objects(
                &self.store.checkpoint(header)?,
                &want.selection,
            )?);
        }
        let mut objects: Vec<_> = tree.repo_files().map(|(_, r)| r.clone()).collect();
        objects.extend(payload.iter().cloned());
        let _object_hold = self.store.hold(&objects)?;
        self.fetch(&objects, want).await?;
        self.remember(reference, tag.as_ref())?;
        let identity = match &want.selection {
            super::Selection::All => root.sha256.clone(),
            _ => crate::sha256::hex_digest(
                format!("{}:{:?}:{:?}", root.sha256, wanted, want.selection).as_bytes(),
            ),
        };
        let used = std::iter::once(&root).chain(&headers).chain(&objects);
        self.store.used(&identity, used)?;
        let dest = local_dir
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.store.root().join("snapshots").join(identity));
        // The selected closure, not every payload that happens to be in the shared cache.
        let chosen: Vec<_> = (tree.variants())
            .filter(|(v, _)| wanted.as_deref().is_none_or(|want| want == *v))
            .collect();
        self.store.write_folder(&tree, &dest, &chosen, &payload)?;
        self.store.maintain(0)?;
        Ok(dest)
    }

    pub async fn file(
        &self,
        reference: &Ref,
        filename: &str,
        local_dir: Option<&Path>,
    ) -> Result<PathBuf> {
        let Root { object: root, tag, .. } = self.root(reference).await?;
        let _root_hold = self.store.hold(std::slice::from_ref(&root))?;
        self.fetch(std::slice::from_ref(&root), &Download::default())
            .await?;
        let tree = self.store.tree(&root)?;
        let object = tree.files.get(filename).ok_or_else(|| Failure {
            kind: Kind::NotFound,
            detail: format!("no repo file {filename:?}"),
        })?;
        let _object_hold = self.store.hold(std::slice::from_ref(object))?;
        self.fetch(std::slice::from_ref(object), &Download::default())
            .await?;
        self.remember(reference, tag.as_ref())?;
        self.store.used(&root.sha256, [&root, object])?;
        let folder = local_dir
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.store.root().join("snapshots").join(&root.sha256));
        let dest = folder.join(filename);
        self.store.materialize(object, &dest)?;
        self.store.maintain(0)?;
        Ok(dest)
    }
}

/// A named variant the commit lacks is not found; an unnamed choice among several is invalid.
fn absent(named: bool, e: crate::format::Error) -> Failure {
    let kind = if named { Kind::NotFound } else { Kind::Invalid };
    Failure { kind, detail: e.to_string() }
}
