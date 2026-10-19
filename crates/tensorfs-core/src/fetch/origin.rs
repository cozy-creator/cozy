//! Where objects come from. An origin names a model's tree (`resolve`) and says where its
//! objects can be read (`locate`); the Fetcher downloads, verifies and admits them.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;

use super::token::TokenProvider;
use crate::err::{refuse, Code, Refusal, Result};
use crate::format::{ObjectRef, Repo, Tree};

#[async_trait]
pub trait Origin: Send + Sync {
    /// The tree of `name` (`<org>/<model>`) at `release` (the latest when `None`), for its
    /// `variant` (an origin that serves one variant's closure needs it).
    async fn resolve(
        &self,
        name: &str,
        release: Option<&str>,
        variant: Option<&str>,
    ) -> Result<ObjectRef>;

    /// Where each of `objects` can be read, in order. The Fetcher asks once per object as it
    /// starts it, and again when a location expires or is refused; an origin whose answer
    /// costs a round trip batches and caches.
    async fn locate(&self, objects: &[ObjectRef]) -> Result<Vec<Location>>;

    /// Whether anyone may read these objects: only then are they written back to a shared
    /// volume.
    fn public(&self) -> bool;
}

/// Where one object can be read.
#[derive(Clone)]
pub enum Location {
    /// An HTTP(S) GET, with the headers it needs, valid until `expires`.
    Url {
        url: String,
        headers: Vec<(String, String)>,
        expires: Option<SystemTime>,
    },
    /// A provider presents each GET afresh, including same-origin redirects and restarts.
    Authenticated {
        url: String,
        provider: Arc<dyn TokenProvider>,
    },
    /// A file on a mounted volume.
    File(PathBuf),
}

pub(crate) struct HttpLocation<'a> {
    pub url: &'a str,
    pub headers: &'a [(String, String)],
    pub provider: Option<&'a dyn TokenProvider>,
}

impl Location {
    pub(crate) fn expired(&self) -> bool {
        matches!(self, Location::Url { expires: Some(at), .. } if *at <= SystemTime::now())
    }

    pub(crate) fn http(&self) -> Option<HttpLocation<'_>> {
        match self {
            Self::Url { url, headers, .. } => Some(HttpLocation { url, headers, provider: None }),
            Self::Authenticated { url, provider } => Some(HttpLocation { url, headers: &[], provider: Some(provider.as_ref()) }),
            Self::File(_) => None,
        }
    }
}

impl std::fmt::Debug for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File(path) => f.debug_tuple("File").field(path).finish(),
            Self::Url { url, expires, .. } => f.debug_struct("Url").field("url", url).field("expires", expires).finish_non_exhaustive(),
            Self::Authenticated { url, .. } => f.debug_struct("Authenticated").field("url", url).finish_non_exhaustive(),
        }
    }
}

/// `<org>/<model>`, each a plain path segment.
pub(crate) fn split_name(name: &str) -> Result<(&str, &str)> {
    let plain = |s: &str| {
        !s.is_empty()
            && !s.starts_with('.')
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    };
    match name.split_once('/') {
        Some((org, model)) if plain(org) && plain(model) => Ok((org, model)),
        _ => refuse(Code::KEY_GRAMMAR, format!("{name:?} is not <org>/<model>")),
    }
}

/// A refs document's tree for `release`.
pub(crate) fn pick(name: &str, bytes: &[u8], release: Option<&str>) -> Result<ObjectRef> {
    let repo = Repo::from_json(bytes).map_err(|error| Refusal {
        code: Code::UNKNOWN_FORMAT,
        detail: format!("{name}: {error}"),
    })?;
    Ok(release_of(&repo, release)?.1)
}

/// `release` of `repo` (the newest not yanked when `None`): its version and tree.
pub fn release_of(repo: &Repo, release: Option<&str>) -> Result<(String, ObjectRef)> {
    let Some(chosen) = repo.release(release) else {
        let has: Vec<&str> = repo.releases.iter().map(|r| r.version.as_str()).collect();
        let wanted = release.unwrap_or("that is not yanked");
        return refuse(
            Code::RELEASE_ABSENT,
            format!("{} has no release {wanted} (has {has:?})", repo.name),
        );
    };
    if chosen.yanked {
        tracing::warn!("{}@{} is yanked", repo.name, chosen.version);
    }
    Ok((chosen.version.clone(), chosen.tree.clone()))
}

/// `variant` of `tree` (its only one when `None`): its name and header.
pub fn variant_of_tree(tree: &Tree, variant: Option<&str>) -> Result<(String, ObjectRef)> {
    match tree.variant(variant) {
        Ok((v, header)) => Ok((v.to_owned(), header.clone())),
        Err(e) => refuse(Code::REF_NOT_FOUND, e.to_string()),
    }
}

/// What a name resolves to: a release, its tree, and the chosen variant's header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chosen {
    pub version: String,
    pub tree: ObjectRef,
    pub variant: String,
    pub header: ObjectRef,
}

/// `release` of `repo` (the newest not yanked when `None`), its tree (read by `tree`), and
/// that tree's `variant` (its only one when `None`).
pub fn choose(
    repo: &Repo,
    release: Option<&str>,
    variant: Option<&str>,
    tree: impl FnOnce(&ObjectRef) -> Result<Tree>,
) -> Result<Chosen> {
    let (version, at) = release_of(repo, release)?;
    let (variant, header) = variant_of_tree(&tree(&at)?, variant)?;
    Ok(Chosen {
        version,
        tree: at,
        variant,
        header,
    })
}
