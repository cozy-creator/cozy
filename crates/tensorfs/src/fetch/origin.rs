//! Where objects come from. An origin serves a model's repo document (`repo_of`) and says where
//! its objects can be read (`locate`); the Fetcher downloads, verifies and admits them.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;

use super::token::TokenProvider;
use crate::err::{refuse, Code, Refusal, Result};
use crate::format::{ObjectRef, Repo, Resolution, Tag, Tree};

#[async_trait]
pub trait Origin: Send + Sync {
    /// The repo document of `name` (`<org>/<model>`); `REF_NOT_FOUND` when there is none.
    async fn repo_of(&self, name: &str) -> Result<Repo>;

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

/// A refs document `bytes` of `name`.
pub(crate) fn parse_repo(name: &str, bytes: &[u8]) -> Result<Repo> {
    Repo::from_json(bytes).map_err(|error| Refusal {
        code: Code::UNKNOWN_FORMAT,
        detail: format!("{name}: {error}"),
    })
}

/// `revision` of `repo` resolved once (spec §7): its tag and, for a header digest, the variant
/// that names it (`REVISION_ABSENT` when nothing does). `tree` reads a commit's tree, only for a
/// header digest.
pub async fn resolve<E: From<Refusal>>(
    repo: &Repo,
    revision: Option<&str>,
    mut tree: impl AsyncFnMut(&ObjectRef) -> std::result::Result<Tree, E>,
) -> std::result::Result<(Tag, Option<String>), E> {
    let found = match repo.resolve(revision) {
        Some(Resolution::Tag(tag)) => Some((tag, None)),
        Some(Resolution::Header(sha256, tags)) => {
            let mut found = None;
            for tag in tags {
                if let Some(variant) = tree(&tag.commit).await?.cites(&sha256) {
                    found = Some((tag, Some(variant.to_owned())));
                    break;
                }
            }
            found
        }
        None => None,
    };
    let Some((tag, variant)) = found else {
        let has: Vec<&str> = repo.tags.iter().map(|t| t.name.as_str()).collect();
        let wanted = revision.unwrap_or("default (no public, non-yanked version)");
        return Err(Refusal { code: Code::REVISION_ABSENT, detail: format!("{} has no revision {wanted} (tags {has:?})", repo.name) }.into());
    };
    if tag.yanked {
        tracing::warn!("{}@{} is yanked", repo.name, tag.name);
    }
    Ok((tag.clone(), variant))
}

/// `variant` of `tree` (its only one when `None`): its name and header.
pub fn variant_of_tree(tree: &Tree, variant: Option<&str>) -> Result<(String, ObjectRef)> {
    match tree.variant(variant) {
        Ok((v, header)) => Ok((v.to_owned(), header.clone())),
        Err(e) => refuse(Code::REF_NOT_FOUND, e.to_string()),
    }
}

/// What a name resolves to: a tag, its commit, and the chosen variant's header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chosen {
    pub tag: Tag,
    pub variant: String,
    pub header: ObjectRef,
}

/// `revision` of `repo` resolved once, and `variant` of its commit (its only one when `None`;
/// for a header digest, the variant naming it). `tree` reads a commit's tree.
pub async fn choose<E: From<Refusal>>(
    repo: &Repo,
    revision: Option<&str>,
    variant: Option<&str>,
    mut tree: impl AsyncFnMut(&ObjectRef) -> std::result::Result<Tree, E>,
) -> std::result::Result<Chosen, E> {
    let (tag, cited) = resolve(repo, revision, &mut tree).await?;
    let tree = tree(&tag.commit).await?;
    let variant = pick_variant(&tree, revision, cited, variant)?;
    let (variant, header) = variant_of_tree(&tree, variant.as_deref())?;
    Ok(Chosen { tag, variant, header })
}

/// The variant to open: the one a header digest named (an explicit one must agree), else `asked`.
pub fn pick_variant(tree: &Tree, revision: Option<&str>, cited: Option<String>, asked: Option<&str>) -> Result<Option<String>> {
    let asked = asked.map(|name| variant_of_tree(tree, Some(name)).map(|(path,_)|path)).transpose()?;
    match (cited, asked) {
        (Some(cited), Some(asked)) if cited != asked => {
            refuse(Code::REF_NOT_FOUND, format!("{} is variant {cited:?}, not {asked:?}", revision.unwrap_or_default()))
        }
        (cited, asked) => Ok(cited.or(asked)),
    }
}
