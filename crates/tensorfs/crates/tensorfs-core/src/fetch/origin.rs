//! Where objects come from. An origin names a model's manifest (`resolve`) and says where its
//! objects can be read (`locate`); the Fetcher downloads, verifies and admits them.

use std::path::PathBuf;
use std::time::SystemTime;

use async_trait::async_trait;

use super::net::Client;
use crate::err::{refuse, Code, Refusal, Result};
use crate::format::{repo_key, ObjectRef, Repo};

/// The most bytes a refs document may have.
const REPO_BYTES: u64 = 16 << 20;

#[async_trait]
pub trait Origin: Send + Sync {
    /// The manifest of `name` (`<org>/<model>`) at `release` (the latest when `None`) and
    /// `lane` (the release's only lane when `None`).
    async fn resolve(
        &self,
        name: &str,
        release: Option<&str>,
        lane: Option<&str>,
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
#[derive(Debug, Clone)]
pub enum Location {
    /// An HTTP(S) GET, with the headers it needs, valid until `expires`.
    Url {
        url: String,
        headers: Vec<(String, String)>,
        expires: Option<SystemTime>,
    },
    /// A file on a mounted volume.
    File(PathBuf),
}

impl Location {
    pub(crate) fn expired(&self) -> bool {
        matches!(self, Location::Url { expires: Some(at), .. } if *at <= SystemTime::now())
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

/// A refs document's manifest for `release` and `lane`.
pub(crate) fn pick(
    name: &str,
    bytes: &[u8],
    release: Option<&str>,
    lane: Option<&str>,
) -> Result<ObjectRef> {
    let repo = Repo::from_json(bytes).map_err(|error| Refusal {
        code: Code::UNKNOWN_FORMAT,
        detail: format!("{name}: {error}"),
    })?;
    Ok(choose(&repo, release, lane)?.2)
}

/// `release` of `repo` (the newest not yanked when `None`), its lane `lane` (the only one when `None`) and
/// that lane's manifest.
pub fn choose(
    repo: &Repo,
    release: Option<&str>,
    lane: Option<&str>,
) -> Result<(String, String, ObjectRef)> {
    let name = &repo.name;
    let releases = || {
        repo.releases
            .iter()
            .map(|r| r.version.as_str())
            .collect::<Vec<_>>()
    };
    let Some(chosen) = repo.release(release) else {
        let wanted = release.unwrap_or("that is not yanked");
        return refuse(
            Code::RELEASE_ABSENT,
            format!("{name} has no release {wanted} (has {:?})", releases()),
        );
    };
    if chosen.yanked {
        tracing::warn!("{name}@{} is yanked", chosen.version);
    }
    let lanes = || chosen.lanes.keys().cloned().collect::<Vec<_>>();
    let lane = match lane {
        Some(lane) => lane.to_string(),
        None if chosen.lanes.len() == 1 => lanes().remove(0),
        None => {
            return refuse(
                Code::REF_NOT_FOUND,
                format!("{name}@{}: name a lane (has {:?})", chosen.version, lanes()),
            )
        }
    };
    match chosen.lanes.get(&lane) {
        Some(manifest) => Ok((chosen.version.clone(), lane, manifest.clone())),
        None => refuse(
            Code::REF_NOT_FOUND,
            format!(
                "{name}@{} has no lane {lane:?} (has {:?})",
                chosen.version,
                lanes()
            ),
        ),
    }
}

/// A static origin over HTTP: `{base}/refs/<org>/<model>.json` (lowercase) and `{base}/blobs/<aa>/<bb>/<sha256>`.
/// Any HTTP server, bucket or repository holding that layout serves it.
pub struct HttpOrigin {
    client: Client,
    base: String,
    bearer: Option<String>,
}

impl HttpOrigin {
    pub fn new(client: &Client, base: &str) -> Result<HttpOrigin> {
        let base = base.trim_end_matches('/').to_string();
        client.policy().admit(&base)?;
        Ok(HttpOrigin {
            client: client.clone(),
            base,
            bearer: None,
        })
    }

    /// Sends `Authorization: Bearer <token>`; the origin is then private.
    pub fn bearer(mut self, token: impl Into<String>) -> HttpOrigin {
        self.bearer = Some(token.into());
        self
    }

    fn headers(&self) -> Vec<(String, String)> {
        self.bearer
            .iter()
            .map(|token| ("authorization".to_string(), format!("Bearer {token}")))
            .collect()
    }
}

#[async_trait]
impl Origin for HttpOrigin {
    async fn resolve(
        &self,
        name: &str,
        release: Option<&str>,
        lane: Option<&str>,
    ) -> Result<ObjectRef> {
        split_name(name)?;
        let url = format!("{}/{}", self.base, repo_key(name));
        let bytes = self
            .client
            .document(&url, &self.headers(), REPO_BYTES)
            .await?;
        pick(name, &bytes, release, lane)
    }

    async fn locate(&self, objects: &[ObjectRef]) -> Result<Vec<Location>> {
        let headers = self.headers();
        Ok((objects.iter())
            .map(|object| Location::Url {
                url: format!("{}/{}", self.base, object.key()),
                headers: headers.clone(),
                expires: None,
            })
            .collect())
    }

    fn public(&self) -> bool {
        self.bearer.is_none()
    }
}
