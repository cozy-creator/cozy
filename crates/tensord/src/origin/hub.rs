//! The distribution endpoint, with the run's token provider only for its private operations.
use std::sync::Arc;
use tensorfs_core::{
    err::{refuse, Code, Refusal, Result},
    fetch::{release_of, Endpoint},
    format::ObjectRef,
};
use super::{Key, Machine, Remote};
use crate::{hub::{self, Catalog, Op}, stop::Stop};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    pub model: String,
    pub release: String,
    pub variant: String,
    pub tree: ObjectRef,
    pub public: bool,
}

fn origin(source: &hub::Source, catalog: &Catalog) -> Result<(Endpoint, Key)> {
    let trust = source.ca_der.as_deref().map(|der| Arc::new(hub::pem(der).into_bytes()));
    let local = hub::origin_key(&source.origin).is_some_and(|key| loopback(&key));
    Ok((catalog.endpoint()?, (trust, local)))
}

pub fn remote(
    _machine: &Machine,
    source: &hub::Source,
    repository: &str,
    release: &str,
    _lane: &str,
    tree: &ObjectRef,
) -> std::result::Result<(Remote, Arc<Catalog>), (&'static str, String)> {
    let catalog = Arc::new(match release.is_empty() {
        true => Catalog::scoped(source, Op::Read { model: repository, manifest: &tree.id() })?,
        false => Catalog::new(source).map_err(|e| ("catalog_read_failed", e.0))?,
    });
    let (origin, (trust, local)) = origin(source, &catalog).map_err(|e| ("catalog_read_failed", e.detail))?;
    Ok((Remote { origin: Arc::new(origin), trust, local }, catalog))
}

pub fn resolve_named(machine: &Machine, source: &hub::Source, repository: &str, release: &str, lanes: &[String]) -> Result<Resolved> {
    let catalog = Catalog::new(source).map_err(|e| Refusal { code: Code::HUB_REFUSED, detail: e.0 })?;
    let (endpoint, (trust, local)) = origin(source, &catalog)?;
    let endpoint = Arc::new(endpoint);
    let name = repository.to_ascii_lowercase();
    let repo = {
        let (endpoint, asked) = (endpoint.clone(), name.clone());
        machine.block(&Stop::default(), async move { endpoint.repo(&asked).await })?
            .ok_or_else(|| Refusal { code: Code::REF_NOT_FOUND, detail: format!("no repo {name}") })?.0
    };
    let (release, tree) = release_of(&repo, (!release.is_empty()).then_some(release))?;
    let remote = Remote { origin: endpoint, trust, local };
    let mut answer = refuse(Code::REF_NOT_FOUND, format!("{name}: no variant asked"));
    for lane in lanes {
        answer = machine.variant(&remote, &tree, (!lane.is_empty()).then_some(lane), &Stop::default());
        if !matches!(&answer, Err(e) if e.code == Code::REF_NOT_FOUND) { break; }
    }
    let (variant, _) = answer?;
    Ok(Resolved { model: name, release, variant, tree, public: true })
}

pub fn resolve_hash(machine: &Machine, source: &hub::Source, repository: &str, _lane: &str, sha256: &str) -> std::result::Result<ObjectRef, (&'static str, String)> {
    let id = format!("sha256:{sha256}");
    let catalog = Catalog::for_op(source, Op::Read { model: repository, manifest: &id })?;
    let endpoint = catalog.endpoint().map_err(|e| catalog.reason("model_resolution_failed", e))?;
    let digest = sha256.to_owned();
    let object = machine.block(&Stop::default(), async move { endpoint.object(&digest).await })
        .map_err(|e| catalog.reason("model_resolution_failed", e))?;
    object.ok_or_else(|| ("model_resolution_failed", format!("{id} is absent or unreadable")))
}

fn loopback(key: &str) -> bool {
    let host = key.split_once("://").and_then(|(_, rest)| rest.rsplit_once(':')).map(|(host, _)| host);
    matches!(host, Some("localhost" | "127.0.0.1" | "[::1]"))
}
