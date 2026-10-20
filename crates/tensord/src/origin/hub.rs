//! A model at a Hub: TensorFS's `TensorhubOrigin` under the run's Cozy credentials. A release
//! names a tree, and a lane is a variant in it; released lanes are read anonymously, the owner's
//! tree by hash under the capability the owner's CLI signed, traded for a DPoP-bound token and
//! presented to the Hub host only. Every Hub call runs on the fetch runtime.
use std::sync::Arc;

use async_trait::async_trait;
use tensorfs::{
    err::{refuse, Code, Refusal, Result},
    fetch::{token::Request, TokenProvider, TensorhubOrigin},
    format::ObjectRef,
};

pub use tensorfs::fetch::Resolved;

use super::{Key, Machine, Remote};
use crate::{
    hub::{self, Catalog, Op},
    stop::Stop,
};

/// The run's capability on the library's Hub calls.
struct Presented(Arc<Catalog>);

#[async_trait]
impl TokenProvider for Presented {
    async fn headers(&self, request: &Request<'_>) -> Result<Vec<(String, String)>> {
        // The token is traded on the Hub runtime: off the fetch runtime's workers.
        let (catalog, method, url) = (self.0.clone(), request.method.to_string(), request.url.to_string());
        tokio::task::spawn_blocking(move || catalog.present(&method, &url)).await
            .map_err(|error| Refusal { code: Code::CREDENTIAL_REQUIRED, detail: error.to_string() })
    }

    fn retry(&self, sent: &Request<'_>, status: u16, headers: &reqwest::header::HeaderMap) -> Result<bool> {
        Ok(self.0.observe(sent.url, sent.headers, status, headers))
    }
}

/// `source`'s Hub as a library origin over the machine's client for it (its CA, whether it is
/// on loopback), presenting `catalog`'s credential when it has one.
fn origin(machine: &Machine, source: &hub::Source, catalog: &Arc<Catalog>) -> Result<(TensorhubOrigin, Key)> {
    let trust = source.ca_der.as_deref().map(|der| Arc::new(hub::pem(der).into_bytes()));
    let local = hub::origin_key(&source.origin).is_some_and(|key| loopback(&key));
    let mut origin = TensorhubOrigin::new(&machine.client(trust.as_ref(), local)?, &source.origin)?;
    if catalog.presents() {
        origin = origin.credential(Arc::new(Presented(catalog.clone())));
    }
    Ok((origin, (trust, local)))
}

/// `repository`'s `tree` at `source`'s Hub: a release's, read in `lane`, or (no release) the
/// owner's tree by hash, read under the run's capability when it names that read.
/// Nothing is asked of the Hub until a request needs it. Answers the catalog its calls present
/// under (a failed fetch's reason, `Catalog::reason`).
pub fn remote(
    machine: &Machine,
    source: &hub::Source,
    repository: &str,
    release: &str,
    lane: &str,
    tree: &ObjectRef,
) -> std::result::Result<(Remote, Arc<Catalog>), (&'static str, String)> {
    let catalog = Arc::new(match release.is_empty() {
        true => Catalog::scoped(source, Op::Read { model: repository, manifest: &tree.id() })?,
        false => Catalog::new(source).map_err(|e| ("catalog_read_failed", e.0))?,
    });
    let (origin, (trust, local)) = self::origin(machine, source, &catalog).map_err(|e| ("catalog_read_failed", e.detail))?;
    let origin = Arc::new(origin.at(repository, (!release.is_empty()).then_some(release), lane, tree));
    Ok((Remote { origin, trust, local }, catalog))
}

/// What the Hub names `repository` at `release` (empty: its newest): the first of `lanes` its
/// release publishes (an empty lane: its only one). A lane the release lacks answers
/// `REF_NOT_FOUND`, and the next is asked.
pub fn resolve_named(machine: &Machine, source: &hub::Source, repository: &str, release: &str, lanes: &[String]) -> Result<Resolved> {
    let catalog = Arc::new(Catalog::new(source).map_err(|e| Refusal { code: Code::HUB_REFUSED, detail: e.0 })?);
    let hub = Arc::new(origin(machine, source, &catalog)?.0);
    let (repository, release, lanes) = (repository.to_string(), release.to_string(), lanes.to_vec());
    machine.block(&Stop::default(), async move {
        let release = (!release.is_empty()).then_some(release.as_str());
        let mut answer = refuse(Code::REF_NOT_FOUND, format!("{repository}: no lane asked"));
        for lane in &lanes {
            answer = hub.named(&repository, release, Some(lane)).await;
            if !matches!(&answer, Err(e) if e.code == Code::REF_NOT_FOUND) {
                break;
            }
        }
        answer
    })
}

/// The owner's tree `sha256` of `repository`, named by hash alone, as the Hub states it (its
/// length), read under the run's capability when it names that read.
pub fn resolve_hash(machine: &Machine, source: &hub::Source, repository: &str, lane: &str, sha256: &str) -> std::result::Result<ObjectRef, (&'static str, String)> {
    let id = format!("sha256:{sha256}");
    let catalog = Arc::new(Catalog::scoped(source, Op::Read { model: repository, manifest: &id })?);
    let hub = Arc::new(origin(machine, source, &catalog).map_err(|e| ("catalog_read_failed", e.detail))?.0);
    let (repository, lane, asked) = (repository.to_string(), lane.to_string(), id.clone());
    let named = machine.block(&Stop::default(), async move { hub.named(&repository, Some(&asked), Some(&lane)).await });
    let named = named.map_err(|e| catalog.reason("model_resolution_failed", e))?;
    match named.tree.sha256 == sha256 {
        true => Ok(named.tree),
        false => Err(("model_resolution_failed", format!("{id} answered {}", named.tree.id()))),
    }
}

fn loopback(key: &str) -> bool {
    let host = key.split_once("://").and_then(|(_, rest)| rest.rsplit_once(':')).map(|(host, _)| host);
    matches!(host, Some("localhost" | "127.0.0.1" | "[::1]"))
}
