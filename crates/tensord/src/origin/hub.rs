//! A model at a Hub: TensorFS's distribution `Endpoint` at the Hub's origin, presenting the
//! owner's newest ordinary bearer when the request carried one (the owner's unreleased trees),
//! else anonymous (released content). A release names a tree; a lane is a variant in it. Every Hub
//! call runs on the fetch runtime.
use std::sync::Arc;

use tensorfs::{
    err::{refuse, Code, Result},
    fetch::{release_of, Endpoint, TokenProvider},
    format::ObjectRef,
};

use super::{Key, Machine, Remote};
use crate::{hub, stop::Stop};

/// `source`'s Hub as an endpoint over the machine's client for it (its CA, whether it is on
/// loopback).
fn endpoint(machine: &Machine, source: &hub::Source) -> Result<(Endpoint, Key)> {
    let trust = source.ca_der.as_deref().map(|der| Arc::new(hub::pem(der).into_bytes()));
    let local = hub::origin_key(&source.origin).is_some_and(|key| loopback(&key));
    let client = machine.client(trust.as_ref(), local)?;
    let endpoint = match &source.bearer {
        Some(bearer) => Endpoint::new(&client, &source.origin, Some(bearer.clone() as Arc<dyn TokenProvider>))?,
        None => Endpoint::new(&client, &source.origin, None)?.anonymous(),
    };
    Ok((endpoint, (trust, local)))
}

/// Where the objects of `source`'s Hub are read. Nothing is asked of the Hub until a fetch
/// needs it.
pub fn remote(machine: &Machine, source: &hub::Source) -> Result<Remote> {
    let (endpoint, (trust, local)) = endpoint(machine, source)?;
    Ok(Remote { origin: Arc::new(endpoint), trust, local })
}

/// `repository`'s release `release` (empty: its newest not yanked) at `source`'s Hub: its
/// version and tree.
pub fn release(machine: &Machine, source: &hub::Source, repository: &str, release: &str) -> Result<(String, ObjectRef)> {
    let (endpoint, _) = endpoint(machine, source)?;
    let (repository, release) = (repository.to_string(), release.to_string());
    machine.block(&Stop::default(), async move {
        match endpoint.repo(&repository).await? {
            Some((repo, _)) => release_of(&repo, (!release.is_empty()).then_some(release.as_str())),
            None => refuse(Code::REF_NOT_FOUND, format!("no model {repository} at {}", endpoint.url())),
        }
    })
}

/// The object `sha256` at `source`'s Hub, with its length, as the request may read it.
pub fn object(machine: &Machine, source: &hub::Source, sha256: &str) -> Result<ObjectRef> {
    let (endpoint, _) = endpoint(machine, source)?;
    let sha256 = sha256.to_string();
    machine.block(&Stop::default(), async move {
        match endpoint.object(&sha256).await? {
            Some(object) => Ok(object),
            None => refuse(Code::REF_NOT_FOUND, format!("sha256:{sha256} is not readable at {}", endpoint.url())),
        }
    })
}

fn loopback(key: &str) -> bool {
    let host = key.split_once("://").and_then(|(_, rest)| rest.rsplit_once(':')).map(|(host, _)| host);
    matches!(host, Some("localhost" | "127.0.0.1" | "[::1]"))
}
