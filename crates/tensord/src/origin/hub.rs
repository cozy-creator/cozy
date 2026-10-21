//! A model at a Hub: TensorFS's distribution `Endpoint` at the Hub's origin, presenting the
//! owner's newest ordinary bearer when the request carried one (the owner's unreleased trees),
//! else anonymous (released content). A tag names a commit (a tree); a lane is a variant in it.
//! Every Hub call runs on the fetch runtime.
use std::sync::Arc;

use tensorfs::{
    err::{refuse, Code, Result},
    fetch::{Endpoint, TokenProvider},
    format::{ObjectRef, Resolution},
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

/// `repository`'s tag `tag` (empty: its default, the newest public version not yanked) at
/// `source`'s Hub: its name and commit.
pub fn release(machine: &Machine, source: &hub::Source, repository: &str, tag: &str) -> Result<(String, ObjectRef)> {
    let (endpoint, _) = endpoint(machine, source)?;
    let (repository, tag) = (repository.to_string(), tag.to_string());
    machine.block(&Stop::default(), async move {
        let Some((repo, _)) = endpoint.repo(&repository).await? else {
            return refuse(Code::REF_NOT_FOUND, format!("no model {repository} at {}", endpoint.url()));
        };
        match repo.resolve((!tag.is_empty()).then_some(tag.as_str())) {
            Some(Resolution::Tag(found)) => Ok((found.name.clone(), found.commit.clone())),
            _ => refuse(Code::REF_NOT_FOUND, format!("{repository} has no tag {tag:?} at {}", endpoint.url())),
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
