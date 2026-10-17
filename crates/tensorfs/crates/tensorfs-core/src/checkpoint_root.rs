//! Validate an exact, already-materialized repository checkpoint as an execution input.

use std::fs::OpenOptions;
use std::io::Read;

use crate::catalog::WriterGuard;
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{Doc, ObjectRef};
use crate::repository::{Repository, RepositoryName};
use crate::store::Store;

pub fn repository(value: &str) -> Result<RepositoryName> {
    let (org, name) = value.split_once('/').ok_or_else(|| Refusal {
        code: Code::KEY_GRAMMAR,
        detail: "checkpoint source repository must be org/name".into(),
    })?;
    RepositoryName::new(org, name)
}

fn io(error: std::io::Error) -> Refusal {
    crate::store::classify_io("checkpoint input", error)
}

fn verify(store: &Store, manifest: &ObjectRef) -> Result<()> {
    let model = store.read_manifest(manifest)?;
    let closure = crate::checkpoint::walk_cozytensors(store, &model)?;
    closure.require_resident(store)?;
    for object in closure.distinct() {
        let record = store.record_valid(&object.sha256).map_err(|why| Refusal {
            code: Code::OBJECT_CORRUPT,
            detail: format!("checkpoint input closure has no current verification record: {why}"),
        })?;
        if record.length != object.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                "checkpoint input closure record changed length",
            );
        }
    }
    Ok(())
}

/// Validate a checkpoint input: exact repository membership and a verified resident closure.
pub fn check_source(store: &Store, source: &str, manifest: &ObjectRef) -> Result<()> {
    let repository = repository(source)?;
    let _gc = WriterGuard::acquire(store.root())?;
    verify_source(store, &repository, manifest)
}

fn verify_source(store: &Store, repository: &RepositoryName, manifest: &ObjectRef) -> Result<()> {
    let file = cozy_platform::durable::open_nofollow(&store.repository_path(repository), OpenOptions::new().read(true))
        .map_err(std::io::Error::from)
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Refusal {
                    code: Code::REPOSITORY_ABSENT,
                    detail: "checkpoint source repository is absent".into(),
                }
            } else {
                io(error)
            }
        })?;
    let mut body = Vec::new();
    file.take(Repository::MAX_BYTES as u64 + 1)
        .read_to_end(&mut body)
        .map_err(io)?;
    let actual = Repository::parse(&body)?;
    if &actual.repo != repository
        || !actual
            .checkpoints
            .iter()
            .any(|row| &row.manifest == manifest)
    {
        return refuse(
            Code::ROOT_ABSENT,
            "checkpoint has no exact source repository membership",
        );
    }
    verify(store, manifest)
}
