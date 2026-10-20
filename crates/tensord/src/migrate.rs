//! Dev builds only (feature `migrate`, xs-045 U12; deleted with the old format): checkpoints from
//! before TensorFS's format cut, rewritten into this machine's store by `tensorfs::migrate`
//! (the last reader of the old format) and checked tensor by tensor.
//!
//! - [`checkpoint`]: an old checkpoint at the run's Hub, on a machine that does nothing else (a
//!   rental). The old objects it repacks land in the store as plain blobs, where the migration
//!   reads them; the large ones the new checkpoint reuses stay at the Hub, which holds them, so
//!   the run's publish neither fetches nor sends them. Each migration first drops what the
//!   previous one left.
//! - [`local`]: every `local/` model of a pre-cut store on this machine, under the same name; the
//!   old objects it reuses move in by hard link (a copy only across filesystems), so the
//!   conversion costs only its new bytes.
use serde_json::{json, Value};
use tensorfs::{
    format::ObjectRef,
    migrate::{self, v1},
    store2::Store,
};

use crate::{
    api::domain, hub, objects::Refused, origin::priority, published::Publisher, stop::Stop, storage,
};

/// Whether the run's model choice is a migration (`protocol::machine::MIGRATE`).
pub fn marked(choice: &domain::ModelChoice) -> bool {
    choice.profiles.iter().any(|p| p == protocol::machine::MIGRATE)
}

/// The pre-cut store a run's one marked choice names as its source: a `--local` migration.
pub fn local_store(models: &[domain::ModelChoice]) -> Option<&str> {
    match models {
        [choice] if marked(choice) && !choice.source.is_empty() => Some(&choice.source),
        _ => None,
    }
}

fn refused(code: &'static str, message: impl Into<String>) -> Refused {
    Refused { code, message: message.into() }
}

/// The migration of `lane` in `src` into `store`: its new manifest, pinned under the writer's
/// key `migrate-<old manifest>`, and its report.
fn rewrite(src: &v1::Source, store: &Store, lane: v1::Lane) -> Result<(ObjectRef, Value), Refused> {
    let report = migrate::lane(src, Some(store), &lane, false);
    let doc = Value::Object(report.doc);
    match report.manifests.1.filter(|_| report.pass) {
        Some(made) => Ok((made, doc)),
        None => Err(refused("migration_failed", format!("{}: {doc}", lane.manifest.sha256))),
    }
}

/// What earlier migrations left here: published by now (the CLI runs one at a time), so their
/// pins go and every unreferenced object goes at once, the old ones fetched for them included.
/// No grace: the Hub path runs on a machine that does nothing else, so disk holds one lane.
fn clear(store: &Store) -> Result<(), Refused> {
    let failed = |e: tensorfs::store2::Error| refused("migration_failed", e.to_string());
    for (id, _) in store.pins().map_err(failed)?.into_iter().filter(|(id, _)| id.starts_with("migrate-")) {
        store.unpin(&id).map_err(failed)?;
    }
    store.clone().with_grace(std::time::Duration::ZERO).gc().map(drop).map_err(failed)
}

/// `choice`, an old checkpoint of `choice.repository` (a release lane, or the owner's by hash),
/// fetched from `hub` and migrated here: the new manifest and the run's row.
pub fn checkpoint(
    publisher: &Publisher,
    hub: &hub::Source,
    choice: &domain::ModelChoice,
    stop: Option<&Stop>,
    observe: &(dyn Fn(&str, u64, u64) + Sync),
) -> Result<(ObjectRef, Value), Refused> {
    let manifest = choice.manifest.as_ref().map(|m| ObjectRef {
        sha256: tensorfs::sha256::hex(&m.digest),
        length: m.length,
    });
    let manifest = manifest.ok_or_else(|| refused("invalid_request", "a migration names the old checkpoint's manifest"))?;
    let (remote, catalog) = crate::origin::remote(
        publisher.machine(),
        hub,
        &choice.repository,
        &choice.release,
        &choice.lane,
        &manifest,
    )
    .map_err(|(code, message)| refused(code, message))?;
    let (store, never) = (publisher.store(), Stop::default());
    clear(store)?;
    let fetch = |objects: &[ObjectRef], stage: &str| -> Result<u64, Refused> {
        let absent: Vec<ObjectRef> = objects.iter().filter(|o| !storage::present(store, o)).cloned().collect();
        let bytes = absent.iter().map(|o| o.length).sum::<u64>();
        if bytes > 0 && storage::admits(store, bytes).is_err() {
            storage::relieve(store, &[], bytes).map_err(|e| refused("migration_fetch_failed", e.to_string()))?;
        }
        observe(stage, 0, bytes);
        publisher
            .machine()
            .objects(&remote, &absent, priority::NOW, stop.unwrap_or(&never))
            .map_err(|e| {
                let (code, message) = catalog.reason("migration_fetch_failed", e);
                refused(code, message)
            })?;
        observe(stage, bytes, bytes);
        Ok(bytes)
    };
    let src = v1::Source::new(store.root());
    let read = |object: &ObjectRef| src.read(object).map_err(|f| refused("migration_failed", f.detail));
    let mut fetched = fetch(std::slice::from_ref(&manifest), "fetching the old manifest")?;
    let old = v1::manifest(&read(&manifest)?).map_err(|f| refused("migration_failed", f.detail))?;
    let header = old.header.clone().ok_or_else(|| refused("migration_failed", "the old manifest has no header"))?;
    fetched += fetch(std::slice::from_ref(&header), "fetching the old header")?;
    let h = v1::header(&read(&header)?).map_err(|f| refused("migration_failed", f.detail))?;
    fetched += fetch(&migrate::reads(&h), &format!("fetching {}", choice.repository))?;
    observe(&format!("migrating {}", choice.repository), 0, 0);
    let named = match choice.release.is_empty() {
        true => format!("{}#{}", choice.repository, manifest.id()),
        false => format!("{}@{} {}", choice.repository, choice.release, choice.lane),
    };
    let (made, doc) = rewrite(&src, store, v1::Lane { manifest, names: vec![named] })?;
    let row = json!({"parameter": choice.parameter, "repository": choice.repository,
        "manifest": made.id(), "migrated": doc, "fetched_bytes": fetched});
    Ok((made, row))
}

/// Every `local/` model of the pre-cut store at `root` (read only), migrated into this
/// machine's store under its name, with the old objects it reuses linked in: one row each.
pub fn local(publisher: &Publisher, root: &str, observe: &(dyn Fn(&str, u64, u64) + Sync)) -> Result<Vec<Value>, Refused> {
    let (src, store) = (v1::Source::new(root), publisher.store());
    let failed = |f: tensorfs::failure::Failure| refused("migration_failed", f.detail);
    let repos = src.repos().map_err(failed)?;
    let models: Vec<(&str, &ObjectRef)> = (repos.iter())
        .filter_map(|r| Some((r.name.strip_prefix("local/")?, r.checkpoints.last()?)))
        .collect();
    let mut rows = Vec::new();
    for (i, (name, manifest)) in models.iter().enumerate() {
        observe(&format!("migrating local/{name}"), i as u64, models.len() as u64);
        let lane = v1::Lane { manifest: (*manifest).clone(), names: vec![format!("local/{name}")] };
        let (made, doc) = rewrite(&src, store, lane)?;
        let closure = store.closure(&made).map_err(|e| refused("migration_failed", e.to_string()))?;
        let adopted = migrate::adopt(&src, store, &closure).map_err(failed)?;
        storage::set_local(store, name, &made, &Default::default()).map_err(|e| refused("local_alias_refused", e.to_string()))?;
        let key = format!("migrate-{}", manifest.sha256);
        store.unpin(&key).map_err(|e| refused("migration_failed", e.to_string()))?;
        rows.push(json!({"repository": format!("local/{name}"), "manifest": made.id(), "migrated": doc,
            "linked": {"objects": adopted.linked.0, "bytes": adopted.linked.1},
            "copied": {"objects": adopted.copied.0, "bytes": adopted.copied.1}}));
    }
    Ok(rows)
}
