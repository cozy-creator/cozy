//! Offline, one-time store1 → store3 conversion. The old filesystem records, not SQLite's
//! rebuildable projection, are authoritative. No old reader is installed in the serving path.
//!
//! Call only after stopping old clients. The owner and recovery claims exclude an old machine,
//! active writers and external read leases; the owner descriptor is handed to the new machine.
//! Original blobs remain hard-linked in the private work directory, outside store3 GC. Original
//! repos, catalog, incomplete roots and checkouts remain untouched. The retired manifests and
//! original format remain there for an explicit offline rollback, never an automatic downgrade.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};

use platform::durable;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::d10::Migrator;
use crate::failure::{fail, Kind, Result};
use crate::format::{ObjectRef, Repo};
use crate::store::{fault, Durability, Store};

const WORK: &str = ".store1-migration";
const PREPARED: &str = "prepared.json";
const FORMAT3: &[u8] = b"{\"format\":\"tensorfs.store/3\"}";
const FENCE: &[u8] = b"TensorFS store1 manifests were retained in .store1-migration/legacy-manifests. This store was converted; use TensorFS 0.13 or an explicit offline rollback.\n";

/// Every old manifest has an explicit new tree; identities are never silently redirected.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Checkpoint {
    pub manifest: ObjectRef,
    pub tree: ObjectRef,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Report {
    pub repos: usize,
    pub checkpoints: Vec<Checkpoint>,
    /// A stopped old SDK's exact source-selection identity, retained as a private tag.
    pub aliases: BTreeMap<String, String>,
}

/// Keep this descriptor until the new machine relinquishes the store. Reclaiming owner.lock
/// through a second descriptor would fail against our own claim and create a handoff race.
pub struct Upgrade {
    pub report: Report,
    pub owner: File,
}

#[derive(Serialize, Deserialize)]
struct Prepared {
    format: String,
    original_format: Option<Vec<u8>>,
    report: Report,
    /// Complete relative metadata paths and bytes. Publication is replayed exactly after death.
    records: BTreeMap<PathBuf, Vec<u8>>,
}

fn invalid<T>(detail: impl Into<String>) -> Result<T> {
    fail(Kind::Invalid, detail)
}

fn real_dir(path: &Path) -> Result<()> {
    let m = fs::symlink_metadata(path)?;
    if !m.is_dir() || m.file_type().is_symlink() {
        return invalid(format!("{} is not a real directory", path.display()));
    }
    Ok(())
}

fn regular(path: &Path) -> Result<fs::Metadata> {
    let m = fs::symlink_metadata(path)?;
    if !m.is_file() || m.file_type().is_symlink() {
        return invalid(format!("{} is not a regular file", path.display()));
    }
    Ok(m)
}

fn list(path: &Path) -> Result<Vec<PathBuf>> {
    real_dir(path)?;
    let mut paths = fs::read_dir(path)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    Ok(paths)
}

fn tree_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for path in list(root)? {
        let m = fs::symlink_metadata(&path)?;
        if m.is_dir() && !m.file_type().is_symlink() {
            files.extend(tree_files(&path)?);
        } else {
            regular(&path)?;
            files.push(path);
        }
    }
    Ok(files)
}

fn sync_dirs(root: &Path) -> Result<()> {
    for path in list(root)? {
        if fs::symlink_metadata(&path)?.is_dir() {
            sync_dirs(&path)?;
        }
    }
    durable::sync_dir(root).map_err(std::io::Error::from)?;
    Ok(())
}

fn digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn record(root: &Path) -> Result<Option<Vec<u8>>> {
    match fs::read(root.join("format")) {
        Ok(b) => Ok(Some(b)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn current(bytes: &[u8]) -> bool {
    serde_json::from_slice::<Value>(bytes)
        .ok()
        .and_then(|v| v["format"].as_str().map(str::to_owned))
        .is_some_and(|s| crate::store::same_major(&s, "tensorfs.store/3"))
}

/// Read-only: once true, an interrupted activation must resume forward rather than relaunch
/// an old SDK. It must not be inferred merely from a candidate's software version.
pub fn cutover_started(root: &Path) -> Result<bool> {
    match fs::symlink_metadata(root.join(WORK).join("legacy-manifests")) {
        Ok(m) if m.is_dir() && !m.file_type().is_symlink() => Ok(true),
        Ok(_) => invalid("store1 migration's retired manifests are not a real directory"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

fn claim(path: &Path) -> Result<File> {
    let file = durable::open_nofollow(
        path,
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false),
    )
    .map_err(std::io::Error::from)?;
    if !file.metadata()?.is_file() {
        return invalid(format!("{} is not a regular claim file", path.display()));
    }
    if !platform::lock::try_claim(&file).map_err(std::io::Error::from)? {
        return fail(
            Kind::Conflict,
            format!(
                "{} is held; stop all old TensorFS clients before migration",
                path.display()
            ),
        );
    }
    Ok(file)
}

/// Upgrades a stopped store in place, without network access, reset or tensor-byte copies.
/// None means an absent/empty root or an already-current store. Unsupported data refuses
/// before cutover. A failed/interrupted cutover is replayable from its durable prepared record.
pub fn migrate(root: &Path) -> Result<Option<Upgrade>> {
    match fs::symlink_metadata(root) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
        Ok(_) => real_dir(root)?,
    }
    if fs::read_dir(root)?.next().is_none() {
        return Ok(None);
    }
    let root = root.canonicalize()?;
    let original = record(&root)?;
    if original.as_deref().is_some_and(current) {
        return Ok(None);
    }
    let resuming = cutover_started(&root)?;
    if let Some(b) = &original {
        if b != b"store-format 1\nclaim-protocol 1\n" {
            return fail(
                Kind::Format,
                format!(
                    "{} is not a store1 format record",
                    root.join("format").display()
                ),
            );
        }
    }
    for name in ["repos", "blobs"] {
        real_dir(&root.join(name))?;
    }
    if !resuming {
        real_dir(&root.join("manifests"))?;
    }
    // The actual published 0.10 predates the format record. Its catalog is a projection,
    // retained opaque, not opened or repaired. Distinguish it from arbitrary directories.
    if original.is_none() {
        let path = root.join("tensorfs.sqlite");
        regular(&path)?;
        let mut head = [0; 16];
        File::open(&path)?.read_exact(&mut head)?;
        if &head != b"SQLite format 3\0" {
            return fail(
                Kind::Format,
                format!("{} is not a store1 SQLite catalog", path.display()),
            );
        }
    }
    let owner = claim(&root.join("owner.lock"))?;
    let writers = root.join("tmp/writers");
    fs::create_dir_all(&writers)?;
    let _recovery = claim(&writers.join("recovery.lock"))?;
    // Another migrator could have completed while this caller was inspecting the root.
    if record(&root)?.as_deref().is_some_and(current) {
        return Ok(None);
    }
    let work = root.join(WORK);
    durable::private_dir(&work).map_err(std::io::Error::from)?;
    let prepared_path = work.join(PREPARED);
    let prepared: Prepared = if resuming {
        let p: Prepared = serde_json::from_slice(&fs::read(&prepared_path)?).map_err(|e| {
            crate::failure::Failure {
                kind: Kind::Invalid,
                detail: e.to_string(),
            }
        })?;
        if p.format != "tensorfs.store1-migration/1" || p.original_format != original {
            return invalid("store1 prepared record does not match the interrupted source");
        }
        p
    } else {
        let p = prepare(&root, &work, original)?;
        durable::replace(
            &prepared_path,
            &serde_json::to_vec(&p).expect("serializable migration"),
            0o600,
        )
        .map_err(std::io::Error::from)?;
        p
    };
    fault::point("store1.prepared");
    publish(&root, &work, &prepared)?;
    fault::point("store1.published");
    let retired = work.join("legacy-manifests");
    if !cutover_started(&root)? {
        fs::rename(root.join("manifests"), &retired)?;
        durable::sync_dir(&work).map_err(std::io::Error::from)?;
        durable::sync_dir(&root).map_err(std::io::Error::from)?;
    }
    fault::point("store1.retired");
    durable::replace(&root.join("manifests"), FENCE, 0o600).map_err(std::io::Error::from)?;
    fault::point("store1.fenced");
    durable::replace(&root.join("format"), FORMAT3, 0o644).map_err(std::io::Error::from)?;
    fault::point("store1.committed");
    Ok(Some(Upgrade {
        report: prepared.report,
        owner,
    }))
}

fn object(v: &Value, what: &str) -> Result<ObjectRef> {
    let o: ObjectRef = serde_json::from_value(v.clone()).map_err(|e| crate::failure::Failure {
        kind: Kind::Invalid,
        detail: format!("{what}: {e}"),
    })?;
    if !digest(&o.sha256) || o.length == 0 {
        return invalid(format!("{what}: invalid object reference"));
    }
    Ok(o)
}

fn read(root: &Path, o: &ObjectRef) -> Result<Vec<u8>> {
    let shard = PathBuf::from(&o.sha256[..2]).join(&o.sha256[2..4]);
    let blob = root.join("blobs").join(&shard).join(&o.sha256);
    let path = if blob.is_file() {
        blob
    } else {
        root.join("manifests")
            .join(shard)
            .join(format!("{}.json", o.sha256))
    };
    regular(&path)?;
    let b = fs::read(&path)?;
    if ObjectRef::of(&b) != *o {
        return fail(
            Kind::Integrity,
            format!("{} does not match {}", path.display(), o.id()),
        );
    }
    Ok(b)
}

fn prepare(root: &Path, work: &Path, original_format: Option<Vec<u8>>) -> Result<Prepared> {
    let stage = Store::open(work.join("store3"))?.with_durability(Durability::Durable);
    // These links also retain old incomplete roots and SQL-only transaction metadata's bytes
    // after current GC. Nothing deletes this private rollback area as part of an upgrade.
    for path in tree_files(&root.join("blobs"))? {
        let rel = path.strip_prefix(root).expect("inside old store");
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(digest)
        {
            link(&path, &stage.root().join(rel))?;
        } else {
            return invalid(format!("{} is not a store1 blob", path.display()));
        }
    }
    let mut all = BTreeSet::new();
    for path in tree_files(&root.join("manifests"))? {
        let bytes = fs::read(&path)?;
        let o = ObjectRef::of(&bytes);
        if path.file_stem().and_then(|n| n.to_str()) != Some(o.sha256.as_str())
            || super::v1::manifest(&bytes)?.is_none()
        {
            return invalid(format!(
                "{} is not a digest-named v1 checkpoint manifest",
                path.display()
            ));
        }
        all.insert(o);
    }
    let mut report = Report {
        repos: 0,
        checkpoints: Vec::new(),
        aliases: BTreeMap::new(),
    };
    let mut cited = BTreeSet::new();
    let mut migrator = Migrator::offline();
    let mut names = BTreeSet::new();
    for path in tree_files(&root.join("repos"))? {
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            return invalid(format!("{} is not a store1 repository", path.display()));
        }
        let bytes = fs::read(&path)?;
        let old: Value = serde_json::from_slice(&bytes).map_err(|e| crate::failure::Failure {
            kind: Kind::Invalid,
            detail: format!("{}: {e}", path.display()),
        })?;
        let org = old["repo"]["org"]
            .as_str()
            .ok_or_else(|| crate::failure::Failure {
                kind: Kind::Invalid,
                detail: "store1 repo has no org".into(),
            })?;
        let model = old["repo"]["name"]
            .as_str()
            .ok_or_else(|| crate::failure::Failure {
                kind: Kind::Invalid,
                detail: "store1 repo has no name".into(),
            })?;
        let name = format!("{org}/{model}");
        if path != root.join(format!("repos/{name}.json")) {
            return invalid(format!(
                "{} does not match repository {name}",
                path.display()
            ));
        }
        if !names.insert(name.to_ascii_lowercase()) {
            return invalid(format!("duplicate store1 repository {name}"));
        }
        let checkpoints = old["checkpoints"]
            .as_array()
            .ok_or_else(|| crate::failure::Failure {
                kind: Kind::Invalid,
                detail: format!("{name}: checkpoints is not an array"),
            })?;
        let releases = old["releases"]
            .as_array()
            .ok_or_else(|| crate::failure::Failure {
                kind: Kind::Invalid,
                detail: format!("{name}: releases is not an array"),
            })?;
        let roots = checkpoints
            .iter()
            .map(|c| object(&c["manifest"], &name))
            .collect::<Result<Vec<_>>>()?;
        if roots.iter().collect::<BTreeSet<_>>().len() != roots.len()
            || checkpoints.iter().any(|c| !c["published"].is_boolean())
        {
            return invalid(format!(
                "{name}: duplicate checkpoint or invalid published fact"
            ));
        }
        cited.extend(roots.iter().cloned());
        let mut versions = Vec::new();
        for r in releases {
            if !r["version"].is_string()
                || !r["yanked"].is_boolean()
                || !r["revision"].as_u64().is_some_and(|r| r > 0)
            {
                return invalid(format!(
                    "{name}: invalid release version, revision or yanked fact"
                ));
            }
            let mut lanes = BTreeMap::new();
            for l in r["lanes"]
                .as_array()
                .ok_or_else(|| crate::failure::Failure {
                    kind: Kind::Invalid,
                    detail: format!("{name}: release lanes is not an array"),
                })?
            {
                let lane = l["lane"].as_str().ok_or_else(|| crate::failure::Failure {
                    kind: Kind::Invalid,
                    detail: format!("{name}: lane has no name"),
                })?;
                let o = object(&l["manifest"], &name)?;
                if !checkpoints.iter().any(|c| {
                    c["published"] == true && object(&c["manifest"], &name).is_ok_and(|m| m == o)
                }) || lanes.insert(lane, o).is_some()
                {
                    return invalid(format!(
                        "{name}: duplicate lane or unretained release checkpoint"
                    ));
                }
            }
            versions.push(json!({"version": r["version"], "lanes": lanes, "yanked": r["yanked"]}));
        }
        let local = org.eq_ignore_ascii_case("local");
        let selection = if local {
            if checkpoints.len() != 1 || !releases.is_empty() {
                return invalid(format!(
                    "{name}: local alias must have exactly one checkpoint and no release"
                ));
            }
            Some(
                checkpoints[0]["source_selection"]
                    .as_str()
                    .filter(|s| digest(s))
                    .ok_or_else(|| crate::failure::Failure {
                        kind: Kind::Invalid,
                        detail: format!("{name}: local alias has no exact source_selection"),
                    })?
                    .to_owned(),
            )
        } else {
            None
        };
        let doc = json!({"name": name, "checkpoints": roots, "releases": versions});
        let mut repo = convert(root, &stage, &mut migrator, doc, &mut report)?;
        if let Some(selection) = selection {
            let tree = report
                .checkpoints
                .iter()
                .find(|c| c.manifest == roots[0])
                .expect("converted checkpoint")
                .tree
                .clone();
            repo.set_tag("main", tree.clone());
            repo.set_tag(&selection, tree);
            report.aliases.insert(name.clone(), selection);
        }
        stage.update_ref(&name, |_| Some(repo))?;
        report.repos += 1;
    }
    let orphaned: Vec<_> = all.difference(&cited).cloned().collect();
    if !orphaned.is_empty() {
        let name = "local/store1-recovered";
        if names.contains(name) {
            return fail(
                Kind::Conflict,
                format!("{name} already exists; no retained checkpoint was renamed"),
            );
        }
        let repo = convert(
            root,
            &stage,
            &mut migrator,
            json!({"name": name, "checkpoints": orphaned}),
            &mut report,
        )?;
        stage.update_ref(name, |_| Some(repo))?;
        names.insert(name.into());
        report.repos += 1;
    }
    report
        .checkpoints
        .sort_by(|a, b| a.manifest.cmp(&b.manifest));
    report.checkpoints.dedup_by(|a, b| a.manifest == b.manifest);
    let mut records = BTreeMap::new();
    for dir in ["refs", "uses"] {
        for path in tree_files(&stage.root().join(dir))? {
            let bytes = fs::read(&path)?;
            if dir == "refs" && !names.contains(&Repo::from_json(&bytes)?.name.to_ascii_lowercase())
            {
                continue;
            }
            records.insert(
                path.strip_prefix(stage.root())
                    .expect("inside stage")
                    .to_owned(),
                bytes,
            );
        }
    }
    sync_dirs(stage.root())?;
    Ok(Prepared {
        format: "tensorfs.store1-migration/1".into(),
        original_format,
        report,
        records,
    })
}

fn convert(
    root: &Path,
    stage: &Store,
    migrator: &mut Migrator<'_>,
    doc: Value,
    report: &mut Report,
) -> Result<Repo> {
    let mut migrated =
        migrator.repo(&serde_json::to_vec(&doc).expect("JSON"), &|o| read(root, o))?;
    for (o, b) in &migrated.objects {
        stage.put(&b[..], Some(o))?;
    }
    // A released checkpoint's identity is also retained privately, including an old public
    // checkpoint no release currently names. The original published facts stay in repos/.
    for reference in doc["checkpoints"].as_array().into_iter().flatten() {
        let manifest = object(reference, "checkpoint")?;
        let one = migrator.repo(
            &serde_json::to_vec(&json!({"name": doc["name"], "checkpoints": [&manifest]}))
                .expect("JSON"),
            &|o| read(root, o),
        )?;
        let tree = one.repo.tags[0].commit.clone();
        for tag in one.repo.tags {
            match migrated.repo.tag(&tag.name) {
                Some(old) if old != &tag => {
                    return fail(
                        Kind::Conflict,
                        format!("{} has conflicting tag {}", migrated.repo.name, tag.name),
                    )
                }
                Some(_) => {}
                None => migrated.repo.tags.push(tag),
            }
        }
        for (o, b) in one.objects {
            stage.put(&b[..], Some(&o))?;
        }
        report.checkpoints.push(Checkpoint { manifest, tree });
    }
    Ok(migrated.repo)
}

fn link(from: &Path, to: &Path) -> Result<()> {
    regular(from)?;
    fs::create_dir_all(to.parent().expect("object parent"))?;
    match fs::hard_link(from, to) {
        Ok(()) => {
            durable::sync_dir(to.parent().expect("object parent")).map_err(std::io::Error::from)?
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            if regular(from)?.len() != regular(to)?.len()
                || durable::FileId::at(from).map_err(std::io::Error::from)?
                    != durable::FileId::at(to).map_err(std::io::Error::from)?
            {
                return fail(
                    Kind::Integrity,
                    format!("{} differs from retained source", to.display()),
                );
            }
        }
        Err(e) => return Err(e.into()),
    }
    Ok(())
}

fn publish(root: &Path, work: &Path, p: &Prepared) -> Result<()> {
    let stage = work.join("store3");
    for path in tree_files(&stage.join("blobs"))? {
        link(
            &path,
            &root.join(path.strip_prefix(&stage).expect("stage object")),
        )?;
    }
    for dir in ["refs", "holds", "files", "uses", "tmp"] {
        fs::create_dir_all(root.join(dir))?;
    }
    for (relative, bytes) in &p.records {
        if relative.is_absolute()
            || !(relative.starts_with("refs") || relative.starts_with("uses"))
            || relative
                .components()
                .any(|c| !matches!(c, std::path::Component::Normal(_)))
        {
            return invalid("prepared record has an invalid relative path");
        }
        let path = root.join(relative);
        fs::create_dir_all(path.parent().expect("metadata parent"))?;
        match fs::read(&path) {
            Ok(old) if old != *bytes => {
                return fail(
                    Kind::Conflict,
                    format!("{} differs from the prepared migration", path.display()),
                )
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                durable::replace(&path, bytes, 0o600).map_err(std::io::Error::from)?
            }
            Err(e) => return Err(e.into()),
        }
    }
    for path in tree_files(&root.join("refs"))? {
        if !p
            .records
            .contains_key(path.strip_prefix(root).expect("inside root"))
        {
            return fail(
                Kind::Conflict,
                format!("{} is outside the prepared migration", path.display()),
            );
        }
    }
    for dir in ["blobs", "refs", "holds", "files", "uses"] {
        sync_dirs(&root.join(dir))?;
    }
    durable::sync_dir(root).map_err(std::io::Error::from)?;
    Ok(())
}
