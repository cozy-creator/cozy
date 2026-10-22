//! A model upload's retained input tree as a plain CozyTensors folder. Only files the caller
//! staged are admitted; missing tensor objects stay missing until the upload's endpoint check.
use std::{fs::{self, File}, io::{self, Read, Write}, path::{Path, PathBuf}};

use tensorfs::{format::ObjectRef, store::{Checkpoint, Store}};

use super::{domain, refused, InputFile, Refused, Runs, Source, Spec, Stoppable};

pub(super) struct Folder {
    pub store: Store,
    pub tree: ObjectRef,
    pub header: ObjectRef,
    pub variants: Vec<String>,
    // Opening every variant retains the complete tree and its present closure during upload.
    _checkpoints: Vec<Checkpoint>,
    _staging: Staging,
}

const LEASE: &str = "cozy.model-folder-stage/1\n";

/// A lease proves this operation is live. Each directory is claimed once, before it receives
/// bytes, so an unlocked old lease can only belong to an interrupted operation. The next folder
/// upload reaps those directories; it never touches an active or unrecognized directory.
struct Staging {
    root: PathBuf,
    lease: Option<File>,
}

impl Staging {
    fn new(parent: &Path) -> io::Result<Self> {
        fs::create_dir_all(parent)?;
        Self::reap(parent)?;
        // Reapers ignore initialization until the marker is durable and its lock is held.
        let temporary = tempfile::Builder::new().prefix(".initializing-").tempdir_in(parent)?;
        let mut lease = File::options().read(true).write(true).create_new(true).open(temporary.path().join("lease"))?;
        lease.lock()?;
        lease.write_all(LEASE.as_bytes())?;
        lease.sync_all()?;
        platform::durable::sync_dir(temporary.path())?;
        let root = parent.join(uuid::Uuid::new_v4().simple().to_string());
        fs::rename(temporary.path(), &root)?;
        let _ = temporary.keep();
        let staging = Self { root, lease: Some(lease) };
        platform::durable::sync_dir(parent)?;
        Ok(staging)
    }

    fn reap(parent: &Path) -> io::Result<()> {
        for entry in fs::read_dir(parent)?.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.len() != 32 || !name.bytes().all(|c| c.is_ascii_hexdigit()) || !entry.file_type()?.is_dir() {
                continue;
            }
            let path = entry.path();
            let marker = path.join("lease");
            if !fs::symlink_metadata(&marker).is_ok_and(|m| m.is_file()) {
                continue;
            }
            let Ok(mut lease) = File::options().read(true).write(true).open(marker) else { continue };
            if lease.try_lock().is_err() {
                continue;
            }
            let mut signature = String::new();
            if (&mut lease).take(128).read_to_string(&mut signature).is_err() || signature != LEASE {
                continue;
            }
            // A claimed directory is never reopened by an operation. Close before removal for
            // Windows; a concurrent reaper may remove the same abandoned path harmlessly.
            drop(lease);
            if let Err(error) = fs::remove_dir_all(&path) {
                if error.kind() != io::ErrorKind::NotFound {
                    return Err(error);
                }
            }
        }
        Ok(())
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        drop(self.lease.take());
        if let Err(error) = fs::remove_dir_all(&self.root) {
            if error.kind() != io::ErrorKind::NotFound {
                tracing::warn!("model folder staging {}: {error}", self.root.display());
            }
        }
    }
}

pub(super) fn input<'a>(spec: &'a Spec, choice: &domain::ModelChoice) -> Result<Option<&'a InputFile>, Refused> {
    let Some(field) = choice.source.strip_prefix("input://") else { return Ok(None) };
    let invalid = || refused("model_folder_invalid", "a folder upload names one retained tree input, without model selectors or conversion profiles");
    let destination = spec.weights_destination.trim_start_matches("model://");
    if field.is_empty() || !spec.warm || !matches!(&spec.source, Source::Models)
        || spec.remove || spec.set.is_some() || spec.models.len() != 1
        || destination.is_empty() || destination.starts_with("local/")
        || !choice.repository.is_empty() || !choice.release.is_empty() || !choice.lane.is_empty()
        || choice.manifest.is_some() || !choice.profiles.is_empty() || !choice.adapters.is_empty()
    {
        return Err(invalid());
    }
    let mut inputs = spec.inputs.iter().filter(|input| input.input_id == field);
    let input = inputs.next().ok_or_else(invalid)?;
    if inputs.next().is_some() || input.media_type != crate::gpu_service::TREE_MEDIA {
        return Err(invalid());
    }
    Ok(Some(input))
}

pub(super) fn prepare(
    runs: &Runs, actor: &str, spec: &Spec, choice: &domain::ModelChoice, stoppable: &Stoppable,
) -> Result<Option<Folder>, Refused> {
    let Some(input) = input(spec, choice)? else { return Ok(None) };
    let missing = || refused("model_folder_absent", "the retained model folder input is absent or changed");
    let (manifest, length) = runs.objects.path(actor, &input.digest)?.ok_or_else(missing)?;
    if length != input.length {
        return Err(missing());
    }
    let files = crate::trees::parse(&fs::read(manifest)?)?;
    let scratch = runs.objects.cas().root().join("tmp/model-folders");
    let staging = Staging::new(&scratch)?;
    let folder = staging.root.join("folder");
    fs::create_dir(&folder)?;
    for (name, object) in files {
        if stoppable.stop.is_cancelled() {
            return Err(refused(crate::published::CANCELED, "the folder upload was canceled before admission"));
        }
        let (source, length) = runs.objects.path(actor, &object.id())?.ok_or_else(missing)?;
        if length != object.length {
            return Err(missing());
        }
        let path = folder.join(name);
        fs::create_dir_all(path.parent().expect("a tree member has a parent"))?;
        // The input CAS is immutable. TensorFS reads these links and verifies admission; it
        // never rewrites the folder. Copy only if the two stores occupy different volumes.
        if fs::hard_link(&source, &path).is_err() {
            fs::copy(&source, &path)?;
        }
    }
    let invalid = |error| refused("model_folder_invalid", format!("{error}"));
    // This payload view contains ONLY this caller's staged files. The shared model cache may
    // hold another actor's tensor bytes: citing their hash in a fabricated header grants no
    // right to upload them. An unstaged object can only be reused by the endpoint's own HEAD.
    let store = Store::open(staging.root.join("store")).map_err(invalid)?;
    let (tree, _) = store.admit_folder(&folder).map_err(invalid)?;
    let parsed = store.tree(&tree).map_err(invalid)?;
    let variants: Vec<String> = parsed.variants().map(|(name, _)| name.to_owned()).collect();
    if variants.is_empty() {
        return Err(refused("model_folder_invalid", "the folder contains no CozyTensors variant headers"));
    }
    let checkpoints: Vec<Checkpoint> = variants.iter()
        .map(|variant| store.variant(&tree, Some(variant)).map_err(invalid))
        .collect::<Result<_, _>>()?;
    Ok(Some(Folder {
        header: checkpoints[0].id().clone(), tree, variants, store, _checkpoints: checkpoints,
        _staging: staging,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_recovery_removes_only_abandoned_owned_directories() {
        let root = tempfile::tempdir().unwrap();
        let active = Staging::new(root.path()).unwrap();
        fs::write(active.root.join("payload"), b"live").unwrap();
        let mut abandoned = Staging::new(root.path()).unwrap();
        fs::write(abandoned.root.join("payload"), b"interrupted").unwrap();
        let abandoned_path = abandoned.root.clone();
        // Simulate process loss: the kernel releases the lease, but no Drop removes the data.
        drop(abandoned.lease.take());
        std::mem::forget(abandoned);
        let unknown = root.path().join("0".repeat(32));
        fs::create_dir(&unknown).unwrap();
        fs::write(unknown.join("lease"), b"not our lease").unwrap();
        let recovering = Staging::new(root.path()).unwrap();
        assert!(!abandoned_path.exists());
        assert_eq!(fs::read(active.root.join("payload")).unwrap(), b"live");
        assert!(unknown.exists(), "an unknown directory is never collected");
        let recovered = recovering.root.clone();
        drop(recovering);
        assert!(!recovered.exists(), "normal completion removes staging immediately");
    }
}
