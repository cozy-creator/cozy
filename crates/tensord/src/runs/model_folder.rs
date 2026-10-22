//! A model upload's retained input tree as a plain CozyTensors folder. Only files the caller
//! staged are admitted; missing tensor objects stay missing until the upload's endpoint check.
use std::fs;

use tensorfs::{format::ObjectRef, store::{Checkpoint, Store}};

use super::{domain, refused, InputFile, Refused, Runs, Source, Spec, Stoppable};

pub(super) struct Folder {
    pub store: Store,
    pub tree: ObjectRef,
    pub header: ObjectRef,
    pub variants: Vec<String>,
    // Opening every variant retains the complete tree and its present closure during upload.
    _checkpoints: Vec<Checkpoint>,
    _staging: tempfile::TempDir,
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
    let scratch = runs.objects.cas().root().join("tmp");
    fs::create_dir_all(&scratch)?;
    let staging = tempfile::Builder::new().prefix("model-folder-").tempdir_in(scratch)?;
    let folder = staging.path().join("folder");
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
    let store = Store::open(staging.path().join("store")).map_err(invalid)?;
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
