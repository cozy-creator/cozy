//! `cozy model list|download|upload`: models on machines. Download and upload are warm runs on
//! the machine `--on` names. The machine fetches,
//! makes and publishes the bytes itself, so no weight passes through this computer; a local file
//! goes to the machine as the run's input.
use std::io::Read as _;
use std::path::Path;
use std::process::ExitCode;

use protocol::calcifer::{Kind as RunKind, Run, StagedInput, Status, SubmitRun};
use protocol::error::{Error, Kind};
use protocol::ids::Digest;
use protocol::machine::{InputFile, ModelChoice, RunKind as SpecKind, RunSpec};
use serde::Serialize;
use sha2::{Digest as _, Sha256};

use super::refs::{self, ModelRef, Repo};
use super::{Download, List, Upload};
use crate::Context;
use crate::output::{self, Record, Table};
use crate::run::Placement;

fn usage(code: &str, message: impl Into<String>) -> Error {
    Error::new(Kind::Usage, code, message)
}

struct ModelInput {
    choice: ModelChoice,
    staged: Vec<StagedInput>,
    files: Vec<InputFile>,
}

impl From<(ModelChoice, Option<StagedInput>)> for ModelInput {
    fn from((choice, staged): (ModelChoice, Option<StagedInput>)) -> Self {
        Self { choice, staged: staged.into_iter().collect(), files: vec![] }
    }
}

/// A plain CozyTensors folder travels as an ordinary retained input tree. The machine admits
/// its headers without importing or fetching the tensor objects a thin folder omits.
fn folder(path: &Path, profiles: &[String], manifests: &Path, memo: &Path) -> Result<ModelInput, Error> {
    if !profiles.is_empty() {
        return Err(usage("model_profile_unused", "a CozyTensors folder is already converted; omit --source-profile"));
    }
    let staged = crate::run::tree::stage(path, Some(memo), manifests)?;
    let input = InputFile {
        field: "model".into(),
        digest: staged.manifest.digest.to_string(),
        length: staged.manifest.length,
        media_type: crate::run::tree::MEDIA.into(),
        order: 0,
    };
    let mut files = staged.files;
    files.push(staged.manifest);
    Ok(ModelInput {
        choice: ModelChoice { parameter: "model".into(), source: "input://model".into(), ..Default::default() },
        staged: files,
        files: vec![input],
    })
}

/// The model a warm run makes present: a reference, or a local `.safetensors` file.
fn source(raw: &str, profiles: &[String]) -> Result<(ModelChoice, Option<StagedInput>), Error> {
    let (mut choice, staged) = if raw.starts_with('.') || Path::new(raw).is_absolute() {
        let staged = measure(Path::new(raw))?;
        let name = Path::new(raw).file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
        let choice = ModelChoice {
            parameter: "model".into(),
            source: format!("object://{}/{}", staged.digest, escape(&name)),
            ..Default::default()
        };
        (choice, Some(staged))
    } else {
        (ModelRef::parse(raw)?.choice("model"), None)
    };
    if !profiles.is_empty() && choice.source.is_empty() {
        return Err(usage("model_profile_unused", "--source-profile converts a provider source or a local file"));
    }
    choice.profiles = profiles.to_vec();
    Ok((choice, staged))
}

/// A local model file, measured once: one regular `.safetensors` file.
fn measure(path: &Path) -> Result<StagedInput, Error> {
    let refused = |why: String| Error::new(Kind::Validation, "model_file_refused", why);
    let regular = std::fs::symlink_metadata(path).is_ok_and(|m| m.is_file());
    if !regular || !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("safetensors")) {
        return Err(refused(format!("{} is not one regular .safetensors file", path.display())));
    }
    let mut file = std::fs::File::open(path).map_err(|e| refused(e.to_string()))?;
    let (mut hash, mut buf, mut length) = (Sha256::new(), vec![0; 1 << 20], 0u64);
    loop {
        match file.read(&mut buf).map_err(|e| refused(e.to_string()))? {
            0 => break,
            n => {
                hash.update(&buf[..n]);
                length += n as u64;
            }
        }
    }
    Ok(StagedInput {
        path: std::path::absolute(path).map_err(|e| refused(e.to_string()))?,
        digest: Digest::from_bytes(hash.finalize().into()),
        length,
    })
}

/// A file name as one URI path segment.
fn escape(name: &str) -> String {
    name.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b'-' | b'+' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// Submits one warm run of `choice` to `destination` (a `local/` alias, a Hub repository, or
/// none) and follows it.
async fn warm(
    ctx: &Context,
    placement: &Placement,
    detach: bool,
    target: String,
    input: ModelInput,
    destination: String,
    bits: u32,
) -> Result<ExitCode, Error> {
    let spec = RunSpec {
        kind: SpecKind::Warm,
        bits,
        models: vec![input.choice],
        inputs: input.files,
        weights_destination: destination,
        ..Default::default()
    };
    let submit = SubmitRun {
        agreed: None,
        on: placement.on(),
        hub: ctx.hub()?.origin.into(),
        kind: RunKind::Warm,
        target,
        spec,
        inputs: input.staged,
        out: None,
        warnings: vec![],
    };
    let submitted = crate::run::submit(ctx, submit, None).await?;
    let run = crate::run::follow(ctx, submitted.run, detach).await?;
    made(ctx, &run);
    Ok(crate::run::exit_code(&run))
}

/// What the machine made (TensorD's `result.models[0]`), after `follow`'s record; `--json`
/// carries it in the run's result.
fn made(ctx: &Context, run: &Run) {
    let Some(model) = run.result.as_ref().map(|r| &r["models"][0]).filter(|m| m.is_object()) else {
        return;
    };
    if ctx.json || run.status != Status::Completed {
        return;
    }
    let text = |key: &str| model[key].as_str().filter(|s| !s.is_empty());
    let mut record = Record::new();
    for key in ["repository", "release", "lane", "checkpoint", "resolved"] {
        if let Some(value) = text(key) {
            record = record.field(key, value);
        }
    }
    let published = &model["published"];
    let (destination, checkpoint) = (published["destination"].as_str(), published["checkpoint"].as_str());
    if let (Some(destination), Some(checkpoint)) = (destination, checkpoint) {
        record = record.field("published", format!("{destination}#{checkpoint}"));
    }
    record.print(ctx);
    if let (Some(destination), Some(checkpoint)) = (destination, checkpoint)
        && !destination.starts_with("local/")
    {
        let model = destination.split('@').next().unwrap_or(destination);
        let at = published["tag"].as_str().filter(|t| !t.is_empty()).unwrap_or(checkpoint);
        output::next(ctx, &[&format!("cozy model publish {model} --release <version> --commit {at}")]);
    }
}

pub(crate) async fn download(ctx: &Context, args: Download) -> Result<ExitCode, Error> {
    let destination = match args.alias.as_deref().map(Repo::parse).transpose()? {
        Some(repo) if repo.org == "local" => repo.to_string(),
        Some(_) => return Err(usage("model_alias_invalid", "the name a machine keeps a model under is local/<name>")),
        None => String::new(),
    };
    let made = source(&args.model, &[])?;
    if made.1.is_some() && destination.is_empty() {
        return Err(usage("model_alias_missing", "a local file is kept under a name: add local/<name>"));
    }
    warm(ctx, &args.placement, args.detach, args.model.clone(), made.into(), destination, args.bits.as_deref().map_or(0, |b| b.parse().expect("clap checked bits"))).await
}

pub(crate) async fn upload(ctx: &Context, args: Upload) -> Result<ExitCode, Error> {
    let destination = refs::destination(&args.destination)?;
    if !ctx.signed_in(&ctx.hub()?) {
        // The machine publishes under this computer's sign-in; refuse before anything is made.
        return Err(Error::new(Kind::Credential, "not_logged_in", "an upload publishes as your account: sign in first")
            .next(ctx.login(&ctx.hub()?)));
    }
    let path = Path::new(&args.source);
    let made = if (args.source.starts_with('.') || path.is_absolute()) && path.is_dir() {
        folder(path, &args.profiles, &ctx.home.root().join("cache/trees"), &ctx.home.root().join("cache/digests"))?
    } else {
        source(&args.source, &args.profiles)?.into()
    };
    let target = format!("{} → {destination}", args.source);
    warm(ctx, &args.placement, args.detach, target, made, destination, 0).await
}

/// One checkpoint a machine holds, and its repository's bytes there.
#[derive(Serialize)]
struct Held {
    machine: String,
    model: String,
    release: String,
    lane: String,
    checkpoint: String,
    yanked: bool,
    bytes: u64,
    /// The repository's bytes other held models share.
    shared_bytes: u64,
}

/// What every machine holds (or the one `--on` names), from its latest Status through Calcifer.
pub(crate) async fn list(ctx: &Context, args: List) -> Result<ExitCode, Error> {
    let machines = ctx.calcifer().await?.machines().await?.machines;
    if let Some(on) = args.on.as_ref().filter(|on| !machines.iter().any(|m| m.name == **on)) {
        return Err(Error::new(Kind::NotFound, "machine_not_found", format!("no machine named {on}")).next("cozy machine list"));
    }
    let (mut rows, mut stores, mut silent) = (vec![], vec![], vec![]);
    for machine in machines.iter().filter(|m| args.on.as_ref().is_none_or(|on| *on == m.name)) {
        let Some(status) = &machine.status else {
            silent.push(machine.name.as_str());
            continue;
        };
        stores.push(format!("{}: {} in {} models", machine.name, output::bytes(status.models_bytes), status.models.len()));
        for model in &status.models {
            let held = |release: &str, lane: &str, checkpoint: &str, yanked| Held {
                machine: machine.name.clone(),
                model: model.repository.clone(),
                release: release.into(),
                lane: lane.into(),
                checkpoint: checkpoint.into(),
                yanked,
                bytes: model.total_bytes,
                shared_bytes: model.total_bytes.saturating_sub(model.unique_bytes),
            };
            if model.checkpoints.is_empty() {
                rows.push(held("", "", "", false));
            }
            rows.extend(model.checkpoints.iter().map(|c| held(&c.release, &c.lane, &c.manifest, c.yanked)));
        }
    }
    if ctx.json {
        output::json(&serde_json::json!({ "models": rows }));
        return Ok(ExitCode::SUCCESS);
    }
    let mut table = Table::new(ctx, ["machine", "model", "release", "lane", "size", "shared"]);
    for row in &rows {
        let lane = if row.yanked { format!("{} (yanked)", row.lane) } else { row.lane.clone() };
        let (size, shared) = (output::bytes(row.bytes), output::bytes(row.shared_bytes));
        table = table.row([row.machine.as_str(), &row.model, &row.release, &lane, &size, &shared]);
    }
    // A store's total counts shared content once.
    for store in stores {
        table = table.note(store);
    }
    if !silent.is_empty() {
        table = table.note(format!("no Status yet from {}", silent.join(", ")));
    }
    table.print();
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_folder_is_one_retained_tree_input_with_every_local_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("model");
        std::fs::create_dir_all(root.join("chunks")).unwrap();
        for (name, bytes) in [("README.md", b"readme".as_slice()), ("LICENSE", b"license"),
            ("model.bf16.cozytensors", b"header1"), ("model.int8.cozytensors", b"header2"),
            ("chunks/plan", b"plan")]
        {
            std::fs::write(root.join(name), bytes).unwrap();
        }
        let input = folder(&root, &[], &dir.path().join("manifests"), &dir.path().join("digests")).unwrap();
        assert_eq!(input.choice.source, "input://model");
        assert_eq!(input.files.len(), 1);
        assert_eq!(input.files[0].media_type, crate::run::tree::MEDIA);
        assert_eq!(input.staged.len(), 6, "all five files and their immutable manifest");
        let manifest = input.staged.last().unwrap();
        assert_eq!(input.files[0].digest, manifest.digest.to_string());
        let document: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest.path).unwrap()).unwrap();
        let names: Vec<_> = document["entries"].as_array().unwrap().iter().map(|e| e["path"].as_str().unwrap()).collect();
        assert_eq!(names, ["LICENSE", "README.md", "chunks/plan", "model.bf16.cozytensors", "model.int8.cozytensors"]);
        assert!(folder(&root, &["convert/1".into()], &dir.path().join("manifests"), &dir.path().join("digests")).is_err());
    }

    #[test]
    fn a_local_file_is_named_by_its_content_and_an_escaped_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("my lora.safetensors");
        std::fs::write(&path, b"tensor bytes").unwrap();
        let (choice, staged) = source(path.to_str().unwrap(), &["as-is/1".into()]).unwrap();
        let staged = staged.unwrap();
        assert_eq!(staged.digest, Digest::of(b"tensor bytes"));
        assert_eq!(staged.length, 12);
        assert_eq!(choice.source, format!("object://{}/my%20lora.safetensors", staged.digest));
        assert_eq!(choice.profiles, ["as-is/1"]);
        std::fs::write(dir.path().join("x.bin"), b"x").unwrap();
        let refused = source(dir.path().join("x.bin").to_str().unwrap(), &[]).unwrap_err();
        assert_eq!(refused.code, "model_file_refused");
        assert_eq!(source("fidika/sdxl", &["as-is/1".into()]).unwrap_err().code, "model_profile_unused");
    }
}
