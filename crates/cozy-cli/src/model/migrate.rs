//! `cozy model migrate`: dev builds only (feature `migrate`, xs-045 U12; deleted with the old
//! format). Moves models' checkpoints from before TensorFS's format cut onto the new format, by
//! each model's own rows at the Hub, its yanked releases and unreleased checkpoints included:
//!
//! 1. the machine `--on` names fetches each old checkpoint, rewrites it, checks every tensor
//!    bit-identical and publishes the result as a private checkpoint (a warm run, like an upload);
//! 2. each release with a lane on an old checkpoint moves to the new ones under its revision,
//!    yanked releases too (they stay yanked);
//! 3. the old checkpoints leave the model, so its repo file names none.
//!
//! Each old checkpoint's audit JSON lands in `--audit`. Running it again converges: the Hub
//! already holds what was published, and rows already moved are not old. `--local` instead
//! rewrites the `local/` models of a pre-cut store on the machine into its own store.
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use cozy_proto::calcifer::{Kind as RunKind, Run, Status, SubmitRun};
use cozy_proto::error::{Error, Kind};
use cozy_proto::machine::{MIGRATE, ModelChoice, RunKind as SpecKind, RunSpec};
use hub_client::types::ModelReleaseUpdate;
use serde::Deserialize;
use serde_json::{Value, json};

use super::refs::ModelRef;
use crate::Context;
use crate::context::{hub_answer, hub_error};
use crate::output;

#[derive(clap::Args, Debug)]
pub(crate) struct Args {
    /// Models (org/name), or one lane's checkpoint (org/name@release/lane), at the Hub.
    #[arg(required_unless_present = "local", conflicts_with = "local")]
    models: Vec<String>,
    /// Instead: the local/ models of the pre-cut store at this path on the machine.
    #[arg(long, value_name = "STORE")]
    local: Option<PathBuf>,
    /// Where each old checkpoint's audit JSON is written.
    #[arg(long, value_name = "DIR", default_value = "migrate-audit")]
    audit: PathBuf,
    #[command(flatten)]
    placement: crate::run::Placement,
}

/// One index row of a model at the Hub (`GET /v1/models/{org}/{name}/rows`).
#[derive(Deserialize, Clone, Debug)]
struct Row {
    release: String,
    lane: String,
    revision: i64,
    yanked: bool,
    checkpoint_id: String,
    manifest_length: u64,
    v1: bool,
}

fn failed(code: &str, message: impl Into<String>) -> Error {
    Error::new(Kind::Conflict, code, message)
}

/// Submits one migration run and follows it to its end.
async fn run(ctx: &Context, args: &Args, target: String, spec: RunSpec) -> Result<Run, Error> {
    let submit = SubmitRun {
        on: args.placement.on(),
        hub: ctx.hub()?.origin.into(),
        kind: RunKind::Warm,
        target,
        spec,
        inputs: vec![],
        out: None,
        warnings: vec![],
    };
    let submitted = crate::run::submit(ctx, submit, None).await?;
    crate::run::follow(ctx, submitted.run, false).await
}

fn write(path: &Path, doc: &Value) -> Result<(), Error> {
    let io = |e: std::io::Error| Error::new(Kind::Internal, "audit_write_failed", format!("{}: {e}", path.display()));
    std::fs::create_dir_all(path.parent().unwrap_or(Path::new("."))).map_err(io)?;
    std::fs::write(path, serde_json::to_vec_pretty(doc).expect("JSON")).map_err(io)
}

pub(crate) async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    if let Some(store) = &args.local {
        return local(ctx, &args, store).await;
    }
    let hub = ctx.hub()?;
    let client = ctx.hub_signed_in(&hub)?;
    let (mut models, mut ok) = (vec![], true);
    for spec in &args.models {
        let (repo, lane) = match ModelRef::parse(spec)? {
            ModelRef::Hub { repo, release: None, lane: None } => (repo, None),
            ModelRef::Hub { repo, release: Some(r), lane: Some(l) } => (repo, Some((r, l))),
            _ => return Err(Error::new(Kind::Usage, "model_ref_invalid", format!("{spec:?}: org/name or org/name@release/lane"))),
        };
        let rows: Vec<Row> = match serde_json::from_value(
            client.model_rows(&repo.org, &repo.name).await.map_err(hub_error)?["rows"].take(),
        ) {
            Ok(rows) => rows,
            Err(e) => return Err(failed("model_rows_invalid", format!("{repo}: {e}"))),
        };
        // The old checkpoints, each with every row that names it.
        let mut old: BTreeMap<String, Vec<Row>> = BTreeMap::new();
        for row in rows.iter().filter(|r| r.v1) {
            old.entry(row.checkpoint_id.clone()).or_default().push(row.clone());
        }
        if let Some((release, lane)) = &lane {
            old.retain(|_, named| named.iter().any(|r| r.release == *release && r.lane == *lane));
        }
        let mut moved: BTreeMap<String, (String, Value)> = BTreeMap::new();
        let mut audits: BTreeMap<String, Value> = BTreeMap::new();
        for (manifest, named) in &old {
            // A release lane reads publicly; an unreleased one, by hash under the run's capability.
            let at = named.iter().find(|r| !r.release.is_empty()).unwrap_or(&named[0]);
            let choice = ModelChoice {
                parameter: "model".into(),
                repository: repo.to_string(),
                release: at.release.clone(),
                lane: at.lane.clone(),
                manifest: manifest.clone(),
                manifest_length: at.manifest_length,
                profiles: vec![MIGRATE.into()],
                ..Default::default()
            };
            let spec = RunSpec {
                kind: SpecKind::Warm,
                models: vec![choice],
                weights_destination: repo.to_string(),
                ..Default::default()
            };
            let ran = run(ctx, &args, format!("migrate {repo}#{manifest}"), spec).await?;
            let model = ran.result.as_ref().map(|r| r["models"][0].clone()).unwrap_or_default();
            let made = model["published"]["checkpoint"].as_str().filter(|_| ran.status == Status::Completed);
            let mut audit = json!({"model": repo.to_string(), "hub": hub.origin.to_string(), "old": manifest,
                "rows": named.iter().map(|r| json!({"release": r.release, "lane": r.lane, "revision": r.revision,
                    "yanked": r.yanked})).collect::<Vec<_>>(),
                "run": ran.number, "machine": ran.machine, "migration": model["migrated"], "repointed": []});
            match made {
                Some(made) => {
                    audit["new"] = made.into();
                    moved.insert(manifest.clone(), (made.to_owned(), model));
                }
                None => {
                    ok = false;
                    audit["error"] = json!(ran.reason);
                }
            }
            audits.insert(manifest.clone(), audit);
        }
        // Each release naming a moved checkpoint, under its revision; a yanked one stays yanked.
        let mut releases: BTreeMap<&str, (i64, HashMap<String, String>)> = BTreeMap::new();
        for row in rows.iter().filter(|r| !r.release.is_empty()) {
            if let Some((made, _)) = moved.get(&row.checkpoint_id) {
                let release = releases.entry(&row.release).or_default();
                release.0 = release.0.max(row.revision);
                release.1.insert(row.lane.clone(), made.clone());
            }
        }
        let mut repointed = vec![];
        for (release, (revision, set_lanes)) in releases {
            let update = ModelReleaseUpdate { expected_revision: revision, set_lanes, remove_lanes: vec![] };
            let answer = hub_answer(client.update_model_release(&repo.org, &repo.name, release, &update).await)?;
            let lanes: BTreeMap<String, String> = answer.lanes.unwrap_or_default().into_iter().map(|l| (l.lane, l.checkpoint_id)).collect();
            for (old, audit) in audits.iter_mut() {
                let named = |l: &str| rows.iter().any(|r| r.release == release && r.lane == l && r.checkpoint_id == *old);
                let mine: BTreeMap<&String, &String> = lanes.iter().filter(|(l, _)| named(l)).collect();
                if !mine.is_empty() {
                    let entry = json!({"release": release, "revision": answer.revision, "yanked": answer.yanked, "lanes": mine});
                    audit["repointed"].as_array_mut().expect("a list").push(entry);
                }
            }
            repointed.push(json!({"release": release, "revision": answer.revision, "yanked": answer.yanked, "lanes": lanes}));
        }
        // The old checkpoints leave the model: its repo file then names none.
        for old in moved.keys() {
            let removed = hub_answer(client.remove_model_checkpoint(&repo.org, &repo.name, old).await)?;
            audits.get_mut(old).expect("audited")["removed"] = removed.removed.into();
        }
        for (old, audit) in &audits {
            let file = args.audit.join(&repo.org).join(&repo.name).join(format!("{}.json", old.trim_start_matches("sha256:")));
            write(&file, audit)?;
        }
        output::say(ctx, &format!("{repo}: {} of {} old checkpoints migrated", moved.len(), old.len()));
        models.push(json!({"model": repo.to_string(), "checkpoints": audits.into_values().collect::<Vec<_>>(),
            "releases": repointed}));
    }
    output::json(&json!({"models": models, "audit": args.audit}));
    Ok(if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE })
}

/// The local/ models of the pre-cut store at `store` on the machine, into its own store.
async fn local(ctx: &Context, args: &Args, store: &Path) -> Result<ExitCode, Error> {
    let choice = ModelChoice {
        parameter: "model".into(),
        source: store.to_string_lossy().into_owned(),
        profiles: vec![MIGRATE.into()],
        ..Default::default()
    };
    let spec = RunSpec { kind: SpecKind::Warm, models: vec![choice], ..Default::default() };
    let ran = run(ctx, args, format!("migrate local/* from {}", store.display()), spec).await?;
    let models = ran.result.as_ref().map(|r| r["models"].clone()).unwrap_or_default();
    for model in models.as_array().into_iter().flatten() {
        let name = model["repository"].as_str().unwrap_or_default().replace('/', "__");
        write(&args.audit.join(format!("{name}.json")), model)?;
    }
    output::json(&json!({"run": ran.number, "status": ran.status, "models": models, "audit": args.audit}));
    Ok(crate::run::exit_code(&ran))
}
