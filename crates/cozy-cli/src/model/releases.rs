//! `cozy model publish|retarget|yank|delete`: a model's releases on the Hub, as its owner.
use std::collections::{BTreeMap, HashMap};
use std::process::ExitCode;

use cozy_proto::error::{Error, Kind};
use hub_client::types::{ModelLaneRetarget, ModelRelease, ModelReleaseUpdate};
use serde::Serialize;

use super::refs::{ModelRef, Repo, checkpoint, label};
use super::{Delete, Publish, Retarget, Yank};
use crate::Context;
use crate::context::hub_answer;
use crate::output::{self, Record};

fn usage(code: &str, message: impl Into<String>) -> Error {
    Error::new(Kind::Usage, code, message)
}

/// A release as the Hub answered a change to it.
#[derive(Serialize)]
struct Changed {
    model: String,
    release: String,
    revision: i64,
    yanked: bool,
    lanes: BTreeMap<String, String>,
    /// False when the release already was as asked.
    changed: bool,
}

fn show(ctx: &Context, repo: &Repo, answer: ModelRelease) -> Result<ExitCode, Error> {
    let lanes = answer.lanes.unwrap_or_default().into_iter().map(|l| (l.lane, l.checkpoint_id));
    let view = Changed {
        model: repo.to_string(),
        release: answer.release,
        revision: answer.revision,
        yanked: answer.yanked,
        lanes: lanes.collect(),
        changed: answer.changed,
    };
    if ctx.json {
        output::json(&view);
        return Ok(ExitCode::SUCCESS);
    }
    let mut record = Record::new()
        .field("model", &view.model)
        .field("release", &view.release)
        .field("revision", view.revision)
        .field("yanked", view.yanked)
        .field("changed", view.changed);
    for (lane, checkpoint) in &view.lanes {
        record = record.field(&format!("lane {lane}"), checkpoint);
    }
    record.print(ctx);
    Ok(ExitCode::SUCCESS)
}

pub(crate) async fn publish(ctx: &Context, args: Publish) -> Result<ExitCode, Error> {
    let repo = Repo::hub(&args.model)?;
    let release = label("--release", &args.release)?;
    let mut set = HashMap::new();
    for spec in &args.lanes {
        let (name, id) = spec
            .split_once('=')
            .ok_or_else(|| usage("model_lane_invalid", format!("--lane {spec:?} is not name=sha256:<checkpoint>")))?;
        if set.insert(label("--lane", name)?, checkpoint(id)?).is_some() {
            return Err(usage("model_lane_invalid", format!("--lane names {name:?} twice")));
        }
    }
    let mut remove = Vec::new();
    for name in &args.remove {
        let name = label("--remove-lane", name)?;
        if set.contains_key(&name) || remove.contains(&name) {
            return Err(usage("model_lane_invalid", format!("lane {name:?} is named twice")));
        }
        remove.push(name);
    }
    if set.is_empty() && remove.is_empty() {
        return Err(usage("model_publish_empty", "give --lane name=sha256:… or --remove-lane name"));
    }
    let client = ctx.hub_signed_in(&ctx.hub()?)?;
    // The write names the revision this read saw: a concurrent change refuses instead of losing.
    let current = match hub_answer(client.get_model(&repo.org, &repo.name).await) {
        Ok(card) => card.releases.into_iter().find(|r| r.release == release),
        Err(e) if e.kind == Kind::NotFound => None,
        Err(e) => return Err(e),
    };
    let mut lanes: HashMap<&str, &str> = current
        .iter()
        .flat_map(|r| r.lanes.iter().map(|l| (l.lane.as_str(), l.manifest_id.as_str())))
        .collect();
    lanes.extend(set.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    lanes.retain(|lane, _| !remove.iter().any(|r| r == lane));
    if lanes.is_empty() {
        return Err(usage("model_release_empty", format!("{repo}@{release} would keep no lane"))
            .remedy("yank the release to withdraw it whole")
            .next(format!("cozy model yank {repo} --release {release}")));
    }
    let update = ModelReleaseUpdate {
        expected_revision: current.as_ref().and_then(|r| r.revision).unwrap_or(0),
        set_lanes: set,
        remove_lanes: remove,
    };
    let answer = hub_answer(client.update_model_release(&repo.org, &repo.name, &release, &update).await)?;
    show(ctx, &repo, answer)
}

pub(crate) async fn retarget(ctx: &Context, args: Retarget) -> Result<ExitCode, Error> {
    let repo = Repo::hub(&args.model)?;
    let release = label("--release", &args.release)?;
    let lane = label("--lane", &args.lane)?;
    let body = ModelLaneRetarget { checkpoint_id: checkpoint(&args.to)? };
    let client = ctx.hub_signed_in(&ctx.hub()?)?;
    let answer = hub_answer(client.retarget_model_lane(&repo.org, &repo.name, &release, &lane, &body).await)?;
    show(ctx, &repo, answer)
}

pub(crate) async fn yank(ctx: &Context, args: Yank) -> Result<ExitCode, Error> {
    let repo = Repo::hub(&args.model)?;
    let release = label("--release", &args.release)?;
    let client = ctx.hub_signed_in(&ctx.hub()?)?;
    let answer = hub_answer(client.yank_model_release(&repo.org, &repo.name, &release).await)?;
    show(ctx, &repo, answer)
}

#[derive(Serialize)]
struct Deleted {
    model: String,
    /// One unreleased checkpoint; absent when the whole model went.
    #[serde(skip_serializing_if = "Option::is_none")]
    checkpoint: Option<String>,
    /// False when it was already gone.
    removed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    checkpoints: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reclaimable_bytes: Option<i64>,
}

/// Never prompts: `--yes` is the confirmation, and without it nothing is sent.
pub(crate) async fn delete(ctx: &Context, args: Delete) -> Result<ExitCode, Error> {
    let (repo, one) = match ModelRef::parse(&args.model)? {
        ModelRef::Checkpoint { repo, manifest } => (repo, Some(manifest)),
        ModelRef::Hub { repo, release: None, lane: None } => (repo, None),
        _ => return Err(usage("model_ref_invalid", format!("{:?}: delete takes org/name or org/name#sha256:…", args.model))),
    };
    let hub = ctx.hub()?;
    if !args.yes {
        return Err(usage("model_delete_unconfirmed", format!("deleting {} from {} cannot be undone", args.model, hub.origin))
            .remedy("run it again with --yes"));
    }
    let client = ctx.hub_signed_in(&hub)?;
    let view = match one {
        Some(manifest) => {
            let answer = hub_answer(client.remove_model_checkpoint(&repo.org, &repo.name, &manifest).await)?;
            Deleted {
                model: repo.to_string(),
                removed: answer.removed,
                checkpoint: Some(manifest),
                checkpoints: None,
                reclaimable_bytes: None,
            }
        }
        None => {
            let answer = hub_answer(client.delete_model(&repo.org, &repo.name).await)?;
            Deleted {
                model: answer.model,
                checkpoint: None,
                removed: true,
                checkpoints: Some(answer.checkpoints),
                reclaimable_bytes: Some(answer.reclaimable_bytes),
            }
        }
    };
    if ctx.json {
        output::json(&view);
    } else {
        Record::of(&view).note("The Hub's GC reclaims only objects no other model references.").print(ctx);
    }
    Ok(ExitCode::SUCCESS)
}
