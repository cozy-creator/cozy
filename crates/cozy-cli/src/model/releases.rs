//! `cozy model publish|retarget|yank|delete`: a model's releases on the Hub, as its owner.
use std::collections::{BTreeMap, HashMap};
use std::process::ExitCode;

use cozy_proto::error::{Error, Kind};
use hub_client::types::{ModelLaneRetarget, ModelRelease, ModelReleaseUpdate};
use serde::Serialize;

use super::refs::{Repo, checkpoint, label};
use super::{Delete, Publish, Retarget, Yank};
use crate::Context;
use crate::context::hub;
use crate::output::{self, Record};

fn usage(code: &str, message: String) -> Error {
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
    let view = Changed {
        model: repo.to_string(),
        release: answer.release,
        revision: answer.revision,
        yanked: answer.yanked,
        lanes: answer
            .lanes
            .unwrap_or_default()
            .into_iter()
            .map(|l| (l.lane, l.checkpoint_id))
            .collect(),
        changed: answer.changed,
    };
    if ctx.json {
        return output::json(&view);
    }
    let mut record = Record::new()
        .field("model", &view.model)
        .field("release", &view.release)
        .field("revision", view.revision)
        .field("changed", view.changed);
    if view.yanked {
        record = record.field("status", "yanked");
    }
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
        return Err(usage(
            "model_publish_empty",
            "model publish changes nothing: give --lane name=sha256:… or --remove-lane name".into(),
        ));
    }
    let at = ctx.hub()?;
    let client = ctx.hub_signed_in(&at)?;
    // The write is checked against the revision this read saw: a concurrent publish refuses.
    let card = hub(client.get_model(&repo.org, &repo.name).await);
    let current = match card {
        Ok(card) => card.releases.into_iter().find(|r| r.release == release),
        Err(e) if e.kind == Kind::NotFound => None,
        Err(e) => return Err(e),
    };
    let mut lanes: HashMap<String, String> = current
        .iter()
        .flat_map(|r| r.lanes.iter().map(|l| (l.lane.clone(), l.manifest_id.clone())))
        .collect();
    lanes.extend(set.clone());
    lanes.retain(|lane, _| !remove.contains(lane));
    if lanes.is_empty() {
        return Err(usage(
            "model_release_empty",
            format!("{repo}@{release} would keep no lane"),
        )
        .remedy("yank the release to withdraw it whole")
        .next(format!("cozy model yank {repo} --release {release}")));
    }
    let update = ModelReleaseUpdate {
        expected_revision: current.and_then(|r| r.revision).unwrap_or(0),
        set_lanes: set,
        remove_lanes: remove,
    };
    let answer = hub(client.update_model_release(&repo.org, &repo.name, &release, &update).await)?;
    show(ctx, &repo, answer)
}

pub(crate) async fn retarget(ctx: &Context, args: Retarget) -> Result<ExitCode, Error> {
    let repo = Repo::hub(&args.model)?;
    let release = label("--release", &args.release)?;
    let lane = label("--lane", &args.lane)?;
    let body = ModelLaneRetarget {
        checkpoint_id: checkpoint(&args.to)?,
    };
    let at = ctx.hub()?;
    let client = ctx.hub_signed_in(&at)?;
    let answer = hub(client.retarget_model_lane(&repo.org, &repo.name, &release, &lane, &body).await)?;
    show(ctx, &repo, answer)
}

pub(crate) async fn yank(ctx: &Context, args: Yank) -> Result<ExitCode, Error> {
    let repo = Repo::hub(&args.model)?;
    let release = label("--release", &args.release)?;
    let at = ctx.hub()?;
    let answer = hub(ctx.hub_signed_in(&at)?.yank_model_release(&repo.org, &repo.name, &release).await)?;
    show(ctx, &repo, answer)
}

#[derive(Serialize)]
#[serde(untagged)]
enum Deleted {
    Checkpoint {
        model: String,
        checkpoint: String,
        /// False when it was already gone.
        removed: bool,
    },
    Model {
        model: String,
        checkpoints: i64,
        reclaimable_objects: i64,
        reclaimable_bytes: i64,
    },
}

/// Never prompts: `--yes` is the confirmation, and without it nothing is sent.
pub(crate) async fn delete(ctx: &Context, args: Delete) -> Result<ExitCode, Error> {
    let (name, one) = match args.model.split_once('#') {
        Some((name, id)) => (name, Some(checkpoint(id)?)),
        None => (args.model.as_str(), None),
    };
    let repo = Repo::hub(name)?;
    let at = ctx.hub()?;
    if !args.yes {
        return Err(usage(
            "model_delete_unconfirmed",
            format!("deleting {} from {} cannot be undone", args.model, at.origin),
        )
        .remedy("run it again with --yes"));
    }
    let client = ctx.hub_signed_in(&at)?;
    let view = match one {
        Some(manifest) => {
            let answer = hub(client.remove_model_checkpoint(&repo.org, &repo.name, &manifest).await)?;
            Deleted::Checkpoint {
                model: repo.to_string(),
                checkpoint: manifest,
                removed: answer.removed,
            }
        }
        None => {
            let answer = hub(client.delete_model(&repo.org, &repo.name).await)?;
            Deleted::Model {
                model: answer.model,
                checkpoints: answer.checkpoints,
                reclaimable_objects: answer.reclaimable_objects,
                reclaimable_bytes: answer.reclaimable_bytes,
            }
        }
    };
    if ctx.json {
        return output::json(&view);
    }
    let record = match &view {
        Deleted::Checkpoint { model, checkpoint, removed } => Record::new()
            .field("model", model)
            .field("checkpoint", checkpoint)
            .field("status", if *removed { "deleted" } else { "already gone" }),
        Deleted::Model { model, checkpoints, reclaimable_bytes, .. } => Record::new()
            .field("model", model)
            .field("checkpoints", checkpoints)
            .field("reclaimable", output::bytes(*reclaimable_bytes as u64))
            .field("status", "deleted"),
    };
    record.print(ctx);
    output::note(ctx, "the Hub's GC reclaims only objects no other model references");
    Ok(ExitCode::SUCCESS)
}
