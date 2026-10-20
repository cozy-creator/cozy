//! `cozy model quantize`: the serving package's quantizer, run as an ordinary conversion call
//! (`cozy run <package>/<fn> <model> <org/name>`). The package owns the recipe; Runtime writes
//! the lane through Runtime's TensorFS store and returns its checkpoint reference.
use std::process::ExitCode;

use protocol::calcifer::{On, Run, Status};
use protocol::config::{Hub, Origin};
use protocol::error::{Error, Kind};

use super::Quantize;
use super::refs::{ModelRef, Repo};
use crate::Context;
use crate::context::hub_answer;
use crate::output::{self, Record};
use crate::run::interface::{Callable, Interface};

/// The callables of `interface` that write `lane` from a checkpoint with `components`: a
/// conversion whose one weights output is named for the lane and whose model slot reads only
/// components the checkpoint carries.
fn quantizers<'a>(interface: &'a Interface, lane: &str, components: &[String]) -> Vec<&'a Callable> {
    interface
        .callables
        .iter()
        .filter(|c| !c.internal && c.is_conversion())
        .filter(|c| matches!(c.weights_outputs.as_slice(), [one] if one.output_id == lane))
        .filter(|c| c.models[0].component_use.values().flatten().all(|need| components.contains(need)))
        .collect()
}

/// The exact checkpoint to quantize spelled for the run, the components it carries and its
/// repository. A Hub
/// model names one lane: the newest available release when none is named, and its only lane
/// unless one is named.
async fn checkpoint(ctx: &Context, hub: &Hub, model: &ModelRef) -> Result<(String, Vec<String>, Repo), Error> {
    let client = ctx.hub_client(hub)?;
    match model {
        ModelRef::Checkpoint { repo, manifest } => {
            let resolved = hub_answer(client.resolve_model(None, &format!("{repo}@{manifest}")).await)?;
            Ok((model.to_string(), resolved.components.unwrap_or_default(), repo.clone()))
        }
        ModelRef::Hub { repo, release, lane } => {
            let card = hub_answer(client.get_model(&repo.org, &repo.name).await)?;
            let found = card.releases.iter().find(|r| match release {
                Some(want) => &r.release == want,
                None => r.yanked != Some(true) && r.yanked_at.as_deref().is_none_or(str::is_empty),
            });
            let found = found.ok_or_else(|| {
                Error::new(Kind::NotFound, "model_release_not_found", format!("{repo} has no release {}", release.as_deref().unwrap_or("available")))
                    .next(format!("cozy model info {repo}"))
            })?;
            let lanes: Vec<_> = found.lanes.iter().filter(|l| lane.as_ref().is_none_or(|want| &l.lane == want)).collect();
            let [one] = lanes.as_slice() else {
                let names: Vec<_> = found.lanes.iter().map(|l| l.lane.as_str()).collect();
                return Err(Error::new(Kind::Usage, "model_lane_unnamed", format!("{repo}@{} has lanes {}: name the one to quantize", found.release, names.join(", ")))
                    .remedy(format!("write {repo}@{}/<lane>", found.release)));
            };
            Ok((format!("{repo}@{}/{}", found.release, one.lane), one.components.clone().unwrap_or_default(), repo.clone()))
        }
        ModelRef::Local(_) | ModelRef::Provider(_) => Err(Error::new(Kind::Usage, "model_quantize_source", format!("{model} is not on the Hub: quantize reads a Hub model"))
            .next(format!("cozy model upload {model} <org/name>"))),
    }
}

/// The one (package, release, function) that writes `lane` from this checkpoint: `--package`'s,
/// else among the published packages the target machine holds.
async fn quantizer(ctx: &Context, hub: &Hub, args: &Quantize, lane: &str, components: &[String]) -> Result<(String, String, String), Error> {
    let candidates: Vec<(String, Option<String>)> = match &args.package {
        Some(package) => {
            let (name, release) = package.split_once('@').map_or((package.as_str(), None), |(n, r)| (n, Some(r.to_owned())));
            vec![(Repo::parse(name)?.to_string(), release)]
        }
        None => {
            let On::Machine(name) = args.placement.on() else {
                return Err(Error::new(Kind::Usage, "model_quantizer_unnamed", "a new or any machine holds no packages yet: name the quantizer's package")
                    .remedy("add --package org/name"));
            };
            let machines = ctx.calcifer().await?.machines().await?.machines;
            let machine = machines.into_iter().find(|m| m.name == name).ok_or_else(|| {
                Error::new(Kind::NotFound, "machine_not_found", format!("no machine named {name}")).next("cozy machine list")
            })?;
            let environments = machine.status.map(|s| s.environments).unwrap_or_default();
            environments
                .into_iter()
                .filter(|e| Origin::parse(&e.hub).is_ok_and(|o| o == hub.origin) && !e.package.starts_with("local/"))
                .map(|e| (e.package, Some(e.release).filter(|r| !r.is_empty())))
                .collect()
        }
    };
    let mut found = vec![];
    for (package, release) in candidates {
        let card = crate::run::resolve(ctx, hub, &package, release.as_deref()).await?.card;
        for callable in quantizers(&card.interface, lane, components) {
            found.push((package.clone(), card.release.clone(), callable.name.clone()));
        }
    }
    found.dedup();
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err(Error::new(Kind::NotFound, "model_quantizer_not_found", format!("no package here quantizes this model ({}) to {lane}", components.join(", ")))
            .remedy(format!("a package's quantizer is a function that takes the model and declares one weights output named {lane}; name its package with --package org/name"))),
        _ => {
            let named: Vec<_> = found.iter().map(|(p, r, f)| format!("{p}/{f}@{r}")).collect();
            Err(Error::new(Kind::Usage, "model_quantizer_ambiguous", format!("{} all quantize it to {lane}", named.join(", ")))
                .remedy("choose one with --package org/name[@release]"))
        }
    }
}

pub(crate) async fn quantize(ctx: &Context, args: Quantize) -> Result<ExitCode, Error> {
    let lane = match (args.int8, args.fp8) {
        (true, _) => "int8",
        (_, true) => "fp8",
        _ => "nvfp4",
    };
    let model = ModelRef::parse(&args.model)?;
    let destination = args.destination.as_deref().map(Repo::hub).transpose()?;
    let hub = ctx.hub()?;
    let (source, components, repo) = checkpoint(ctx, &hub, &model).await?;
    let destination = destination.unwrap_or_else(|| repo.clone());
    let (package, release, function) = quantizer(ctx, &hub, &args, lane, &components).await?;
    let inputs = vec![source.clone(), destination.to_string()];
    let run = crate::run::call(ctx, &package, Some(&release), &function, inputs, &args.placement, args.detach).await?;
    if !ctx.json && run.status == Status::Completed {
        match written_checkpoint(&run, lane) {
            Some(WrittenCheckpoint::Reference(reference)) => {
                let (model, tree) = (&reference.repo_id, &reference.revision);
                let mut record = Record::new().field("checkpoint", format!("{model}@{tree}"));
                if !reference.variant.is_empty() {
                    record = record.field("variant", &reference.variant);
                }
                record.print(ctx);
                // The release it was made from moves to the commit that adds its variant.
                let pinned = source.split_once('@').map(|(_, rest)| rest.split('/').next().unwrap_or(rest));
                let release = pinned.filter(|_| destination == repo).unwrap_or("<version>");
                output::next(ctx, &[&format!("cozy model publish {model} --release {release} --commit {tree}")]);
            }
            Some(WrittenCheckpoint::Header(header)) => {
                Record::new().field("checkpoint header", header).print(ctx);
            }
            Some(WrittenCheckpoint::Legacy(digest)) => {
                Record::new().field("checkpoint", format!("{destination}#{digest}")).print(ctx);
            }
            None => {}
        }
    }
    Ok(crate::run::exit_code(&run))
}

enum WrittenCheckpoint<'a> {
    Reference(&'a protocol::machine::CheckpointRef),
    Header(&'a str),
    Legacy(&'a str),
}

/// A typed output names the repository TREE. The result's new checkpoint field names only a
/// HEADER; the nested manifest form belongs to older saved results.
fn written_checkpoint<'a>(run: &'a Run, lane: &str) -> Option<WrittenCheckpoint<'a>> {
    if let Some(reference) = run.output.iter().filter(|item| item.name == lane)
        .find_map(|item| item.checkpoint.as_ref()) {
        return Some(WrittenCheckpoint::Reference(reference));
    }
    let result = run.result.as_ref()?;
    if let Some(header) = result["checkpoint"].as_str() {
        return Some(WrittenCheckpoint::Header(header));
    }
    result["manifest"]["digest"].as_str().map(WrittenCheckpoint::Legacy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::interface::tests::interface;

    fn names(found: Vec<&Callable>) -> Vec<&str> {
        found.into_iter().map(|c| c.name.as_str()).collect()
    }

    #[test]
    fn the_quantizer_is_the_conversion_named_for_the_lane_that_reads_the_checkpoint() {
        let klein = interface("flux-3-klein-0.5.0");
        let base: Vec<String> = ["text_encoder", "transformer", "vae"].map(String::from).into();
        let upsampler = vec!["model".to_owned()];
        assert_eq!(names(quantizers(&klein, "int8", &base)), ["int8"]);
        assert_eq!(names(quantizers(&klein, "nvfp4", &upsampler)), ["upsampler_nvfp4"]);
        assert!(quantizers(&klein, "fp8", &base).is_empty());
        let sdxl = interface("sdxl-2.10.0");
        let parts: Vec<String> = ["text_encoder", "text_encoder_2", "unet", "vae"].map(String::from).into();
        assert_eq!(names(quantizers(&sdxl, "fp8", &parts)), ["fp8"]);
        assert!(quantizers(&sdxl, "int8", &parts[..3]).is_empty(), "a checkpoint without its vae");
    }

    #[test]
    fn typed_output_trees_take_precedence_over_result_headers_and_old_manifests() {
        let mut run: Run = serde_json::from_value(serde_json::json!({
            "id": protocol::ids::RunId::mint([7; 12]), "number": 7, "kind": "call",
            "target": "quantize", "hub": "https://tensorhub.com", "status": "completed",
            "created_at_ms": 1,
            "result": {"checkpoint": "sha256:header", "manifest": {"digest": "sha256:legacy"}},
            "output": [{
                "id": "7/fp8", "name": "fp8", "index": 0, "rev": 1, "length": 0,
                "media_type": "", "status": "completed",
                "checkpoint": {"repo_id": "actual/output", "revision": "sha256:tree", "variant": "fp8"}
            }]
        })).unwrap();
        assert!(matches!(written_checkpoint(&run, "fp8"), Some(WrittenCheckpoint::Reference(reference))
            if reference.repo_id == "actual/output" && reference.revision == "sha256:tree"));
        run.output.clear();
        assert!(matches!(written_checkpoint(&run, "fp8"), Some(WrittenCheckpoint::Header("sha256:header"))));
        run.result = Some(serde_json::json!({"manifest": {"digest": "sha256:legacy"}}));
        assert!(matches!(written_checkpoint(&run, "fp8"), Some(WrittenCheckpoint::Legacy("sha256:legacy"))));
    }
}
