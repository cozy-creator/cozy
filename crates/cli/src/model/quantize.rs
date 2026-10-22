//! `cozy model quantize`: the serving package's quantizer, run as an ordinary conversion call
//! (`cozy run <package>/<fn> <model> <org/name>`). The package owns the recipe; Runtime writes
//! the lane through Runtime's TensorFS store and returns its checkpoint reference.
use std::process::ExitCode;

use hub_client::types::{ModelCard, ModelCatalogRelease};
use protocol::calcifer::{On, Run, Status};
use protocol::config::{Hub, Origin};
use protocol::error::{Error, Kind};

use super::Quantize;
use super::refs::{self, ModelRef, Repo};
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

/// A 16-bit lane: original weights a 4-bit lane may be written from.
const ORIGINAL: [&str; 3] = ["bf16", "fp16", "f32"];

/// A 4-bit lane is written only from original weights (owner, 2026-10-22): from a release holding
/// none, such as an ingested Q4_K GGUF or an nvfp4 or w4a8 checkpoint, it would stack two rounds
/// of 4-bit error. Refused, naming the newest release that holds them.
fn original_weights(repo: &Repo, card: &ModelCard, source: &ModelCatalogRelease, lane: &str) -> Result<(), Error> {
    let original = |r: &ModelCatalogRelease| r.lanes.iter().any(|l| l.precision.as_deref().is_some_and(|p| ORIGINAL.contains(&p)));
    if !matches!(lane, "nvfp4" | "w4a8") || original(source) {
        return Ok(());
    }
    let held: Vec<String> = source.lanes.iter()
        .map(|l| format!("{} ({})", l.lane, l.precision.as_deref().filter(|p| !p.is_empty()).unwrap_or("encoded"))).collect();
    let error = Error::new(Kind::Usage, "model_quantize_source_quantized", format!(
        "{repo}@{} holds no original weights ({}): a {lane} lane from them stacks two rounds of 4-bit error",
        source.release, held.join(", ")));
    let releases = card.releases.iter().filter(|r| r.yanked != Some(true) && original(r));
    Err(match tensorfs::format::newest_by(releases, |r| r.release.as_str()) {
        Some(release) => error.next(format!("cozy model quantize {repo}@{} --{lane}", release.release)),
        None => error.remedy(format!("upload the original weights (cozy model upload <path> {repo}) and quantize those")),
    })
}

/// The checkpoint to quantize spelled for the run, the components it carries and its repository.
/// A Hub model is its release (the newest available when none is named): the quantizer's slot
/// selects its variant by the one rule, as any run's does; the components are every variant's.
async fn checkpoint(ctx: &Context, hub: &Hub, model: &ModelRef, lane: &str) -> Result<(String, Vec<String>, Repo), Error> {
    let client = ctx.hub_client(hub)?;
    match model {
        ModelRef::Checkpoint { repo, manifest } => {
            let resolved = hub_answer(client.resolve_model(None, &format!("{repo}@{manifest}")).await)?;
            Ok((model.to_string(), resolved.components.unwrap_or_default(), repo.clone()))
        }
        ModelRef::Hub { repo, release } => {
            let card = hub_answer(client.get_model(&repo.org, &repo.name).await)?;
            let found = match release {
                Some(want) => card.releases.iter().find(|r| &r.release == want),
                None => {
                    let live = card.releases.iter().filter(|r| r.yanked != Some(true) && r.yanked_at.as_deref().is_none_or(str::is_empty));
                    tensorfs::format::newest_by(live, |r| r.release.as_str())
                }
            };
            let found = found.ok_or_else(|| {
                Error::new(Kind::NotFound, "model_release_not_found", format!("{repo} has no release {}", release.as_deref().unwrap_or("available")))
                    .next(format!("cozy model info {repo}"))
            })?;
            original_weights(repo, &card, found, lane)?;
            let mut each = found.lanes.iter().map(|l| l.components.clone().unwrap_or_default());
            let first = each.next().ok_or_else(|| {
                Error::new(Kind::NotFound, "model_lane_not_found", format!("{repo}@{} has no variant", found.release))
            })?;
            let components = each.fold(first, |held, next| held.into_iter().filter(|c| next.contains(c)).collect());
            Ok((format!("{repo}@{}", found.release), components, repo.clone()))
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
    let lane = match (args.int8, args.fp8, args.w4a8) {
        (true, ..) => "int8",
        (_, true, _) => "fp8",
        (.., true) => "w4a8",
        _ => "nvfp4",
    };
    let model = ModelRef::parse(&args.model)?;
    let destination = args.destination.as_deref().map(refs::destination).transpose()?;
    let hub = ctx.hub()?;
    let (source, components, repo) = checkpoint(ctx, &hub, &model, lane).await?;
    // The variant goes onto the tag's tree (main unless named): publishing the tag releases all.
    let destination = destination.unwrap_or_else(|| format!("{repo}@main"));
    let (package, release, function) = quantizer(ctx, &hub, &args, lane, &components).await?;
    let inputs = vec![source.clone(), destination.clone()];
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
                // The release it was made from moves to the commit the machine wrote, which holds
                // its variant too: exactly that one, wherever its tag moves next.
                let into = destination.split_once('@').map_or(destination.as_str(), |(into, _)| into);
                let pinned = source.split_once('@').map(|(_, rest)| rest.split('/').next().unwrap_or(rest));
                let release = pinned.filter(|_| into == repo.to_string()).unwrap_or("<version>");
                output::next(ctx, &[&format!("cozy model publish {model} --release {release} --commit {tree}")]);
            }
            Some(WrittenCheckpoint::Header(header)) => {
                Record::new().field("checkpoint header", header).print(ctx);
            }
            None => {}
        }
    }
    Ok(crate::run::exit_code(&run))
}

enum WrittenCheckpoint<'a> {
    Reference(&'a protocol::machine::CheckpointRef),
    Header(&'a str),
}

/// A typed output names the repository TREE; the result's checkpoint field names only a HEADER.
fn written_checkpoint<'a>(run: &'a Run, lane: &str) -> Option<WrittenCheckpoint<'a>> {
    if let Some(reference) = run.output.iter().filter(|item| item.name == lane)
        .find_map(|item| item.checkpoint.as_ref()) {
        return Some(WrittenCheckpoint::Reference(reference));
    }
    let result = run.result.as_ref()?;
    result["checkpoint"].as_str().map(WrittenCheckpoint::Header)
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

    /// A 4-bit lane is written only from original weights: a release of 4-bit variants (an
    /// ingested GGUF, an nvfp4 checkpoint) is refused, naming the newest release holding them.
    #[test]
    fn a_four_bit_lane_is_never_written_from_four_bit_weights() {
        let lane = |path: &str, precision: &str| serde_json::json!({"lane": path, "precision": precision,
            "manifest_id": "sha256:00", "manifest_length": 1, "components": ["transformer"], "objects": 1, "bytes": 1});
        let card: ModelCard = serde_json::from_value(serde_json::json!({
            "model": {"org": "acme", "name": "x", "created_at": "2026-10-22T00:00:00Z"}, "throughput": [],
            "releases": [
                {"release": "1.1.0", "lanes": [lane("model.q4_k_m.cozytensors", ""), lane("model.nvfp4.cozytensors", "nvfp4")]},
                {"release": "1.0.0", "lanes": [lane("model.bf16.cozytensors", "bf16"), lane("model.int8.cozytensors", "int8")]},
                {"release": "0.9.0", "lanes": [lane("model.bf16.cozytensors", "bf16")]}]})).unwrap();
        let repo = Repo::parse("acme/x").unwrap();
        let (gguf, original) = (&card.releases[0], &card.releases[1]);
        for four in ["w4a8", "nvfp4"] {
            let refused = original_weights(&repo, &card, gguf, four).unwrap_err();
            assert_eq!(refused.code, "model_quantize_source_quantized");
            assert!(refused.to_string().contains("stacks two rounds of 4-bit error"), "{refused}");
            assert!(format!("{refused:?}").contains(&format!("cozy model quantize acme/x@1.0.0 --{four}")), "{refused:?}");
            original_weights(&repo, &card, original, four).unwrap();
        }
        original_weights(&repo, &card, gguf, "int8").unwrap();
        let only: ModelCard = serde_json::from_value(serde_json::json!({
            "model": {"org": "acme", "name": "x", "created_at": "2026-10-22T00:00:00Z"}, "throughput": [],
            "releases": [{"release": "1.0.0", "lanes": [lane("model.w4a8.cozytensors", "w4a8")]}]})).unwrap();
        let refused = original_weights(&repo, &only, &only.releases[0], "nvfp4").unwrap_err();
        assert!(format!("{refused:?}").contains("cozy model upload"), "{refused:?}");
    }

    #[test]
    fn typed_output_trees_take_precedence_over_result_headers_and_old_manifests() {
        let mut run: Run = serde_json::from_value(serde_json::json!({
            "id": protocol::ids::RunId::mint([7; 12]), "number": 7, "kind": "call",
            "target": "quantize", "hub": "https://tensorhub.com", "status": "completed",
            "created_at_ms": 1,
            "result": {"checkpoint": "sha256:header"},
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
    }
}
