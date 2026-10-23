//! `cozy model quantize`: the lane a named package's declared recipe writes from a checkpoint. A
//! lane flag picks the package's recipe of that lane for this model; `--recipe` names one. The
//! package is always named: packages that serve the same model can each declare a recipe for it.
//! The package's `quantize` (the Runtime's) runs it on the machine, which writes the lane, its
//! provenance in its header, and returns its checkpoint reference.
use std::process::ExitCode;

use hub_client::types::{ModelCard, ModelCatalogRelease};
use protocol::calcifer::{Run, Status};
use protocol::config::Hub;
use protocol::error::{Error, Kind};

use super::Quantize;
use super::refs::{self, ModelRef, Repo};
use crate::Context;
use crate::context::hub_answer;
use crate::output::{self, Record};
use crate::run::interface::{Interface, Recipe, RECIPES};

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
        ModelRef::Local(_) | ModelRef::Provider(_) => Err(not_on_hub(model)),
    }
}

fn not_on_hub(model: &ModelRef) -> Error {
    Error::new(Kind::Usage, "model_quantize_source", format!("{model} is not on the Hub: quantize reads a Hub model"))
        .next(format!("cozy model upload {model} <org/name>"))
}

/// The package named for the recipe: `org/name` and the release asked for, if any.
fn package(args: &Quantize) -> Result<(String, Option<String>), Error> {
    let Some(named) = &args.package else {
        return Err(Error::new(Kind::Usage, "model_quantize_package",
            "name the package whose recipe writes the lane: packages that serve the same model can each declare one")
            .remedy("add --package org/name[@release]"));
    };
    let named = crate::package::PackageRef::parse(named)?;
    Ok((named.package(), named.release))
}

/// `name` as recipe names fold: `drafter-head` is `drafter_head`.
fn fold(name: &str) -> String {
    name.replace('-', "_").to_lowercase()
}

/// The lane asked for: its flag's, or the named recipe's.
fn lane(args: &Quantize, interface: &Interface, package: &str) -> Result<String, Error> {
    if let Some(name) = &args.recipe {
        return (interface.recipes.iter().find(|r| r.name == fold(name)))
            .map(|r| r.lane.clone())
            .ok_or_else(|| Error::new(Kind::NotFound, "model_recipe_not_found", format!("{package} declares no recipe {name}"))
                .remedy(declared(interface)));
    }
    let lane = match (args.int8, args.fp8, args.w4a8) {
        (true, ..) => "int8",
        (_, true, _) => "fp8",
        (.., true) => "w4a8",
        _ => "nvfp4",
    };
    Ok(lane.to_owned())
}

/// What a package declares, for a refusal: each recipe's flag and the model it reads.
fn declared(interface: &Interface) -> String {
    if interface.recipes.is_empty() {
        return "it declares no recipes; name the package that does with --package org/name".into();
    }
    let each: Vec<String> = (interface.recipes.iter())
        .map(|r| {
            let mut reads: Vec<&str> = r.model.component_use.values().flatten().map(String::as_str).collect();
            reads.sort_unstable();
            reads.dedup();
            let reads = if reads.is_empty() { "any checkpoint".to_owned() } else { reads.join(", ") };
            format!("{} ({}: {reads})", r.flag(), r.model.class)
        })
        .collect();
    format!("its recipes: {}", each.join("; "))
}

pub(crate) async fn quantize(ctx: &Context, args: Quantize) -> Result<ExitCode, Error> {
    let (package, release) = package(&args)?;
    let model = ModelRef::parse(&args.model)?;
    if matches!(model, ModelRef::Local(_) | ModelRef::Provider(_)) {
        return Err(not_on_hub(&model));
    }
    let destination = args.destination.as_deref().map(refs::destination).transpose()?;
    let hub = ctx.hub()?;
    let card = crate::run::resolve(ctx, &hub, &package, release.as_deref()).await?.card;
    let lane = lane(&args, &card.interface, &package)?;
    let (source, components, repo) = checkpoint(ctx, &hub, &model, &lane).await?;
    let wanted = |r: &&Recipe| r.lane == lane && args.recipe.as_deref().is_none_or(|n| r.name == fold(n));
    if !card.interface.recipes.iter().filter(wanted).any(|r| r.reads(&components)) {
        let what = args.recipe.as_deref().unwrap_or(&lane);
        return Err(Error::new(Kind::NotFound, "model_quantizer_not_found", format!(
            "{package}@{} has no {what} recipe for this model ({})", card.release, components.join(", ")))
            .remedy(declared(&card.interface)));
    }
    // The variant goes onto the tag's tree (main unless named): publishing the tag releases all.
    let destination = destination.unwrap_or_else(|| format!("{repo}@main"));
    let mut terms = vec![source.clone(), destination.clone(), format!("lane={lane}"),
        format!("package={package}"), format!("release={}", card.release)];
    terms.extend(args.recipe.iter().map(|name| format!("recipe={name}")));
    let run = crate::run::call(ctx, &package, Some(&card.release), RECIPES, terms, &args.placement, args.detach).await?;
    if !ctx.json && run.status == Status::Completed {
        match written_checkpoint(&run, &lane) {
            Some(WrittenCheckpoint::Reference(reference)) => {
                let (model, tree) = (&reference.repo_id, &reference.revision);
                let mut record = Record::new().field("checkpoint", format!("{model}@{tree}"));
                if !reference.variant.is_empty() {
                    record = record.field("variant", &reference.variant);
                }
                record.print(ctx);
                // The release it was made from moves to the commit the machine wrote, which
                // holds its variant too: exactly that one, wherever its tag moves next.
                let into = destination.split_once('@').map_or(destination.as_str(), |(into, _)| into);
                let pinned = source.split_once('@').map(|(_, rest)| rest.split('/').next().unwrap_or(rest));
                let release = pinned.filter(|_| into == repo.to_string()).unwrap_or("<version>");
                output::next(ctx, &[&format!("cozy model publish {model} --release {release} --commit {tree}")]);
            }
            Some(WrittenCheckpoint::Header(header)) => Record::new().field("checkpoint header", header).print(ctx),
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

    /// Klein's shape: two models' int8 recipes and the drafter's selection, as an interface
    /// records them.
    fn klein() -> Interface {
        let recipe = |name: &str, lane: &str, class: &str, stage: &str, reads: &[&str]| serde_json::json!({
            "name": name, "lane": lane, "module": "flux_3_klein.quantize", "export": name.to_uppercase(),
            "model": {"class": class, "component_use": {stage: reads}}, "plan": {"kind": "quantization"}});
        Interface::parse(&serde_json::json!({"format": "cozy.package.interface/1", "application": "k:app",
            "entrypoints": [], "recipes": [
                recipe("drafter_head", "bf16", "Drafter", "draft", &["model"]),
                recipe("int8", "int8", "Klein", "denoise", &["transformer", "text_encoder"]),
                recipe("upsampler_int8", "int8", "Upsampler", "decode", &["model"]),
                {"name": "unreadable"}]}))
        .unwrap()
    }

    #[test]
    fn a_recipe_reads_a_checkpoint_holding_its_models_components() {
        let klein = klein();
        let held = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        let reads = |name: &str, names: &[&str]| klein.recipes.iter().find(|r| r.name == name).unwrap().reads(&held(names));
        assert!(reads("int8", &["transformer", "text_encoder", "vae"]));
        assert!(!reads("int8", &["model"]));
        assert!(reads("upsampler_int8", &["model"]) && reads("drafter_head", &["model"]));
        assert_eq!(fold("Drafter-Head"), "drafter_head");
        assert_eq!(declared(&klein), "its recipes: --recipe drafter_head (Drafter: model); \
            --int8 (Klein: text_encoder, transformer); --int8 (Upsampler: model)");
        assert!(declared(&Interface::default()).contains("declares no recipes"));
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
