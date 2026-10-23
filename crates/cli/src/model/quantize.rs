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
/// none, such as an ingested Q4_K GGUF or a 4-bit checkpoint, it would stack two rounds of 4-bit
/// error. Refused, naming the newest release that holds them.
fn original_weights(repo: &Repo, card: &ModelCard, source: &ModelCatalogRelease, lane: &str) -> Result<(), Error> {
    let original = |r: &ModelCatalogRelease| r.lanes.iter().any(|l| l.precision.as_deref().is_some_and(|p| ORIGINAL.contains(&p)));
    if !matches!(lane, "nvfp4" | "w4a8" | "svdq-nvfp4") || original(source) {
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
    let lane = match (args.int8, args.fp8, args.w4a8, args.svdq_nvfp4) {
        (true, ..) => "int8",
        (_, true, ..) => "fp8",
        (_, _, true, _) => "w4a8",
        (.., true) => "svdq-nvfp4",
        _ => "nvfp4",
    };
    Ok(lane.to_owned())
}

/// The recipes of `lane` (named `recipe` if given) that read a checkpoint holding `components`.
fn recipes<'i>(interface: &'i Interface, lane: &str, recipe: Option<&str>, components: &[String]) -> Vec<&'i Recipe> {
    (interface.recipes.iter())
        .filter(|r| r.lane == lane && recipe.is_none_or(|n| r.name == fold(n)) && r.reads(components))
        .collect()
}

/// When one of `found` calibrates on the model itself: `model.<slot>=<source>` for each model the
/// package's `quantize` constructs (on demand). Else nothing: another recipe's model is never bound.
fn calibration(interface: &Interface, found: &[&Recipe], source: &str) -> Vec<String> {
    if found.iter().all(|r| r.calibrate.is_empty()) {
        return vec![];
    }
    (interface.callables.iter()).filter(|c| c.name == RECIPES).flat_map(|c| &c.models)
        .filter(|slot| !slot.weights).map(|slot| format!("model.{}={source}", slot.param)).collect()
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
    let found = recipes(&card.interface, &lane, args.recipe.as_deref(), &components);
    if found.is_empty() {
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
    // A calibration reads the original weights, never an encoded lane: the run takes 16 bits.
    let calibrated = calibration(&card.interface, &found, &source);
    let bits = (!calibrated.is_empty()).then_some("16");
    terms.extend(calibrated);
    let run = crate::run::call(ctx, &package, Some(&card.release), RECIPES, terms, bits, &args.placement, args.detach).await?;
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
