//! `cozy model quantize`: the lane a named package's declared recipe writes from a checkpoint. A
//! lane flag picks the package's recipe of that lane for this model; `--recipe` names one. The
//! package is always named, a release or a local directory: packages that serve the same model
//! can each declare a recipe for it.
//! The package's `quantize` (the Runtime's) runs it on the machine, which writes the lane, its
//! provenance in its header, and returns its checkpoint reference.
use std::path::PathBuf;
use std::process::ExitCode;

use hub_client::types::ModelCatalogRelease;
use protocol::calcifer::{Run, Status};
use protocol::config::Hub;
use protocol::error::{Error, Kind};
use tensorfs::Endpoint;
use tensorfs::format::{Header, ObjectRef, Tree};
use tensorfs::precision::Profile;

use super::refs::{self, ModelRef, Repo};
use super::{Quantize, releases};
use crate::Context;
use crate::context::hub_answer;
use crate::output::{self, Record};
use crate::run::Callee;
use crate::run::interface::{Interface, Recipe, RECIPES};

/// A 16-bit lane: original weights a 4-bit lane may be written from.
const ORIGINAL: [&str; 3] = ["bf16", "fp16", "f32"];

/// The checkpoint a quantize reads: spelled for the run, the components every variant holds, its
/// repository, and the release it is (a public tag), if one.
struct Source {
    spelled: String,
    components: Vec<String>,
    repo: Repo,
    release: Option<String>,
}

/// A 4-bit lane is written only from original weights (owner, 2026-10-22): from a checkpoint
/// holding none, such as an ingested Q4_K GGUF or a 4-bit checkpoint, it would stack two rounds of
/// 4-bit error. Refused, naming the newest release that holds them.
async fn original_weights(ctx: &Context, hub: &Hub, source: &str, repo: &Repo, held: &[Variant], lane: &str) -> Result<(), Error> {
    if !matches!(lane, "nvfp4" | "w4a8" | "svdq-nvfp4") || held.iter().any(|v| ORIGINAL.contains(&v.precision.as_str())) {
        return Ok(());
    }
    let each: Vec<String> = (held.iter())
        .map(|v| format!("{} ({})", v.path, if v.precision.is_empty() { "encoded" } else { &v.precision })).collect();
    let error = Error::new(Kind::Usage, "model_quantize_source_quantized", format!(
        "{source} holds no original weights ({}): a {lane} lane from them stacks two rounds of 4-bit error", each.join(", ")));
    let card = hub_answer(ctx.hub_client(hub)?.get_model(&repo.org, &repo.name).await).ok();
    let original = |r: &&ModelCatalogRelease| r.yanked != Some(true)
        && r.lanes.iter().any(|l| l.precision.as_deref().is_some_and(|p| ORIGINAL.contains(&p)));
    let releases = card.iter().flat_map(|c| c.releases.iter().filter(original));
    Err(match tensorfs::format::newest_by(releases, |r| r.release.as_str()) {
        Some(release) => error.next(format!("cozy model quantize {repo}@{} --{lane}", release.release)),
        None => error.remedy(format!("upload the original weights (cozy model upload <path> {repo}) and quantize those")),
    })
}

/// A variant as its header says: its path, precision lane (`bf16`, `int8`, …) and components.
struct Variant {
    path: String,
    precision: String,
    components: Vec<String>,
}

/// Each variant of `commit`, by its header.
async fn variants(endpoint: &Endpoint, commit: &ObjectRef) -> tensorfs::Result<Vec<Variant>> {
    let tree = Tree::from_json(&endpoint.document(commit).await?)?;
    let mut held = vec![];
    for (path, header) in tree.variants() {
        let header = Header::from_cbor(&endpoint.document(header).await?)?;
        let components = header.components().iter().map(|c| c.name.clone()).collect();
        held.push(Variant { path: path.to_owned(), precision: Profile::of(&header).family, components });
    }
    Ok(held)
}

/// The checkpoint to quantize, by TensorFS's one rule, the one a run's `model.<slot>=org/name@x`
/// takes on its machine: `@x` a release or any tag its owner reads (`main` after an upload), none
/// the repo's default, `#sha256:…` a commit or a variant header. Its variants' headers say what
/// it holds.
async fn checkpoint(ctx: &Context, hub: &Hub, model: &ModelRef, lane: &str) -> Result<Source, Error> {
    let (repo, revision) = match model {
        ModelRef::Hub { repo, release } => (repo, release.as_deref()),
        ModelRef::Checkpoint { repo, manifest } => (repo, Some(manifest.as_str())),
        ModelRef::Local(_) | ModelRef::Provider(_) => return Err(not_on_hub(model)),
    };
    let (name, endpoint) = (repo.to_string(), releases::endpoint(ctx, hub).await?);
    let absent = |what: String| Error::new(Kind::NotFound, "model_revision_not_found", what).next(format!("cozy model info {repo}"));
    let doc = (endpoint.repo(&name).await.map_err(releases::failed)?).map(|(doc, _)| doc)
        .ok_or_else(|| absent(format!("{repo} has no commits")))?;
    let (commit, tag) = match revision {
        None => doc.default_tag().map(|t| (t.commit.clone(), Some(t)))
            .ok_or_else(|| absent(format!("{repo} has no release and no main")))?,
        Some(revision) => match endpoint.commit(&name, revision).await {
            Ok((commit, through)) => (commit, through.and_then(|t| doc.tag(&t))),
            Err(f) if f.kind == tensorfs::Kind::NotFound => return Err(absent(format!("{repo} has no release or tag {revision}"))),
            Err(f) => return Err(releases::failed(f)),
        },
    };
    let held = variants(&endpoint, &commit).await.map_err(releases::failed)?;
    let spelled = match (model, tag) {
        (ModelRef::Hub { .. }, Some(tag)) => format!("{repo}@{}", tag.name),
        _ => model.to_string(),
    };
    original_weights(ctx, hub, &spelled, repo, &held, lane).await?;
    let mut each = held.into_iter().map(|v| v.components);
    let first = each.next().ok_or_else(|| {
        Error::new(Kind::NotFound, "model_lane_not_found", format!("{spelled} has no variant"))
    })?;
    let components = each.fold(first, |held, next| held.into_iter().filter(|c| next.contains(c)).collect());
    let release = tag.filter(|t| t.public && matches!(model, ModelRef::Hub { .. })).map(|t| t.name.clone());
    Ok(Source { spelled, components, repo: repo.clone(), release })
}

fn not_on_hub(model: &ModelRef) -> Error {
    Error::new(Kind::Usage, "model_quantize_source", format!("{model} is not on the Hub: quantize reads a Hub model"))
        .next(format!("cozy model upload {model} <org/name>"))
}

/// The package named for the recipe: a release on the Hub, or local code nobody published.
enum Named {
    /// `org/name` and the release asked for, if any.
    Hub(String, Option<String>),
    /// A package directory, called as `cozy run ./dir` calls it.
    Local(PathBuf),
}

fn package(args: &Quantize) -> Result<Named, Error> {
    let Some(named) = &args.package else {
        return Err(Error::new(Kind::Usage, "model_quantize_package",
            "name the package whose recipe writes the lane: packages that serve the same model can each declare one")
            .remedy("add --package org/name[@release], or --package ./dir for a package not yet published"));
    };
    if crate::run::target::local_path(named) {
        let cwd = std::env::current_dir()
            .map_err(|e| Error::new(Kind::Internal, "cwd", format!("the working directory: {e}")))?;
        return Ok(Named::Local(cwd.join(named.trim())));
    }
    let named = crate::package::PackageRef::parse(named)?;
    Ok(Named::Hub(named.package(), named.release))
}

/// `name` as recipe names fold: `drafter-head` is `drafter_head`.
fn fold(name: &str) -> String {
    name.replace('-', "_").to_lowercase()
}

/// The lanes asked for, in order: `--format`'s, or the named recipe's one.
fn lanes(args: &Quantize, interface: &Interface, package: &str) -> Result<Vec<String>, Error> {
    if let Some(name) = &args.recipe {
        return (interface.recipes.iter().find(|r| r.name == fold(name)))
            .map(|r| vec![r.lane.clone()])
            .ok_or_else(|| Error::new(Kind::NotFound, "model_recipe_not_found", format!("{package} declares no recipe {name}"))
                .remedy(declared(interface)));
    }
    let mut lanes = args.format.clone();
    lanes.dedup();
    Ok(lanes)
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
    let named = package(&args)?;
    let model = ModelRef::parse(&args.model)?;
    if matches!(model, ModelRef::Local(_) | ModelRef::Provider(_)) {
        return Err(not_on_hub(&model));
    }
    let destination = args.destination.as_deref().map(refs::destination).transpose()?;
    let hub = ctx.hub()?;
    let (card, prepared);
    let callee = match &named {
        Named::Hub(package, release) => {
            card = crate::run::resolve(ctx, &hub, package, release.as_deref()).await?.card;
            Callee::Release { package, card: &card }
        }
        Named::Local(path) => {
            prepared = crate::run::local::prepare(ctx, path, args.placement.on(), false).await?;
            Callee::Local(&prepared)
        }
    };
    let interface = callee.interface();
    // One run per lane, in the order asked; the first that does not complete ends the rest.
    let mut code = ExitCode::SUCCESS;
    for lane in lanes(&args, interface, callee.package())? {
        code = write(ctx, &args, &hub, &model, destination.as_deref(), &callee, &lane).await?;
        if code != ExitCode::SUCCESS {
            break;
        }
    }
    Ok(code)
}

/// One lane of `model`, written by `callee`'s recipe into `destination` (the model's own tag tree
/// when none), as one run.
async fn write(
    ctx: &Context,
    args: &Quantize,
    hub: &Hub,
    model: &ModelRef,
    destination: Option<&str>,
    callee: &Callee<'_>,
    lane: &str,
) -> Result<ExitCode, Error> {
    let interface = callee.interface();
    let source = checkpoint(ctx, hub, model, lane).await?;
    let found = recipes(interface, lane, args.recipe.as_deref(), &source.components);
    if found.is_empty() {
        let what = args.recipe.as_deref().unwrap_or(lane);
        return Err(Error::new(Kind::NotFound, "model_quantizer_not_found", format!(
            "{callee} has no {what} recipe for this model ({})", source.components.join(", ")))
            .remedy(declared(interface)));
    }
    // The variant goes onto the tag's tree (main unless named): publishing the tag releases all.
    // Local code's lane records its package as `local/<name>` and its release as the source's
    // manifest digest: an unpublished recipe, exactly.
    let destination = destination.map_or_else(|| format!("{}@main", source.repo), str::to_owned);
    let mut terms = vec![source.spelled.clone(), destination.clone(), format!("lane={lane}"),
        format!("package={}", callee.package()), format!("release={}", callee.release())];
    terms.extend(args.recipe.iter().map(|name| format!("recipe={name}")));
    // A calibration reads the original weights, never an encoded lane: the run takes 16 bits.
    let calibrated = calibration(interface, &found, &source.spelled);
    let bits = (!calibrated.is_empty()).then_some("16");
    terms.extend(calibrated);
    let run = crate::run::call(ctx, callee, RECIPES, terms, bits, &args.placement).await?;
    if !ctx.json && run.status == Status::Completed {
        match written_checkpoint(&run, lane) {
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
                let release = source.release.as_deref().filter(|_| into == source.repo.to_string()).unwrap_or("<version>");
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
