//! `cozy model quantize`: the serving package's quantizer, run as an ordinary conversion call
//! (`cozy run <package>/<fn> <model> <org/name>`). The package owns the recipe; the machine writes
//! the lane and uploads it as a private checkpoint.
use std::process::ExitCode;

use cozy_proto::calcifer::On;
use cozy_proto::config::Hub;
use cozy_proto::error::{Error, Kind};

use super::Quantize;
use super::refs::{ModelRef, Repo};
use crate::Context;
use crate::context::hub_answer;
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
                .filter(|e| e.hub == hub.origin && !e.package.starts_with("local/"))
                .map(|e| (e.package, Some(e.release).filter(|r| !r.is_empty())))
                .collect()
        }
    };
    let mut found = vec![];
    for (package, release) in candidates {
        let card = crate::run::card(ctx, hub, &package, release.as_deref()).await?;
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
    let destination = destination.unwrap_or(repo);
    let (package, release, function) = quantizer(ctx, &hub, &args, lane, &components).await?;
    let inputs = vec![source, destination.to_string()];
    crate::run::call(ctx, &package, Some(&release), &function, inputs, &args.placement, args.detach).await
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
}
