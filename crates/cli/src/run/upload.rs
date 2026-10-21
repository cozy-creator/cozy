//! `cozy run upload N[#output] org/model`: a run's weights output as a private checkpoint in
//! the owner's repository, uploaded by the machine that holds it, without running it again.
use std::process::ExitCode;

use protocol::{
    calcifer::{Kind as RunKind, On, OutputItem, Run, SubmitRun},
    error::{Error, Kind},
    machine::{self as v1, ModelChoice},
};

use super::{exit_code, follow, submit};
use crate::{Context, model::refs};

pub async fn exec(
    ctx: &Context,
    reference: &str,
    model: &str,
    detach: bool,
) -> Result<ExitCode, Error> {
    let (number, output) = match reference.split_once('#') {
        Some((number, output)) => (number, Some(output)),
        None => (reference, None),
    };
    let destination = refs::destination(model)?;
    let run = ctx.calcifer().await?.run(number).await?;
    let item = weights(&run, output)?;
    let machine = run.machine.clone().ok_or_else(|| {
        Error::new(
            Kind::Conflict,
            "run_unplaced",
            format!("run {} never reached a machine", run.number),
        )
    })?;
    let hub = ctx.config.resolve(&run.hub)?;
    if !ctx.signed_in(&hub) {
        return Err(Error::new(
            Kind::Credential,
            "hub_sign_in_required",
            format!(
                "an upload goes to your account at {}, which needs your sign-in",
                hub.origin
            ),
        )
        .next("cozy login <email>"));
    }
    let submission = submission(&run, item, machine, &destination);
    let submitted = submit(ctx, submission, None).await?;
    Ok(exit_code(&follow(ctx, submitted.run, detach).await?))
}

/// The weights output named `output`, else the run's only one.
fn weights<'a>(run: &'a Run, output: Option<&str>) -> Result<&'a OutputItem, Error> {
    let held: Vec<&OutputItem> = run
        .output
        .iter()
        .filter(|item| item.is_checkpoint() && output.is_none_or(|name| item.name == name))
        .collect();
    match held.as_slice() {
        [item] => Ok(item),
        [] => Err(Error::new(
            Kind::NotFound,
            "run_output_absent",
            match output {
                Some(name) => format!("run {} wrote no weights output {name:?}", run.number),
                None => format!("run {} wrote no weights output", run.number),
            },
        )),
        several => Err(Error::new(
            Kind::Usage,
            "run_output_ambiguous",
            format!(
                "run {} wrote {} weights outputs: name one as {}#<output>",
                run.number,
                several.len(),
                run.number
            ),
        )),
    }
}

/// A WARM run on the run's machine: the output's checkpoint by hash, into `destination`.
fn submission(run: &Run, item: &OutputItem, machine: String, destination: &str) -> SubmitRun {
    let reference = (item.checkpoint.as_ref()).expect("a weights output names its checkpoint");
    let model = ModelChoice {
        parameter: item.name.clone(),
        repository: reference.repo_id.clone(),
        manifest: reference.revision.clone(),
        lane: reference.variant.clone(),
        ..Default::default()
    };
    let spec = v1::RunSpec {
        kind: v1::RunKind::Warm,
        models: vec![model],
        weights_destination: destination.to_owned(),
        ..Default::default()
    };
    SubmitRun {
        agreed: None,
        on: On::Machine(machine),
        hub: run.hub.clone(),
        kind: RunKind::Warm,
        target: format!("upload {}#{} to {destination}", run.number, item.name),
        spec,
        inputs: vec![],
        out: None,
        warnings: vec![],
    }
}

#[cfg(test)]
mod tests {
    use protocol::{
        calcifer::{ItemStatus, Status},
        ids::{Digest, RunId},
    };

    use super::*;

    /// A weights output's marker in these fixtures: done, it names its checkpoint.
    const WEIGHTS: &str = "weights";

    fn run(outputs: &[(&str, &str, bool)]) -> Run {
        let output = outputs
            .iter()
            .enumerate()
            .map(|(i, (name, media_type, done))| OutputItem {
                id: format!("7/{name}"),
                name: (*name).to_owned(),
                index: 0,
                rev: 1,
                length: 100 + i as u64,
                media_type: (*media_type).to_owned(),
                label: String::new(),
                status: ItemStatus::Completed,
                sha256: None,
                path: None,
                export_error: None,
                checkpoint: (*media_type == WEIGHTS && *done).then(|| v1::CheckpointRef {
                    repo_id: "local/quantized".into(),
                    revision: Digest::of(name.as_bytes()).to_string(),
                    variant: (*name).to_owned(),
                    ..Default::default()
                }),
            })
            .collect();
        serde_json::from_value::<Run>(serde_json::json!({
            "id": RunId::mint([7; 12]), "number": 7, "kind": "call", "target": "fidika/sdxl/int8",
            "hub": "https://tensorhub.com", "machine": "kiyui", "status": Status::Completed, "created_at_ms": 1,
        }))
        .map(|mut run| {
            run.output = output;
            run
        })
        .unwrap()
    }

    #[test]
    fn the_weights_output_is_named_or_the_only_one() {
        let one = run(&[("int8", WEIGHTS, true), ("preview", "image/webp", true)]);
        assert_eq!(weights(&one, None).unwrap().name, "int8");
        assert_eq!(
            weights(&one, Some("preview")).unwrap_err().code,
            "run_output_absent"
        );
        let two = run(&[("int8", WEIGHTS, true), ("fp8", WEIGHTS, true)]);
        assert_eq!(
            weights(&two, None).unwrap_err().code,
            "run_output_ambiguous"
        );
        assert_eq!(weights(&two, Some("fp8")).unwrap().name, "fp8");
    }

    #[test]
    fn an_upload_is_a_warm_run_on_the_runs_machine() {
        let held = run(&[("int8", WEIGHTS, true)]);
        let destination = refs::destination("Paul/SDXL-int8").unwrap();
        let submit = submission(&held, &held.output[0], "kiyui".into(), &destination);
        assert_eq!(
            (submit.on, submit.kind),
            (On::Machine("kiyui".into()), RunKind::Warm)
        );
        assert_eq!(submit.target, "upload 7#int8 to paul/sdxl-int8@main");
        let spec = submit.spec;
        assert_eq!(
            (spec.kind, spec.weights_destination.as_str()),
            (v1::RunKind::Warm, "paul/sdxl-int8@main")
        );
        let model = &spec.models[0];
        assert_eq!(
            (
                model.parameter.as_str(),
                model.repository.as_str(),
                model.lane.as_str()
            ),
            ("int8", "local/quantized", "int8")
        );
        assert_eq!(model.manifest, Digest::of(b"int8").to_string());
        assert!(model.release.is_empty());
        assert!(refs::destination("local/x").is_err());
    }

    #[test]
    fn a_retained_typed_checkpoint_retries_its_tree_and_variant_without_byte_metadata() {
        let mut held = run(&[("model", "", false)]);
        held.status = Status::Failed;
        let reference = v1::CheckpointRef {
            repo_id: "local/derived".into(),
            revision: format!("sha256:{}", "a".repeat(64)),
            variant: "f32".into(),
            tag: String::new(),
        };
        held.output[0].checkpoint = Some(reference.clone());
        let item = weights(&held, Some("model")).unwrap();
        let destination = refs::destination("owner/retry").unwrap();
        let sent = submission(&held, item, "original-machine".into(), &destination);
        assert_eq!(sent.on, On::Machine("original-machine".into()));
        assert_eq!(sent.spec.kind, v1::RunKind::Warm);
        let choice = &sent.spec.models[0];
        assert_eq!(
            (&choice.repository, &choice.manifest, &choice.lane),
            (&reference.repo_id, &reference.revision, &reference.variant)
        );
        assert!(choice.release.is_empty());
        assert_eq!(choice.manifest_length, 0);
        assert!(sent.spec.source.is_none());
        held.output
            .push(run(&[("other", WEIGHTS, true)]).output.remove(0));
        assert_eq!(
            weights(&held, None).unwrap_err().code,
            "run_output_ambiguous"
        );
    }
}
