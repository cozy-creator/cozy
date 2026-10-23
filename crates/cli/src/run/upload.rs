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
