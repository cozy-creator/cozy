//! Local code: `cozy run ./dir[/fn]` and `cozy run ./script.py`, a package nobody published. The
//! machine installs it and answers its interface; the CLI binds the terms against that and calls
//! it, just as it calls a release. No Python runs here.
use std::{path::Path, process::ExitCode};

use cozy_proto::{
    calcifer::{Kind as RunKind, On, Run, Status, SubmitRun},
    error::{Error, Kind},
    machine as v1,
};
use prost::Message as _;

use super::{
    RunArgs, describe,
    follow::{self, settle, submit},
    interface::Interface,
    spec,
};
use crate::{Context, output, package::local::Local};

/// What the machine installed: the package's interface and where the call must go.
struct Installed {
    interface: Interface,
    /// The machine's name for the installation; an older machine names none.
    installation: Option<String>,
    machine: String,
}

pub async fn run(
    ctx: &Context,
    path: &Path,
    function: Option<String>,
    args: RunArgs,
) -> Result<ExitCode, Error> {
    if !path.exists() {
        return Err(Error::new(
            Kind::NotFound,
            "target_not_found",
            format!("{} does not exist", path.display()),
        ));
    }
    let hub = ctx.hub()?;
    let on = args.placement.on();
    // This computer's machine reads the files in place; any other is sent them.
    let in_place = on == On::Machine("local".into());
    let local = crate::package::local::source(ctx, path, in_place).await?;
    let installed = install(ctx, &hub.origin, on, &local).await?;
    let script = path.extension().is_some_and(|e| e == "py");
    let public = installed.interface.public();
    let function = match (function, public.as_slice()) {
        (Some(function), _) => function,
        (None, [only]) if script => (*only).to_owned(),
        (None, _) => {
            super::functions_of(ctx, &local.package, "local", &installed.interface);
            return Ok(ExitCode::SUCCESS);
        }
    };
    let callable = super::callable(&local.package, &installed.interface, &function)?;
    let target = format!("{}/{}", local.package, callable.name);
    if args.describe {
        anstream::print!("{}", describe::contract(&target, callable));
        return Ok(ExitCode::SUCCESS);
    }
    let terms = super::terms(ctx, &args)?;
    let bound = super::bind_callable(ctx, &target, callable, &terms)?;
    if let Some(warning) = bound.warning(&target) {
        output::warn(ctx, &warning);
    }
    let call = spec::Call {
        package: &local.package,
        source: spec::Source::Local {
            source: &local.source,
            installation: installed.installation.as_deref(),
        },
        callable,
        gpus: args.gpus.unwrap_or(0),
        bits: super::bits(&args),
    };
    // The call goes where the code was installed, with its files once more: a machine that
    // dropped them meanwhile is sent only what it lacks.
    let on = On::Machine(installed.machine);
    let spec = match args.warm {
        Some(level) => call.warm(level, &bound.models)?,
        None => call.spec(&bound)?,
    };
    let bound = args.warm.is_none().then_some(&bound);
    let out = args.out.as_ref().map(|dir| terms.cwd.join(dir));
    let mut submission = call.submission(on, &hub.origin, &spec, bound, out);
    submission.inputs.extend(local.inputs.iter().cloned());
    super::send(ctx, submission, &terms.key, args.detach).await
}

/// Installs the code on its machine: a WARM run naming no function, whose result is the
/// installation and its interface.
async fn install(ctx: &Context, hub: &str, on: On, local: &Local) -> Result<Installed, Error> {
    let spec = v1::RunSpec {
        kind: v1::RunKind::Warm.into(),
        source: Some(v1::run_spec::Source::Local(local.source.clone())),
        ..Default::default()
    };
    let submission = SubmitRun {
        on,
        hub: hub.to_owned(),
        kind: RunKind::Warm,
        target: format!("install {}", local.package),
        spec: spec.encode_to_vec(),
        inputs: local.inputs.clone(),
        out: None,
        warnings: vec![],
    };
    let submitted = submit(ctx, submission, None).await?;
    let run = settle(ctx, submitted.run).await?;
    installed(run)
}

fn installed(run: Run) -> Result<Installed, Error> {
    if run.status != Status::Completed {
        let (code, message) = run.reason.map_or_else(
            || {
                (
                    "local_install_unfinished".to_owned(),
                    format!("its install is {}", follow::status_word(run.status)),
                )
            },
            |r| (r.code, r.message),
        );
        return Err(Error::new(
            Kind::Unavailable,
            "local_install_failed",
            format!("the machine did not install this code: {message}"),
        )
        .cause(code)
        .next(format!("cozy run show {}", run.number)));
    }
    let result = run.result.unwrap_or_default();
    let interface = Interface::parse(&result["interface"]).map_err(|why| {
        Error::new(
            Kind::Unavailable,
            "local_interface_unreadable",
            format!("the machine's description of this code is unreadable: {why}"),
        )
    })?;
    let installation = result["installation"].as_str().map(str::to_owned);
    let machine = run.machine.ok_or_else(|| {
        Error::new(
            Kind::Internal,
            "local_install_unplaced",
            "the install ran on no machine",
        )
    })?;
    Ok(Installed {
        interface,
        installation,
        machine,
    })
}

#[cfg(test)]
mod tests {
    use cozy_proto::{calcifer::Reason, ids::RunId};

    use super::*;

    fn run(status: Status, result: serde_json::Value) -> Run {
        let mut run: Run = serde_json::from_value(serde_json::json!({
            "id": RunId::mint([3; 12]), "number": 4, "kind": "warm", "target": "install local/tiny",
            "hub": "https://tensorhub.com", "machine": "local", "status": status, "created_at_ms": 1,
        }))
        .unwrap();
        run.result = Some(result);
        run
    }

    #[test]
    fn an_install_answers_its_interface_installation_and_machine() {
        let card = super::super::interface::tests::card("sdxl-2.10.0");
        let result = serde_json::json!({"package": "local/tiny", "release": "0.1.0", "installation": "local-0123",
            "interface": card["package_interface"], "bytes": {"hub": 0, "volume": 0}, "models": []});
        let installed = installed(run(Status::Completed, result)).unwrap();
        assert_eq!(
            (
                installed.installation.as_deref(),
                installed.machine.as_str()
            ),
            (Some("local-0123"), "local")
        );
        assert_eq!(installed.interface.public(), ["fp8", "generate", "int8"]);
    }

    #[test]
    fn a_failed_install_says_why() {
        let mut failed = run(Status::Failed, serde_json::Value::Null);
        failed.reason = Some(Reason {
            code: "uv_failed".into(),
            message: "no solution".into(),
        });
        let error = installed(failed).err().unwrap();
        assert_eq!(
            (error.code.as_str(), error.cause.as_deref()),
            ("local_install_failed", Some("uv_failed"))
        );
        assert!(error.message.contains("no solution"));
    }
}
