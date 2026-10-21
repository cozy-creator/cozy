//! Local code: `cozy run ./dir[/fn]` and `cozy run ./script.py`, a package nobody published. The
//! machine installs it and answers its interface; the CLI binds the terms against that and calls
//! it, just as it calls a release. No Python runs here. The interface is the source's (a static
//! description), so it is kept by the source manifest's digest: a source sent before is called
//! at once, and its machine makes its environment ready inside the call.
use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

use platform::{durable, paths::Home};

use protocol::{
    calcifer::{Kind as RunKind, On, Run, Status, SubmitRun},
    error::{Error, Kind},
    machine as v1,
};

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
    /// The machine's name for the installation; none for a kept interface.
    installation: Option<String>,
    /// Where it was installed; none for a kept interface.
    machine: Option<String>,
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
    // A warm set names the machine's installation, which only an install answers.
    let installed = match kept(&ctx.home, &local).filter(|_| args.warm.is_none()) {
        Some(interface) => Installed {
            interface,
            installation: None,
            machine: None,
        },
        None => install(ctx, &hub.origin, on.clone(), &local).await?,
    };
    let script = path.extension().is_some_and(|e| e == "py");
    let public = installed.interface.public();
    let function = match (function, public.as_slice()) {
        (Some(function), _) => function,
        (None, [only]) if script => (*only).to_owned(),
        (None, _) => {
            super::unnamed(&local.package, None, &installed.interface, &args)?;
            super::functions_of(ctx, &local.package, ("local", ""), None, &installed.interface);
            return Ok(ExitCode::SUCCESS);
        }
    };
    let callable = super::callable(&local.package, &installed.interface, &function)?;
    let target = format!("{}/{}", local.package, callable.name);
    if args.describe {
        anstream::print!(
            "{}",
            describe::contract(&target, &installed.interface, callable)
        );
        return Ok(ExitCode::SUCCESS);
    }
    let terms = super::terms(ctx, &args)?;
    let bound = super::bind::bind(&target, None, callable, &terms)?;
    if let Some(warning) = bound.warning(&target) {
        output::warn(&warning);
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
        attention_kernel: args.attention_kernel.as_deref().unwrap_or_default(),
    };
    // The call goes where the code was installed, with its files once more: a machine that
    // dropped them meanwhile is sent only what it lacks, and installs them in the call.
    let on = installed.machine.map_or(on, On::Machine);
    let spec = match args.warm {
        Some(level) => call.warm(level, &bound.models)?,
        None => call.spec(&bound)?,
    };
    if args.warm.is_none() {
        super::gpuless(ctx, &on, &installed.interface, callable, &target, call.bits).await;
    }
    let bound = args.warm.is_none().then_some(&bound);
    let out = args.out.as_ref().map(|dir| terms.cwd.join(dir));
    let mut submission = call.submission(on, &hub.origin, &spec, bound, out);
    submission.inputs.extend(local.inputs.iter().cloned());
    let run = super::send(ctx, submission, &terms.key, args.detach).await?;
    Ok(super::exit_code(&run))
}

/// Installs the code on its machine: a WARM run naming no function, whose result is the
/// installation and its interface, then kept. It is named for what the machine does: an install
/// when the environment's inputs (metadata, lock, path dependencies) are new here, else a sync of
/// the changed source.
async fn install(ctx: &Context, hub: &str, on: On, local: &Local) -> Result<Installed, Error> {
    let target = match synced(&ctx.home, local) {
        true => format!("sync {} (source changed)", local.package),
        false => format!("install {}", local.package),
    };
    let spec = v1::RunSpec {
        kind: v1::RunKind::Warm,
        source: Some(v1::Source::Local(local.source.clone())),
        ..Default::default()
    };
    let submission = SubmitRun {
        agreed: None,
        on,
        hub: hub.to_owned(),
        kind: RunKind::Warm,
        target,
        spec,
        inputs: local.inputs.clone(),
        out: None,
        warnings: vec![],
    };
    let submitted = submit(ctx, submission, None, false).await?;
    let run = settle(ctx, submitted.run).await?;
    let result = run.result.clone().unwrap_or_default();
    let installed = installed(run)?;
    keep(&ctx.home, local, &result["interface"]);
    Ok(installed)
}

/// Where local code's interfaces are kept, beside release cards: one file per source manifest,
/// named for its environment's inputs, so a changed source with the same environment is seen.
fn dir(home: &Home, local: &Local) -> PathBuf {
    let mut dir = home.root().join("packages");
    dir.extend(local.package.split('/'));
    dir
}

fn file(home: &Home, local: &Local) -> PathBuf {
    let manifest = local.source.manifest.trim_start_matches("sha256:");
    dir(home, local).join(format!(
        "{}-{manifest}.json",
        &hex(&local.environment)[..16]
    ))
}

fn hex(digest: &protocol::ids::Digest) -> String {
    digest.to_string().trim_start_matches("sha256:").to_owned()
}

/// The interface kept for this exact source.
fn kept(home: &Home, local: &Local) -> Option<Interface> {
    let document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(file(home, local)).ok()?).ok()?;
    Interface::parse(&document).ok()
}

/// Whether some kept source of this package has the same environment inputs.
fn synced(home: &Home, local: &Local) -> bool {
    let prefix = format!("{}-", &hex(&local.environment)[..16]);
    std::fs::read_dir(dir(home, local))
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| e.file_name().to_string_lossy().starts_with(&prefix))
}

/// Keeps `interface` for this source, and only the newest 16 sources of the package.
fn keep(home: &Home, local: &Local, interface: &serde_json::Value) {
    const KEPT: usize = 16;
    let dir = dir(home, local);
    let kept = durable::private_dir(&dir).and_then(|()| {
        durable::replace(&file(home, local), interface.to_string().as_bytes(), 0o600)
    });
    // An interface that cannot be kept is asked again next time; the run goes on.
    drop(kept);
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    files.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
    for (_, old) in files.into_iter().skip(KEPT) {
        let _ = std::fs::remove_file(old);
    }
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
        machine: Some(machine),
    })
}

#[cfg(test)]
mod tests {
    use protocol::{calcifer::Reason, ids::RunId};

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
                installed.machine.as_deref()
            ),
            (Some("local-0123"), Some("local"))
        );
        assert_eq!(installed.interface.public(), ["generate", "fp8", "int8"]);
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
