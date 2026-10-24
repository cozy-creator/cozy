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
use crate::{Context, package::local::Local};

/// What the machine installed: the package's interface and where the call must go.
struct Installed {
    interface: Interface,
    /// The machine's name for the installation; none for a kept interface.
    installation: Option<String>,
    /// Where it was installed; none for a kept interface.
    machine: Option<String>,
}

/// Local code made callable: installed on its machine, or its interface kept from an earlier
/// install of this exact source.
pub(crate) struct Prepared {
    local: Local,
    installed: Installed,
    on: On,
}

impl Prepared {
    /// `local/<name>`.
    pub(crate) fn package(&self) -> &str {
        &self.local.package
    }

    pub(crate) fn interface(&self) -> &Interface {
        &self.installed.interface
    }

    /// The source manifest's digest: this exact code.
    pub(crate) fn source(&self) -> &str {
        &self.local.source.manifest
    }

    /// Calls `function`. The call goes where the code was installed, with its files once more:
    /// a machine that dropped them meanwhile is sent only what it lacks, and installs them in
    /// the call.
    pub(super) async fn call(&self, ctx: &Context, function: &str, args: RunArgs) -> Result<Run, Error> {
        let code = super::Code {
            package: &self.local.package,
            interface: &self.installed.interface,
            release: None,
            source: spec::Source::Local {
                source: &self.local.source,
                installation: self.installed.installation.as_deref(),
            },
            files: &self.local.inputs,
        };
        let on = self.installed.machine.clone().map_or_else(|| self.on.clone(), On::Machine);
        super::call_code(ctx, code, function, on, args).await
    }
}

/// The code at `path` made callable `on` its machine. A warm set names the machine's
/// installation, which only an install answers.
pub(crate) async fn prepare(ctx: &Context, path: &Path, on: On, warm: bool) -> Result<Prepared, Error> {
    if !path.exists() {
        return Err(Error::new(
            Kind::NotFound,
            "target_not_found",
            format!("{} does not exist", path.display()),
        ));
    }
    let hub = ctx.hub()?;
    let local = crate::package::local::source(ctx, path, &on).await?;
    let installed = match kept(&ctx.home, &local).filter(|_| !warm) {
        Some(interface) => Installed {
            interface,
            installation: None,
            machine: None,
        },
        None => install(ctx, &hub.origin, on.clone(), &local).await?,
    };
    Ok(Prepared { local, installed, on })
}

pub(super) async fn run(
    ctx: &Context,
    path: &Path,
    function: Option<String>,
    args: RunArgs,
) -> Result<ExitCode, Error> {
    let prepared = prepare(ctx, path, args.placement.on(), args.warm.is_some()).await?;
    let (package, interface) = (prepared.package(), prepared.interface());
    let script = path.extension().is_some_and(|e| e == "py");
    // A script's one callable runs bare, a recipe too: local code is its author's to run.
    let callables: Vec<&str> = (interface.callables.iter()).filter(|c| !c.internal).map(|c| c.name.as_str()).collect();
    let function = match (function, callables.as_slice()) {
        (Some(function), _) => function,
        (None, [only]) if script => (*only).to_owned(),
        (None, _) => {
            super::unnamed(package, None, interface, &args)?;
            super::functions_of(ctx, package, ("local", ""), None, interface);
            return Ok(ExitCode::SUCCESS);
        }
    };
    if args.describe {
        let callable = super::callable(package, interface, &function)?;
        if ctx.json {
            let document = std::fs::read(file(&ctx.home, &prepared.local)).ok();
            let document: serde_json::Value = document.and_then(|bytes| serde_json::from_slice(&bytes).ok()).unwrap_or_default();
            crate::output::json(&super::catalog::request(&document, &callable.name));
            return Ok(ExitCode::SUCCESS);
        }
        let target = format!("{package}/{}", callable.name);
        let contract = describe::contract(&target, interface, callable);
        anstream::print!("{contract}");
        return Ok(ExitCode::SUCCESS);
    }
    let run = prepared.call(ctx, &function, args).await?;
    Ok(super::exit_code(&run))
}

/// Installs the code on its machine: a WARM run naming no function, whose result is the
/// installation and its interface, then kept. It is named for what the machine does: an install
/// when the environment's inputs (metadata, lock, path dependencies) are new here, else a sync of
/// the changed source.
async fn install(ctx: &Context, hub: &str, on: On, local: &Local) -> Result<Installed, Error> {
    let submitted = submit(ctx, install_run(&ctx.home, hub, on, local), None).await?;
    let run = settle(ctx, submitted.run).await?;
    let result = run.result.clone().unwrap_or_default();
    let installed = installed(run)?;
    keep(&ctx.home, local, &result["interface"]);
    Ok(installed)
}

/// The run that installs local code `on` a machine: a WARM run naming no function.
pub(crate) fn install_run(home: &Home, hub: &str, on: On, local: &Local) -> SubmitRun {
    let target = match synced(home, local) {
        true => format!("sync {} (source changed)", local.package),
        false => format!("install {}", local.package),
    };
    let spec = v1::RunSpec {
        kind: v1::RunKind::Warm,
        source: Some(v1::Source::Local(local.source.clone())),
        ..Default::default()
    };
    SubmitRun {
        agreed: None,
        on,
        hub: hub.to_owned(),
        kind: RunKind::Warm,
        target,
        spec,
        inputs: local.inputs.clone(),
        out: None,
        warnings: vec![],
    }
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
