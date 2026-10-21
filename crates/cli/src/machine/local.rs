//! This computer's machine: `cozy machine enable`, `disable` and `authorize`. Enabling runs
//! `cozy machine serve --persistent` as this user's service.
use std::{
    io,
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use platform::{
    lock::{Kind as LockKind, LockFile},
    process::Birth,
    service::{ServiceSpec, Services},
};
use protocol::{
    calcifer::{self, Event, Machine, NewMachine, On, Run, RunList, Status, SubmitRun},
    config::MachineRoot,
    error::{Error, Kind},
    machine as v1,
};
use serde::Serialize;

use crate::{Context, calcifer_client::Client, output};

/// The user service `cozy machine enable` installs.
const SERVICE: &str = "machine";
/// How often a start is looked at, and how often the person hears it is still starting.
const SAMPLE: Duration = Duration::from_millis(250);
const REPORT: Duration = Duration::from_secs(10);

/// `cozy machine enable`: this computer's machine as this user's service, recorded as `local`.
/// A service runs only for the login home under a user service manager; otherwise the machine
/// runs detached, with one warning. Enabling an enabled machine changes nothing and says its state.
pub async fn enable(ctx: &Context) -> Result<ExitCode, Error> {
    let services = services(ctx);
    let managed = services.available();
    let calcifer = ctx.calcifer().await?;
    let root = machine_root(ctx)?;
    let spec = spec(ctx, &root)?;
    let running = !free(&root);
    let changed = match managed {
        true => services
            .ensure(&spec)
            .map_err(|e| service_error("machine_start_failed", e))?,
        false => false,
    };
    let started = !running || changed;
    if started {
        let before = served_at(&root);
        let log = start_log(&root);
        let dir = log.parent().expect("a log has a directory");
        std::fs::create_dir_all(dir).map_err(|e| {
            Error::new(
                Kind::Internal,
                "machine_start_failed",
                format!("{}: {e}", dir.display()),
            )
        })?;
        start(
            ctx,
            managed.then_some(&services),
            &spec,
            &root,
            running,
            before,
        )
        .await?;
    }
    if !managed {
        output::warn(match services.reachable() {
            true => FOREIGN_HOME,
            false => UNMANAGED,
        });
    }
    let local: Machine = calcifer
        .post(
            "/v1/machines",
            &NewMachine::Local { root: root.clone() },
            None,
        )
        .await?;
    if ctx.json {
        output::json(&serde_json::json!({"machine": local, "changed": started}));
        return Ok(ExitCode::SUCCESS);
    }
    let mut record = output::Record::new()
        .field("machine", &local.name)
        .field(
            "service",
            match managed {
                true => format!("cozy-{SERVICE}: running"),
                false => "detached".into(),
            },
        )
        .field("root", root.display().to_string());
    if let Some(address) = &local.address {
        record = record.field("address", address);
    }
    record
        .note(match started {
            true => "this computer is a machine now",
            false => "it was already enabled: nothing changed",
        })
        .print(ctx);
    output::next(
        ctx,
        &[
            "cozy machine show local",
            "cozy run <org/package> --on local",
        ],
    );
    Ok(ExitCode::SUCCESS)
}

/// `cozy machine disable`: stops this computer's machine once its runs end, and keeps every file
/// it holds; enable starts it again on them.
pub async fn disable(ctx: &Context) -> Result<ExitCode, Error> {
    let services = services(ctx);
    let calcifer = ctx.calcifer().await?;
    let root = machine_root(ctx)?;
    let was = !free(&root);
    if was {
        idle(ctx, &calcifer).await?;
    }
    match services.available() {
        true => services
            .uninstall(SERVICE)
            .map_err(|e| service_error("machine_stop_failed", e))?,
        false => end_detached(&root)?,
    }
    match calcifer
        .delete::<serde_json::Value>("/v1/machines/local")
        .await
    {
        Err(e) if e.kind == Kind::NotFound => {}
        answer => drop(answer?),
    }
    if ctx.json {
        output::json(&serde_json::json!({"machine": "local", "changed": was}));
        return Ok(ExitCode::SUCCESS);
    }
    output::Record::new()
        .field("machine", "local")
        .field("service", "stopped")
        .note(match was {
            true => "stopped; its packages, models and outputs stay, and enable starts it on them",
            false => "it was not running",
        })
        .print(ctx);
    Ok(ExitCode::SUCCESS)
}

/// A session without a user manager: Linux only (launchd and the `Run` key always answer).
const UNMANAGED: &str = "no service manager: the local machine won't restart after logout or \
     reboot; `sudo loginctl enable-linger $USER` (see `cozy doctor`)";
const FOREIGN_HOME: &str = "this home is not your login home's: the local machine runs detached \
     and won't restart after logout or reboot (see `cozy doctor`)";
/// What makes the machine this computer's service, which a detached start is not.
const AS_A_SERVICE: &str = match cfg!(target_os = "linux") {
    true => "sudo loginctl enable-linger $USER, then cozy machine enable",
    false => "cozy machine enable as this computer's user, from your login home",
};

/// This computer's services: a systemd user unit, a launchd LaunchAgent or the HKCU `Run` key.
/// The machine runs as one only for the login home under its user's manager
/// (`Services::available`); otherwise it runs detached.
fn services(ctx: &Context) -> Services {
    Services::new(&ctx.home, &ctx.user_home)
}

/// The service: the installed cozy as `machine serve --persistent`, saying anything before its
/// own log opens into `start.log` beside it.
fn spec(ctx: &Context, root: &Path) -> Result<ServiceSpec, Error> {
    Ok(ServiceSpec {
        name: SERVICE.into(),
        program: crate::calcifer_client::installed(&ctx.home)?,
        args: vec!["machine".into(), "serve".into(), "--persistent".into()],
        log: Some(start_log(root)),
    })
}

fn start_log(root: &Path) -> PathBuf {
    MachineRoot::new(root).state().join("start.log")
}

/// Starts the machine detached from this command, as `spec` says (in this command's environment,
/// whose HOME is the one enabling it), saying anything before its own log opens into its start log.
fn detach(spec: &ServiceSpec) -> Result<platform::process::Detached, Error> {
    platform::process::detach(&spec.program, &spec.args, spec.log.as_deref()).map_err(|e| {
        Error::new(Kind::Unavailable, "machine_start_failed", e.to_string()).remedy(AS_A_SERVICE)
    })
}

/// Starts the machine as `services`' service (or detached without one) and waits until it serves:
/// a start that fails, or a machine that exits or is restarted before it serves, is refused.
async fn start(
    ctx: &Context,
    services: Option<&Services>,
    spec: &ServiceSpec,
    root: &Path,
    running: bool,
    before: Option<std::time::SystemTime>,
) -> Result<(), Error> {
    let failed = |why: String| {
        Error::new(Kind::Unavailable, "machine_start_failed", why)
            .next(format!("cat {}", start_log(root).display()))
    };
    match services {
        Some(services) => {
            if running {
                services.stop(SERVICE).map_err(|e| failed(e.to_string()))?;
            }
            services.start(spec).map_err(|e| failed(e.to_string()))?;
            ready(ctx, root, before, || {
                services
                    .failed(SERVICE)
                    .then(|| failed(format!("cozy-{SERVICE}.service exited before it served")))
            })
            .await
        }
        None => {
            let mut child = detach(spec)?;
            ready(ctx, root, before, || match child.try_wait() {
                Ok(Some(status)) => Some(failed(format!(
                    "the machine exited ({status}) before it served"
                ))),
                _ => None,
            })
            .await
        }
    }
}

/// Ends a detached machine: the process its root's lock names, and everything it started.
fn end_detached(root: &Path) -> Result<(), Error> {
    let failed = |e: String| Error::new(Kind::Unavailable, "machine_stop_failed", e);
    let lock = LockFile::new(MachineRoot::new(root).lock());
    if free(root) {
        return Ok(());
    }
    let holder = lock.holder().map_err(|e| failed(e.to_string()))?;
    let birth = holder
        .and_then(|record| serde_json::from_slice::<platform::process::Birth>(&record).ok())
        .ok_or_else(|| failed(format!("{} names no process", root.display())))?;
    if let Some(process) =
        platform::process::Exact::open(&birth).map_err(|e| failed(e.to_string()))?
    {
        process.kill().map_err(|e| failed(e.to_string()))?;
        process.wait().map_err(|e| failed(e.to_string()))?;
    }
    // Its service lets go of the lock as it exits, after its parent.
    lock.lock(LockKind::Exclusive)
        .map(drop)
        .map_err(|e| failed(e.to_string()))
}

/// The machine's root as the service will read it: the config's `[machine] root` as written now.
fn machine_root(ctx: &Context) -> Result<PathBuf, Error> {
    Ok(machine_config(ctx)?.root(&ctx.home))
}

fn machine_config(ctx: &Context) -> Result<protocol::config::MachineConfig, Error> {
    let path = ctx.home.config();
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let (config, _) = protocol::config::Config::parse(&text, &path.display().to_string())?;
    Ok(config.machine)
}

/// For `cozy uninstall`: the machine's root, and the model store it fills (this computer's
/// TensorFS home, as the machine resolves it); none when the config names a store, which may be
/// another's.
pub(crate) fn root_and_store(ctx: &Context) -> Result<(PathBuf, Option<PathBuf>), Error> {
    let config = machine_config(ctx)?;
    let root = config.root(&ctx.home);
    let shared = || tensorfs::home::resolve(None).map_err(std::io::Error::other);
    let store = match config.store {
        Some(_) => None,
        None => Some(config.store(&MachineRoot::new(&root), false, shared)
            .map_err(|e| Error::new(Kind::Internal, "store_unresolved", e.to_string()))?),
    };
    Ok((root, store))
}

/// For `cozy uninstall`: ends the machine now, with whatever it runs, removes its service, and
/// waits until every process it was has exited.
pub(crate) fn end(ctx: &Context, root: &Path) -> Result<(), Error> {
    let tree = processes(root);
    let services = services(ctx);
    if services.available() {
        services
            .uninstall(SERVICE)
            .map_err(|e| service_error("machine_stop_failed", e))?;
    }
    platform::process::end_tree(&tree)
        .map_err(|e| Error::new(Kind::Unavailable, "machine_stop_failed", e.to_string()))?;
    end_detached(root)
}

/// The running machine's processes, eldest first: the holder of its root's lock and all it
/// started, and on Windows the stable parent above it, which would start a killed service again.
fn processes(root: &Path) -> Vec<platform::process::Exact> {
    let lock = LockFile::new(MachineRoot::new(root).lock());
    let holder = (!free(root))
        .then(|| lock.holder().ok().flatten())
        .flatten();
    let Some(birth) = holder.and_then(|record| serde_json::from_slice::<Birth>(&record).ok())
    else {
        return vec![];
    };
    // A parent is born before its child; an older PID's reuse is not.
    #[cfg(windows)]
    let birth = (platform::process::parent_of(birth.pid).ok())
        .and_then(|pid| Birth::of(pid).ok())
        .filter(|parent| parent.start_ticks <= birth.start_ticks)
        .unwrap_or(birth);
    platform::process::tree(birth.pid)
}

/// Whether nothing holds the machine's root: no machine has run there, or none runs now.
fn free(root: &Path) -> bool {
    let path = MachineRoot::new(root).lock();
    !path.exists()
        || matches!(
            LockFile::new(path).try_lock(LockKind::Exclusive),
            Ok(Some(_))
        )
}

/// When the machine at `root` last wrote that it serves: its `api-ready.json`'s mtime.
fn served_at(root: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(MachineRoot::new(root).ready())
        .and_then(|m| m.modified())
        .ok()
}

/// Waits until the machine writes that it serves after `before`, saying every so often that it
/// still starts and where its log is. It ends on the machine's own answer, never on a clock.
async fn ready(
    ctx: &Context,
    root: &Path,
    before: Option<std::time::SystemTime>,
    mut failed: impl FnMut() -> Option<Error>,
) -> Result<(), Error> {
    let began = Instant::now();
    let mut reported = began;
    loop {
        if served_at(root).is_some_and(|at| Some(at) > before) && !free(root) {
            return Ok(());
        }
        if let Some(error) = failed() {
            return Err(error);
        }
        tokio::time::sleep(SAMPLE).await;
        if reported.elapsed() >= REPORT {
            reported = Instant::now();
            output::say(
                ctx,
                &format!(
                    "waiting for the machine to serve ({}s so far); {} says what it is doing",
                    began.elapsed().as_secs(),
                    start_log(root).display()
                ),
            );
        }
    }
}

/// Waits until no run is in progress on `local`, naming each and following it to its end.
async fn idle(ctx: &Context, calcifer: &Client) -> Result<(), Error> {
    loop {
        let list: RunList = calcifer
            .runs(&[("status", "in_progress"), ("limit", "200")])
            .await?;
        let live: Vec<_> = list
            .runs
            .into_iter()
            .filter(|run| run.machine.as_deref() == Some("local"))
            .collect();
        let Some(first) = live.first() else {
            return Ok(());
        };
        let named: Vec<String> = live
            .iter()
            .map(|run| format!("run {}", run.number))
            .collect();
        output::say(
            ctx,
            &format!(
                "waiting for {} on local to end; Ctrl-C leaves everything as it is",
                named.join(", ")
            ),
        );
        let mut events = calcifer.events(first.id.as_str(), 0);
        let mut stage = String::new();
        while let Some(event) = events.next().await {
            match event? {
                Event::Progress { progress } if progress.stage != stage => {
                    stage.clone_from(&progress.stage);
                    output::say(ctx, &format!("  run {}: {stage}", first.number));
                }
                Event::Completed { .. } | Event::Failed { .. } | Event::Canceled { .. } => break,
                _ => {}
            }
        }
    }
}

/// What this computer's machine did with the binary `cozy update` installed (design §5).
#[derive(Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub(crate) enum Adopted {
    NotEnabled,
    /// None runs now: it starts on the new binary.
    NotRunning,
    Restarted {
        version: String,
    },
    /// It ends the `runs` it accepted first, then restarts itself: update run `run`.
    Waits {
        runs: u64,
        run: u64,
    },
    /// It kept its binary: rolled back, refused, or a machine that predates binary updates.
    Kept {
        code: String,
        message: String,
        run: Option<u64>,
    },
}

/// Asks this computer's machine to run the binary `cozy update` installed: an update run on
/// `local` closes admission, waits for the runs it accepted, restarts the service on the new
/// binary and rolls back to `.cozy-previous` if that never serves. Followed until it serves on it
/// or says how many runs it waits for.
pub(crate) async fn adopt(ctx: &Context, version: &str) -> Adopted {
    match adopting(ctx, version).await {
        Ok(adopted) => adopted,
        Err(e) => Adopted::Kept {
            code: e.code,
            message: e.message,
            run: None,
        },
    }
}

async fn adopting(ctx: &Context, version: &str) -> Result<Adopted, Error> {
    let calcifer = ctx.calcifer().await?;
    let local: Machine = match calcifer.get("/v1/machines/local").await {
        Ok(local) => local,
        Err(e) if e.kind == Kind::NotFound => return Ok(Adopted::NotEnabled),
        Err(e) => return Err(e),
    };
    if free(&machine_root(ctx)?) {
        return Ok(Adopted::NotRunning);
    }
    let hub = match local.hub.as_str() {
        "" => ctx.hub()?,
        origin => ctx.config.resolve(origin)?,
    };
    let payload = serde_json::json!({ "cozy": version });
    let submit = SubmitRun {
        agreed: None,
        on: On::Machine(local.name),
        hub: hub.origin.to_string(),
        kind: calcifer::Kind::Update,
        target: format!("cozy {version}"),
        spec: v1::RunSpec {
            kind: v1::RunKind::Update,
            payload: Some(v1::Json::parse(payload.to_string()).expect("JSON")),
            ..Default::default()
        },
        inputs: vec![],
        out: None,
        warnings: vec![],
    };
    let run = crate::run::submit(ctx, submit, None).await?.run;
    let mut events = calcifer.events(run.id.as_str(), 0);
    while let Some(event) = events.next().await {
        match event? {
            Event::Progress { progress }
                if progress.stage == "waiting_activation"
                    && progress.total > progress.completed =>
            {
                let runs = progress.total - progress.completed;
                return Ok(Adopted::Waits {
                    runs,
                    run: run.number,
                });
            }
            Event::Completed { run, .. }
            | Event::Failed { run, .. }
            | Event::Canceled { run, .. } => {
                return Ok(settled(run));
            }
            _ => {}
        }
    }
    Ok(settled(calcifer.run(run.id.as_str()).await?))
}

/// An update run's end: the version its machine serves on (`to.cozy`, which the new binary
/// reports once it serves), or why it kept the binary it had.
fn settled(run: Run) -> Adopted {
    let to = run.result.as_ref().and_then(|r| r["to"]["cozy"].as_str());
    match (&run.status, to, run.reason) {
        (Status::Completed, Some(version), _) => Adopted::Restarted { version: version.into() },
        (Status::Completed, None, _) => Adopted::Kept {
            code: "binary_update_unsupported".into(),
            message: "the machine predates binary updates; `cozy machine disable` and `cozy machine enable` restart it".into(),
            run: Some(run.number),
        },
        (Status::Failed | Status::Canceled, _, Some(reason)) => Adopted::Kept {
            code: reason.code,
            message: reason.message,
            run: Some(run.number),
        },
        (status, _, _) => Adopted::Kept {
            code: "update_unsettled".into(),
            message: format!("the update run is {status:?}"),
            run: Some(run.number),
        },
    }
}

fn service_error(code: &str, error: platform::Error) -> Error {
    Error::new(Kind::Unavailable, code, error.to_string()).next("journalctl --user -u cozy-machine")
}

/// `cozy machine authorize <public-key>`: lets the controller with this Ed25519 key (as `cozy
/// machine add` printed it on that computer) use this computer's machine. TensorD reads the file
/// again when a signer is unknown: no restart.
pub fn authorize(ctx: &Context, key: &str) -> Result<ExitCode, Error> {
    let key = key.trim();
    let changed = authorized(ctx, key)?;
    let path = ctx.home.keys().join("authorized");
    if ctx.json {
        output::json(&serde_json::json!({"authorized": key, "changed": changed}));
    } else {
        output::Record::new()
            .field("authorized", key)
            .field("in", path.display().to_string())
            .note(match changed {
                false => "it was already authorized",
                true => "that controller can now use this machine; nothing restarts",
            })
            .print(ctx);
    }
    Ok(ExitCode::SUCCESS)
}

/// Adds `key` to `keys/authorized` unless it is there: true when it was added.
fn authorized(ctx: &Context, key: &str) -> Result<bool, Error> {
    let valid = URL_SAFE_NO_PAD
        .decode(key)
        .ok()
        .and_then(|raw| <[u8; 32]>::try_from(raw).ok())
        .is_some_and(|raw| ed25519_dalek::VerifyingKey::from_bytes(&raw).is_ok());
    if !valid {
        return Err(Error::new(
            Kind::Validation,
            "public_key_invalid",
            format!("{key:?} is not a controller's public key (32 bytes of unpadded base64url)"),
        )
        .next("on that computer: install cozy (`curl -LsSf https://tensorhub.com/install.sh | sh`), then cozy machine add <name> <address> --fingerprint <sha256:…>"));
    }
    let path = ctx.home.keys().join("authorized");
    let unwritable = |e: io::Error| {
        Error::new(
            Kind::Internal,
            "keys_unwritable",
            format!("{}: {e}", path.display()),
        )
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(unwritable(e)),
    };
    let held = text
        .lines()
        .any(|line| line.split('#').next().unwrap_or_default().trim() == key);
    if !held {
        let mut text = text;
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(key);
        text.push('\n');
        write_private(&path, &text).map_err(unwritable)?;
    }
    Ok(!held)
}

/// Writes `text` to `path` whole at 0600, its directory 0700.
fn write_private(path: &Path, text: &str) -> io::Result<()> {
    let dir = path.parent().expect("a keys file has a directory");
    platform::durable::private_dir(dir).map_err(io::Error::other)?;
    platform::durable::replace(path, text.as_bytes(), 0o600).map_err(io::Error::other)
}
