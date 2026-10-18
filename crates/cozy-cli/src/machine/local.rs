//! This computer's machine: `cozy machine enable`, `disable` and `authorize`. Enabling runs
//! `cozy machine serve --persistent` as this user's service; a machine an older cozy started (a
//! transient `tensord-*` unit) is taken over on the same root, store, ports, worker id and keys,
//! once it is idle, so nothing it holds is fetched again.
use std::{
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use cozy_platform::{
    lock::{Kind as LockKind, LockFile},
    service::{Legacy, ServiceSpec, Services},
};
use cozy_proto::{
    calcifer::{Event, Machine, NewMachine, RunList},
    error::{Error, Kind},
};
use toml_edit::{Item, Table, value};

use crate::{Context, calcifer_client::Client, output};

/// The user service `cozy machine enable` installs.
const SERVICE: &str = "machine";
/// How often a start is looked at, and how often the person hears it is still starting.
const SAMPLE: Duration = Duration::from_millis(250);
const REPORT: Duration = Duration::from_secs(10);

/// An older cozy's machine environment, and the `[machine]` key each carries into.
const CARRIED: [(&str, &str); 7] = [
    ("COZY_MACHINE_ROOT", "root"),
    ("COZY_TENSORFS_ROOT", "store"),
    ("COZY_WORKER_INTERNAL_PORT", "port"),
    ("COZY_MEDIA_INTERNAL_PORT", "webrtc_port"),
    ("COZY_WEBRTC_INTERNAL_PORT", "webrtc_port"),
    ("COZY_LISTEN_HOST", "listen_host"),
    ("COZY_WORKER_ID", "worker_id"),
];

/// `cozy machine enable`: this computer's machine as this user's service, recorded as `local`.
/// A machine an older cozy started is taken over once its runs end; Ctrl-C before then leaves
/// everything as it was. A service runs only for the login home under a user service manager;
/// otherwise the machine runs detached, with one warning. Enabling an enabled machine changes
/// nothing and says its state.
pub async fn enable(ctx: &Context) -> Result<ExitCode, Error> {
    let services = services(ctx)?;
    let managed = services.available();
    let calcifer = ctx.calcifer().await?;
    // An older cozy's machine is taken over while it runs, or once if its settings were never
    // carried; a stopped one whose settings are the config's is history.
    let legacy = legacy(ctx, &services, "machine_takeover_failed")?
        .filter(|legacy| legacy.active || !configured(ctx));
    if let Some(legacy) = &legacy {
        if legacy.active {
            idle(ctx, &calcifer).await?;
        }
        carry(ctx, legacy)?;
        services
            .retire(legacy)
            .map_err(|e| service_error("machine_takeover_failed", e))?;
        output::say(
            ctx,
            &format!("stopped {}, which an older cozy started", legacy.unit),
        );
    }
    // The config as the service reads it: the carried `[machine]` keys, if any were written.
    let root = machine_root(ctx)?;
    let spec = spec(&root)?;
    let running = !free(&root);
    let changed = match managed {
        true => services
            .ensure(&spec)
            .map_err(|e| service_error("machine_start_failed", e))?,
        false => false,
    };
    let started = !running || changed || legacy.is_some();
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
        let came = start(
            ctx,
            managed.then_some(&services),
            &spec,
            &root,
            running,
            before,
        )
        .await;
        // A machine that does not start leaves the older one running as it did, never none.
        if let (Err(e), Some(legacy)) = (&came, &legacy) {
            if managed {
                let _ = services.uninstall(SERVICE);
            }
            let revived = match services.revive(legacy) {
                Ok(()) => format!("{} runs again as it did", legacy.unit),
                Err(again) => format!("{} did not start again either: {again}", legacy.unit),
            };
            return Err(Error {
                remedy: Some(revived),
                ..e.clone()
            });
        }
        came?;
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
        .note(match (started, legacy.is_some()) {
            (_, true) => {
                "it took over the machine an older cozy started: same root, store, ports and keys"
            }
            (true, false) => "this computer is a machine now",
            (false, false) => "it was already enabled: nothing changed",
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
    let services = services(ctx)?;
    let calcifer = ctx.calcifer().await?;
    let legacy = legacy(ctx, &services, "machine_stop_failed")?;
    let root = machine_root(ctx)?;
    let was = !free(&root) || legacy.as_ref().is_some_and(|l| l.active);
    if was {
        idle(ctx, &calcifer).await?;
    }
    if let Some(legacy) = &legacy {
        services
            .retire(legacy)
            .map_err(|e| service_error("machine_stop_failed", e))?;
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

const UNMANAGED: &str = "no service manager: the local machine won't restart after logout or \
     reboot; `sudo loginctl enable-linger $USER` (see `cozy doctor`)";
const FOREIGN_HOME: &str = "this home is not your login home's: the local machine runs detached \
     and won't restart after logout or reboot (see `cozy doctor`)";

/// This computer's services. The machine runs as one only for the login home under a user
/// service manager (`Services::available`); otherwise it runs detached.
fn services(ctx: &Context) -> Result<Services, Error> {
    if std::env::consts::OS != "linux" {
        return Err(Error::new(
            Kind::Usage,
            "machine_unsupported",
            "this computer's machine runs on Linux for now",
        )
        .next("cozy machine rent"));
    }
    Ok(Services::new(&ctx.home, &ctx.user_home))
}

/// The machine an older cozy started under this session's user manager, wherever one answers.
fn legacy(ctx: &Context, services: &Services, code: &str) -> Result<Option<Legacy>, Error> {
    match services.reachable() {
        true => services
            .legacy(&ctx.home.machine())
            .map_err(|e| service_error(code, e)),
        false => Ok(None),
    }
}

/// The service: this cozy as `machine serve --persistent`, saying anything before its own log
/// opens into `start.log` beside it.
fn spec(root: &Path) -> Result<ServiceSpec, Error> {
    let program = std::env::current_exe()
        .map_err(|e| Error::new(Kind::Internal, "current_exe", e.to_string()))?;
    Ok(ServiceSpec {
        name: SERVICE.into(),
        program,
        args: vec!["machine".into(), "serve".into(), "--persistent".into()],
        log: Some(start_log(root)),
    })
}

fn start_log(root: &Path) -> PathBuf {
    root.join("var/lib/cozy/machine/start.log")
}

/// Starts the machine detached from this command, as `spec` says (in this command's environment,
/// whose HOME is the one enabling it), saying anything before its own log opens into its start log.
fn detach(spec: &ServiceSpec) -> Result<cozy_platform::process::Detached, Error> {
    cozy_platform::process::detach(&spec.program, &spec.args, spec.log.as_deref()).map_err(|e| {
        Error::new(Kind::Unavailable, "machine_start_failed", e.to_string())
            .remedy("sudo loginctl enable-linger $USER, then cozy machine enable")
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
    let lock = LockFile::new(root.join("var/lib/cozy/machine/agent.lock"));
    if free(root) {
        return Ok(());
    }
    let holder = lock.holder().map_err(|e| failed(e.to_string()))?;
    let birth = holder
        .and_then(|record| serde_json::from_slice::<cozy_platform::process::Birth>(&record).ok())
        .ok_or_else(|| failed(format!("{} names no process", root.display())))?;
    if let Some(process) =
        cozy_platform::process::Exact::open(&birth).map_err(|e| failed(e.to_string()))?
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
    let path = ctx.home.config();
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let (config, _) = cozy_proto::config::Config::parse(&text, &path.display().to_string())?;
    Ok(config.machine.root(&ctx.home))
}

/// Whether the config names the machine's root: settings an enable carried, or the person's.
fn configured(ctx: &Context) -> bool {
    let path = ctx.home.config();
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    cozy_proto::config::Config::parse(&text, &path.display().to_string())
        .is_ok_and(|(config, _)| config.machine.root.is_some())
}

/// Whether nothing holds the machine's root: no machine has run there, or none runs now.
fn free(root: &Path) -> bool {
    let path = root.join("var/lib/cozy/machine/agent.lock");
    !path.exists()
        || matches!(
            LockFile::new(path).try_lock(LockKind::Exclusive),
            Ok(Some(_))
        )
}

fn ready_path(root: &Path) -> PathBuf {
    root.join("var/lib/cozy/tensord/api-ready.json")
}

/// When the machine at `root` last wrote that it serves: its `api-ready.json`'s mtime.
fn served_at(root: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(ready_path(root))
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

/// Writes the older machine's settings into `[machine]` and its controller keys into
/// `keys/authorized`, so the service serves the same root, store, ports and worker id.
fn carry(ctx: &Context, legacy: &Legacy) -> Result<(), Error> {
    let env: &BTreeMap<String, String> = &legacy.environment;
    crate::hub::edit(ctx, |doc| {
        if !doc.get("machine").is_some_and(Item::is_table) {
            doc["machine"] = Item::Table(Table::new());
        }
        let machine = doc["machine"].as_table_mut().expect("machine is a table");
        for (name, key) in CARRIED {
            let Some(text) = env.get(name).filter(|v| !v.is_empty()) else {
                continue;
            };
            machine[key] = match key {
                "port" | "webrtc_port" => value(text.parse::<i64>().map_err(|_| {
                    Error::new(
                        Kind::Validation,
                        "machine_takeover_failed",
                        format!("{} has {name}={text}, which is no port", legacy.unit),
                    )
                })?),
                _ => value(text.as_str()),
            };
        }
        Ok(())
    })?;
    for key in env
        .get("COZY_AUTHORIZED_KEYS")
        .map(|keys| {
            keys.split(',')
                .map(str::trim)
                .filter(|k| !k.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
    {
        authorized(ctx, key)?;
    }
    Ok(())
}

fn service_error(code: &str, error: cozy_platform::Error) -> Error {
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
        .next("on that computer: cozy machine add <name> <address> --fingerprint <sha256:…>"));
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
    cozy_platform::durable::private_dir(dir).map_err(io::Error::other)?;
    cozy_platform::durable::replace(path, text.as_bytes(), 0o600).map_err(io::Error::other)
}
