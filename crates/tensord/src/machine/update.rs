//! A software update, as `Run kind: update` on `cozy.machine.v1` and (until cutover)
//! `runtime-update/1` behind the CLI's maintenance routes: stage and verify a Runtime/TensorFS
//! pair, wait for measured idleness, then restart the service on it in place. Executors' package
//! environments take the new pair. A `cozy` binary renamed over the launch path (`cozy update`)
//! activates with any update: the parent execs it (`super::binary`). The previous pair and
//! binary stay; a candidate that never proves readiness is rolled back by the parent. Boot id,
//! leaf, journal and outputs are kept. A cancel ends an update
//! that has not begun activating (`canceled`); a pair the machine already runs is unchanged:
//! the same version, or for a wheel file the same bytes (sha256), never just the same version.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex},
};
use crate::process::Supervised;
use crate::{origin::Machine, stop::Stop};

/// The file in a selected pair naming its development Runtime build (`0.24.6+e9baf7a4`),
/// which runs report: a named wheel file is not a published release.
pub const RUNTIME_BUILD: &str = "runtime-build";

/// The service's exit status asking its parent to start the service again on the selected pair.
pub const REPLACE_EXIT: i32 = 75;
/// This binary's release, which an update's `from` and `to` name as `cozy`.
const VERSION: &str = env!("CARGO_PKG_VERSION");
const MAX_WHEEL_BYTES: u64 = 256 << 20;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Pair {
    pub runtime: String,
    pub tensorfs: String,
    /// Each wheel's content digest (hex sha256), in an update's from/to only.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub runtime_sha256: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tensorfs_sha256: String,
    /// The `cozy` binary's release, in an update's from/to only.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub cozy: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Status {
    pub operation: String,
    /// waiting | preparing | waiting_activation | installing | starting | succeeded | rolled_back | failed | canceled
    pub state: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
    /// A failure's own reason code (`wheel_digest_mismatch`); empty reads `update_failed`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub code: String,
    pub from: Pair,
    pub to: Pair,
    #[serde(default)]
    pub pinned: bool,
    /// A toolchain install, not a software pair.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub toolchain: bool,
    /// While `waiting_activation`: the admitted runs still in flight.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub waiting_for: u32,
    /// Every state it passed through: the update's run log.
    #[serde(default)]
    pub history: Vec<Step>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub state: String,
    pub at_ms: i64,
    /// A `waiting_activation` step's runs: those admitted when admission closed, and how many
    /// of them have ended since.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub total: u32,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub completed: u32,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}
impl Status {
    pub fn terminal(&self) -> bool {
        matches!(self.state.as_str(), "succeeded" | "rolled_back" | "failed" | "canceled")
    }
    fn enter(&mut self, state: &str) {
        if self.state != state || self.history.is_empty() {
            self.state = state.into();
            self.history.push(Step {
                state: state.into(),
                at_ms: super::lifecycle::now_ms(),
                ..Default::default()
            });
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Choice {
    #[serde(default)]
    pub file: String,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub version: String,
    /// Where the machine fetches `file` itself, held to `sha256`.
    #[serde(default)]
    pub url: String,
}

impl Choice {
    /// `distribution`'s wheel at an https `url`, held to `sha256`: the machine fetches it.
    pub fn fetched(distribution: &str, url: &str, sha256: &str) -> Result<Choice, String> {
        let path = url.strip_prefix("https://").map(|rest| rest.split(['?', '#']).next().unwrap_or_default());
        let file = path.and_then(|p| p.split_once('/')).and_then(|(_, p)| p.rsplit('/').next()).unwrap_or_default();
        if path.is_none() {
            return Err(format!("{url} is not an https URL"));
        }
        if wheel_name(file).is_none_or(|(d, _)| d != distribution) {
            return Err(format!("{url} names no {distribution} wheel ({distribution}-<version>-….whl)"));
        }
        let sha256 = sha256.to_ascii_lowercase();
        if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("{file}'s sha256 is 64 hex digits, not {sha256:?}"));
        }
        Ok(Choice { file: file.into(), sha256, url: url.into(), ..Default::default() })
    }
}

/// An update failure with its own reason code.
#[derive(Debug)]
pub struct Coded {
    pub code: &'static str,
    pub message: String,
}
impl std::fmt::Display for Coded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for Coded {}

#[derive(Clone, Debug, Deserialize)]
pub struct Request {
    pub operation: String,
    #[serde(default)]
    pub pin: Option<bool>,
    pub runtime: Option<Choice>,
    pub tensorfs: Option<Choice>,
    /// The `cozy` release the caller installed at the launch path (`cozy update`): a guard,
    /// checked before anything drains. A swapped binary activates with or without it.
    #[serde(default)]
    pub cozy: Option<String>,
    /// Install the build toolchain (`super::toolchain`) instead of a software pair.
    #[serde(default)]
    pub toolchain: bool,
}

/// The activation in flight: what a rollback restores.
#[derive(Serialize, Deserialize)]
struct Pending {
    status: Status,
    sdk_before: Option<PathBuf>,
    /// A binary activation's launch path, whose `.cozy-previous` a rollback renames back.
    #[serde(default)]
    binary: Option<PathBuf>,
}

/// What the stable parent does after recovering or rolling back an activation.
#[derive(Debug, PartialEq, Eq)]
pub enum Recovered {
    /// There was none.
    Nothing,
    /// Links (and a binary) were restored; this image starts its service.
    Restored,
    /// The binary at this launch path is not this image's and was selected: exec it.
    Exec(PathBuf),
}

/// Where the machine keeps its software: everything under its own engine directory.
#[derive(Clone)]
pub struct Paths {
    engine: PathBuf,
    image_wheels: PathBuf,
    uv: PathBuf,
}
impl Paths {
    pub fn new(engine: &Path, root: &Path) -> Self {
        // A development image may vendor this machine's own executor SDK beside the Go agent's.
        let own = root.join("opt/cozy/machine/wheels");
        let image_wheels = if own.is_dir() {
            own
        } else {
            root.join("opt/cozy/wheels")
        };
        Self {
            engine: engine.into(),
            image_wheels,
            uv: super::client::uv(root),
        }
    }
    fn update(&self, name: &str) -> PathBuf {
        self.engine.join("update").join(name)
    }
    fn current_sdk_link(&self) -> PathBuf {
        self.engine.join("sdk/current")
    }
    /// The executors' Runtime/TensorFS wheels: the last activated pair, else the image's.
    pub fn sdk(&self) -> PathBuf {
        let link = self.current_sdk_link();
        match fs::symlink_metadata(&link) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => self.image_wheels.clone(),
            // A selected but unavailable SDK is not an instruction to use the image's pair.
            _ => link,
        }
    }
}

/// A pre-cohort machine update may have activated the TensorD its Runtime wheel bundled
/// (`agent/current`). This machine is the cozy binary and never runs another: the activation
/// is removed, once.
pub fn retire_agent(paths: &Paths) {
    let agent = paths.engine.join("agent");
    if fs::symlink_metadata(&agent).is_err() {
        return;
    }
    match fs::remove_dir_all(&agent) {
        Ok(()) => tracing::info!("retired the TensorD a pre-cohort update activated ({})", agent.display()),
        Err(error) => tracing::warn!("{}: {error}; it is never run", agent.display()),
    }
}

/// Restores what a failed activation replaced: the SDK link, and a binary activation's previous
/// binary, renamed back over the launch path.
pub fn rollback_pending(paths: &Paths, cause: &str) -> io::Result<Recovered> {
    let Some(pending) = read_pending(paths)? else {
        return Ok(Recovered::Nothing);
    };
    let mut status = latest(paths, pending.status)?;
    status.error = cause.into();
    status.enter("rolled_back");
    // Persist the decision before changing the link. A crash during rollback then resumes
    // rollback, even when the candidate had already published its starting marker.
    write_json(&paths.update("status.json"), &status)?;
    relink(&paths.current_sdk_link(), pending.sdk_before.as_deref())?;
    if let Some(launch) = &pending.binary {
        // Gone once renamed: a rollback resumed after this step finds the launch path restored.
        match fs::rename(super::binary::previous(launch), launch) {
            Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
            _ => fs::File::open(launch.parent().unwrap_or(Path::new("/")))?.sync_all()?,
        }
    }
    remove_pending(paths)?;
    Ok(match pending.binary.filter(|_| swapped()) {
        Some(launch) => Recovered::Exec(launch),
        None => Recovered::Restored,
    })
}

/// `pair` as this binary runs it.
fn this_binary(pair: Pair) -> Pair {
    Pair { cozy: VERSION.into(), ..pair }
}

/// Whether the file at this process's launch path is not this image.
fn swapped() -> bool {
    super::binary::launch().is_some_and(super::binary::Launch::swapped)
}

/// Recovers a publication interrupted before its durable `starting` marker; the parent calls
/// this before each service it forks. A succeeded marker means readiness already committed the
/// update: only its cleanup remains. A published binary activation this image has not yet
/// handed to is `Exec`.
pub fn recover_activation(paths: &Paths) -> io::Result<Recovered> {
    let Some(pending) = read_pending(paths)? else {
        return Ok(Recovered::Nothing);
    };
    let status = latest(paths, pending.status)?;
    match status.state.as_str() {
        "starting" => {
            // A binary activation alone leaves the SDK link as it was.
            if previous_link(&paths.current_sdk_link())? != pending.sdk_before {
                for distribution in ["cozy_runtime", "tensorfs"] {
                    if let Err(error) = existing_wheel(&paths.current_sdk_link(), distribution) {
                        return rollback_pending(paths, &format!("activated SDK unavailable: {error}"));
                    }
                }
            }
            Ok(match pending.binary.filter(|_| swapped()) {
                Some(launch) => Recovered::Exec(launch),
                None => Recovered::Nothing,
            })
        },
        "succeeded" => {
            remove_pending(paths)?;
            Ok(Recovered::Nothing)
        }
        "rolled_back" => rollback_pending(paths, &status.error),
        "waiting" | "preparing" | "waiting_activation" | "installing" | "failed" | "canceled" => {
            rollback_pending(paths, "machine_restarted: software publication did not finish")
        }
        state => Err(invalid(&format!("cannot recover update state {state}"))),
    }
}

fn read_pending(paths: &Paths) -> io::Result<Option<Pending>> {
    match fs::read(paths.update("pending.json")) {
        Ok(raw) => Ok(Some(serde_json::from_slice(&raw)?)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn remove_pending(paths: &Paths) -> io::Result<()> {
    fs::remove_file(paths.update("pending.json"))?;
    cozy_platform::durable::sync_dir(&paths.update("")).map_err(io::Error::from)
}

/// The activation's status as last written (its later steps), else as it was when pending.
/// A missing status has its pending fallback; an unreadable status must not commit by accident.
fn latest(paths: &Paths, pending: Status) -> io::Result<Status> {
    match fs::read(paths.update("status.json")) {
        Ok(raw) => {
            let status: Status = serde_json::from_slice(&raw)?;
            Ok(if status.operation == pending.operation { status } else { pending })
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(pending),
        Err(error) => Err(error),
    }
}

pub struct Updates {
    paths: Paths,
    status: Mutex<Option<Status>>,
    idle: Box<dyn Fn() -> bool + Send + Sync>,
    lifecycle: Option<Arc<super::lifecycle::Lifecycle>>,
    /// The downloader published wheels land through (`pypi::wheel`).
    machine: Option<Arc<Machine>>,
    /// The running update's stop: its cancel ends the download in flight.
    stop: Mutex<Stop>,
    /// The build toolchain as last probed: at start and after each install.
    toolchain: Mutex<Option<cozy_proto::machine::Toolchain>>,
}

impl Updates {
    pub fn open(
        paths: Paths,
        idle: Box<dyn Fn() -> bool + Send + Sync>,
        lifecycle: Option<Arc<super::lifecycle::Lifecycle>>,
        machine: Option<Arc<Machine>>,
    ) -> io::Result<Arc<Self>> {
        fs::create_dir_all(paths.update("staged"))?;
        cozy_platform::durable::sync_dir(&paths.engine)?;
        let mut status = fs::read(paths.update("status.json"))
            .ok()
            .and_then(|raw| serde_json::from_slice::<Status>(&raw).ok());
        // An update the service stopped during before activation ran no further: it failed.
        let pending = paths.update("pending.json").is_file();
        if let Some(stopped) = status.as_mut().filter(|s| !s.terminal() && !pending) {
            stopped.error = "machine_restarted: the machine stopped before the update activated".into();
            stopped.enter("failed");
            write_json(&paths.update("status.json"), stopped)?;
        }
        Ok(Arc::new(Self {
            paths,
            status: Mutex::new(status),
            idle,
            lifecycle,
            machine,
            stop: Mutex::default(),
            toolchain: Mutex::default(),
        }))
    }

    /// The build toolchain, probed once and again after each install.
    pub fn toolchain(&self) -> cozy_proto::machine::Toolchain {
        let mut held = self.toolchain.lock().unwrap();
        if held.is_none() {
            *held = Some(self.probe_toolchain());
        }
        let mut toolchain = held.clone().unwrap_or_default();
        if let Some(status) = self.status.lock().unwrap().as_ref().filter(|s| s.toolchain) {
            match status.state.as_str() {
                "failed" => toolchain.detail.clone_from(&status.error),
                state if !status.terminal() && state != "waiting" => toolchain.state = "installing".into(),
                _ => {}
            }
        }
        toolchain
    }

    fn probe_toolchain(&self) -> cozy_proto::machine::Toolchain {
        let environment = cozy_proto::config::environment();
        let path: Vec<PathBuf> = environment.get("PATH").map(|p| std::env::split_paths(p).collect()).unwrap_or_default();
        let cuda_home = environment.get("CUDA_HOME").map(PathBuf::from);
        super::toolchain::probe(&super::toolchain::dir(&self.paths.engine), cuda_home.as_deref(), &path)
    }

    /// Each kernel's verdict on this machine, as the Runtime keeps them under its kernel store.
    pub fn kernels(&self) -> Vec<cozy_proto::machine::KernelVerdict> {
        super::toolchain::verdicts(&self.paths.engine.join("gpu/kernels"))
    }

    /// Where the machine's toolchain lives, which executors find through their environment.
    pub fn toolchain_dir(&self) -> PathBuf {
        super::toolchain::dir(&self.paths.engine)
    }

    /// After this process proved readiness: an activation in flight is committed.
    pub fn commit(&self) -> io::Result<()> {
        let result = self.commit_pending();
        if let Err(error) = &result {
            if let Some(status) = self.status.lock().unwrap().as_mut().filter(|s| !s.terminal()) {
                // This is a live diagnostic, not an invented durable failure. The service can
                // remain healthy while observers learn why this activation cannot settle.
                status.error = format!("update_commit_blocked: {error}");
            }
        }
        result
    }

    fn commit_pending(&self) -> io::Result<()> {
        let Some(pending) = read_pending(&self.paths)? else {
            return Ok(());
        };
        let mut status = latest(&self.paths, pending.status)?;
        if !matches!(status.state.as_str(), "starting" | "succeeded") {
            return Err(invalid("cannot commit an update before software publication finished"));
        }
        status.to = this_binary(identified(&self.paths.sdk())?);
        status.error.clear();
        status.enter("succeeded");
        self.commit_status(status)
    }

    /// The durable success is authoritative even if removing the rollback record fails. A
    /// later startup finishes that cleanup; observers of this live service see success now.
    fn commit_status(&self, status: Status) -> io::Result<()> {
        // Keep the next request out until cleanup finishes; otherwise its new pending record
        // could be removed by this previous operation's cleanup.
        let mut current = self.status.lock().unwrap();
        write_json(&self.paths.update("status.json"), &status)?;
        *current = Some(status);
        remove_pending(&self.paths)
    }

    /// The update `operation` names, if it is this machine's latest.
    pub fn update(&self, operation: &str) -> Option<Status> {
        self.status
            .lock()
            .unwrap()
            .clone()
            .filter(|s| s.operation == operation)
    }

    /// Cancels update `operation` before it activates: it ends `canceled`, its candidate is
    /// removed and admission reopens. Once activating it ends on its own (ResourceBusy).
    pub fn cancel(&self, operation: &str) -> io::Result<Status> {
        let mut current = self.status.lock().unwrap();
        let status = current
            .as_mut()
            .filter(|s| s.operation == operation)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "this machine has no such update"))?;
        if status.terminal() {
            return Ok(status.clone());
        }
        if !matches!(status.state.as_str(), "waiting" | "preparing" | "waiting_activation") {
            return Err(io::Error::new(
                io::ErrorKind::ResourceBusy,
                "this update is activating and ends on its own",
            ));
        }
        let mut canceled = status.clone();
        canceled.error = "canceled by its owner".into();
        canceled.enter("canceled");
        write_json(&self.paths.update("status.json"), &canceled)?;
        *status = canceled.clone();
        self.stop.lock().unwrap().cancel();
        Ok(canceled)
    }

    /// The executors' Runtime/TensorFS pair.
    pub fn software(&self) -> Pair {
        pair_in(&self.paths.sdk())
    }

    /// PUT /v1/machine/runtime/wheels/{file}: kept as staged/<sha256>/<file>.
    pub fn stage(&self, file: &str, body: &mut dyn Read) -> Result<(String, u64), (u16, String)> {
        if self.busy() {
            return Err((
                409,
                "the machine is running a Runtime update; stage wheels after it finishes".into(),
            ));
        }
        if wheel_name(file).is_none() {
            return Err((400, "stage a cozy_runtime or tensorfs wheel".into()));
        }
        let temp = self.paths.update(&format!(
            "staged/.upload-{}",
            hex(&super::identity::random::<8>().map_err(server)?)
        ));
        let mut out = fs::File::create(&temp).map_err(server)?;
        let mut hash = Sha256::new();
        let (mut buffer, mut length) = (vec![0u8; 1 << 20], 0u64);
        loop {
            let n = body
                .read(&mut buffer)
                .map_err(|e| (400, format!("the wheel upload broke: {e}")))?;
            if n == 0 {
                break;
            }
            length += n as u64;
            if length > MAX_WHEEL_BYTES {
                let _ = fs::remove_file(&temp);
                return Err((400, "the wheel exceeds 256 MiB".into()));
            }
            hash.update(&buffer[..n]);
            out.write_all(&buffer[..n]).map_err(server)?;
        }
        out.sync_all().map_err(server)?;
        let sha = hex(&hash.finalize());
        let dir = self.paths.update(&format!("staged/{sha}"));
        fs::create_dir_all(&dir).map_err(server)?;
        fs::rename(&temp, dir.join(file)).map_err(server)?;
        Ok((sha, length))
    }

    fn busy(&self) -> bool {
        self.status
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|s| !s.terminal())
    }

    /// POST /v1/machine/runtime/update: accepts one operation and runs it in the background.
    pub fn request(
        self: &Arc<Self>,
        request: Request,
        exit: fn(i32),
    ) -> Result<Status, (u16, String)> {
        let valid = !request.operation.is_empty()
            && request.operation.len() <= 128
            && request
                .operation
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
            && (request.runtime.is_some() || request.tensorfs.is_some() || request.cozy.is_some()) != request.toolchain;
        if !valid {
            return Err((400, "an update requires operation and a Runtime, TensorFS or cozy, or the toolchain alone".into()));
        }
        let mut current = self.status.lock().unwrap();
        if let Some(status) = current.as_ref() {
            if status.operation == request.operation {
                return Ok(status.clone());
            }
            if !status.terminal() {
                return Err((
                    409,
                    format!(
                        "the machine is running another Runtime update: {}",
                        status.operation
                    ),
                ));
            }
        }
        // A committed terminal outcome may still have cleanup to do. Finish that before a
        // different operation overwrites its status and loses the pending decision's meaning.
        recover_activation(&self.paths).map_err(server)?;
        // An update is not a job: a released or activating machine refuses it, and preparing it
        // never holds the rental's idle release. Activation then closes admission.
        drop(self.lifecycle.as_ref().map(|l| l.admit()).transpose()
            .map_err(|refused| (503, refused.to_string()))?);
        let mut status = Status {
            operation: request.operation.clone(),
            toolchain: request.toolchain,
            from: this_binary(identified(&self.paths.sdk()).map_err(server)?),
            pinned: request.pin.unwrap_or(false),
            ..Status::default()
        };
        status.enter("waiting");
        write_json(&self.paths.update("status.json"), &status).map_err(server)?;
        *current = Some(status.clone());
        *self.stop.lock().unwrap() = Stop::default();
        drop(current);
        let updates = self.clone();
        std::thread::Builder::new()
            .name("runtime-update".into())
            .spawn(move || {
                if let Err(error) = updates.run(&request, exit) {
                    let failed = updates.set(|s| {
                        if !s.terminal() {
                            s.error = format!("{error}; it keeps Runtime {} and TensorFS {}", s.from.runtime, s.from.tensorfs);
                            s.code = error.get_ref().and_then(|e| e.downcast_ref::<Coded>()).map_or("", |c| c.code).into();
                            s.enter("failed");
                        }
                    });
                    // A canceled update's status already says so.
                    match failed {
                        Err(persist) if persist.kind() != io::ErrorKind::Interrupted => {
                            tracing::warn!("Runtime update failed: {error}; status: {persist}")
                        }
                        _ => {}
                    }
                }
            })
            .map_err(server)?;
        Ok(status)
    }

    /// Changes the running update's status; one a cancel ended stays ended (Interrupted).
    fn set(&self, change: impl FnOnce(&mut Status)) -> io::Result<()> {
        let mut current = self.status.lock().unwrap();
        if let Some(mut status) = current.clone() {
            if status.state == "canceled" {
                return Err(io::Error::new(io::ErrorKind::Interrupted, "the update was canceled"));
            }
            change(&mut status);
            write_json(&self.paths.update("status.json"), &status)?;
            *current = Some(status);
        }
        Ok(())
    }

    fn run(&self, request: &Request, exit: fn(i32)) -> io::Result<()> {
        let from = this_binary(identified(&self.paths.sdk())?);
        // A toolchain is installed beside the software, which keeps running: nothing restarts.
        if request.toolchain {
            self.set(|s| {
                s.to = from.clone();
                s.enter("installing");
            })?;
            let stop = self.stop.lock().unwrap().clone();
            let installed = super::toolchain::install(&self.toolchain_dir(), &self.paths.uv, &stop);
            *self.toolchain.lock().unwrap() = Some(self.probe_toolchain());
            installed?;
            return self.set(|s| s.enter("succeeded"));
        }
        // A named wheel file is compared by its bytes: a rebuilt wheel at the running version
        // is installed, the very wheel the machine runs is not.
        let running = |choice: &Option<Choice>, version: &str, sha256: &str| {
            choice.as_ref().is_none_or(|c| match c.file.is_empty() {
                true => c.version == version,
                false => c.sha256 == sha256,
            })
        };
        // A binary renamed over the launch path activates too, once it runs and is the release
        // the caller named, and its previous one is kept to roll back to.
        let binary = self.binary(request)?;
        let pair = !(running(&request.runtime, &from.runtime, &from.runtime_sha256)
            && running(&request.tensorfs, &from.tensorfs, &from.tensorfs_sha256));
        if !pair && binary.is_none() {
            // Nothing to install: the machine already runs this pair and binary.
            return self.set(|s| {
                s.to = from.clone();
                s.enter("succeeded");
            });
        }
        let candidate = match pair {
            true => {
                self.set(|s| s.enter("preparing"))?;
                let sdk = self.paths.engine.join("sdk");
                fs::create_dir_all(&sdk)?;
                // On the first update this also created engine/sdk; persist that parent entry.
                cozy_platform::durable::sync_dir(&self.paths.engine)?;
                // Caller operation IDs are receipt identities, never filesystem paths. Every
                // attempt prepares a fresh immutable generation, a historical operation's retry too.
                let candidate = sdk.join(format!("candidate-{}", hex(&super::identity::random::<16>()?)));
                fs::create_dir(&candidate)?;
                Some(candidate)
            }
            false => None,
        };
        let prepared = (|| -> io::Result<_> {
            let mut to = match &candidate {
                Some(candidate) => self.prepare(request, candidate)?,
                None => from.clone(),
            };
            to.cozy = binary.as_ref().map_or_else(|| from.cozy.clone(), |(_, version)| version.clone());
            self.set(|s| s.to = to.clone())?;
            let activation = self.drain()?;
            self.set(|s| s.enter("installing"))?;
            Ok((to, activation))
        })();
        if let (Err(_), Some(candidate)) = (&prepared, &candidate) {
            // Never selected: nothing refers to it.
            let _ = fs::remove_dir_all(candidate);
        }
        let (to, _activation) = prepared?;
        let pending = Pending {
            status: self.status.lock().unwrap().clone().unwrap_or_default(),
            sdk_before: previous_link(&self.paths.current_sdk_link())?,
            binary: binary.as_ref().map(|(launch, _)| launch.clone()),
        };
        let publication = (|| {
            write_json(&self.paths.update("pending.json"), &pending)?;
            if let Some(candidate) = &candidate {
                relink(&self.paths.current_sdk_link(), Some(candidate))?;
            }
            self.set(|s| s.enter("starting"))
        })();
        if let Err(error) = publication {
            // Admission is still closed. Do not leave a partial SDK/binary pair available to
            // newly accepted work, and do not overwrite a truthful rolled_back outcome.
            match rollback_pending(&self.paths, &format!("software publication failed: {error}")) {
                Ok(Recovered::Nothing) => {}
                Ok(_) => {
                    *self.status.lock().unwrap() = Some(latest(&self.paths, pending.status)?);
                }
                Err(rollback) => {
                    tracing::error!("update publication: {error}; rollback: {rollback}");
                    // The links cannot be trusted. Restart while admission remains closed;
                    // startup recovers before selecting software, or refuses this launch.
                    exit(REPLACE_EXIT);
                }
            }
            return Err(error);
        }
        tracing::info!(
            "update {}: restarting on cozy {}, Runtime {} / TensorFS {}",
            request.operation, to.cozy, to.runtime, to.tensorfs
        );
        exit(if binary.is_some() { super::binary::EXEC_EXIT } else { REPLACE_EXIT });
        Ok(())
    }

    /// The binary this update activates, with its release: the launch path's, when it is not this
    /// image. Refused before anything drains when it does not run, is not the release `cozy`
    /// names, or has no previous binary beside it to roll back to.
    fn binary(&self, request: &Request) -> io::Result<Option<(PathBuf, String)>> {
        let coded = |code, message| io::Error::other(Coded { code, message });
        let Some(launch) = super::binary::launch().filter(|l| l.swapped()) else {
            return match &request.cozy {
                Some(named) if named != VERSION => Err(coded(
                    "binary_mismatch",
                    format!("this machine runs cozy {VERSION} from its launch path, not {named}"),
                )),
                _ => Ok(None),
            };
        };
        let version = super::binary::version(&launch.path).map_err(|e| coded("binary_mismatch", e.to_string()))?;
        if let Some(named) = request.cozy.as_ref().filter(|named| **named != version) {
            return Err(coded("binary_mismatch", format!("{} is cozy {version}, not {named}", launch.path.display())));
        }
        let previous = super::binary::previous(&launch.path);
        if !previous.is_file() {
            return Err(coded(
                "binary_unrecoverable",
                format!("{} is missing: nothing to roll back to", previous.display()),
            ));
        }
        Ok(Some((launch.path.clone(), version)))
    }

    /// Closes admission and waits for the work admitted before to drain, saying how many runs
    /// it waits on as they end. A cancel ends it (Interrupted).
    fn drain(&self) -> io::Result<Option<super::lifecycle::Activation>> {
        // No new work is admitted from here; work admitted before drains first.
        let activation = self.lifecycle.as_ref().map(|l| l.activate()).transpose()
            .map_err(|refused| io::Error::other(refused.to_string()))?;
        let admitted = || self.lifecycle.as_ref().map_or(0, |l| l.admitted() as u32);
        let total = admitted();
        let mut seen = None;
        while !(self.idle)() {
            let left = admitted();
            if seen != Some(left) {
                seen = Some(left);
                self.set(|s| {
                    s.state = "waiting_activation".into();
                    s.waiting_for = left;
                    s.history.push(Step {
                        state: "waiting_activation".into(),
                        at_ms: super::lifecycle::now_ms(),
                        total,
                        completed: total.saturating_sub(left),
                    });
                })?;
            }
            // Waiting makes no progress of its own: the work it waits on is judged by its own.
            if self.status.lock().unwrap().as_ref().is_some_and(Status::terminal) {
                return Err(io::Error::new(io::ErrorKind::Interrupted, "the update was canceled"));
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        self.set(|s| s.waiting_for = 0)?;
        Ok(activation)
    }

    /// Stages and verifies the pair in `candidate`: what the update's `to` names.
    fn prepare(&self, request: &Request, candidate: &Path) -> io::Result<Pair> {
        let runtime = match &request.runtime {
            Some(choice) => self.resolve("cozy_runtime", choice)?,
            None => existing_wheel(&self.paths.sdk(), "cozy_runtime")?,
        };
        let runtime = stage(candidate, &runtime, "cozy_runtime")?;
        if let Some(file) = request.runtime.as_ref().filter(|c| !c.file.is_empty()) {
            fs::write(candidate.join(RUNTIME_BUILD), build(&runtime, &file.sha256)?)?;
        }
        match &request.tensorfs {
            None if request.runtime.is_some() => self.pair(candidate, &runtime)?,
            choice => {
                let tensorfs = match choice {
                    Some(choice) => self.resolve("tensorfs", choice)?,
                    None => existing_wheel(&self.paths.sdk(), "tensorfs")?,
                };
                let tensorfs = stage(candidate, &tensorfs, "tensorfs")?;
                if let Err(reason) = self.compile(&[runtime, tensorfs])? {
                    let names: Vec<String> = wheels(candidate).into_iter().map(|(d, v, _)| format!("{d} {v}")).collect();
                    return Err(invalid(&format!("{} do not install together: {reason}", names.join(" and "))));
                }
            }
        }
        cozy_platform::durable::sync_dir(candidate)?;
        identified(candidate)
    }

    /// A Runtime named alone keeps the machine's TensorFS when the two resolve together, else takes
    /// the TensorFS uv resolves for the Runtime over the index, fetched from PyPI held to its digest.
    /// One that resolves none names `--tensorfs-wheel`.
    fn pair(&self, candidate: &Path, runtime: &Path) -> io::Result<()> {
        if let Ok(current) = existing_wheel(&self.paths.sdk(), "tensorfs") {
            if self.compile(&[runtime.to_owned(), current.clone()])?.is_ok() {
                return stage(candidate, &current, "tensorfs").map(drop);
            }
        }
        let name = runtime.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let unresolved = |why: String| {
            io::Error::other(Coded {
                code: "tensorfs_unresolved",
                message: format!("{name} resolves no TensorFS over the index ({why}); name one with --tensorfs-wheel"),
            })
        };
        let resolved = self.compile(&[runtime.to_owned()])?.map_err(unresolved)?;
        let version = resolved
            .lines()
            .find_map(|line| line.trim().strip_prefix("tensorfs=="))
            .ok_or_else(|| unresolved("it requires none".into()))?;
        let choice = Choice { version: version.into(), ..Default::default() };
        stage(candidate, &self.resolve("tensorfs", &choice)?, "tensorfs").map(drop)
    }

    /// Resolves `wheels` as a package environment installs them (uv, over the index): the pinned
    /// requirements, or uv's reason they do not resolve. The machine never selects a pair its
    /// environments would set aside for a release's locked Runtime.
    fn compile(&self, wheels: &[PathBuf]) -> io::Result<Result<String, String>> {
        let input: String = wheels.iter().map(|p| p.display().to_string() + "\n").collect();
        super::client::ensure_uv(&self.paths.uv)?;
        let mut command = std::process::Command::new(&self.paths.uv);
        let mut child = cozy_platform::process::own_group(&mut command)
            .args(["pip", "compile", "--quiet", "--no-config", "--no-cache", "--no-header", "--no-annotate", "--python", "3.12", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        // uv's own exit says why, should it stop before reading the input.
        let _ = child.stdin.take().expect("piped stdin").write_all(input.as_bytes());
        let (stdout, stderr) = (child.stdout.take().expect("piped stdout"), child.stderr.take().expect("piped stderr"));
        let stop = self.stop.lock().unwrap().clone();
        let (ended, out, reason) = std::thread::scope(|scope| {
            let drain = |mut pipe: Box<dyn Read + Send>| {
                scope.spawn(move || {
                    let mut text = String::new();
                    let _ = pipe.read_to_string(&mut text);
                    text
                })
            };
            let (out, reason) = (drain(Box::new(stdout)), drain(Box::new(stderr)));
            let ended = crate::process::supervise(&mut child, Some(&stop), "uv", &|_| ());
            (ended, out.join().unwrap_or_default(), reason.join().unwrap_or_default())
        });
        match ended? {
            Supervised::Exited(status) if status.success() => Ok(Ok(out)),
            Supervised::Canceled => Err(io::Error::new(io::ErrorKind::Interrupted, "the update was canceled")),
            Supervised::Wedged(verdict) => Err(io::Error::other(format!("uv pip compile: {verdict}"))),
            Supervised::Exited(_) => {
                let reason: Vec<&str> = reason.split_whitespace().filter(|w| !matches!(*w, "×" | "╰─▶")).collect();
                Ok(Err(reason.join(" ").trim_end_matches('.').to_owned()))
            }
        }
    }

    /// A {file, sha256} fetched from its url or staged, or a published {version} fetched from
    /// PyPI, each with its digest checked.
    fn resolve(&self, distribution: &str, choice: &Choice) -> io::Result<PathBuf> {
        if !choice.url.is_empty() {
            let machine = self.machine.as_ref().ok_or_else(|| invalid("this machine fetches no wheels"))?;
            let stop = self.stop.lock().unwrap().clone();
            let dir = self.paths.update("downloads");
            return super::pypi::wheel_at(&choice.url, &choice.sha256, &choice.file, &dir, machine, &stop);
        }
        if !choice.file.is_empty() {
            let path = self
                .paths
                .update(&format!("staged/{}/{}", choice.sha256, choice.file));
            if choice.file.contains('/')
                || !path.is_file()
                || hex(&Sha256::digest(fs::read(&path)?)) != choice.sha256
            {
                return Err(invalid(&format!(
                    "{} was not staged with digest {}",
                    choice.file, choice.sha256
                )));
            }
            return Ok(path);
        }
        if choice.version.is_empty() {
            return Err(invalid("a pair member names a staged file or a version"));
        }
        let machine = self.machine.as_ref().ok_or_else(|| invalid("this machine fetches no published wheels"))?;
        let stop = self.stop.lock().unwrap().clone();
        super::pypi::wheel(distribution, &choice.version, &self.paths.update("downloads"), machine, &stop)
    }
}

/// Copies a verified `distribution` wheel into `candidate`, durably: where it is now. The copy is
/// written and flushed through its own handle, since a wheel from the store is read-only (a copy
/// keeps that) and Windows flushes only through a writable handle.
fn stage(candidate: &Path, wheel: &Path, distribution: &str) -> io::Result<PathBuf> {
    let name = wheel.file_name().ok_or_else(|| invalid("wheel name"))?;
    verify_wheel(wheel, distribution)?;
    let prepared = candidate.join(name);
    let mut staged = fs::File::create(&prepared)?;
    io::copy(&mut fs::File::open(wheel)?, &mut staged)?;
    staged.sync_all()?;
    Ok(prepared)
}

fn existing_wheel(dir: &Path, distribution: &str) -> io::Result<PathBuf> {
    wheels(dir)
        .into_iter()
        .find(|(d, _, path)| d == distribution && path.is_file())
        .map(|(_, _, path)| path)
        .ok_or_else(|| invalid(&format!("no installed {distribution} wheel")))
}

/// Checks the wheel's METADATA names the distribution and version its file name does. A
/// `tensord` a Runtime wheel still bundles is never this machine's: its binary moves only with the
/// `cozy` release.
fn verify_wheel(path: &Path, distribution: &str) -> io::Result<()> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let (wheel_distribution, version) =
        wheel_name(name).ok_or_else(|| invalid("not a Runtime or TensorFS wheel"))?;
    if wheel_distribution != distribution {
        return Err(invalid(&format!("{name} is not a {distribution} wheel")));
    }
    let mut archive = zip::ZipArchive::new(fs::File::open(path)?).map_err(io::Error::other)?;
    let metadata = format!("{distribution}-{version}.dist-info/METADATA");
    let mut text = String::new();
    archive
        .by_name(&metadata)
        .map_err(|_| invalid(&format!("{name} has no {metadata}")))?
        .read_to_string(&mut text)?;
    let field = |key: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(key))
            .map(str::trim)
            .unwrap_or_default()
            .to_owned()
    };
    if field("Name:").to_lowercase().replace(['-', '.'], "_") != distribution
        || field("Version:") != version
    {
        return Err(invalid(&format!(
            "{name} METADATA names another distribution or version"
        )));
    }
    Ok(())
}

/// A development Runtime wheel's version and build: the source commit it embeds, else its
/// bytes' digest.
fn build(path: &Path, sha256: &str) -> io::Result<String> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let (_, version) = wheel_name(name).ok_or_else(|| invalid("not a Runtime wheel"))?;
    let mut archive = zip::ZipArchive::new(fs::File::open(path)?).map_err(io::Error::other)?;
    let mut provenance = String::new();
    if let Ok(mut file) = archive.by_name("cozy_runtime/_build_provenance.py") {
        file.read_to_string(&mut provenance)?;
    }
    let commit = provenance.lines().find_map(|l| l.strip_prefix("COMMIT = ")).map(|c| c.trim().trim_matches('"'));
    let id = commit.filter(|c| c.len() >= 8 && c.chars().all(|c| c.is_ascii_hexdigit())).unwrap_or(sha256);
    Ok(format!("{version}+{}", &id[..8]))
}

/// `cozy_runtime-0.18.102-cp312-…whl` → ("cozy_runtime", "0.18.102").
fn wheel_name(file: &str) -> Option<(String, String)> {
    let stem = file.strip_suffix(".whl")?;
    let mut parts = stem.split('-');
    let (distribution, version) = (parts.next()?.to_lowercase(), parts.next()?.to_owned());
    matches!(distribution.as_str(), "cozy_runtime" | "tensorfs").then_some((distribution, version))
}

fn wheels(dir: &Path) -> Vec<(String, String, PathBuf)> {
    let mut found: Vec<_> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let (d, v) = wheel_name(&name)?;
            Some((d, v, e.path()))
        })
        .collect();
    found.sort();
    found
}

pub fn pair_in(dir: &Path) -> Pair {
    let mut pair = Pair::default();
    for (distribution, version, _) in wheels(dir) {
        match distribution.as_str() {
            "cozy_runtime" => pair.runtime = version,
            _ => pair.tensorfs = version,
        }
    }
    pair
}

/// The pair with each wheel's content digest: what an update compares and reports.
fn identified(dir: &Path) -> io::Result<Pair> {
    let mut pair = pair_in(dir);
    for (distribution, _, path) in wheels(dir) {
        let mut hash = Sha256::new();
        io::copy(&mut fs::File::open(path)?, &mut hash)?;
        match distribution.as_str() {
            "cozy_runtime" => pair.runtime_sha256 = hex(&hash.finalize()),
            _ => pair.tensorfs_sha256 = hex(&hash.finalize()),
        }
    }
    Ok(pair)
}

fn previous_link(link: &Path) -> io::Result<Option<PathBuf>> {
    match fs::read_link(link) {
        Ok(target) => Ok(Some(target)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// Points `link` at `target` atomically, or removes it. Windows renames nothing over a directory
/// link, so there the old link goes first; a crash between leaves no link, and the pending
/// activation's recovery links it again.
pub(crate) fn relink(link: &Path, target: Option<&Path>) -> io::Result<()> {
    let Some(target) = target else {
        return match unlink(link) {
            Ok(()) => Ok(cozy_platform::durable::sync_dir(link.parent().expect("link has a parent"))?),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        };
    };
    let dir = link
        .parent()
        .ok_or_else(|| invalid("a link needs a directory"))?;
    fs::create_dir_all(dir)?;
    // The first activation creates the link's directory. Its entry must survive with the link.
    cozy_platform::durable::sync_dir(dir.parent().ok_or_else(|| invalid("a link directory needs a parent"))?)?;
    let temp = dir.join(format!(".link-{}", hex(&super::identity::random::<8>()?)));
    cozy_platform::durable::symlink(target, &temp)?;
    #[cfg(windows)]
    match unlink(link) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
        _ => {}
    }
    fs::rename(&temp, link)?;
    Ok(cozy_platform::durable::sync_dir(dir)?)
}

/// Removes the link itself, never its target: a Windows directory link is a directory entry.
fn unlink(link: &Path) -> io::Result<()> {
    #[cfg(windows)]
    if std::os::windows::fs::FileTypeExt::is_symlink_dir(&fs::symlink_metadata(link)?.file_type()) {
        return fs::remove_dir(link);
    }
    fs::remove_file(link)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    super::identity::write_atomic(path, &serde_json::to_vec(value)?, 0o600)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn invalid(detail: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, detail.to_owned())
}

fn server(error: io::Error) -> (u16, String) {
    (500, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal wheel: METADATA, optionally a bundled script.
    fn wheel(dir: &Path, distribution: &str, version: &str, script: Option<&[u8]>) -> PathBuf {
        requiring(dir, distribution, version, script, "")
    }

    fn requiring(dir: &Path, distribution: &str, version: &str, script: Option<&[u8]>, requires: &str) -> PathBuf {
        let path = dir.join(format!("{distribution}-{version}-py3-none-any.whl"));
        let mut zip = zip::ZipWriter::new(fs::File::create(&path).unwrap());
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file(
            format!("{distribution}-{version}.dist-info/METADATA"),
            options,
        )
        .unwrap();
        zip.write_all(
            format!(
                "Metadata-Version: 2.1\nName: {}\nVersion: {version}\n{requires}",
                distribution.replace('_', "-")
            )
            .as_bytes(),
        )
        .unwrap();
        if let Some(script) = script {
            zip.start_file(
                format!("{distribution}-{version}.data/scripts/tensord"),
                options,
            )
            .unwrap();
            zip.write_all(script).unwrap();
        }
        zip.finish().unwrap();
        path
    }

    #[test]
    fn wheels_verify_by_metadata() {
        let dir = std::env::temp_dir().join(format!("cozy-update-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // A Runtime wheel that still bundles a tensord verifies; the bundled one is never run.
        let script = b"#!/bin/sh\necho '{\"name\":\"tensord\"}'\n";
        let runtime = wheel(&dir, "cozy_runtime", "0.18.103", Some(script));
        verify_wheel(&runtime, "cozy_runtime").unwrap();
        assert!(verify_wheel(&runtime, "tensorfs").is_err());
        let wrong = dir.join("tensorfs-0.3.94-py3-none-any.whl");
        fs::copy(wheel(&dir, "tensorfs", "0.3.93", None), &wrong).unwrap();
        assert!(
            verify_wheel(&wrong, "tensorfs").is_err(),
            "METADATA must agree with the file name"
        );
        assert_eq!(pair_in(&dir).runtime, "0.18.103");
        let built = dir.join("cozy_runtime-0.18.104-py3-none-any.whl");
        let mut zip = zip::ZipWriter::new(fs::File::create(&built).unwrap());
        zip.start_file("cozy_runtime/_build_provenance.py", zip::write::SimpleFileOptions::default()).unwrap();
        zip.write_all(b"COMMIT = \"e9baf7a4fe4b855009b2eb288c6dda336c1fe2f0\"\n").unwrap();
        zip.finish().unwrap();
        assert_eq!(build(&built, &"0".repeat(64)).unwrap(), "0.18.104+e9baf7a4", "a build names its commit");
        fs::remove_dir_all(dir).unwrap();
    }

    /// A wheel fetched from PyPI is the store's read-only blob; staging copies it all the same.
    #[test]
    fn a_read_only_wheel_stages() {
        let dir = std::env::temp_dir().join(format!("cozy-stage-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let candidate = dir.join("candidate");
        fs::create_dir_all(&candidate).unwrap();
        let fetched = wheel(&dir, "tensorfs", "0.11.0", None);
        let mut permissions = fs::metadata(&fetched).unwrap().permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&fetched, permissions).unwrap();
        let staged = stage(&candidate, &fetched, "tensorfs").unwrap();
        assert_eq!(fs::read(&staged).unwrap(), fs::read(&fetched).unwrap());
        let mut permissions = fs::metadata(&fetched).unwrap().permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        permissions.set_readonly(false);
        fs::set_permissions(&fetched, permissions).unwrap();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_failed_activation_rolls_back_links_and_reports_it() {
        let dir = std::env::temp_dir().join(format!("cozy-rollback-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let paths = Paths::new(&dir.join("engine"), &dir);
        let (old, new) = (dir.join("engine/sdk/old"), dir.join("engine/sdk/new"));
        fs::create_dir_all(&old).unwrap();
        fs::create_dir_all(&new).unwrap();
        relink(&paths.current_sdk_link(), Some(&old)).unwrap();
        let status = Status {
            operation: "op".into(),
            state: "starting".into(),
            ..Status::default()
        };
        write_json(
            &paths.update("pending.json"),
            &Pending {
                status,
                sdk_before: Some(old.clone()),
                binary: None,
            },
        )
        .unwrap();
        relink(&paths.current_sdk_link(), Some(&new)).unwrap();
        assert_eq!(rollback_pending(&paths, "candidate never became ready").unwrap(), Recovered::Restored);
        assert_eq!(fs::read_link(paths.current_sdk_link()).unwrap(), old);
        let status: Status =
            serde_json::from_slice(&fs::read(paths.update("status.json")).unwrap()).unwrap();
        assert_eq!(
            (status.state.as_str(), status.error.as_str()),
            ("rolled_back", "candidate never became ready")
        );
        assert_eq!(rollback_pending(&paths, "again").unwrap(), Recovered::Nothing);
        fs::remove_dir_all(dir).unwrap();
    }

    fn state(updates: &Updates, id: &str, until: &str) {
        loop {
            let status = updates.update(id).unwrap();
            if status.state == until {
                return;
            }
            assert!(!status.terminal(), "{}: {}", status.state, status.error);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// Activation admits nothing new while work admitted before it drains, so no run is
    /// accepted while the service exits; it reopens once the update is done.
    #[test]
    fn activation_closes_admission_while_admitted_work_drains() {
        let root = std::env::temp_dir().join(format!("cm-activation-{}", uuid::Uuid::new_v4()));
        let paths = Paths::new(&root.join("engine"), &root);
        fs::create_dir_all(&paths.image_wheels).unwrap();
        wheel(&paths.image_wheels, "cozy_runtime", "0.1.0", None);
        wheel(&paths.image_wheels, "tensorfs", "0.1.0", None);
        let lifecycle = super::super::lifecycle::Lifecycle::open(root.join("idle.json"), false, true, super::super::lifecycle::IDLE_GRACE_MS).unwrap();
        let earlier = lifecycle.admit().unwrap();
        let drained = lifecycle.clone();
        let idle = Box::new(move || drained.admitted() == 0);
        let updates = Updates::open(paths.clone(), idle, Some(lifecycle.clone()), None).unwrap();
        let candidate = wheel(&root, "cozy_runtime", "0.2.0", None);
        let file = candidate.file_name().unwrap().to_str().unwrap().to_owned();
        let (sha256, _) = updates.stage(&file, &mut fs::File::open(&candidate).unwrap()).unwrap();
        let runtime = Some(Choice { file, sha256, ..Default::default() });
        let request = Request { operation: "u1".into(), pin: None, runtime, tensorfs: None, cozy: None, toolchain: false };
        updates.request(request, |code| assert_eq!(code, REPLACE_EXIT)).unwrap();
        state(&updates, "u1", "waiting_activation");
        // It says how many runs it waits on: the run, and its event, count them.
        let waiting = updates.update("u1").unwrap();
        let step = waiting.history.last().unwrap();
        assert_eq!((waiting.waiting_for, step.total, step.completed), (1, 1, 0), "{waiting:?}");
        assert!(lifecycle.admit().is_err(), "a new run was admitted during activation");
        assert!(!paths.update("pending.json").is_file());
        drop(earlier);
        state(&updates, "u1", "starting");
        while lifecycle.admit().is_err() {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        fs::remove_dir_all(root).unwrap();
    }

    /// A cancel ends an update before it activates: it ends `canceled`, its candidate is
    /// removed, nothing is selected and admission reopens. Once activating, it is refused.
    #[test]
    fn a_canceled_update_ends_before_activation() {
        let root = std::env::temp_dir().join(format!("cm-canceled-update-{}", uuid::Uuid::new_v4()));
        let paths = Paths::new(&root.join("engine"), &root);
        fs::create_dir_all(&paths.image_wheels).unwrap();
        wheel(&paths.image_wheels, "cozy_runtime", "0.1.0", None);
        wheel(&paths.image_wheels, "tensorfs", "0.1.0", None);
        let lifecycle = super::super::lifecycle::Lifecycle::open(root.join("idle.json"), false, true, super::super::lifecycle::IDLE_GRACE_MS).unwrap();
        let earlier = lifecycle.admit().unwrap();
        let drained = lifecycle.clone();
        let updates = Updates::open(paths.clone(), Box::new(move || drained.admitted() == 0), Some(lifecycle.clone()), None).unwrap();
        let candidate = wheel(&root, "cozy_runtime", "0.2.0", None);
        let request = update_request(&updates, "u1", &candidate);
        updates.request(request, |_| panic!("a canceled update activated")).unwrap();
        state(&updates, "u1", "waiting_activation");
        assert_eq!(updates.cancel("u1").unwrap().state, "canceled");
        // Admission reopens and the candidate goes, with the earlier run still admitted.
        while lifecycle.admit().is_err() || fs::read_dir(root.join("engine/sdk")).unwrap().next().is_some() {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        drop(earlier);
        let persisted: Status = serde_json::from_slice(&fs::read(paths.update("status.json")).unwrap()).unwrap();
        assert_eq!(persisted.state, "canceled");
        assert!(fs::symlink_metadata(paths.current_sdk_link()).is_err(), "a canceled update selected software");
        assert_eq!(updates.cancel("u1").unwrap().state, "canceled", "a repeated cancel answers its end");
        assert_eq!(updates.cancel("other").unwrap_err().kind(), io::ErrorKind::NotFound);
        fs::remove_dir_all(root).unwrap();

        let activating = Activation::prepare();
        let updates = Updates::open(activating.paths.clone(), Box::new(|| true), None, None).unwrap();
        assert_eq!(updates.cancel("transaction").unwrap_err().kind(), io::ErrorKind::ResourceBusy);
        assert_eq!(updates.update("transaction").unwrap().state, "installing");
    }

    /// A pair the machine already runs installs nothing: the update succeeds unchanged at once.
    #[test]
    fn the_running_pair_is_an_unchanged_update() {
        let root = std::env::temp_dir().join(format!("cm-unchanged-update-{}", uuid::Uuid::new_v4()));
        let paths = Paths::new(&root.join("engine"), &root);
        fs::create_dir_all(&paths.image_wheels).unwrap();
        wheel(&paths.image_wheels, "cozy_runtime", "0.1.0", None);
        wheel(&paths.image_wheels, "tensorfs", "0.1.0", None);
        let updates = Updates::open(paths.clone(), Box::new(|| true), None, None).unwrap();
        let request = Request {
            operation: "same".into(), pin: None,
            runtime: Some(Choice { version: "0.1.0".into(), ..Default::default() }),
            tensorfs: Some(Choice { version: "0.1.0".into(), ..Default::default() }),
            cozy: None,
            toolchain: false,
        };
        updates.request(request, |_| panic!("an unchanged update restarted the service")).unwrap();
        loop {
            let status = updates.update("same").unwrap();
            if status.terminal() {
                assert_eq!((status.state.as_str(), &status.to), ("succeeded", &status.from));
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(!root.join("engine/sdk").exists());
        fs::remove_dir_all(root).unwrap();
    }

    /// An update the service stopped during before activation is failed, not left running.
    #[test]
    fn an_update_stopped_before_activation_fails() {
        let root = std::env::temp_dir().join(format!("cm-stopped-update-{}", uuid::Uuid::new_v4()));
        let paths = Paths::new(&root.join("engine"), &root);
        let mut status = Status { operation: "u1".into(), ..Default::default() };
        status.enter("preparing");
        fs::create_dir_all(paths.update("")).unwrap();
        write_json(&paths.update("status.json"), &status).unwrap();
        let updates = Updates::open(paths, Box::new(|| true), None, None).unwrap();
        let stopped = updates.update("u1").unwrap();
        assert_eq!(stopped.state, "failed");
        assert!(stopped.error.starts_with("machine_restarted"));
        fs::remove_dir_all(root).unwrap();
    }

    struct Activation {
        root: PathBuf,
        paths: Paths,
        old: PathBuf,
        new: PathBuf,
    }
    impl Activation {
        fn prepare() -> Self {
            let root = std::env::temp_dir().join(format!("cm-publication-{}", uuid::Uuid::new_v4()));
            let paths = Paths::new(&root.join("engine"), &root);
            let old = root.join("engine/sdk/old");
            let new = root.join("engine/sdk/new");
            for (dir, version) in [(&old, "0.1.0"), (&new, "0.2.0")] {
                fs::create_dir_all(dir).unwrap();
                wheel(dir, "cozy_runtime", version, None);
                wheel(dir, "tensorfs", version, None);
            }
            relink(&paths.current_sdk_link(), Some(&old)).unwrap();
            let mut status = Status { operation: "transaction".into(), ..Default::default() };
            status.enter("installing");
            write_json(&paths.update("status.json"), &status).unwrap();
            write_json(&paths.update("pending.json"), &Pending {
                status,
                sdk_before: Some(old.clone()),
                binary: None,
            }).unwrap();
            Self { root, paths, old, new }
        }
        fn state(&self, state: &str) {
            let mut status: Status = serde_json::from_slice(&fs::read(self.paths.update("status.json")).unwrap()).unwrap();
            status.enter(state);
            write_json(&self.paths.update("status.json"), &status).unwrap();
        }
        fn links(&self, dir: &Path) {
            assert_eq!(fs::read_link(self.paths.current_sdk_link()).unwrap(), dir);
        }
    }
    impl Drop for Activation {
        fn drop(&mut self) { let _ = fs::remove_dir_all(&self.root); }
    }

    #[test]
    fn every_unpublished_activation_boundary_rolls_back_before_service_selection() {
        // Process death after pending, or after the SDK link but before the durable starting
        // marker: neither may be reported as successfully activated.
        for changed_link in [false, true] {
            let transaction = Activation::prepare();
            if changed_link {
                relink(&transaction.paths.current_sdk_link(), Some(&transaction.new)).unwrap();
            }
            let updates = Updates::open(transaction.paths.clone(), Box::new(|| true), None, None).unwrap();
            assert!(updates.commit().is_err(), "unpublished state committed");
            assert_eq!(recover_activation(&transaction.paths).unwrap(), Recovered::Restored);
            transaction.links(&transaction.old);
            assert!(!transaction.paths.update("pending.json").exists());
            let recovered = Updates::open(transaction.paths.clone(), Box::new(|| true), None, None).unwrap();
            assert_eq!(recovered.update("transaction").unwrap().state, "rolled_back");
            assert_eq!(recover_activation(&transaction.paths).unwrap(), Recovered::Nothing);
        }
    }

    #[test]
    fn published_activation_waits_for_readiness_and_committed_cleanup_keeps_it() {
        let transaction = Activation::prepare();
        relink(&transaction.paths.current_sdk_link(), Some(&transaction.new)).unwrap();
        transaction.state("starting");
        assert_eq!(recover_activation(&transaction.paths).unwrap(), Recovered::Nothing);
        assert!(transaction.paths.update("pending.json").exists());
        transaction.links(&transaction.new);
        let updates = Updates::open(transaction.paths.clone(), Box::new(|| true), None, None).unwrap();
        // Keep the old pending bytes to simulate death after writing succeeded, before removal.
        let pending = fs::read(transaction.paths.update("pending.json")).unwrap();
        updates.commit().unwrap();
        assert_eq!(updates.update("transaction").unwrap().state, "succeeded");
        fs::write(transaction.paths.update("pending.json"), pending).unwrap();
        assert_eq!(recover_activation(&transaction.paths).unwrap(), Recovered::Nothing);
        transaction.links(&transaction.new);
        assert!(!transaction.paths.update("pending.json").exists());
    }

    #[test]
    fn rolled_back_cleanup_can_resume_without_changing_the_outcome() {
        let transaction = Activation::prepare();
        let pending = fs::read(transaction.paths.update("pending.json")).unwrap();
        rollback_pending(&transaction.paths, "candidate could not start").unwrap();
        let settled = fs::read(transaction.paths.update("status.json")).unwrap();
        fs::write(transaction.paths.update("pending.json"), pending).unwrap();
        assert_eq!(recover_activation(&transaction.paths).unwrap(), Recovered::Restored);
        transaction.links(&transaction.old);
        let expected: Status = serde_json::from_slice(&settled).unwrap();
        let actual: Status = serde_json::from_slice(&fs::read(transaction.paths.update("status.json")).unwrap()).unwrap();
        assert_eq!(actual.state, expected.state);
        assert_eq!(actual.history, expected.history);
        assert_eq!(actual.error, expected.error);
    }

    #[test]
    fn interruption_during_rollback_resumes_the_same_decision() {
        for restored_links in 0..=1 {
            let transaction = Activation::prepare();
            relink(&transaction.paths.current_sdk_link(), Some(&transaction.new)).unwrap();
                transaction.state("starting");
            transaction.state("rolled_back");
            if restored_links == 1 {
                relink(&transaction.paths.current_sdk_link(), Some(&transaction.old)).unwrap();
            }
            assert_eq!(recover_activation(&transaction.paths).unwrap(), Recovered::Restored);
            transaction.links(&transaction.old);
            assert!(!transaction.paths.update("pending.json").exists());
        }
    }

    #[test]
    fn unreadable_transaction_files_do_not_mean_no_pending_update() {
        let transaction = Activation::prepare();
        fs::write(transaction.paths.update("status.json"), b"truncated").unwrap();
        assert!(recover_activation(&transaction.paths).is_err());
        transaction.links(&transaction.old);
        fs::remove_file(transaction.paths.update("pending.json")).unwrap();
        fs::create_dir(transaction.paths.update("pending.json")).unwrap();
        assert!(recover_activation(&transaction.paths).is_err());
        assert!(rollback_pending(&transaction.paths, "test").is_err());
    }

    #[test]
    fn committed_success_is_visible_even_when_pending_cleanup_fails() {
        let transaction = Activation::prepare();
        transaction.state("starting");
        let updates = Updates::open(transaction.paths.clone(), Box::new(|| true), None, None).unwrap();
        let mut succeeded = updates.update("transaction").unwrap();
        succeeded.enter("succeeded");
        // Fail the actual cleanup operation after the commit decision has already been read.
        fs::remove_file(transaction.paths.update("pending.json")).unwrap();
        fs::create_dir(transaction.paths.update("pending.json")).unwrap();
        assert!(updates.commit_status(succeeded).is_err());
        assert_eq!(updates.update("transaction").unwrap().state, "succeeded");
        let persisted: Status = serde_json::from_slice(&fs::read(transaction.paths.update("status.json")).unwrap()).unwrap();
        assert_eq!(persisted.state, "succeeded");
        let next = Request {
            operation: "next".into(), pin: None,
            runtime: Some(Choice { version: "0.2.0".into(), ..Default::default() }),
            tensorfs: None,
            cozy: None,
            toolchain: false,
        };
        let rejected = updates.request(next, |_| panic!("unsettled cleanup started an update")).unwrap_err();
        assert_eq!(rejected.0, 500);
        assert_eq!(updates.update("transaction").unwrap().state, "succeeded");
    }

    #[test]
    fn failed_commit_is_observable_without_a_fabricated_terminal_outcome() {
        let transaction = Activation::prepare();
        transaction.state("starting");
        let updates = Updates::open(transaction.paths.clone(), Box::new(|| true), None, None).unwrap();
        let before = fs::read(transaction.paths.update("status.json")).unwrap();
        fs::remove_file(transaction.paths.update("status.json")).unwrap();
        fs::create_dir(transaction.paths.update("status.json")).unwrap();
        assert!(updates.commit().is_err());
        let blocked = updates.update("transaction").unwrap();
        assert_eq!(blocked.state, "starting");
        assert!(!blocked.terminal());
        assert!(blocked.error.starts_with("update_commit_blocked:"));
        assert!(transaction.paths.update("pending.json").is_file());
        fs::remove_dir(transaction.paths.update("status.json")).unwrap();
        fs::write(transaction.paths.update("status.json"), before).unwrap();
        updates.commit().unwrap();
        let committed = updates.update("transaction").unwrap();
        assert_eq!(committed.state, "succeeded");
        assert!(committed.error.is_empty());
    }

    fn update_request(updates: &Arc<Updates>, id: &str, wheel: &Path) -> Request {
        let file = wheel.file_name().unwrap().to_str().unwrap().to_owned();
        let (sha256, _) = updates.stage(&file, &mut fs::File::open(wheel).unwrap()).unwrap();
        Request {
            operation: id.into(), pin: None,
            runtime: Some(Choice { file, sha256, ..Default::default() }), tensorfs: None, cozy: None, toolchain: false,
        }
    }

    #[test]
    fn operation_ids_never_name_or_delete_software_directories() {
        let transaction = Activation::prepare();
        rollback_pending(&transaction.paths, "fixture").unwrap();
        let updates = Updates::open(transaction.paths.clone(), Box::new(|| true), None, None).unwrap();
        let original = fs::read_dir(&transaction.old).unwrap().count();
        for (id, version) in [(".", "0.3.0"), ("..", "0.3.1")] {
            let candidate = wheel(&transaction.root, "cozy_runtime", version, None);
            let request = update_request(&updates, id, &candidate);
            updates.request(request, |code| assert_eq!(code, REPLACE_EXIT)).unwrap();
            state(&updates, id, "starting");
            updates.commit().unwrap();
            assert_eq!(fs::read_dir(&transaction.old).unwrap().count(), original);
            assert!(transaction.paths.update("status.json").is_file());
            let selected = fs::read_link(transaction.paths.current_sdk_link()).unwrap();
            assert!(selected.file_name().unwrap().to_str().unwrap().starts_with("candidate-"));
        }
    }

    #[test]
    fn retrying_a_historical_operation_preserves_its_selected_generation() {
        let transaction = Activation::prepare();
        rollback_pending(&transaction.paths, "fixture").unwrap();
        let updates = Updates::open(transaction.paths.clone(), Box::new(|| true), None, None).unwrap();
        let candidate = wheel(&transaction.root, "cozy_runtime", "0.3.0", None);
        let request = update_request(&updates, "A", &candidate);
        updates.request(request, |code| assert_eq!(code, REPLACE_EXIT)).unwrap();
        state(&updates, "A", "starting");
        updates.commit().unwrap();
        let selected = fs::read_link(transaction.paths.current_sdk_link()).unwrap();
        let selected_wheel = existing_wheel(&selected, "cozy_runtime").unwrap();
        let bytes = fs::read(&selected_wheel).unwrap();
        // A subsequent failure becomes the latest operation but must not make A's live
        // generation available for destructive replacement when the caller retries A.
        let wrong = wheel(&transaction.root, "tensorfs", "0.4.0", None);
        let request = update_request(&updates, "B", &wrong);
        updates.request(request, |_| panic!("invalid candidate activated")).unwrap();
        state(&updates, "B", "failed");
        assert_eq!(fs::read_link(transaction.paths.current_sdk_link()).unwrap(), selected);
        // Retrying A names the wheel the machine runs: unchanged, its generation kept.
        let request = update_request(&updates, "A", &candidate);
        updates.request(request, |_| panic!("the running wheel restarted the service")).unwrap();
        state(&updates, "A", "succeeded");
        assert_eq!(fs::read(&selected_wheel).unwrap(), bytes);
        assert_eq!(fs::read_link(transaction.paths.current_sdk_link()).unwrap(), selected);
    }

    /// A named wheel at the running version installs when its bytes differ; the very wheel the
    /// machine runs installs nothing. The result's from/to say which, by content.
    #[test]
    fn a_named_wheel_is_compared_by_its_bytes_not_its_version() {
        let transaction = Activation::prepare();
        rollback_pending(&transaction.paths, "fixture").unwrap();
        let updates = Updates::open(transaction.paths.clone(), Box::new(|| true), None, None).unwrap();
        let rebuilt = transaction.root.join("rebuilt");
        fs::create_dir_all(&rebuilt).unwrap();
        let candidate = wheel(&rebuilt, "cozy_runtime", "0.1.0", Some(b"rebuilt"));
        let request = update_request(&updates, "rebuilt", &candidate);
        updates.request(request, |code| assert_eq!(code, REPLACE_EXIT)).unwrap();
        state(&updates, "rebuilt", "starting");
        updates.commit().unwrap();
        let installed = updates.update("rebuilt").unwrap();
        assert_eq!((installed.from.runtime.as_str(), installed.to.runtime.as_str()), ("0.1.0", "0.1.0"));
        assert_ne!(installed.from, installed.to, "a rebuilt wheel at the running version reads unchanged");
        assert_eq!(installed.to.runtime_sha256, hex(&Sha256::digest(fs::read(&candidate).unwrap())));
        // A named wheel is a development build its runs name: by its bytes, without provenance.
        let build = fs::read_to_string(transaction.paths.sdk().join(RUNTIME_BUILD)).unwrap();
        assert_eq!(build, format!("0.1.0+{}", &installed.to.runtime_sha256[..8]));
        assert_eq!(installed.from.tensorfs_sha256, installed.to.tensorfs_sha256);
        let request = update_request(&updates, "again", &candidate);
        updates.request(request, |_| panic!("the running wheel restarted the service")).unwrap();
        state(&updates, "again", "succeeded");
        let again = updates.update("again").unwrap();
        assert_eq!(again.from, again.to);
        assert_eq!(again.to, installed.to);
    }

    /// A pair that does not resolve fails its update with uv's reason and selects nothing.
    #[test]
    fn a_pair_that_does_not_resolve_fails_with_the_resolvers_reason() {
        let root = std::env::temp_dir().join(format!("cm-unresolved-update-{}", uuid::Uuid::new_v4()));
        let paths = Paths::new(&root.join("engine"), &root);
        fs::create_dir_all(&paths.image_wheels).unwrap();
        wheel(&paths.image_wheels, "cozy_runtime", "0.1.0", None);
        wheel(&paths.image_wheels, "tensorfs", "0.1.0", None);
        let updates = Updates::open(paths.clone(), Box::new(|| true), None, None).unwrap();
        let candidate = requiring(&root, "cozy_runtime", "0.2.0", None, "Requires-Dist: tensorfs>=0.2\n");
        let mut request = update_request(&updates, "u1", &candidate);
        let tensorfs = wheel(&root, "tensorfs", "0.1.0", None);
        let (sha256, _) = updates.stage("tensorfs-0.1.0-py3-none-any.whl", &mut fs::File::open(tensorfs).unwrap()).unwrap();
        request.tensorfs = Some(Choice { file: "tensorfs-0.1.0-py3-none-any.whl".into(), sha256, ..Default::default() });
        updates.request(request, |_| panic!("an unresolved pair activated")).unwrap();
        let failed = loop {
            match updates.update("u1").unwrap() {
                status if status.terminal() => break status,
                _ => std::thread::sleep(std::time::Duration::from_millis(10)),
            }
        };
        assert_eq!(failed.state, "failed");
        assert!(failed.error.starts_with("cozy_runtime 0.2.0 and tensorfs 0.1.0 do not install together: "), "{}", failed.error);
        assert!(failed.error.contains("tensorfs>=0.2"), "{}", failed.error);
        assert!(failed.error.ends_with("; it keeps Runtime 0.1.0 and TensorFS 0.1.0"), "{}", failed.error);
        assert!(fs::symlink_metadata(paths.current_sdk_link()).is_err(), "an unresolved pair was selected");
        fs::remove_dir_all(root).unwrap();
    }

    /// A Runtime named alone whose TensorFS neither the machine's nor any over the index meets
    /// fails naming `--tensorfs-wheel`, and selects nothing.
    #[test]
    fn a_runtime_alone_that_resolves_no_tensorfs_names_the_flag() {
        let root = std::env::temp_dir().join(format!("cm-unpaired-update-{}", uuid::Uuid::new_v4()));
        let paths = Paths::new(&root.join("engine"), &root);
        fs::create_dir_all(&paths.image_wheels).unwrap();
        wheel(&paths.image_wheels, "cozy_runtime", "0.1.0", None);
        wheel(&paths.image_wheels, "tensorfs", "0.1.0", None);
        let updates = Updates::open(paths.clone(), Box::new(|| true), None, None).unwrap();
        let candidate = requiring(&root, "cozy_runtime", "0.2.0", None, "Requires-Dist: tensorfs>=999\n");
        updates.request(update_request(&updates, "u1", &candidate), |_| panic!("an unpaired Runtime activated")).unwrap();
        let failed = loop {
            match updates.update("u1").unwrap() {
                status if status.terminal() => break status,
                _ => std::thread::sleep(std::time::Duration::from_millis(10)),
            }
        };
        assert_eq!((failed.state.as_str(), failed.code.as_str()), ("failed", "tensorfs_unresolved"), "{}", failed.error);
        assert!(failed.error.contains("name one with --tensorfs-wheel"), "{}", failed.error);
        assert!(failed.error.ends_with("; it keeps Runtime 0.1.0 and TensorFS 0.1.0"), "{}", failed.error);
        assert!(fs::symlink_metadata(paths.current_sdk_link()).is_err(), "an unpaired Runtime was selected");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_failed_status_write_does_not_advance_the_in_memory_state() {
        let transaction = Activation::prepare();
        let updates = Updates::open(transaction.paths.clone(), Box::new(|| true), None, None).unwrap();
        fs::remove_file(transaction.paths.update("status.json")).unwrap();
        fs::create_dir(transaction.paths.update("status.json")).unwrap();
        assert!(updates.set(|s| s.enter("starting")).is_err());
        assert_eq!(updates.update("transaction").unwrap().state, "installing");
    }

}
