//! Existing Runtime device-executor control seam. Models and codecs remain in Python SDK.
use platform::process::{Birth, Exact};
use crate::{
    execution::open_artifact,
    launch_identity::{LaunchIdentity, Seal},
    process::{reap_group, tail, Liveness, Meter, Watch, Watching},
    seam,
};
pub use protocol::executor::{
    Activate, Adapter, Answer, Bearer, Binding, ByteGrant, CallInterface, CallProgress, CheckpointItem,
    Command, Construction, Event, Fork, Frame, Hello, HostFrame, InputFile, InputMetadata, Invoke,
    Kind, Level, Load, LoadFacts, Map, MemoSettings, Metrics, ModelLoad, ModelPrefetch, Outcome,
    Output, PrepareRequest, Probe, Publish, PublishPart, Request, Residency, ResultRef,
    StageMemoStore, Start, Unload, RESULT_DOCUMENT,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use platform::{
    durable::Mode as _,
    ipc::{LocalListener, LocalStream},
};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ExitStatus, Stdio},
    sync::{Arc, Mutex},
    time::Duration,
};
#[cfg(unix)]
use std::os::{fd::AsRawFd, unix::process::ExitStatusExt};

#[derive(Clone, Debug)]
pub struct ExecutorConfig {
    pub python: PathBuf,
    pub root: PathBuf,
    pub socket: PathBuf,
    /// Explicitly configured locations. Credentials never enter this child.
    pub environment: BTreeMap<String, String>,
    pub seal: Seal,
    pub generation_hold: Option<Arc<File>>,
    pub identity: Option<LaunchIdentity>,
    /// Give the executor its own scope in this namespace: a cgroup when the host delegates
    /// one, else a token its descendants inherit.
    pub cgroup_namespace: Option<String>,
}

/// The lease protocol (design varena-leases), where varena serves one.
#[cfg(unix)]
const LEASES: Option<u32> = Some(varena::lease::PROTOCOL);
#[cfg(not(unix))]
const LEASES: Option<u32> = None;

/// How an executor's GPU memory is served, by the memory protocol its Hello speaks. An executor
/// is never refused for it: skew degrades to standalone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Contract {
    /// Chunks leased over each GPU's listener.
    Leases,
    /// Any other protocol: its own pages, a fixed, never-asked charge on its GPU.
    Standalone,
}

/// How an executor's memory is served, by the protocol its Hello speaks.
pub fn contract(hello: &Hello) -> Contract {
    match Some(hello.memory_protocol) == LEASES {
        true => Contract::Leases,
        false => Contract::Standalone,
    }
}

/// Refuses a Start naming leases to an executor that does not speak them.
fn accepts(hello: &Hello, command: &Command) -> io::Result<()> {
    if matches!(command, Command::Start(Start { leases: Some(_), .. })) && contract(hello) != Contract::Leases {
        return Err(io::Error::new(io::ErrorKind::Unsupported, format!(
            "Runtime {} speaks memory protocol {}: it reads no leases",
            hello.runtime_version, hello.memory_protocol,
        )));
    }
    Ok(())
}

/// A refusal of one executor request, its detail bounded.
pub fn refused(seq: u64, code: &str, detail: impl Into<String>) -> Answer {
    Answer::refused(seq, code, crate::triage::ends(&detail.into(), 1024))
}


/// A Start's `memory_protocol`: the lease protocol where it names leases this machine serves,
/// else 0.
pub fn memory_protocol(leases: Option<u32>) -> u32 {
    leases.and(LEASES).unwrap_or(0)
}

/// The run a command executes and the spool its executor reads and writes.
fn spool(command: &Command) -> Option<(&str, &Path)> {
    match command {
        Command::Invoke(invoke) => Some((invoke.request_id.as_str(), invoke.spool.as_path())),
        _ => None,
    }
}



pub trait Services {
    fn progress(&mut self, _: &Frame) {}
    fn request(
        &mut self,
        frame: &Frame,
        descriptor: Option<File>,
    ) -> io::Result<(Answer, Option<File>)> {
        drop(descriptor);
        Ok((Answer::unavailable(frame.seq), None))
    }
}

pub struct Baseline;
impl Services for Baseline {}

/// The executor's seam for frames sent outside any command's exchange, under the lock every
/// command and answer is written with. Gone with the executor.
#[derive(Clone)]
pub struct Line(std::sync::Weak<Mutex<LocalStream>>);
impl Line {
    /// Sends `command`, answered by nothing; false once the executor is gone.
    pub fn send(&self, command: &Command) -> bool {
        let Some(writer) = self.0.upgrade() else {
            return false;
        };
        let bytes = command_bytes(command);
        bytes.and_then(|bytes| seam::write_body(&mut writer.lock().unwrap(), &bytes)).is_ok()
    }
    pub fn same(&self, other: &Line) -> bool {
        self.0.ptr_eq(&other.0)
    }
}

/// `command`'s wire bytes. Only the typed TensorFS options may carry supplied access: the bytes
/// keep the token, every other field is scanned.
fn command_bytes(command: &Command) -> io::Result<Vec<u8>> {
    let mut value = serde_json::to_value(command)?;
    let bytes = serde_json::to_vec(&value)?;
    if let (true, Some(options)) = (
        match command {
            Command::Load(load) => load.tensorfs.is_some(),
            Command::Invoke(invoke) => invoke.tensorfs.is_some(),
            Command::Bearer(_) => true,
            _ => false,
        },
        value.get_mut("tensorfs").and_then(Value::as_object_mut),
    ) {
        options.remove("token");
    }
    scan(&value)?;
    Ok(bytes)
}

pub struct DeviceExecutor {
    child: Option<Child>,
    /// Forked from an import-only executor (`fork/1`), which is its parent and reaps it.
    forked: bool,
    exact: Exact,
    /// Read here; written only through `writer`, which a `Line` shares.
    stream: LocalStream,
    writer: Arc<Mutex<LocalStream>>,
    pub birth: Birth,
    pub hello: Hello,
    /// Sampling resolution of the wedge rule. Tests shorten it; nothing else changes it.
    pub liveness: Liveness,
    /// Stillness while this answers true is not this executor's: it waits on runs watched
    /// by their own executors (a job's unfinished children).
    pub waits: Option<crate::process::Waits>,
    /// Its GPUs' work, which judges its invocations beside its CPU, bytes and frames
    /// (`Meter::Work`); None on a lane with no GPUs.
    pub activity: Option<crate::process::Activity>,
    /// What its invocations wait on (GPU memory asked of other processes), which a wedge's
    /// verdict names; None on a lane with no GPUs.
    pub place: Option<crate::process::Place>,
    root: PathBuf,
    socket: PathBuf,
    _generation_hold: Option<Arc<File>>,
    codec: Arc<Codec>,
    retained: Vec<Box<dyn Send>>,
    identity: Option<LaunchIdentity>,
    watched: Watched,
    /// Longest gap between frames this executor has shown during invocations.
    worst_gap: Duration,
    /// Where its stderr stood when the request it serves began.
    log_from: u64,
    memo: Option<Arc<crate::memo::Memo>>,
}

/// Cooperative first: the executor stops at its next safe point. The running invocation's
/// pace then decides whether it stopped and must be killed.
#[derive(Clone)]
pub struct Cancellation {
    root: PathBuf,
    identity: Option<LaunchIdentity>,
    watched: Watched,
}
impl Cancellation {
    pub fn cancel(&self, request_id: &str) -> io::Result<()> {
        let temporary = self
            .root
            .join(format!("executor.cancel.{}.pending", uuid::Uuid::new_v4()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(request_id.as_bytes())?;
        file.sync_all()?;
        if let Some(identity) = self.identity {
            identity.readable(&temporary)?;
        }
        fs::rename(temporary, self.root.join("executor.cancel"))?;
        platform::durable::sync_dir(&self.root)?;
        if let Some(watch) = self.watched.lock().unwrap().as_ref() {
            watch.canceled();
        }
        Ok(())
    }
}

type Watched = Arc<Mutex<Option<Arc<Watch>>>>;

pub enum Forked {
    Ready(Box<DeviceExecutor>),
    /// No process was made; the configuration is free for a spawn.
    Refused(Box<ExecutorConfig>, String),
    /// The parent's channel failed (it died): no process was made, and it forks no more.
    Lost(Box<ExecutorConfig>, String),
}

/// An exit status from its code.
fn exit_status(code: i32) -> ExitStatus {
    #[cfg(unix)]
    return ExitStatus::from_raw(code);
    #[cfg(windows)]
    return std::os::windows::process::ExitStatusExt::from_raw(code as u32);
}

/// A forked executor's wait status while it is its parent's zombie (`/proc` `exit_code`).
/// Windows forks nothing, so there is none.
#[cfg(windows)]
fn zombie_status(_: &Birth) -> Option<ExitStatus> {
    None
}

/// A forked executor's wait status while it is its parent's zombie (`/proc` `exit_code`).
#[cfg(unix)]
fn zombie_status(birth: &Birth) -> Option<ExitStatus> {
    let stat = fs::read_to_string(format!("/proc/{}/stat", birth.pid)).ok()?;
    let fields: Vec<&str> = stat.rsplit_once(')')?.1.split_whitespace().collect();
    let start: u64 = fields.get(19)?.parse().ok()?;
    let code: i32 = fields.get(49)?.parse().ok()?;
    (fields.first() == Some(&"Z") && start == birth.start_ticks).then(|| exit_status(code))
}

/// The executor's root and listening socket, and the sealed environment it starts with.
fn endpoint(config: &ExecutorConfig) -> io::Result<(LocalListener, BTreeMap<String, String>)> {
    fs::create_dir_all(&config.root)?;
    platform::durable::set_mode(&config.root, 0o700)?;
    if let Some(identity) = config.identity {
        identity.own(&config.root)?;
    }
    let listener = LocalListener::bind(&config.socket)?;
    // A Unix socket is a file whose mode and owner admit the executor; a Windows pipe's own
    // access list admits only this user.
    #[cfg(unix)]
    platform::durable::set_mode(&config.socket, 0o600)?;
    if let Some(identity) = config.identity {
        identity.socket(&config.socket)?;
    }
    Ok((listener, config.seal.environment(&config.environment)))
}

/// Where an executor dials the listener at `root/name`. A Unix socket path has a short limit
/// (`sun_path`: 104 bytes on macOS); past it, the socket lives in this user's private directory
/// under /tmp, named by the hash of the path it stands for. A Windows pipe takes any path.
pub fn socket_path(root: &Path, name: &str) -> io::Result<PathBuf> {
    let natural = root.join(name);
    #[cfg(all(unix, not(target_os = "linux")))]
    if natural.as_os_str().len() > 100 {
        use std::os::unix::fs::MetadataExt;
        let dir = PathBuf::from(format!("/tmp/cozy-{}", crate::launch_identity::euid()));
        platform::durable::private_dir(&dir)?;
        let metadata = fs::symlink_metadata(&dir)?;
        if !metadata.is_dir() || metadata.uid() != crate::launch_identity::euid() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "the short socket directory belongs to another user",
            ));
        }
        let digest = tensorfs::sha256::hex_digest(natural.as_os_str().as_encoded_bytes());
        return Ok(dir.join(&digest[..24]));
    }
    Ok(natural)
}

/// The executor's containment scope, when its config names one. No scopes on this OS (macOS,
/// Windows until its Job Object backend): none, and the executor's tree ends by its group.
fn scope(config: &ExecutorConfig) -> io::Result<Option<Arc<platform::contain::Scope>>> {
    let Some(namespace) = &config.cgroup_namespace else {
        return Ok(None);
    };
    match platform::contain::Scope::create(namespace) {
        Ok(scope) => Ok(Some(Arc::new(scope))),
        Err(error) if error.is_unsupported() => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn parent_of(pid: u32) -> io::Result<u32> {
    Ok(platform::process::parent_of(pid)?)
}

/// The executor's end of the channel is gone: a write broke or no reply came.
pub fn channel_lost(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset | io::ErrorKind::UnexpectedEof
    )
}

/// The executor exited before it connected: no authored code ran. Deterministic causes
/// (an SDK without this module, a broken environment) fail the run with this evidence.
#[derive(Debug)]
pub struct EndedBeforeStart {
    pub status: ExitStatus,
    pub stderr_tail: String,
}
impl std::fmt::Display for EndedBeforeStart {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "executor exited before connecting ({}): {}",
            self.status, self.stderr_tail
        )
    }
}
impl std::error::Error for EndedBeforeStart {}

/// How an executor ended: status, a kill's measurement, descendants killed after it.
pub type Ended = crate::process::Reaped;

#[derive(Clone)]
pub struct CodecConfig {
    pub python: PathBuf,
    pub environment: BTreeMap<String, String>,
    pub generation_hold: Option<Arc<File>>,
}

/// The selected SDK's encoder for codecs this machine does not encode itself (audio,
/// video): one helper process per executor, started at its first such output and ended
/// with the executor (or when the machine dies: it exits on stdin EOF).
pub struct Codec {
    config: CodecConfig,
    log: PathBuf,
    helper: Mutex<Option<CodecHelper>>,
}
struct CodecHelper {
    child: Child,
    input: std::process::ChildStdin,
    output: std::io::BufReader<std::process::ChildStdout>,
}
impl Drop for CodecHelper {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
impl Codec {
    pub fn python(&self) -> &Path {
        &self.config.python
    }
    pub fn new(config: CodecConfig, log: PathBuf) -> Self {
        Self {
            config,
            log,
            helper: Mutex::new(None),
        }
    }
    fn spawn(&self) -> io::Result<CodecHelper> {
        #[cfg(unix)]
        let hold = self
            .config
            .generation_hold
            .as_ref()
            .map(|hold| hold.as_raw_fd());
        let mut command = std::process::Command::new(&self.config.python);
        platform::process::inherit_nothing(&mut command);
        command
            .args(["-I", "-m", "cozy_runtime.cli.main", "frame-codec"])
            .env_clear()
            .envs(&self.config.environment)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .mode(0o600)
                    .open(&self.log)?,
            );
        // The retained generation hold stays open in the codec.
        #[cfg(unix)]
        platform::process::keep_open(&mut command, hold.into_iter().collect());
        let mut child = command.spawn()?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("codec stdin missing"))?;
        let output = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("codec stdout missing"))?;
        Ok(CodecHelper {
            child,
            input,
            output: std::io::BufReader::new(output),
        })
    }
    /// Encodes `frames` with the Runtime's `cozy-runtime frame-codec`, each into its owned spool
    /// name: the media type of each, by handle.
    fn encode(&self, spool: &Path, frames: &[HostFrame]) -> io::Result<BTreeMap<String, String>> {
        use std::io::BufRead;
        #[derive(Serialize)]
        struct Encode<'a> {
            #[serde(flatten)]
            frame: &'a HostFrame,
            name: String,
        }
        let frames = (frames.iter())
            .map(|frame| Ok(Encode { frame, name: spool_name(&frame.handle)? }))
            .collect::<io::Result<Vec<_>>>()?;
        let mut helper = self.helper.lock().unwrap();
        if helper.is_none() {
            *helper = Some(self.spawn()?);
        }
        let request = serde_json::json!({"spool": spool, "frames": frames});
        let mut line = serde_json::to_vec(&request)?;
        line.push(b'\n');
        let mut answer = String::new();
        let exchanged = (|| {
            let current = helper.as_mut().expect("spawned codec helper");
            current.input.write_all(&line)?;
            current.input.flush()?;
            current.output.read_line(&mut answer)
        })();
        match exchanged {
            Ok(n) if n > 0 => {}
            Ok(_) | Err(_) => {
                helper.take(); // the next output starts a fresh helper
                return Err(io::Error::other(format!(
                    "selected SDK encoder stopped; see {}",
                    self.log.display()
                )));
            }
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Answer {
            Done { media: BTreeMap<String, String> },
            Refused { error: String },
        }
        match serde_json::from_str::<Answer>(&answer).map_err(io::Error::other)? {
            Answer::Done { media } => Ok(media),
            Answer::Refused { error } => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("selected SDK encoder refused: {error}"),
            )),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AssetBinding {
    pub output_id: String,
    pub asset_ref: String,
    pub name: String,
    pub kind: String,
    pub media_type: String,
    pub length: u64,
}

/// Rows a run keeps, and the characters of each of their texts.
const MAX_ADJUSTMENTS: usize = 64;
const MAX_ADJUSTMENT_TEXT: usize = 512;

/// The invoke reply's adjustment rows. They are advisory: a malformed row is skipped with a
/// warning and never fails the run.
pub fn adjustments(id: &str, reply: &Frame) -> Vec<crate::journal::Adjustment> {
    let rows = match &reply.adjustments {
        Value::Null => return vec![],
        Value::Array(rows) => rows,
        other => {
            let other = crate::triage::ends(&other.to_string(), 200);
            tracing::warn!("run {id}: adjustments skipped: not a list: {other}");
            return vec![];
        }
    };
    if rows.len() > MAX_ADJUSTMENTS {
        tracing::warn!("run {id}: {} adjustment rows; the first {MAX_ADJUSTMENTS} are kept", rows.len());
    }
    let clip = |text: &mut String| *text = crate::triage::ends(text, MAX_ADJUSTMENT_TEXT);
    rows.iter()
        .take(MAX_ADJUSTMENTS)
        .filter_map(|row| match crate::journal::Adjustment::deserialize(row) {
            Ok(mut row) => {
                for text in [&mut row.field, &mut row.requested, &mut row.applied, &mut row.reason, &mut row.source] {
                    clip(text);
                }
                Some(row)
            }
            Err(error) => {
                let row = crate::triage::ends(&row.to_string(), 200);
                tracing::warn!("run {id}: adjustment row skipped ({error}): {row}");
                None
            }
        })
        .collect()
}

pub fn postprocess(
    codec: &Codec,
    spool: &Path,
    reply: &Frame,
) -> io::Result<(Value, Vec<AssetBinding>)> {
    let result = read_result(spool, reply)?;
    // A tree is held as its manifest and member files; the encoders bind every other output.
    // Custody (`Engine::managed_result`) reads each bound file once, hashing as it copies.
    let (trees, files): (Vec<_>, Vec<_>) =
        reply.outputs.iter().cloned().partition(|o| o.kind == "tree");
    let reply = &Frame {
        outputs: files,
        ..reply.clone()
    };
    // Every frame is encoded by the SDK's one encoder, the one an author's test runs.
    let media = match reply.frames.is_empty() {
        true => BTreeMap::new(),
        false => codec.encode(spool, &reply.frames)?,
    };
    let mut bindings = bind(spool, reply, &media)?;
    for tree in &trees {
        bindings.push(tree_binding(spool, tree)?);
    }
    Ok((result, bindings))
}
/// An output tree as the machine holds it: each regular file of its spool directory linked as
/// the spool's `sha256-<hex>`, and a manifest naming them by path (`gpu_service::TREE_MEDIA`),
/// which the output binds to. A tree forwarded from a child (`sha256:<manifest>`) is already
/// both.
fn tree_binding(spool: &Path, output: &Output) -> io::Result<AssetBinding> {
    let invalid = |why: &str| io::Error::new(io::ErrorKind::InvalidData, format!("output tree: {why}"));
    let hash = |path: &Path| -> io::Result<(String, u64)> {
        let mut file = open_artifact(spool, path)?;
        let mut hash = tensorfs::sha256::Sha256::new();
        let length = io::copy(&mut file, &mut HashWriter(&mut hash))?;
        Ok((tensorfs::sha256::hex(&hash.finish()), length))
    };
    let name = match output.asset_ref.strip_prefix("sha256:") {
        Some(hex) if hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()) => {
            format!("sha256-{hex}")
        }
        _ => {
            let tail: Vec<_> = output.asset_ref.rsplitn(3, '/').collect();
            let [n, "tree", _] = tail.as_slice() else {
                return Err(invalid("its reference names no spool directory"));
            };
            let directory = format!("tree-{n}");
            let mut files = std::collections::BTreeMap::new();
            let mut pending = vec![std::path::PathBuf::new()];
            while let Some(relative) = pending.pop() {
                for entry in fs::read_dir(spool.join(&directory).join(&relative))? {
                    let entry = entry?;
                    let (kind, path) = (entry.file_type()?, relative.join(entry.file_name()));
                    if kind.is_dir() {
                        pending.push(path);
                        continue;
                    } else if !kind.is_file() {
                        return Err(invalid("it holds something other than files and directories"));
                    }
                    let member = Path::new(&directory).join(&path);
                    let (sha, length) = hash(&member)?;
                    match fs::hard_link(spool.join(&member), spool.join(format!("sha256-{sha}"))) {
                        Err(error) if error.kind() != io::ErrorKind::AlreadyExists => return Err(error),
                        _ => (),
                    }
                    let path = path.to_str().ok_or_else(|| invalid("a path is not UTF-8"))?;
                    files.insert(path.to_owned(), tensorfs::format::ObjectRef { sha256: sha, length });
                }
            }
            if files.is_empty() {
                return Err(invalid("it holds no file"));
            }
            let manifest = crate::trees::document(&files);
            let name = format!("{directory}-manifest");
            fs::write(spool.join(&name), manifest)?;
            name
        }
    };
    let mut bytes = Vec::new();
    open_artifact(spool, Path::new(&name))?.read_to_end(&mut bytes)?;
    let files = crate::trees::parse(&bytes).map_err(|e| invalid(&e.to_string()))?;
    let mut content = 0;
    for object in files.values() {
        let member = format!("sha256-{}", object.sha256);
        if open_artifact(spool, Path::new(&member))?.metadata()?.len() != object.length {
            return Err(invalid("a member differs from its manifest"));
        }
        content += object.length;
    }
    if output.size_bytes.is_some_and(|size| size != content) {
        return Err(invalid("its files differ from the size the SDK reported"));
    }
    Ok(AssetBinding {
        output_id: output.output_id.clone(),
        asset_ref: output.asset_ref.clone(),
        name,
        kind: "tree".into(),
        media_type: crate::gpu_service::TREE_MEDIA.into(),
        length: bytes.len() as u64,
    })
}

struct HashWriter<'a>(&'a mut tensorfs::sha256::Sha256);
impl Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The spool file an output's reference names: a file a child returned, which this run hands
/// on as its own (`jobs::grant`), else `<kind>-<name>`.
fn spool_name(reference: &str) -> io::Result<String> {
    if let Some(hex) = (reference.strip_prefix("sha256:"))
        .filter(|h| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        return Ok(format!("sha256-{hex}"));
    }
    match reference.rsplitn(3, '/').collect::<Vec<_>>().as_slice() {
        [name, kind, _] if !name.is_empty() && !name.contains(['/', '\0']) && *name != "." && *name != ".." => {
            Ok(format!("{kind}-{name}"))
        }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "SDK output has no owned spool binding")),
    }
}

/// Binds every output to its spool file, its frames' media types from `media`.
fn bind(spool: &Path, reply: &Frame, media: &BTreeMap<String, String>) -> io::Result<Vec<AssetBinding>> {
    let invalid = |detail: &str| io::Error::new(io::ErrorKind::InvalidData, detail.to_string());
    let mut total = 0u64;
    let mut bindings = vec![];
    for output in &reply.outputs {
        let name = spool_name(&output.asset_ref)?;
        // Its bytes are read once, by custody; here only their length is bound.
        let length = open_artifact(spool, Path::new(&name))?.metadata()?.len();
        if output.size_bytes.is_some_and(|n| n != length) {
            return Err(invalid("SDK output bytes changed before custody"));
        }
        total += length;
        if reply.max_output_bytes.is_some_and(|limit| total > limit) {
            return Err(invalid(
                "encoded output exceeds the authored aggregate allowance",
            ));
        }
        bindings.push(AssetBinding {
            output_id: output.output_id.clone(),
            asset_ref: output.asset_ref.clone(),
            name,
            kind: output.kind.clone(),
            media_type: media.get(&output.asset_ref).unwrap_or(&output.media_type).clone(),
            length,
        });
    }
    Ok(bindings)
}

impl DeviceExecutor {
    pub fn spawn(config: ExecutorConfig) -> io::Result<Self> {
        Self::spawn_observed(config, |_, _| Ok(()))
    }

    pub fn spawn_observed(
        config: ExecutorConfig,
        on_birth: impl FnOnce(&Birth, &Cancellation) -> io::Result<()>,
    ) -> io::Result<Self> {
        Self::spawn_internal(config, None, on_birth)
    }
    pub fn spawn_owned(
        config: ExecutorConfig,
        launcher: &crate::child_launcher::ChildLauncher,
        on_birth: impl FnOnce(&Birth, &Cancellation) -> io::Result<()>,
    ) -> io::Result<Self> {
        Self::spawn_internal(config, Some(launcher), on_birth)
    }
    fn spawn_internal(
        config: ExecutorConfig,
        launcher: Option<&crate::child_launcher::ChildLauncher>,
        on_birth: impl FnOnce(&Birth, &Cancellation) -> io::Result<()>,
    ) -> io::Result<Self> {
        let (listener, environment) = endpoint(&config)?;
        let scope = scope(&config)?;
        let mut command =
            crate::launch_identity::trampoline(&config.python, config.identity, scope.as_deref())?;
        command
            .args(["-I", "-m", "cozy_runtime.internal.executor", "--socket"])
            .arg(&config.socket)
            .arg("--root")
            .arg(&config.root)
            .env_clear()
            .envs(&environment)
            .envs(scope.as_deref().and_then(platform::contain::Scope::environment))
            .envs(crate::launch_identity::interpreter_environment(&config.python)?)
            .stdin(Stdio::null())
            .stdout(File::create(config.root.join("stdout.log"))?)
            .stderr(File::create(config.root.join("stderr.log"))?);
        // Its generation and kernel namespace stay held while it lives, past this machine's death.
        crate::launch_identity::inherit(
            &mut command,
            &[
                config.generation_hold.as_ref(),
                config.seal.kernel_hold.as_ref(),
            ],
        );
        let child = match launcher {
            Some(launcher) => launcher.spawn(command)?,
            None => command.spawn()?,
        };
        // Windows has no trampoline to join the scope before anything runs: the executor joins
        // it at once, before its first child (the executor starts none until it is ready).
        #[cfg(windows)]
        let child = match scope.as_deref().map(|scope| scope.adopt(child.id())) {
            Some(Err(error)) => {
                let mut child = child;
                let _ = child.kill();
                let _ = child.wait();
                return Err(error.into());
            }
            _ => child,
        };
        let launched = Unready {
            pid: child.id(),
            parent: std::process::id(),
            child: Some(child),
            exact: None,
            birth: None,
            scope,
        };
        Self::connect(config, listener, environment, launched, on_birth)
    }

    /// An executor forked from this import-only one (`fork/1`), its imports done. It dials
    /// its own seam and is checked exactly as a spawned one is, with this process as its
    /// parent. Refused (no process exists) gives the configuration back for a spawn.
    pub fn fork(
        &mut self,
        config: ExecutorConfig,
        on_birth: impl FnOnce(&Birth, &Cancellation) -> io::Result<()>,
    ) -> io::Result<Forked> {
        let (listener, environment) = endpoint(&config)?;
        let scope = scope(&config)?;
        let mut forked = environment.clone();
        if let Some((name, token)) = scope.as_deref().and_then(platform::contain::Scope::environment) {
            forked.insert(name.into(), token.into());
        }
        let reply = self.command(
            &Command::Fork(Fork {
                socket: config.socket.clone(),
                root: config.root.clone(),
                environment: forked,
            }),
            &mut Baseline,
        );
        let pid = match reply {
            Ok(reply) if reply.ok && reply.hello.pid != 0 => reply.hello.pid,
            refused => {
                drop(listener);
                fs::remove_file(&config.socket)?;
                // A child the parent made before its channel failed carries the token too.
                if let Err(error) = scope.as_deref().map_or(Ok(0), platform::contain::Scope::end) {
                    tracing::warn!("refused fork's scope not ended: {error}");
                }
                return Ok(match refused {
                    Ok(reply) => Forked::Refused(
                        Box::new(config),
                        format!("{}: {}", reply.code, reply.detail),
                    ),
                    Err(error) => Forked::Lost(Box::new(config), error.to_string()),
                });
            }
        };
        // A fork starts in its parent's cgroup: move it into its own before it dials, so
        // ending it never touches the import-only parent. A token came with its environment.
        if let Some(scope) = &scope {
            scope.adopt(pid)?;
        }
        let launched = Unready {
            pid,
            parent: self.birth.pid,
            child: None,
            exact: None,
            birth: None,
            scope,
        };
        Self::connect(config, listener, environment, launched, on_birth)
            .map(|executor| Forked::Ready(Box::new(executor)))
    }

    /// Wait for the launched process to dial or end, prove the connection is that process,
    /// and take its Hello. Until then every early return kills and reaps it: nothing
    /// authored has run, so no measurement is needed to end it.
    fn connect(
        config: ExecutorConfig,
        listener: LocalListener,
        environment: BTreeMap<String, String>,
        mut unready: Unready,
        on_birth: impl FnOnce(&Birth, &Cancellation) -> io::Result<()>,
    ) -> io::Result<Self> {
        let codec = Arc::new(Codec::new(
            CodecConfig {
                python: config.python.clone(),
                environment,
                generation_hold: config.generation_hold.clone(),
            },
            config.root.join("codec.stderr.log"),
        ));
        let pid = unready.pid;
        let birth = Birth::of(pid)?;
        let exact = Exact::open(&birth)?
            .ok_or_else(|| io::Error::other("launched executor has no exact birth"))?
            .with_scope(unready.scope.clone());
        unready.exact = Some(exact.try_clone()?);
        unready.birth = Some(birth.clone());
        if unready.child.is_none() && parent_of(pid)? != unready.parent {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "forked executor is not a child of the process it was forked from",
            ));
        }
        let watched: Watched = Arc::default();
        on_birth(
            &birth,
            &Cancellation {
                root: config.root.clone(),
                identity: config.identity,
                watched: watched.clone(),
            },
        )?;
        // Wait for connection OR observed process death. No elapsed-time startup kill.
        let Some(stream) = listener.accept_unless(&exact)? else {
            let status = unready.ended()?;
            return Err(io::Error::other(EndedBeforeStart {
                status,
                stderr_tail: tail(&config.root.join("stderr.log")),
            }));
        };
        let credential = platform::handles::peer(&stream)?;
        let expected_uid = config
            .identity
            .map_or_else(crate::launch_identity::euid, |identity| identity.uid);
        let expected_gid = config
            .identity
            .map_or_else(crate::launch_identity::egid, |identity| identity.gid);
        if credential.pid != pid
            || credential.uid != expected_uid
            || credential.gid != expected_gid
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "device executor connection differs from launched process",
            ));
        }
        let parent = unready.parent;
        let mut executor = Self {
            child: None,
            forked: unready.child.is_none(),
            exact,
            writer: Arc::new(Mutex::new(stream.try_clone()?)),
            stream,
            birth,
            hello: Hello::default(),
            liveness: Liveness::default(),
            waits: None,
            activity: None,
            place: None,
            root: config.root,
            socket: config.socket,
            _generation_hold: config.generation_hold,
            codec,
            retained: Vec::new(),
            identity: config.identity,
            watched,
            worst_gap: Duration::ZERO,
            log_from: 0,
            memo: None,
        };
        let hello = executor.command(&Command::Hello, &mut Baseline)?;
        let mismatched = config.seal.mismatches(&hello.hello.sealed);
        if !hello.ok
            || hello.hello.pid != executor.birth.pid
            || hello.hello.ppid != parent
            || hello.hello.pgid != executor.birth.pid
            || !mismatched.is_empty()
        {
            return Err(io::Error::other(format!(
                "stock executor hello did not identify the launched, sealed process \
                 (pid {}, parent {}, group {}; differing sealed names {mismatched:?})",
                hello.hello.pid, hello.hello.ppid, hello.hello.pgid
            )));
        }
        if let Some(relative) = unready.scope.as_deref().and_then(|s| s.cgroup_relative()) {
            let member = fs::read_to_string(format!("/proc/{}/cgroup", executor.birth.pid))?;
            if !member.lines().any(|line| line == format!("0::{relative}")) {
                return Err(io::Error::other("executor is outside its own cgroup"));
            }
        }
        // Proven: the executor now owns the child; the launch guard no longer kills it.
        executor.child = unready.child.take();
        unready.exact = None;
        unready.pid = 0;
        executor.hello = hello.hello;
        Ok(executor)
    }
    pub fn codec(&self) -> Arc<Codec> {
        self.codec.clone()
    }
    /// The machine tier this executor's memoized stages look up and store in.
    pub fn keep_memo(&mut self, memo: Arc<crate::memo::Memo>) {
        self.memo = Some(memo);
    }
    pub fn root_path(&self) -> &Path {
        &self.root
    }
    /// A request begins: an executor serves many submitters, so a failure quotes only the
    /// stderr written from here on.
    pub fn begin_request(&mut self) {
        self.log_from = fs::metadata(self.root.join("stderr.log")).map_or(0, |m| m.len());
    }
    /// The end of the stderr of the request it serves (its whole life before the first), as
    /// much as its triage bundle keeps.
    pub fn stderr_tail(&self) -> String {
        crate::process::tail_from(&self.root.join("stderr.log"), self.log_from, crate::triage::MAX_STDERR)
    }
    pub fn line(&self) -> Line {
        Line(Arc::downgrade(&self.writer))
    }
    pub fn cancellation(&self) -> Cancellation {
        Cancellation {
            root: self.root.clone(),
            identity: self.identity,
            watched: self.watched.clone(),
        }
    }

    pub fn observer_pidfd(&self) -> io::Result<File> {
        Ok(self.exact.try_clone()?.into_file())
    }

    /// Retain a source/resource until this exact receiver exits, including owner-handle loss.
    /// Observer lifetimes must never own the DeviceExecutor handle.
    pub fn retain_until_exit(&mut self, resource: impl Send + 'static) {
        self.retained.push(Box::new(resource));
    }

    /// One command, observed: a wedged executor is killed on its measured lack of progress,
    /// which closes the channel and returns this call with the measurement.
    pub fn command(
        &mut self,
        command: &Command,
        services: &mut impl Services,
    ) -> io::Result<Frame> {
        // Refuse leases to an executor that does not speak them before writing the command.
        accepts(&self.hello, command)?;
        let meter = match command {
            Command::Invoke(_) => Meter::Work,
            _ => Meter::Burn,
        };
        let watching = Watching::start(
            self.exact.try_clone()?,
            meter,
            self.liveness,
            self.worst_gap,
            self.waits.clone(),
            self.activity.clone(),
            command.name(),
        )?;
        watching.watch().place(self.place.clone());
        *self.watched.lock().unwrap() = Some(watching.watch());
        let result = self.exchange(command, services, &watching.watch());
        *self.watched.lock().unwrap() = None;
        // Whatever the run was computing when its exchange ended, its waiters compute themselves.
        if let (Some(memo), Some((run, _))) = (&self.memo, spool(command)) {
            memo.release(run);
        }
        let (killed, worst_gap) = watching.finish();
        if meter == Meter::Work {
            self.worst_gap = self.worst_gap.max(worst_gap);
        }
        match killed {
            Some(verdict) => Err(io::Error::other(crate::process::Wedged(verdict))),
            None => result,
        }
    }

    fn exchange(
        &mut self,
        command: &Command,
        services: &mut impl Services,
        watch: &Watch,
    ) -> io::Result<Frame> {
        let bytes = command_bytes(command)?;
        seam::write_body(&mut self.writer.lock().unwrap(), &bytes)?;
        let mut last = Advance::default();
        loop {
            let frame = read_frame(&mut self.stream)?
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "stock executor EOF before reply")
                })?;
            if advances(&frame, &mut last) {
                watch.frame();
            }
            match frame.event {
                Some(Event::Request) if !frame.malformed.is_empty() => {
                    refuse_malformed(&mut self.writer.lock().unwrap(), &frame)?
                }
                Some(Event::Progress) if !frame.malformed.is_empty() => (),
                Some(Event::Progress) => {
                    if !frame.stage.is_empty() {
                        watch.at(&frame.stage);
                    }
                    services.progress(&frame)
                }
                Some(Event::Request)
                    if matches!(frame.request, Some(Request::StageMemoLookup(_) | Request::StageMemoStore(_))) =>
                {
                    // A lookup may wait on the run computing its key: not this executor's stillness.
                    watch.serving(true);
                    let answer = match (&self.memo, spool(command)) {
                        (Some(memo), Some((run, spool))) => memo.answer(run, spool, frame.seq, frame.request.as_ref().expect("a memo request")),
                        _ => Answer::unavailable(frame.seq),
                    };
                    watch.serving(false);
                    write_frame(&mut self.writer.lock().unwrap(), &answer)?;
                }
                Some(Event::Request) => {
                    // The machine's own answer time is not the executor's stillness.
                    watch.serving(true);
                    let answered = self.answer(&frame, services);
                    watch.serving(false);
                    answered?;
                }
                Some(Event::Unknown) => (),
                None if frame.reply == command.name() => return Ok(frame),
                // A Runtime without the `bearer` handler refuses it as a command.
                None if frame.reply == "bearer" => (),
                None => return Err(io::Error::other("out-of-order stock executor reply")),
            }
        }
    }

    fn answer(&mut self, frame: &Frame, services: &mut impl Services) -> io::Result<()> {
        let descriptor = match frame.descriptor {
            true => Some(File::from(platform::handles::recv_fd(&self.stream)?)),
            false => None,
        };
        let (mut answer, descriptor) = services.request(frame, descriptor)?;
        answer.event = "answer".into();
        answer.seq = frame.seq;
        answer.descriptor = descriptor.is_some();
        let mut writer = self.writer.lock().unwrap();
        write_frame(&mut writer, &answer)?;
        if let Some(descriptor) = descriptor {
            platform::handles::send_fd(&writer, &descriptor)?;
        }
        Ok(())
    }

    /// Its Start's `memo`: the tiers its stage memo serves, when this machine memoizes.
    pub fn memo_settings(&self) -> Option<MemoSettings> {
        self.memo.as_ref().map(|_| MemoSettings { process_bytes: -1, entry_bytes: crate::memo::ENTRY_BYTES })
    }

    /// Attempt-keyed cooperative cancellation; observer teardown never calls this.
    pub fn cancel(&self, request_id: &str) -> io::Result<()> {
        self.cancellation().cancel(request_id)
    }

    /// Ask the executor to exit, then observe that exact exit (killing only a wedge).
    pub fn shutdown(mut self) -> io::Result<()> {
        let asked = self.command(&Command::Shutdown, &mut Baseline);
        let ended = self.terminate()?;
        asked?;
        if !ended.status.success() {
            return Err(io::Error::other(format!(
                "stock executor shutdown: {}",
                ended.status
            )));
        }
        Ok(())
    }

    /// Close the channel, let the executor stop at its next exchange, kill it only on a
    /// measured wedge, then reap it and release what it held. Blocks until it is gone.
    pub fn terminate(mut self) -> io::Result<Ended> {
        let parts = self.parts();
        parts.end()
    }

    fn parts(&mut self) -> Ending {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
        Ending {
            exact: self.exact.try_clone().map_err(io::Error::from),
            child: self.child.take(),
            forked: std::mem::take(&mut self.forked),
            retained: std::mem::take(&mut self.retained),
            socket: std::mem::take(&mut self.socket),
            liveness: self.liveness,
        }
    }
}

/// A launched process before its Hello proves it: dropped, it is killed (and reaped by its
/// parent: this machine, or the executor it was forked from).
struct Unready {
    /// 0 once the executor owns the process.
    pid: u32,
    /// The process it must be a child of: this machine, or the executor it was forked from.
    parent: u32,
    child: Option<Child>,
    exact: Option<Exact>,
    birth: Option<Birth>,
    /// Its own scope, ended with it if it never becomes an executor.
    scope: Option<Arc<platform::contain::Scope>>,
}
impl Unready {
    /// Its exit status, once the process is seen to have exited.
    fn ended(&mut self) -> io::Result<ExitStatus> {
        match self.child.as_mut() {
            Some(child) => child.wait(),
            None => Ok(self
                .birth
                .as_ref()
                .and_then(zombie_status)
                .unwrap_or_else(|| exit_status(0))),
        }
    }
}
impl Drop for Unready {
    fn drop(&mut self) {
        if self.pid == 0 {
            return;
        }
        let exact = self.exact.take().or_else(|| {
            Birth::of(self.pid)
                .ok()
                .and_then(|birth| Exact::open(&birth).ok().flatten())
        });
        if let Some(exact) = &exact {
            let _ = exact.kill();
        }
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(scope) = self.scope.take() {
            let _ = scope.end();
        }
    }
}

/// Everything an executor's teardown must hold until its exit is observed.
struct Ending {
    exact: io::Result<Exact>,
    child: Option<Child>,
    forked: bool,
    retained: Vec<Box<dyn Send>>,
    socket: PathBuf,
    liveness: Liveness,
}
impl Ending {
    fn end(mut self) -> io::Result<Ended> {
        let exact = match &self.exact {
            Ok(exact) => exact,
            Err(error) => {
                // An unobservable receiver is not authority to release its sources.
                std::mem::forget(std::mem::take(&mut self.retained));
                return Err(io::Error::other(format!(
                    "executor exit unobservable: {error}"
                )));
            }
        };
        // Followers (degree > 1) stay in the leader's group and are waited for too.
        let mut ended = match reap_group(exact, self.child.as_mut(), self.liveness) {
            Ok(ended) => ended,
            Err(error) => {
                std::mem::forget(std::mem::take(&mut self.retained));
                return Err(error);
            }
        };
        if self.forked {
            ended.status = zombie_status(exact.birth()).unwrap_or(ended.status);
        }
        if !self.socket.as_os_str().is_empty() {
            let _ = fs::remove_file(&self.socket);
        }
        drop(std::mem::take(&mut self.retained));
        Ok(ended)
    }
}

/// Dropping the handle ends the executor before returning: whoever releases its slot or
/// reservation next does so after its exit, on every path including early returns.
impl Drop for DeviceExecutor {
    fn drop(&mut self) {
        if self.child.is_none() && !self.forked && self.retained.is_empty() {
            return;
        }
        if let Err(error) = self.parts().end() {
            tracing::warn!("executor teardown: {error}");
        }
    }
}

/// The last progress a call reported: its stage, its fractions (as bits), position, completed
/// units and relayed call.
type Advance = (String, Option<u64>, Option<u64>, Option<u64>, u64, Option<String>, u64);

/// Whether `frame` is real progress of its call: a progress frame that moved its stage, position
/// or completed units (`last` is the one before; a repeat, telemetry or a log line is not), a
/// work count that moved, or a request that makes something toward its output: a product or
/// checkpoint published, weights kept, a child call started. A re-ask for a layout, a memo
/// lookup, a poll is not.
fn advances(frame: &Frame, last: &mut Advance) -> bool {
    match frame.kind {
        Kind::Work => std::mem::replace(&mut last.6, frame.work) != frame.work,
        Kind::Progress if matches!(frame.event, Some(Event::Progress)) && !frame.stage.is_empty() => {
            let fraction = |f: Option<f64>| f.map(f64::to_bits);
            let now = (
                frame.stage.clone(),
                fraction(frame.stage_fraction),
                fraction(frame.overall_fraction),
                frame.position,
                frame.advance,
                frame.call_request.clone(),
                last.6,
            );
            let moved = *last != now;
            *last = now;
            moved
        }
        _ => matches!(frame.request, Some(Request::Publish(_) | Request::CheckpointItem(_) | Request::ChildCall(_))),
    }
}

pub fn read_result(spool: &Path, reply: &Frame) -> io::Result<Value> {
    if !reply.ok
        || reply
            .outcome
            .as_ref()
            .is_none_or(|outcome| outcome.terminal != "succeeded")
    {
        return Err(io::Error::other(
            "stock invocation has no successful typed outcome",
        ));
    }
    if !reply.quiescent || !reply.poisoned.is_empty() {
        return Err(io::Error::other(
            "stock invocation did not settle quiescent",
        ));
    }
    let reference = reply
        .result_ref
        .as_ref()
        .ok_or_else(|| io::Error::other("stock invocation has no retained result reference"))?;
    let mut bytes = Vec::new();
    open_artifact(spool, Path::new(RESULT_DOCUMENT))?.read_to_end(&mut bytes)?;
    if bytes.len() as u64 != reference.length
        || format!("sha256:{}", tensorfs::sha256::hex_digest(&bytes)) != reference.digest
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "spooled result identity differs from executor reference",
        ));
    }
    serde_json::from_slice(&bytes).map_err(io::Error::other)
}
fn write_frame(
    stream: &mut LocalStream,
    value: &impl Serialize,
) -> io::Result<()> {
    let value = serde_json::to_value(value)?;
    scan(&value)?;
    seam::write_body(stream, &serde_json::to_vec(&value)?)
}
/// SDK control framing shared with the CPU source-qualification receiver.
pub fn read_frame(stream: &mut LocalStream) -> io::Result<Option<Frame>> {
    seam::read_body(stream)?.map(|bytes| decode_frame(&bytes)).transpose()
}

/// Every field of a frame is optional, and the Runtime spells an absent one as null as often as
/// it leaves it out (`call_request`, run 5257): a null field reads as absent.
fn decode_frame(bytes: &[u8]) -> io::Result<Frame> {
    let mut fields: serde_json::Map<String, Value> =
        serde_json::from_slice(bytes).map_err(io::Error::other)?;
    fields.retain(|_, value| !value.is_null());
    let event = match fields.get("event").and_then(Value::as_str) {
        Some("request") => Some(Event::Request),
        Some("progress") => Some(Event::Progress),
        _ => None,
    };
    let seq = fields.get("seq").and_then(Value::as_u64);
    let descriptor = fields.get("descriptor").and_then(Value::as_bool).unwrap_or(false);
    let reply = fields.get("reply").and_then(Value::as_str).unwrap_or_default().to_owned();
    let request = (event == Some(Event::Request)).then(|| Request::decode(Value::Object(fields.clone())));
    let frame: Result<Frame, _> = serde_path_to_error::deserialize(Value::Object(fields));
    let frame = match (frame, request) {
        (Ok(_), Some(Err(why))) => Err(why),
        (Ok(frame), request) => Ok(Frame { request: request.and_then(Result::ok), ..frame }),
        (Err(error), _) => Err(error.to_string()),
    };
    frame.or_else(|error| match (event, seq) {
        (Some(Event::Request), Some(seq)) => Ok(Frame {
            event,
            seq,
            descriptor,
            malformed: error.clone(),
            ..Frame::default()
        }),
        (Some(Event::Progress), _) => Ok(Frame {
            event,
            malformed: error,
            ..Frame::default()
        }),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("executor_reply_malformed: the executor's `{reply}` reply: {error}"),
        )),
    })
}

/// A request this machine cannot decode is refused alone, under its own sequence number,
/// after the fd it announced is taken off the socket: the executor's call goes on.
fn refuse_malformed(
    stream: &mut LocalStream,
    frame: &Frame,
) -> io::Result<()> {
    if frame.descriptor {
        drop(platform::handles::recv_fd(stream)?);
    }
    let answer = refused(frame.seq, "malformed_request", frame.malformed.clone());
    write_frame(stream, &answer)
}
pub fn write_answer(
    stream: &mut LocalStream,
    answer: &Answer,
) -> io::Result<()> {
    write_frame(stream, answer)
}
fn scan(value: &Value) -> io::Result<()> {
    match value {
        Value::Object(fields) => {
            for (name, value) in fields {
                if ["token", "credential", "authorization", "secret", "jwt"]
                    .contains(&name.to_lowercase().as_str())
                {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "credential cannot cross device executor seam",
                    ));
                }
                scan(value)?;
            }
        }
        Value::Array(values) => {
            for value in values {
                scan(value)?;
            }
        }
        _ => (),
    }
    Ok(())
}

