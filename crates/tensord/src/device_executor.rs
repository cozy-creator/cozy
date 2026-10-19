//! Existing Runtime device-executor control seam. Models and codecs remain in Python SDK.
use cozy_platform::process::{Birth, Exact};
use crate::{
    execution::open_artifact,
    launch_identity::{LaunchIdentity, Seal},
    process::{reap_group, tail, Liveness, Meter, Watch, Watching},
    protocol,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use cozy_platform::{
    durable::Mode as _,
    ipc::{LocalListener, LocalStream},
};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{Arc, Mutex},
    time::Duration,
};
#[cfg(unix)]
use std::os::{
    fd::AsRawFd,
    unix::process::{CommandExt, ExitStatusExt},
};

pub const RESULT_DOCUMENT: &str = "result.canonical";

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

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Hello {
    pub pid: u32,
    pub ppid: u32,
    pub pgid: u32,
    pub runtime_version: String,
    pub tensorfs_version: String,
    pub sealed: BTreeMap<String, String>,
    pub torch_loaded: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Adapter {
    pub model_id: String,
    pub r#ref: String,
    pub scale: f64,
    pub kind: String,
    pub component: String,
    pub source_component: String,
    pub family: String,
}
impl Default for Adapter {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            r#ref: String::new(),
            scale: 1.0,
            kind: "lora".into(),
            component: String::new(),
            source_component: String::new(),
            family: String::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Binding {
    pub application: String,
    pub package_interface: String,
    pub model_class: String,
    pub model_binding_path: String,
    pub model_parameter_name: String,
    pub model_parameter_names: Vec<String>,
    pub component: String,
    pub components: Vec<String>,
    pub snapshots: BTreeMap<String, String>,
    pub store: String,
    pub snapshot: String,
    pub release: String,
    /// Hardware derivation variant, e.g. sm86; empty lets the SDK measure it. Never a catalog lane.
    pub variant: String,
    pub development: bool,
    pub custody: String,
    pub objective: String,
    pub steps_basis: u64,
    pub placement: String,
    pub package: String,
    pub model: String,
    pub adapters: Vec<Adapter>,
}
impl Default for Binding {
    fn default() -> Self {
        Self {
            application: String::new(),
            package_interface: String::new(),
            model_class: String::new(),
            model_binding_path: String::new(),
            model_parameter_name: String::new(),
            model_parameter_names: Vec::new(),
            component: String::new(),
            components: Vec::new(),
            snapshots: BTreeMap::new(),
            store: String::new(),
            snapshot: String::new(),
            release: String::new(),
            variant: String::new(),
            development: false,
            custody: "canonical".into(),
            objective: String::new(),
            steps_basis: 0,
            placement: String::new(),
            package: String::new(),
            model: String::new(),
            adapters: Vec::new(),
        }
    }
}

/// One model of a many-model construction (an entrypoint with several model slots).
#[derive(Clone, Debug, Serialize)]
pub struct ModelLoad {
    pub binding: Binding,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum DeviceCommand {
    Hello,
    Shutdown,
    Start {
        devices: String,
        application: String,
        package_interface: PathBuf,
        sequence_parallel_degree: u32,
        #[serde(skip_serializing_if = "is_false")]
        import_only: bool,
        /// What its attempts may call: the Runtime installs their callers before it imports
        /// the package and brokers each attempt's calls.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        call_interfaces: Vec<CallInterface>,
        /// Each constructed slot's checkpoint, so Runtime schedules fetches before it imports.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        constructions: Vec<Construction>,
        /// One ledger memfd per GPU of `devices`, in order, follows the frame: rank 0 claims its
        /// row in the first and hands follower r the r-th (design memory-varena §5.5).
        #[serde(skip_serializing_if = "is_zero")]
        ledgers: u32,
        /// `shared` (a local machine: leave the display its floor, give memory back) or
        /// `dedicated` (a rented pod).
        memory_mode: &'static str,
    },
    Load {
        construction: String,
        devices: String,
        sequence_parallel_degree: u32,
        binding: Box<Binding>,
        /// Several model slots in one construction (H3 turbo: base + LoRA); Runtime builds
        /// these models and loads their granted weights, ignoring `binding`.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        models: Vec<ModelLoad>,
        #[serde(skip_serializing_if = "String::is_empty")]
        attention_pin: String,
    },
    Activate {
        construction: String,
    },
    PrepareRequest {
        request_id: String,
        construction: String,
        entrypoint: String,
        payload: Value,
        /// Checked before entry, so a bad pin is the request's refusal, not a fault.
        #[serde(skip_serializing_if = "String::is_empty")]
        attention_kernel: String,
        /// The request's weight precision (16, 8 or 4); 16 holds every attention site exact.
        #[serde(skip_serializing_if = "is_zero")]
        bits: u32,
        /// File inputs by field path (identity only; the bytes arrive with Invoke).
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        input_metadata: BTreeMap<String, Value>,
    },
    Invoke {
        /// Retained asset capacity derived from the actual installed result schema.
        max_output_bytes: u64,
        request_id: String,
        construction: String,
        entrypoint: String,
        spool: PathBuf,
        deadline_s: Option<f64>,
        #[serde(skip_serializing_if = "String::is_empty")]
        attention_kernel: String,
        #[serde(skip_serializing_if = "is_zero")]
        bits: u32,
        /// File inputs by field path: read-only copies in this call's spool.
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        inputs: BTreeMap<String, Value>,
        /// Input trees by reference: (read-only directory, manifest digest).
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        trees: BTreeMap<String, (PathBuf, String)>,
        /// Each `Weights[M]` parameter's checkpoint (`sha256:<hex>`), read through `weights_writer`.
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        sources: BTreeMap<String, String>,
        /// The weights outputs this attempt may write: output id to its byte bound.
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        weights: BTreeMap<String, u64>,
    },
    /// Warming: the loaded construction's models are loaded at `level` (`gpu` or `ram`, varena
    /// filling them), nothing run.
    Map {
        construction: String,
        level: &'static str,
    },
    Unload {
        construction: String,
    },
    Probe {
        collect: bool,
    },
    /// Fork this import-only executor into a new one on its own seam (`fork/1`).
    Fork {
        socket: PathBuf,
        root: PathBuf,
        environment: BTreeMap<String, String>,
    },
}

/// One constructed model slot of a session: its checkpoint and the components it uses.
#[derive(Clone, Debug, Serialize)]
pub struct Construction {
    pub path: String,
    pub manifest: String,
    pub components: Vec<String>,
}

/// One callable an attempt may call: its own package's, or another App's of its environment.
#[derive(Clone, Debug, Serialize)]
pub struct CallInterface {
    pub module: String,
    pub export: String,
    pub interface_path: PathBuf,
    #[serde(rename = "self")]
    pub self_call: bool,
    /// Another package's interface: the Runtime installs its typed callers from it (a caller
    /// imports the callee's function and calls it with its request's fields as arguments).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interface_document: Option<Value>,
    /// A Runtime built-in surface (`operations`): the Runtime binds it from its own code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub builtin: Option<String>,
}
impl DeviceCommand {
    /// The run a command executes and the spool its executor reads and writes.
    fn spool(&self) -> Option<(&str, &Path)> {
        match self {
            Self::Invoke {
                request_id, spool, ..
            } => Some((request_id.as_str(), spool.as_path())),
            _ => None,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Hello => "hello",
            Self::Shutdown => "shutdown",
            Self::Start { .. } => "start",
            Self::Load { .. } => "load",
            Self::Activate { .. } => "activate",
            Self::PrepareRequest { .. } => "prepare_request",
            Self::Invoke { .. } => "invoke",
            Self::Map { .. } => "map",
            Self::Unload { .. } => "unload",
            Self::Probe { .. } => "probe",
            Self::Fork { .. } => "fork",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    Progress,
    Request,
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    ModelSource,
    /// One weight set's bytes: its sealed host layout, or its chunk files.
    ReadGrant,
    /// A sealed host layout the executor filled, for the next executor of its weights.
    Deposit,
    Publish,
    StageMemoLookup,
    StageMemoStore,
    Progress,
    ExecutionActivity,
    ChildCall,
    ChildPoll,
    ChildCancel,
    ChildForget,
    ChildEvents,
    GpuRelease,
    ModelPrefetch,
    /// An attempt's weights: one source read, one output written, or an output adopted.
    WeightsWriter,
    /// A fresh channel over the call's checkpoints held in part (`access::Fetch::channel`).
    FetchChannel,
    /// A `progress` event carrying a retained `log` observation (`Telemetry.log`).
    Log,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ResultRef {
    pub digest: String,
    pub length: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Outcome {
    pub terminal: String,
    #[serde(default)]
    pub origin: String,
    #[serde(default)]
    pub code: String,
    #[serde(default)]
    pub message: String,
    /// The raising traceback of a failed invocation (diagnostics, never a terminal field).
    #[serde(default)]
    pub traceback: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Output {
    pub output_id: String,
    pub asset_ref: String,
    pub kind: String,
    #[serde(default)]
    pub media_type: String,
    #[serde(default)]
    pub size_bytes: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HostFrame {
    pub handle: String,
    pub codec: String,
    pub raw: String,
    pub raw_bytes: u64,
    #[serde(default)]
    pub media_type: String,
    #[serde(default)]
    pub facts: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Metrics {
    pub handler_ms: Option<f64>,
    pub device_lease_ms: Option<f64>,
    pub d2h_wait_ms: Option<f64>,
    pub peak_vram_bytes: Option<i64>,
    pub peak_reserved_bytes: Option<i64>,
    pub activation_peak_bytes: Option<i64>,
    pub activation_peaks: BTreeMap<String, i64>,
    pub allocated_at_start_bytes: Option<i64>,
    pub working_peak_vram_bytes: Option<i64>,
    pub rss_at_end_bytes: Option<i64>,
    pub started_unix: Option<f64>,
    pub gpu_count: Option<u32>,
    pub shape_cell: Option<String>,
}
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct LoadFacts {
    pub cuda_init_ms: Option<f64>,
    pub prepare_ms: Option<f64>,
    pub stages: Vec<(String, f64)>,
    pub warm_ms: Option<f64>,
    pub filled_bytes: Option<u64>,
    pub device_free_bytes: Option<i64>,
    pub device_total_bytes: Option<i64>,
    pub allocator_bytes: Option<i64>,
    pub reserved_bytes: Option<i64>,
    pub rss_bytes: Option<i64>,
    /// The attention kernel each component's sites were given, and what was skipped and why.
    pub attention: Option<Value>,
}

/// One decode per frame; only consumed business fields are represented.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Frame {
    pub event: Option<Event>,
    /// A request or progress frame whose fields did not decode (a negative byte count, run
    /// 10078): why. A request is refused alone under its own sequence number and a progress
    /// frame is dropped; the executor and its other work go on.
    #[serde(skip)]
    pub malformed: String,
    pub reply: String,
    pub ok: bool,
    pub code: String,
    pub detail: String,
    /// A refusal's own terminal (`refused`/`failed`) and origin (`request`/`author`/`runtime`).
    pub terminal: String,
    pub origin: String,
    /// A traced command refusal's raising traceback.
    pub traceback: String,
    #[serde(flatten)]
    pub hello: Hello,
    pub outcome: Option<Outcome>,
    pub quiescent: bool,
    pub poisoned: String,
    pub reused: bool,
    /// `start` of a group (degree > 1): the follower ranks' pids, rank 1 first.
    pub follower_pids: Vec<u32>,
    pub result_ref: Option<ResultRef>,
    pub outputs: Vec<Output>,
    pub frames: Vec<HostFrame>,
    pub max_output_bytes: Option<u64>,
    pub request_id: String,
    pub seq: u64,
    pub kind: Kind,
    pub descriptor: bool,
    pub name: String,
    /// `read_grant`/`deposit`: the sealed host layout's digest.
    pub layout: String,
    pub manifest: String,
    pub sha256: String,
    pub length: u64,
    pub facts: Option<LoadFacts>,
    /// `map`: whether every weight of the construction is on its GPU after it.
    pub complete: Option<bool>,
    pub metrics: Option<Metrics>,
    /// `invoke`: the request's memory per model per step (varena's `RequestStats`); another
    /// frame's `memory` of another shape (an older Hello's list) reads as absent.
    #[serde(deserialize_with = "lenient")]
    pub memory: Option<crate::memory::call::Stats>,
    pub stage: String,
    pub position: Option<u64>,
    pub total: Option<u64>,
    pub stage_fraction: Option<f64>,
    pub overall_fraction: Option<f64>,
    pub step_ms: Option<f64>,
    pub advance: u64,
    /// `progress`: the child call whose stage a job relays (the Runtime sends null for none).
    pub call_request: Option<String>,
    /// A retained observation (`kind` log): `name` is its message, `value` its level and
    /// `fields` its scalars.
    pub value: Value,
    pub fields: Value,
    /// PrepareRequest: the request's normalized features (its shape).
    pub features: BTreeMap<String, Value>,
    // `publish` (Outputs.publish): one product of a declared output.
    pub output: String,
    pub label: String,
    pub asset_ref: String,
    pub asset_kind: String,
    pub media_type: String,
    pub size_bytes: u64,
    pub digest: String,
    pub parts: Vec<PublishPart>,
    /// `start`: the executor's legs, `[[name, ms], ...]`; read leniently.
    pub stages: Value,
    /// `invoke`: the executor's stage and step tracks, its ranks' execution records (GPU,
    /// attention served) and its observation tail; read leniently, kept as the run's
    /// measurements.
    pub attribution: Value,
    pub execution: Value,
    pub observations: Value,
    /// `invoke`: some attention site of the call applied the request's pin.
    pub attention_applied: bool,
    /// `invoke`: the rows of how the request was served other than as sent; read row by row
    /// (`adjustments`).
    pub adjustments: Value,
    // `child_*` (a job's managed calls): the call's index, callee and canonical request.
    pub call_index: u64,
    pub module: String,
    pub export: String,
    pub payload: String,
    pub progress_label: String,
    // `stage_memo_*` (a memoized Model method): its key and scope (`stage`, `length` above), and
    // on a store the spool file it offers with what producing it cost.
    pub key: String,
    pub numerics: String,
    pub local: String,
    pub cost_ms: f64,
    /// `weights_writer`: `source`, `output` or `adopt`, its output and its transaction.
    pub operation: String,
    pub output_slot: String,
    pub transaction: String,
}

impl Frame {
    /// A `log` observation (`Telemetry.log`) as its log line: the level (debug, info, warning
    /// or error) and the message followed by ` key=value` for each field.
    pub fn log(&self) -> Option<(&'static str, String)> {
        if self.kind != Kind::Log {
            return None;
        }
        let level = match self.value.as_str().map(str::to_ascii_lowercase).as_deref() {
            Some("warning" | "warn") => "warning",
            Some("error" | "critical" | "fatal") => "error",
            Some("debug") => "debug",
            _ => "info",
        };
        let mut text = self.name.clone();
        for (key, value) in self.fields.as_object().into_iter().flatten() {
            match value {
                Value::String(s) if !s.is_empty() && !s.contains(char::is_whitespace) => {
                    text.push_str(&format!(" {key}={s}"))
                }
                other => text.push_str(&format!(" {key}={other}")),
            }
        }
        Some((level, text))
    }
}

/// One part of a composite product: a spool file and the media time it adds.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct PublishPart {
    pub local: String,
    pub duration_us: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Answer {
    pub event: &'static str,
    pub seq: u64,
    pub ok: bool,
    pub code: String,
    pub detail: String,
    pub held: bool,
    /// One fd follows the answer. Sent only when true: a `Tier` answer types it as the fd.
    #[serde(skip_serializing_if = "is_false")]
    pub descriptor: bool,
    pub sha256: String,
    pub length: u64,
    /// `read_grant`: the fds that follow the answer.
    #[serde(skip_serializing_if = "is_zero")]
    pub descriptors: u64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub digest: String,
    #[serde(skip_serializing_if = "is_zero")]
    pub sequence: u64,
    /// `CallState` (a managed call): its state, child, canonical result and byte grants.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub state: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub child_request_id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub result: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub byte_grants: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<Value>,
    /// A failed child call's raising traceback, whole (its triage's).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub traceback: String,
    /// `Opened` (a weights output) and `Adopted`: its transaction, run and manifest.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub transaction: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub request_id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub manifest: String,
    #[serde(skip_serializing_if = "is_zero")]
    pub manifest_length: u64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub receipt_digest: String,
    /// A memo hit: the entry's verified copy in the run's spool, and what producing it cost.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub local: String,
    #[serde(skip_serializing_if = "is_zero_ms")]
    pub cost_ms: f64,
    /// The memo scope computed one key twice, differently: the executor stops reusing it.
    #[serde(skip_serializing_if = "is_false")]
    pub disputed: bool,
    /// A memo store: whether the entry was kept, else why not.
    #[serde(skip_serializing_if = "is_false")]
    pub stored: bool,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub reason: String,
    /// `read_grant` without a sealed layout: the SHA-256 of the chunk objects whose fds
    /// follow, sorted by digest (`HostTier::objects_digest`), in place of the list (SDXL's
    /// UNet: about 1,700 objects).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub objects_sha256: String,
}
fn is_zero_ms(value: &f64) -> bool {
    *value == 0.0
}
impl Answer {
    pub fn unavailable(seq: u64) -> Self {
        Self {
            event: "answer",
            seq,
            ok: false,
            code: "capability_unavailable".into(),
            detail: "owner has not qualified this operation".into(),
            held: false,
            descriptor: false,
            sha256: String::new(),
            length: 0,
            descriptors: 0,
            digest: String::new(),
            sequence: 0,
            state: String::new(),
            child_request_id: String::new(),
            result: String::new(),
            byte_grants: Vec::new(),
            progress: None,
            traceback: String::new(),
            transaction: String::new(),
            request_id: String::new(),
            manifest: String::new(),
            manifest_length: 0,
            receipt_digest: String::new(),
            local: String::new(),
            cost_ms: 0.0,
            disputed: false,
            stored: false,
            reason: String::new(),
            objects_sha256: String::new(),
        }
    }
    pub fn ok(seq: u64) -> Self {
        Self {
            ok: true,
            code: String::new(),
            detail: String::new(),
            ..Self::unavailable(seq)
        }
    }
    pub fn refused(seq: u64, code: &str, detail: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            detail: crate::triage::ends(&Into::<String>::into(detail), 1024),
            ..Self::unavailable(seq)
        }
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
    /// `read_grant`: the plan's memfd, when one came with the request; the fds returned follow
    /// the answer (`descriptors`).
    fn read_grant(
        &mut self,
        frame: &Frame,
        plan: Option<File>,
    ) -> io::Result<(Answer, Vec<File>)> {
        drop(plan);
        Ok((Answer::unavailable(frame.seq), Vec::new()))
    }
}

pub struct Baseline;
impl Services for Baseline {}

pub struct DeviceExecutor {
    child: Option<Child>,
    /// Forked from an import-only executor (`fork/1`), which is its parent and reaps it.
    forked: bool,
    exact: Exact,
    stream: LocalStream,
    pub birth: Birth,
    pub hello: Hello,
    /// Sampling resolution of the wedge rule. Tests shorten it; nothing else changes it.
    pub liveness: Liveness,
    /// Stillness while this answers true is not this executor's: it waits on runs watched
    /// by their own executors (a job's unfinished children).
    pub waits: Option<crate::process::Waits>,
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
/// frame meter then decides whether it stopped moving and must be killed.
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
        cozy_platform::durable::sync_dir(&self.root)?;
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
    cozy_platform::durable::set_mode(&config.root, 0o700)?;
    if let Some(identity) = config.identity {
        identity.own(&config.root)?;
    }
    let listener = LocalListener::bind(&config.socket)?;
    // A Unix socket is a file whose mode and owner admit the executor; a Windows pipe's own
    // access list admits only this user.
    #[cfg(unix)]
    cozy_platform::durable::set_mode(&config.socket, 0o600)?;
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
        cozy_platform::durable::private_dir(&dir)?;
        let metadata = fs::symlink_metadata(&dir)?;
        if !metadata.is_dir() || metadata.uid() != crate::launch_identity::euid() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "the short socket directory belongs to another user",
            ));
        }
        let digest = tensorfs_core::sha256::hex_digest(natural.as_os_str().as_encoded_bytes());
        return Ok(dir.join(&digest[..24]));
    }
    Ok(natural)
}

/// The executor's containment scope, when its config names one. No scopes on this OS (macOS,
/// Windows until its Job Object backend): none, and the executor's tree ends by its group.
fn scope(config: &ExecutorConfig) -> io::Result<Option<Arc<cozy_platform::contain::Scope>>> {
    let Some(namespace) = &config.cgroup_namespace else {
        return Ok(None);
    };
    match cozy_platform::contain::Scope::create(namespace) {
        Ok(scope) => Ok(Some(Arc::new(scope))),
        Err(error) if error.is_unsupported() => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn parent_of(pid: u32) -> io::Result<u32> {
    Ok(cozy_platform::process::parent_of(pid)?)
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
        let mut command = Command::new(&self.config.python);
        cozy_platform::process::inherit_nothing(&mut command);
        command
            .args([
                "-I",
                "-c",
                include_str!("../python/tensord_client/device_codec.py"),
            ])
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
        // SAFETY: only async-signal-safe fcntl on the retained generation hold.
        #[cfg(unix)]
        unsafe {
            command.pre_exec(move || {
                if let Some(descriptor) = hold {
                    if libc::fcntl(descriptor, libc::F_SETFD, 0) < 0 {
                        return Err(io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
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
    fn encode(&self, spool: &Path, reply: &Frame) -> io::Result<PostReply> {
        use std::io::BufRead;
        let mut helper = self.helper.lock().unwrap();
        if helper.is_none() {
            *helper = Some(self.spawn()?);
        }
        let request = PostRequest {
            spool,
            frames: &reply.frames,
            outputs: &reply.outputs,
            max_output_bytes: reply.max_output_bytes,
        };
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
            Done(PostReply),
            Refused { error: String },
        }
        match serde_json::from_str::<Answer>(&answer).map_err(io::Error::other)? {
            Answer::Done(post) => Ok(post),
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
#[derive(Debug, Deserialize)]
struct PostReply {
    bindings: Vec<AssetBinding>,
}
#[derive(Serialize)]
struct PostRequest<'a> {
    spool: &'a Path,
    frames: &'a [HostFrame],
    outputs: &'a [Output],
    max_output_bytes: Option<u64>,
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
    let mut post = match encode_native(spool, reply)? {
        Some(bindings) => PostReply { bindings },
        None => codec.encode(spool, reply)?,
    };
    for tree in &trees {
        post.bindings.push(tree_binding(spool, tree)?);
    }
    Ok((result, post.bindings))
}
/// An output tree as the machine holds it: each regular file of its spool directory linked as
/// the spool's `sha256-<hex>`, and a manifest naming them by path (`gpu_service::TREE_MEDIA`),
/// which the output binds to. A tree forwarded from a child (`sha256:<manifest>`) is already
/// both.
fn tree_binding(spool: &Path, output: &Output) -> io::Result<AssetBinding> {
    let invalid = |why: &str| io::Error::new(io::ErrorKind::InvalidData, format!("output tree: {why}"));
    let hash = |path: &Path| -> io::Result<(String, u64)> {
        let mut file = open_artifact(spool, path)?;
        let mut hash = tensorfs_core::sha256::Sha256::new();
        let length = io::copy(&mut file, &mut HashWriter(&mut hash))?;
        Ok((tensorfs_core::sha256::hex(&hash.finish()), length))
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
            let mut entries = vec![];
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
                    entries.push(serde_json::json!({"kind": "file", "path": path,
                        "blob": {"sha256": sha, "length": length}}));
                }
            }
            if entries.is_empty() {
                return Err(invalid("it holds no file"));
            }
            entries.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
            let manifest = crate::boundary_json::exact(&serde_json::json!({"entries": entries}));
            let name = format!("{directory}-manifest");
            fs::write(spool.join(&name), manifest)?;
            name
        }
    };
    let mut bytes = Vec::new();
    open_artifact(spool, Path::new(&name))?.read_to_end(&mut bytes)?;
    let manifest: Value = serde_json::from_slice(&bytes).map_err(|_| invalid("its manifest is not JSON"))?;
    let mut content = 0;
    for entry in manifest["entries"].as_array().ok_or_else(|| invalid("no entries"))? {
        let member = format!("sha256-{}", entry["blob"]["sha256"].as_str().unwrap_or_default());
        let length = open_artifact(spool, Path::new(&member))?.metadata()?.len();
        if length != entry["blob"]["length"].as_u64().unwrap_or(u64::MAX) {
            return Err(invalid("a member differs from its manifest"));
        }
        content += entry["blob"]["length"].as_u64().unwrap_or_default();
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

struct HashWriter<'a>(&'a mut tensorfs_core::sha256::Sha256);
impl Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Encodes registered PNG/WebP frames in this process exactly as the SDK's post thread
/// does (lossless RGB; PNG at zlib's fastest level, WebP lossless) and binds every output
/// to its spool file. None when a frame needs a codec only the SDK has.
fn encode_native(spool: &Path, reply: &Frame) -> io::Result<Option<Vec<AssetBinding>>> {
    if reply
        .frames
        .iter()
        .any(|f| !matches!(f.codec.as_str(), "png" | "webp"))
    {
        return Ok(None);
    }
    let invalid = |detail: &str| io::Error::new(io::ErrorKind::InvalidData, detail.to_string());
    let blob = |reference: &str| -> io::Result<String> {
        // A file a child returned, which this run hands on as its own (`jobs::grant`).
        if let Some(hex) = reference.strip_prefix("sha256:").filter(|h| {
            h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit())
        }) {
            return Ok(format!("sha256-{hex}"));
        }
        let tail: Vec<_> = reference.rsplitn(3, '/').collect();
        match tail.as_slice() {
            [name, kind, _]
                if !name.is_empty()
                    && !name.contains(['/', '\0'])
                    && *name != "."
                    && *name != ".." =>
            {
                Ok(format!("{kind}-{name}"))
            }
            _ => Err(invalid("SDK output has no owned spool binding")),
        }
    };
    let mut media = BTreeMap::new();
    for frame in &reply.frames {
        let mut raw = Vec::new();
        open_artifact(spool, Path::new(&frame.raw))?.read_to_end(&mut raw)?;
        let dimension = |name: &str| {
            frame
                .facts
                .get(name)
                .and_then(Value::as_u64)
                .filter(|v| *v > 0 && *v <= u32::MAX as u64)
                .map(|v| v as u32)
        };
        let (Some(width), Some(height)) = (dimension("width"), dimension("height")) else {
            return Err(invalid("registered frame has no dimensions"));
        };
        if raw.len() as u64 != frame.raw_bytes
            || raw.len() as u64 != width as u64 * height as u64 * 3
        {
            return Err(invalid("registered raw frame length changed"));
        }
        let mut encoded = Vec::new();
        if frame.codec == "png" {
            let mut encoder = png::Encoder::new(&mut encoded, width, height);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.set_compression(png::Compression::Fast);
            encoder
                .write_header()
                .and_then(|mut w| w.write_image_data(&raw))
                .map_err(io::Error::other)?;
        } else {
            image_webp::WebPEncoder::new(&mut encoded)
                .encode(&raw, width, height, image_webp::ColorType::Rgb8)
                .map_err(io::Error::other)?;
        }
        let name = blob(&frame.handle)?;
        let temporary = spool.join(format!(".codec-{}", uuid::Uuid::new_v4().simple()));
        let mut file = cozy_platform::durable::open_nofollow(
            &temporary,
            OpenOptions::new().write(true).create_new(true).mode(0o600),
        )?;
        file.write_all(&encoded)?;
        fs::rename(&temporary, spool.join(&name))?;
        media.insert(
            frame.handle.clone(),
            if frame.codec == "png" {
                "image/png"
            } else {
                "image/webp"
            },
        );
    }
    let mut total = 0u64;
    let mut bindings = vec![];
    for output in &reply.outputs {
        let name = blob(&output.asset_ref)?;
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
            media_type: media
                .get(&output.asset_ref)
                .map(|m| m.to_string())
                .unwrap_or_else(|| output.media_type.clone()),
            length,
        });
    }
    Ok(Some(bindings))
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
            .envs(scope.as_deref().and_then(cozy_platform::contain::Scope::environment))
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
        if let Some((name, token)) = scope.as_deref().and_then(cozy_platform::contain::Scope::environment) {
            forked.insert(name.into(), token.into());
        }
        let reply = self.command(
            &DeviceCommand::Fork {
                socket: config.socket.clone(),
                root: config.root.clone(),
                environment: forked,
            },
            &mut Baseline,
        );
        let pid = match reply {
            Ok(reply) if reply.ok && reply.hello.pid != 0 => reply.hello.pid,
            refused => {
                drop(listener);
                fs::remove_file(&config.socket)?;
                // A child the parent made before its channel failed carries the token too.
                if let Err(error) = scope.as_deref().map_or(Ok(0), cozy_platform::contain::Scope::end) {
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
        let credential = cozy_platform::handles::peer(&stream)?;
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
            stream,
            birth,
            hello: Hello::default(),
            liveness: Liveness::default(),
            waits: None,
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
        let hello = executor.command(&DeviceCommand::Hello, &mut Baseline)?;
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
        command: &DeviceCommand,
        services: &mut impl Services,
    ) -> io::Result<Frame> {
        self.command_with(command, &[], services)
    }

    /// `command` with `fds` following its frame (`Start`'s ledgers).
    pub fn command_with(
        &mut self,
        command: &DeviceCommand,
        fds: &[File],
        services: &mut impl Services,
    ) -> io::Result<Frame> {
        let meter = match command {
            DeviceCommand::Invoke { .. } => Meter::Frames,
            _ => Meter::Burn,
        };
        let watching = Watching::start(
            self.exact.try_clone()?,
            meter,
            self.liveness,
            self.worst_gap,
            self.waits.clone(),
            command.name(),
        )?;
        *self.watched.lock().unwrap() = Some(watching.watch());
        let result = self.exchange(command, fds, services, &watching.watch());
        *self.watched.lock().unwrap() = None;
        // Whatever the run was computing when its exchange ended, its waiters compute themselves.
        if let (Some(memo), Some((run, _))) = (&self.memo, command.spool()) {
            memo.release(run);
        }
        let (killed, worst_gap) = watching.finish();
        if meter == Meter::Frames {
            self.worst_gap = self.worst_gap.max(worst_gap);
        }
        match killed {
            Some(verdict) => Err(io::Error::other(verdict)),
            None => result,
        }
    }

    fn exchange(
        &mut self,
        command: &DeviceCommand,
        fds: &[File],
        services: &mut impl Services,
        watch: &Watch,
    ) -> io::Result<Frame> {
        let mut value = serde_json::to_value(command)?;
        // The executor's memoized stages go to the machine tier.
        if matches!(command, DeviceCommand::Start { .. }) && self.memo.is_some() {
            value["memo"] =
                serde_json::json!({"process_bytes": -1, "entry_bytes": crate::memo::ENTRY_BYTES});
        }
        write_frame(&mut self.stream, &value)?;
        for fd in fds {
            cozy_platform::handles::send_fd(&self.stream, fd)?;
        }
        loop {
            let frame = read_frame(&mut self.stream)?
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "stock executor EOF before reply")
                })?;
            watch.frame();
            match frame.event {
                Some(Event::Request) if !frame.malformed.is_empty() => {
                    refuse_malformed(&mut self.stream, &frame)?
                }
                Some(Event::Progress) if !frame.malformed.is_empty() => (),
                Some(Event::Progress) => services.progress(&frame),
                Some(Event::Request)
                    if matches!(frame.kind, Kind::StageMemoLookup | Kind::StageMemoStore) =>
                {
                    // A lookup may wait on the run computing its key: not this executor's stillness.
                    watch.serving(true);
                    let answer = match (&self.memo, command.spool()) {
                        (Some(memo), Some((run, spool))) => memo.answer(run, spool, &frame),
                        _ => Answer::unavailable(frame.seq),
                    };
                    watch.serving(false);
                    write_frame(&mut self.stream, &answer)?;
                }
                Some(Event::Request) if frame.kind == Kind::ReadGrant => {
                    let plan = if frame.descriptor {
                        Some(File::from(cozy_platform::handles::recv_fd(&self.stream)?))
                    } else {
                        None
                    };
                    watch.serving(true);
                    let answered = services.read_grant(&frame, plan);
                    watch.serving(false);
                    let (mut answer, out) = answered?;
                    answer.event = "answer";
                    answer.seq = frame.seq;
                    answer.descriptor = false;
                    answer.descriptors = out.len() as u64;
                    write_frame(&mut self.stream, &answer)?;
                    for file in &out {
                        cozy_platform::handles::send_fd(&self.stream, file)?;
                    }
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
                None => return Err(io::Error::other("out-of-order stock executor reply")),
            }
        }
    }

    fn answer(&mut self, frame: &Frame, services: &mut impl Services) -> io::Result<()> {
        let descriptor = if frame.descriptor {
            Some(File::from(cozy_platform::handles::recv_fd(&self.stream)?))
        } else {
            None
        };
        let (mut answer, descriptor) = services.request(frame, descriptor)?;
        answer.event = "answer";
        answer.seq = frame.seq;
        answer.descriptor = descriptor.is_some();
        write_frame(&mut self.stream, &answer)?;
        if let Some(descriptor) = descriptor {
            cozy_platform::handles::send_fd(&self.stream, &descriptor)?;
        }
        Ok(())
    }

    /// Attempt-keyed cooperative cancellation; observer teardown never calls this.
    pub fn cancel(&self, request_id: &str) -> io::Result<()> {
        self.cancellation().cancel(request_id)
    }

    /// Ask the executor to exit, then observe that exact exit (killing only a wedge).
    pub fn shutdown(mut self) -> io::Result<()> {
        let asked = self.command(&DeviceCommand::Shutdown, &mut Baseline);
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
    scope: Option<Arc<cozy_platform::contain::Scope>>,
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
        || format!("sha256:{}", tensorfs_core::sha256::hex_digest(&bytes)) != reference.digest
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "spooled result identity differs from executor reference",
        ));
    }
    serde_json::from_slice(&bytes).map_err(io::Error::other)
}
/// A field of the shape this machine reads, else absent: never a reason to refuse the frame.
fn lenient<'de, D: serde::Deserializer<'de>, T: serde::de::DeserializeOwned>(d: D) -> Result<Option<T>, D::Error> {
    Ok(serde_json::from_value(Value::deserialize(d)?).ok())
}

fn is_zero<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}
fn is_false(value: &bool) -> bool {
    !*value
}

fn write_frame(
    stream: &mut LocalStream,
    value: &impl Serialize,
) -> io::Result<()> {
    let value = serde_json::to_value(value)?;
    scan(&value)?;
    protocol::write_body(stream, &serde_json::to_vec(&value)?)
}
/// SDK control framing shared with the CPU source-qualification receiver.
pub fn read_frame(stream: &mut LocalStream) -> io::Result<Option<Frame>> {
    protocol::read_body(stream)?.map(|bytes| decode_frame(&bytes)).transpose()
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
    serde_json::from_value(Value::Object(fields)).or_else(|error| match (event, seq) {
        (Some(Event::Request), Some(seq)) => Ok(Frame {
            event,
            seq,
            descriptor,
            malformed: error.to_string(),
            ..Frame::default()
        }),
        (Some(Event::Progress), _) => Ok(Frame {
            event,
            malformed: error.to_string(),
            ..Frame::default()
        }),
        _ => Err(io::Error::other(error)),
    })
}

/// A request this machine cannot decode is refused alone, under its own sequence number,
/// after the fd it announced is taken off the socket: the executor's call goes on.
fn refuse_malformed(
    stream: &mut LocalStream,
    frame: &Frame,
) -> io::Result<()> {
    if frame.descriptor {
        drop(cozy_platform::handles::recv_fd(stream)?);
    }
    let answer = Answer::refused(frame.seq, "malformed_request", frame.malformed.clone());
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

#[cfg(test)]
mod codec_tests {
    use super::*;

    /// An invoke reply's adjustment rows are advisory: each malformed one is skipped and the
    /// reply, the run and the other rows stand, in the executor's order, each text bounded.
    #[test]
    fn a_malformed_adjustment_row_is_skipped_and_the_reply_stands() {
        let row = |kind: &str, field: &str| serde_json::json!({"field": field, "kind": kind,
            "requested": "21.4", "applied": "15", "reason": "video truncated to 15 s", "source": "policy"});
        let long = "x".repeat(4096);
        let reply = serde_json::json!({"reply": "invoke", "ok": true, "adjustments": [
            row("adjusted", "assets.2.asset"),
            row("shrink", "steps"),
            {"field": "steps", "requested": "50", "applied": "30", "reason": "", "source": "handler"},
            {"field": 7, "kind": "clamp", "requested": "50", "applied": "30", "reason": "", "source": "handler"},
            "steps",
            row("warn", &long),
        ]});
        let (mut runtime, mut machine) = cozy_platform::ipc::pair().unwrap();
        write_frame(&mut runtime, &reply).unwrap();
        let frame = read_frame(&mut machine).unwrap().unwrap();
        assert!(frame.ok && frame.reply == "invoke");
        let kept = adjustments("run-1", &frame);
        assert_eq!(kept.len(), 2, "{kept:?}");
        assert_eq!(kept[0], serde_json::from_value(row("adjusted", "assets.2.asset")).unwrap());
        assert_eq!((kept[1].kind, kept[1].field.chars().count()), (crate::journal::AdjustmentKind::Warn, MAX_ADJUSTMENT_TEXT));

        let rows: Vec<_> = (0..MAX_ADJUSTMENTS + 3).map(|i| row("clamp", &format!("f{i}"))).collect();
        let many = Frame { adjustments: rows.into(), ..Default::default() };
        assert_eq!(adjustments("run-2", &many).len(), MAX_ADJUSTMENTS);
        let none = Frame { adjustments: serde_json::json!({"steps": 30}), ..Default::default() };
        assert!(adjustments("run-3", &none).is_empty() && adjustments("run-4", &Frame::default()).is_empty());
    }

    /// A null in any optional field reads as its absence: a string, a flattened hello field,
    /// a count, a flag, a kind or a list (the Runtime's `None`).
    #[test]
    fn a_memory_field_of_another_shape_reads_as_absent() {
        let hello: Frame = serde_json::from_value(serde_json::json!({"reply": "hello", "ok": true, "pid": 7,
            "memory": [1, 2]})).unwrap();
        assert!(hello.ok && hello.memory.is_none() && hello.hello.pid == 7);
        let invoke: Frame = serde_json::from_value(serde_json::json!({"reply": "invoke", "memory": {"paged_in": 5}})).unwrap();
        assert_eq!(invoke.memory.unwrap().paged_in, 5);
    }

    #[test]
    fn a_null_field_reads_as_absent() {
        let body = serde_json::to_vec(&serde_json::json!({
            "event": "request", "kind": "publish", "seq": 4, "output": "image", "label": null,
            "asset_ref": null, "media_type": null, "digest": null, "code": null, "detail": null,
            "traceback": null, "poisoned": null, "stage": null, "runtime_version": null,
            "call_request": null, "size_bytes": null, "ok": null, "parts": null, "attribution": null,
        }))
        .unwrap();
        let (mut runtime, mut machine) = cozy_platform::ipc::pair().unwrap();
        runtime.write_all(&(body.len() as u32).to_be_bytes()).unwrap();
        runtime.write_all(&body).unwrap();
        let frame = read_frame(&mut machine).unwrap().unwrap();
        assert_eq!((frame.kind, frame.seq, frame.output.as_str()), (Kind::Publish, 4, "image"));
        assert!(frame.label.is_empty() && frame.code.is_empty() && frame.hello.runtime_version.is_empty());
        assert!(frame.call_request.is_none() && frame.size_bytes == 0 && frame.parts.is_empty());
    }

    /// A frame is as large as its structure: a load reply naming every key its checkpoint
    /// skips (FLUX.3 Klein's as-is text encoder) failed the old 64 KiB frame. Only a prefix
    /// over `MAX_FRAME` is refused, before any body byte is read.
    #[test]
    fn a_frame_over_64_kib_crosses_and_a_lying_prefix_is_refused() {
        let keys: Vec<String> = (0..20_000).map(|i| format!("model.visual.blocks.{i}.attn.qkv.weight")).collect();
        let reply = serde_json::json!({"reply": "load", "ok": true,
            "facts": {"ignored_extra_keys": keys, "filled_bytes": 7}});
        assert!(serde_json::to_vec(&reply).unwrap().len() > 512 << 10);
        let (mut runtime, mut machine) = cozy_platform::ipc::pair().unwrap();
        let writer = std::thread::spawn(move || write_frame(&mut runtime, &reply).map(|()| runtime));
        let frame = read_frame(&mut machine).unwrap().unwrap();
        let mut runtime = writer.join().unwrap().unwrap();
        assert!(frame.ok && frame.reply == "load");
        assert_eq!(frame.facts.unwrap().filled_bytes, Some(7));
        runtime.write_all(&(protocol::MAX_FRAME as u32 + 1).to_be_bytes()).unwrap();
        let refused = read_frame(&mut machine).unwrap_err();
        assert_eq!(refused.kind(), io::ErrorKind::InvalidData, "{refused}");
    }

    /// A request whose fields do not decode (a negative byte count, run 10078) is refused alone
    /// under its sequence number with the fd it announced taken off the socket; an undecodable
    /// progress frame is dropped; the next frame reads as before.
    #[test]
    fn a_request_that_does_not_decode_is_refused_alone() {
        let (mut runtime, mut machine) = cozy_platform::ipc::pair().unwrap();
        let malformed = serde_json::json!({"event": "request", "kind": "read_grant", "seq": 9,
            "length": -23068672, "descriptor": true});
        write_frame(&mut runtime, &malformed).unwrap();
        cozy_platform::handles::send_fd(&runtime, &File::open(std::env::current_exe().unwrap()).unwrap()).unwrap();
        let frame = read_frame(&mut machine).unwrap().unwrap();
        assert_eq!((frame.event, frame.seq), (Some(Event::Request), 9));
        assert!(frame.malformed.contains("expected u64"), "{}", frame.malformed);
        refuse_malformed(&mut machine, &frame).unwrap();
        let answer = read_frame(&mut runtime).unwrap().unwrap();
        assert_eq!((answer.seq, answer.ok, answer.code.as_str()), (9, false, "malformed_request"));
        let progress = serde_json::json!({"event": "progress", "stage": "x", "position": -1});
        write_frame(&mut runtime, &progress).unwrap();
        let dropped = read_frame(&mut machine).unwrap().unwrap();
        assert!(dropped.event == Some(Event::Progress) && !dropped.malformed.is_empty());
        let fine = serde_json::json!({"event": "request", "kind": "read_grant", "seq": 10,
            "length": 5});
        write_frame(&mut runtime, &fine).unwrap();
        let next = read_frame(&mut machine).unwrap().unwrap();
        assert!(next.malformed.is_empty());
        assert_eq!((next.kind, next.seq, next.length), (Kind::ReadGrant, 10, 5));
    }

    fn pid(codec: &Codec) -> Option<u32> {
        codec.helper.lock().unwrap().as_ref().map(|h| h.child.id())
    }

    #[test]
    #[ignore = "requires an explicitly configured installed SDK interpreter; no GPU imports"]
    fn one_sdk_encoder_serves_every_output_of_an_executor() {
        let python = PathBuf::from(std::env::var("TENSORD_CPU_TEST_PYTHON").unwrap());
        let root = std::env::temp_dir().join(format!("codec-{}", uuid::Uuid::new_v4().simple()));
        let spool = root.join("spool");
        fs::create_dir_all(&spool).unwrap();
        let codec = Codec::new(
            CodecConfig {
                python,
                environment: BTreeMap::new(),
                generation_hold: None,
            },
            root.join("codec.log"),
        );
        let reply = |index: usize, raw_bytes: u64| {
            let name = format!("image-{index:04}");
            fs::write(
                spool.join(format!("{name}.raw")),
                vec![index as u8 * 40; 4 * 2 * 3],
            )
            .unwrap();
            Frame {
                frames: vec![HostFrame {
                    handle: format!("asset/image/{index:04}"),
                    codec: "png".into(),
                    raw: format!("{name}.raw"),
                    raw_bytes,
                    media_type: "image/png".into(),
                    facts: BTreeMap::from([
                        ("width".into(), 4.into()),
                        ("height".into(), 2.into()),
                    ]),
                }],
                outputs: vec![Output {
                    output_id: format!("out-{index}"),
                    asset_ref: format!("asset/image/{index:04}"),
                    kind: "image".into(),
                    media_type: "image/png".into(),
                    size_bytes: None,
                }],
                ..Default::default()
            }
        };
        let first = codec.encode(&spool, &reply(1, 24)).unwrap();
        let helper = pid(&codec);
        let second = codec.encode(&spool, &reply(2, 24)).unwrap();
        assert_eq!(
            pid(&codec),
            helper,
            "one encoder process serves the executor's outputs"
        );
        for (post, name) in [(first, "image-0001"), (second, "image-0002")] {
            let bytes = fs::read(spool.join(name)).unwrap();
            assert_eq!(post.bindings[0].name, name);
            assert_eq!(post.bindings[0].media_type, "image/png");
            assert_eq!(post.bindings[0].length, bytes.len() as u64);
            assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        }
        // A refused output is that output's failure; the encoder keeps serving.
        assert!(codec.encode(&spool, &reply(3, 25)).is_err());
        assert_eq!(pid(&codec), helper);
        codec.encode(&spool, &reply(4, 24)).unwrap();
        drop(codec);
        fs::remove_dir_all(root).unwrap();
    }
}
