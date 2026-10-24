//! TensorD ↔ executor seam: the commands TensorD sends a Runtime executor, the replies, progress
//! and requests the executor sends back, and TensorD's answers. One definition for both sides.
//!
//! Each side may be another release than the other (the cozy binary and the tensorfs wheel ship
//! separately), so every reader is tolerant: an unknown field is ignored, a missing one takes its
//! default, and only a needed field (one without a default here) fails, by its name.
use std::{collections::BTreeMap, path::PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::machine::TensorFSOptions;

/// The result document's spool name.
pub const RESULT_DOCUMENT: &str = "result.canonical";

// --------------------------------------------------------------------------- commands

/// One command to an executor, tagged by `cmd`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Command {
    Hello,
    Shutdown,
    Start(Start),
    Load(Box<Load>),
    Activate(Activate),
    PrepareRequest(PrepareRequest),
    Invoke(Box<Invoke>),
    /// Warming: the loaded construction's models at `level`, nothing run.
    Map(Map),
    Unload(Unload),
    Probe(Probe),
    /// Fork this import-only executor into a new one on its own seam (`fork/1`).
    Fork(Fork),
    /// `actor`'s newest bearer at `tensorfs.endpoint`, sent outside any exchange and answered by
    /// nothing: the executor's reads and uploads there present it from now on.
    Bearer(Bearer),
}

impl Command {
    /// Every `cmd` TensorD sends.
    pub const NAMES: [&'static str; 12] = [
        "hello", "shutdown", "start", "load", "activate", "prepare_request", "invoke", "map", "unload",
        "probe", "fork", "bearer",
    ];

    /// One command from its wire bytes. An unknown member is ignored and a missing one takes its
    /// default; the error names a needed member's path (`spool`, `bits`) or an unknown `cmd`.
    pub fn decode(data: &[u8]) -> Result<Command, String> {
        fn one<T: serde::de::DeserializeOwned>(fields: Value) -> Result<T, String> {
            serde_path_to_error::deserialize(fields).map_err(|e| e.to_string())
        }
        let fields: Value = serde_json::from_slice(data).map_err(|e| e.to_string())?;
        let cmd = fields.get("cmd").and_then(Value::as_str).unwrap_or_default().to_owned();
        Ok(match cmd.as_str() {
            "hello" => Command::Hello,
            "shutdown" => Command::Shutdown,
            "start" => Command::Start(one(fields)?),
            "load" => Command::Load(Box::new(one(fields)?)),
            "activate" => Command::Activate(one(fields)?),
            "prepare_request" => Command::PrepareRequest(one(fields)?),
            "invoke" => Command::Invoke(Box::new(one(fields)?)),
            "map" => Command::Map(one(fields)?),
            "unload" => Command::Unload(one(fields)?),
            "probe" => Command::Probe(one(fields)?),
            "fork" => Command::Fork(one(fields)?),
            "bearer" => Command::Bearer(one(fields)?),
            other => return Err(format!("unknown command `{other}`")),
        })
    }

    /// Its `cmd`: what its reply names.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Hello => "hello",
            Self::Shutdown => "shutdown",
            Self::Start(_) => "start",
            Self::Load(_) => "load",
            Self::Activate(_) => "activate",
            Self::PrepareRequest(_) => "prepare_request",
            Self::Invoke(_) => "invoke",
            Self::Map(_) => "map",
            Self::Unload(_) => "unload",
            Self::Probe(_) => "probe",
            Self::Fork(_) => "fork",
            Self::Bearer(_) => "bearer",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct Start {
    /// The session's GPUs; empty runs on the CPU and never opens CUDA.
    #[serde(default)]
    pub devices: String,
    #[serde(default)]
    pub application: String,
    #[serde(default)]
    pub package_interface: PathBuf,
    #[serde(default = "one")]
    pub sequence_parallel_degree: u32,
    /// Import torch, the Runtime and the package and stop: no device is touched.
    #[serde(default, skip_serializing_if = "is_false")]
    pub import_only: bool,
    /// What its attempts may call: the Runtime installs their callers before it imports the
    /// package and brokers each attempt's calls.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub call_interfaces: Vec<CallInterface>,
    /// Each constructed slot's checkpoint, so the Runtime schedules fetches before it imports.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constructions: Vec<Construction>,
    /// The namespace (TensorD's uid) each GPU's lease listener (`lease-<uuid>`) is named in, for
    /// an executor speaking the lease protocol; a GPU nobody serves runs its own pages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leases: Option<u32>,
    /// The memory contract it serves: the lease protocol, or 0 (its own memory).
    #[serde(default)]
    pub memory_protocol: u32,
    /// Its stage memo tiers, when the machine memoizes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memo: Option<MemoSettings>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct Load {
    pub construction: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensorfs: Option<TensorFSOptions>,
    /// The request's signer: the Runtime keeps its bearer per signer and endpoint.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub actor: String,
    #[serde(default)]
    pub devices: String,
    #[serde(default = "one")]
    pub sequence_parallel_degree: u32,
    #[serde(default)]
    pub binding: Binding,
    /// Several model slots in one construction (H3 turbo: base + LoRA); the Runtime builds these
    /// models and loads their granted weights, ignoring `binding`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<ModelLoad>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub attention_pin: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct Activate {
    pub construction: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct PrepareRequest {
    pub request_id: String,
    #[serde(default)]
    pub construction: String,
    pub entrypoint: String,
    pub payload: Json,
    /// Checked before entry, so a bad pin is the request's refusal, not a fault.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub attention_kernel: String,
    /// The request's weight precision (16, 8 or 4; 0: the rules choose).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub bits: u32,
    /// File inputs by field path (identity only; the bytes arrive with Invoke).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub input_metadata: BTreeMap<String, InputMetadata>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct Invoke {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensorfs: Option<TensorFSOptions>,
    /// Retained asset capacity derived from the installed result schema.
    #[serde(default = "max_output_bytes")]
    pub max_output_bytes: u64,
    pub request_id: String,
    #[serde(default)]
    pub construction: String,
    pub entrypoint: String,
    pub spool: PathBuf,
    #[serde(default)]
    pub deadline_s: Option<f64>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub attention_kernel: String,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub bits: u32,
    /// File inputs by field path: read-only copies in this call's spool.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub inputs: BTreeMap<String, InputFile>,
    /// Input trees by reference: (read-only directory, manifest digest).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub trees: BTreeMap<String, (PathBuf, String)>,
    /// Each `Weights[M]` parameter's checkpoint (`sha256:<hex>`, its header), which the Runtime
    /// opens from the store itself.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sources: BTreeMap<String, String>,
    /// The weights outputs this attempt may write, and each one's granted new bytes.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub weights: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub destination: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub actor: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub operation: String,
    /// The family whose GPUs its calls share: while it waits on a child call it yields under
    /// this name, so the child is admitted beside it.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub family: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct Map {
    pub construction: String,
    #[serde(default)]
    pub level: Level,
}

/// What a warmed construction's models hold: `gpu` is warm and fills free room now, `ram`
/// leaves its resident pages idle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    #[default]
    Gpu,
    Ram,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct Unload {
    pub construction: String,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct Probe {
    /// Collect cyclic garbage first: a cycle holding a device tensor is not a leak.
    #[serde(default)]
    pub collect: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct Fork {
    pub socket: PathBuf,
    pub root: PathBuf,
    /// The child's whole environment, as the machine seals a spawned executor's.
    pub environment: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct Bearer {
    #[serde(default)]
    pub actor: String,
    pub tensorfs: TensorFSOptions,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
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

/// One model's construction: its class, where its weights are, and how to fill them. A
/// weightless binding names only its release and an empty `model_class`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
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
    /// `canonical` or `local`.
    pub custody: String,
    pub placement: String,
    pub package: String,
    pub model: String,
    pub adapters: Vec<Adapter>,
    /// A group's placeable components and the GPU that keeps each, an index within the group
    /// (0: GPU 0, the serial scope's; g: GPU g keeps it resident for the group), decided by
    /// TensorD at Load.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub homes: BTreeMap<String, u32>,
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
            placement: String::new(),
            package: String::new(),
            model: String::new(),
            adapters: Vec::new(),
            homes: BTreeMap::new(),
        }
    }
}

/// One model of a many-model construction (an entrypoint with several model slots).
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct ModelLoad {
    #[serde(default)]
    pub binding: Binding,
}

/// An executor's stage memo tiers (`memo: stage/1`): its in-memory tier (-1: its own default;
/// 0: off) and the largest entry the machine keeps (0: that tier off).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
#[serde(default)]
pub struct MemoSettings {
    pub process_bytes: i64,
    pub entry_bytes: u64,
}
impl Default for MemoSettings {
    fn default() -> Self {
        Self { process_bytes: -1, entry_bytes: 0 }
    }
}

/// One constructed model slot of a session: its checkpoint and the components it uses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct Construction {
    /// `<fn>.models.<param>`
    pub path: String,
    /// `sha256:<hex>`
    pub manifest: String,
    /// The slot's `components=`; empty is every component.
    #[serde(default)]
    pub components: Vec<String>,
}

/// One callable an attempt may call: its own package's, or another App's of its environment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct CallInterface {
    pub module: String,
    pub export: String,
    #[serde(default)]
    pub interface_path: PathBuf,
    #[serde(default, rename = "self")]
    pub self_call: bool,
    /// Another package's interface: the Runtime installs its typed callers from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interface_document: Option<Json>,
    /// A Runtime built-in surface (`operations`): the Runtime binds it from its own code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub builtin: Option<String>,
}

/// One file input's identity, before its bytes are granted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct InputMetadata {
    pub input_id: String,
    pub media_type: String,
    pub digest: String,
    pub length: u64,
    #[serde(default)]
    pub order: u64,
}

/// One verified input file in the attempt's own spool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct InputFile {
    pub local: PathBuf,
    pub media_type: String,
    pub digest: String,
    pub length: u64,
    #[serde(default)]
    pub order: u64,
}

// --------------------------------------------------------------------------- the executor's frames

/// A frame the executor sends unasked: `progress` or `request`; a reply has none.
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
    Publish,
    /// A weights output's TensorFS reference, which the Runtime committed or uploaded.
    CheckpointItem,
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
    /// A `progress` event carrying a retained `log` observation (`Telemetry.log`).
    Log,
    /// A `progress` event telling the executor's work count (`work`): what its Runtime did for
    /// its calls (blocks entered, graphs replayed, LM steps, construction legs).
    Work,
    #[default]
    #[serde(other)]
    Unknown,
}

/// An executor's `hello` reply: who it is, its versions, its memory protocol and its seal.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Hello {
    pub pid: u32,
    pub ppid: u32,
    pub pgid: u32,
    pub runtime_version: String,
    pub tensorfs_version: String,
    /// Its memory wire contract: the lease protocol leases chunks from each GPU's manager; any
    /// other runs its memory alone.
    pub memory_protocol: u32,
    pub sealed: BTreeMap<String, String>,
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

/// The Runtime's residency report on an invoke reply: its notes and, per model, its graph
/// regions' counters (captures, replays, pool bytes, eager gaps). Relayed whole: fields this
/// side does not know are kept.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Residency {
    pub notes: serde_json::Map<String, Value>,
    pub graphs: BTreeMap<String, serde_json::Map<String, Value>>,
    #[serde(flatten)]
    pub other: serde_json::Map<String, Value>,
}

/// One part of a composite product: a spool file and the media time it adds.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct PublishPart {
    pub local: String,
    pub duration_us: u64,
}

/// One decode per frame of whatever the executor sends (a reply, progress or a request); only
/// consumed business fields are represented, and every one is optional.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Frame {
    pub event: Option<Event>,
    /// A request or progress frame whose fields did not decode: why. A request is refused alone
    /// under its own sequence number and a progress frame is dropped; the executor goes on.
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
    pub sha256: String,
    pub length: u64,
    /// `load` and `invoke` telemetry: one of a shape this side does not read is absent, never the
    /// reply's failure.
    #[serde(deserialize_with = "lenient")]
    pub facts: Option<LoadFacts>,
    #[serde(deserialize_with = "lenient")]
    pub metrics: Option<Metrics>,
    /// `invoke`: varena's record of the request (`Stats`), kept whole; another frame's `memory`
    /// of another shape reads as absent.
    #[serde(deserialize_with = "lenient")]
    pub memory: Option<serde_json::Map<String, Value>>,
    /// `invoke`: the Runtime's residency report, relayed into the run's measurements; one of
    /// another shape reads as absent.
    #[serde(deserialize_with = "lenient")]
    pub residency: Option<Residency>,
    pub stage: String,
    pub position: Option<u64>,
    pub total: Option<u64>,
    pub stage_fraction: Option<f64>,
    pub overall_fraction: Option<f64>,
    pub step_ms: Option<f64>,
    pub advance: u64,
    /// `work`: the executor's work count, which moves while it works for its calls.
    pub work: u64,
    /// `progress`: the child call whose stage a job relays.
    pub call_request: Option<String>,
    /// A retained observation (`kind` log): `name` is its message, `value` its level and `fields`
    /// its scalars.
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
    /// `invoke`: the executor's stage and step tracks, its ranks' execution records and its
    /// observation tail; read leniently, kept as the run's measurements.
    pub attribution: Value,
    pub execution: Value,
    pub observations: Value,
    /// `invoke`: some attention site of the call applied the request's pin.
    pub attention_applied: bool,
    /// `invoke`: the rows of how the request was served other than as sent; read row by row.
    pub adjustments: Value,
    // `child_*` (a job's managed calls): the call's index, callee and canonical request.
    pub call_index: u64,
    pub module: String,
    pub export: String,
    pub payload: String,
    pub progress_label: String,
    // `stage_memo_*` (a memoized Model method): its key and scope, and on a store the spool file
    // it offers with what producing it cost.
    pub key: String,
    pub numerics: String,
    pub local: String,
    pub cost_ms: f64,
    /// `checkpoint_item`: the immutable TensorFS reference the Runtime committed or uploaded.
    pub repo_id: String,
    pub revision: String,
    pub variant: String,
    pub tag: String,
}

impl Frame {
    /// A `log` observation (`Telemetry.log`) as its log line: the level (debug, info, warning or
    /// error) and the message followed by ` key=value` for each field.
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

// --------------------------------------------------------------------------- TensorD's answers

/// TensorD's answer to one executor request, under its `seq`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
#[serde(default)]
pub struct Answer {
    pub event: String,
    pub seq: u64,
    pub ok: bool,
    pub code: String,
    pub detail: String,
    /// One fd follows the answer. Sent only when true.
    #[serde(skip_serializing_if = "is_false")]
    pub descriptor: bool,
    pub sha256: String,
    pub length: u64,
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
    pub byte_grants: Vec<ByteGrant>,
    /// Advisory: a progress of a shape this side does not read is absent, never the answer's
    /// failure.
    #[serde(skip_serializing_if = "Option::is_none", deserialize_with = "lenient")]
    pub progress: Option<CallProgress>,
    /// A failed child call's raising traceback, whole (its triage's).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub traceback: String,
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
}

impl Default for Answer {
    fn default() -> Self {
        Self {
            event: "answer".into(),
            seq: 0,
            ok: false,
            code: String::new(),
            detail: String::new(),
            descriptor: false,
            sha256: String::new(),
            length: 0,
            digest: String::new(),
            sequence: 0,
            state: String::new(),
            child_request_id: String::new(),
            result: String::new(),
            byte_grants: Vec::new(),
            progress: None,
            traceback: String::new(),
            local: String::new(),
            cost_ms: 0.0,
            disputed: false,
            stored: false,
            reason: String::new(),
        }
    }
}

impl Answer {
    pub fn unavailable(seq: u64) -> Self {
        Self {
            seq,
            code: "capability_unavailable".into(),
            detail: "owner has not qualified this operation".into(),
            ..Self::default()
        }
    }
    pub fn ok(seq: u64) -> Self {
        Self { ok: true, code: String::new(), detail: String::new(), ..Self::unavailable(seq) }
    }
    pub fn refused(seq: u64, code: &str, detail: impl Into<String>) -> Self {
        Self { code: code.into(), detail: detail.into(), ..Self::unavailable(seq) }
    }
}

/// A child's result byte, granted into the caller's spool: what the caller checks against the
/// result's own record before it hands out a handle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct ByteGrant {
    pub output_id: String,
    pub kind: String,
    pub digest: String,
    pub local: PathBuf,
    pub length: u64,
    pub content_bytes: u64,
    #[serde(default)]
    pub media_type: String,
}

/// A managed call's newest progress: its sequence and the child's progress frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "tensorfs.executor", frozen, eq, get_all, skip_from_py_object))]
pub struct CallProgress {
    pub sequence: u64,
    pub payload: Json,
}

// --------------------------------------------------------------------------- helpers

/// A field of the shape this side reads, else absent: never a reason to refuse the frame.
fn lenient<'de, D: serde::Deserializer<'de>, T: serde::de::DeserializeOwned>(d: D) -> Result<Option<T>, D::Error> {
    Ok(serde_json::from_value(Value::deserialize(d)?).ok())
}
fn is_zero<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}
fn is_false(value: &bool) -> bool {
    !*value
}
fn is_zero_ms(value: &f64) -> bool {
    *value == 0.0
}
fn one() -> u32 {
    1
}
/// The Runtime's own default retained-output capacity (`MAX_OUTPUT_BYTES`).
fn max_output_bytes() -> u64 {
    256 << 20
}

/// A JSON document carried as it is (a request's payload, an interface): Python reads it as plain
/// dicts, lists and scalars.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Json(pub Value);

#[cfg(feature = "python")]
mod python {
    use pyo3::{
        exceptions::PyValueError,
        prelude::*,
        types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
        IntoPyObjectExt,
    };
    use serde_json::Value;

    use super::*;

    fn value<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
        Ok(match value {
            Value::Null => py.None().into_bound(py),
            Value::Bool(b) => b.into_bound_py_any(py)?,
            Value::Number(n) => match (n.as_i64(), n.as_u64()) {
                (Some(i), _) => i.into_bound_py_any(py)?,
                (_, Some(u)) => u.into_bound_py_any(py)?,
                _ => n.as_f64().unwrap_or(f64::NAN).into_bound_py_any(py)?,
            },
            Value::String(s) => s.into_bound_py_any(py)?,
            Value::Array(items) => {
                let list = PyList::empty(py);
                for item in items {
                    list.append(self::value(py, item)?)?;
                }
                list.into_any()
            }
            Value::Object(fields) => {
                let dict = PyDict::new(py);
                for (key, item) in fields {
                    dict.set_item(key, self::value(py, item)?)?;
                }
                dict.into_any()
            }
        })
    }

    /// A Python value as JSON: a class of this module as its own wire JSON, a path as its text,
    /// else dicts, lists and scalars.
    fn json_of(value: &Bound<'_, PyAny>) -> PyResult<Value> {
        let invalid = |what: &str| PyValueError::new_err(format!("{what} has no JSON form"));
        Ok(if value.is_none() {
            Value::Null
        } else if let Ok(b) = value.cast::<PyBool>() {
            Value::Bool(b.is_true())
        } else if value.is_instance_of::<PyInt>() {
            match value.extract::<i64>() {
                Ok(i) => i.into(),
                Err(_) => value.extract::<u64>()?.into(),
            }
        } else if value.is_instance_of::<PyFloat>() {
            serde_json::Number::from_f64(value.extract::<f64>()?).map(Value::Number).ok_or_else(|| invalid("a non-finite float"))?
        } else if let Ok(s) = value.cast::<PyString>() {
            Value::String(s.to_cow()?.into_owned())
        } else if let Ok(d) = value.cast::<PyDict>() {
            let mut fields = serde_json::Map::new();
            for (key, item) in d.iter() {
                fields.insert(key.extract::<String>()?, json_of(&item)?);
            }
            Value::Object(fields)
        } else if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
            Value::Array(value.try_iter()?.map(|item| json_of(&item?)).collect::<PyResult<_>>()?)
        } else if value.hasattr("__fspath__")? {
            Value::String(value.call_method0("__fspath__")?.extract::<String>()?)
        } else if value.hasattr("json")? && value.get_type().module()?.to_cow()? == "tensorfs.executor" {
            serde_json::from_str(&value.call_method0("json")?.extract::<String>()?).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return Err(invalid(&value.get_type().name()?.to_cow()?));
        })
    }

    /// `this` with `changes` in place of its fields, through serde.
    fn replaced<T: Serialize + serde::de::DeserializeOwned>(this: &T, changes: Option<&Bound<'_, PyDict>>) -> PyResult<T> {
        match serde_json::to_value(this).map_err(|e| PyValueError::new_err(e.to_string()))? {
            Value::Object(fields) => merged(fields, changes),
            _ => Err(PyValueError::new_err("not a record")),
        }
    }

    /// The record `fields` and `changes` make, through serde: a change of the wrong shape is
    /// refused by its path, one naming no field of the record by its name, and a needed field
    /// left out by its name.
    fn merged<T: Serialize + serde::de::DeserializeOwned>(
        mut fields: serde_json::Map<String, Value>,
        changes: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<T> {
        let error = |e: &dyn std::fmt::Display| PyValueError::new_err(e.to_string());
        let mut given = vec![];
        for (key, value) in changes.into_iter().flat_map(|d| d.iter()) {
            let (key, value) = (key.extract::<String>()?, json_of(&value)?);
            fields.insert(key.clone(), value.clone());
            given.push((key, value));
        }
        let result: T = serde_path_to_error::deserialize(Value::Object(fields)).map_err(|e| error(&e))?;
        let after = serde_json::to_value(&result).map_err(|e| error(&e))?;
        // A field serde leaves out when empty reads back absent; any other change must read back.
        let empty = |v: &Value| match v {
            Value::Null => true,
            Value::Bool(b) => !b,
            Value::Number(n) => n.as_f64() == Some(0.0),
            Value::String(s) => s.is_empty(),
            Value::Array(a) => a.is_empty(),
            Value::Object(o) => o.is_empty(),
        };
        if let Some((key, _)) = given.iter().find(|(key, value)| after.get(key).is_none() && !empty(value)) {
            return Err(PyValueError::new_err(format!("no field `{key}`")));
        }
        Ok(result)
    }

    impl<'py> IntoPyObject<'py> for &Json {
        type Target = PyAny;
        type Output = Bound<'py, PyAny>;
        type Error = PyErr;
        fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
            value(py, &self.0)
        }
    }

    impl<'py> IntoPyObject<'py> for Json {
        type Target = PyAny;
        type Output = Bound<'py, PyAny>;
        type Error = PyErr;
        fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
            value(py, &self.0)
        }
    }

    /// A level reads as its wire name, `gpu` or `ram`.
    impl<'py> IntoPyObject<'py> for &Level {
        type Target = PyAny;
        type Output = Bound<'py, PyAny>;
        type Error = PyErr;
        fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
            match self {
                Level::Gpu => "gpu",
                Level::Ram => "ram",
            }
            .into_bound_py_any(py)
        }
    }

    macro_rules! methods {
        ($($t:ty),*) => {$(
            #[pymethods]
            impl $t {
                fn __repr__(&self) -> String {
                    format!("{self:?}")
                }
                /// Its wire JSON.
                fn json(&self) -> PyResult<String> {
                    serde_json::to_string(self).map_err(|e| PyValueError::new_err(e.to_string()))
                }
                /// A copy with `changes` in place of its fields (`msgspec.structs.replace`).
                #[pyo3(signature = (**changes))]
                fn replace(&self, changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
                    replaced(self, changes)
                }
                /// One built from `fields` by name: a field left out takes its default; a needed one
                /// left out is refused by its name.
                #[new]
                #[pyo3(signature = (**fields))]
                fn new(fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
                    merged(serde_json::Map::new(), fields)
                }
            }
        )*};
    }
    methods!(
        Adapter, Binding, ModelLoad, MemoSettings, Construction, CallInterface, InputMetadata, InputFile,
        ByteGrant, CallProgress, Answer
    );

    macro_rules! command_methods {
        ($($t:ident => $wrap:expr),*) => {$(
            #[pymethods]
            impl $t {
                fn __repr__(&self) -> String {
                    format!("{self:?}")
                }
                /// Its frame, `cmd` included.
                fn json(&self) -> PyResult<String> {
                    let wrap: fn($t) -> Command = $wrap;
                    serde_json::to_string(&wrap(self.clone())).map_err(|e| PyValueError::new_err(e.to_string()))
                }
                /// A copy with `changes` in place of its fields (`msgspec.structs.replace`).
                #[pyo3(signature = (**changes))]
                fn replace(&self, changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
                    replaced(self, changes)
                }
                /// One built from `fields` by name: a field left out takes its default; a needed one
                /// left out is refused by its name.
                #[new]
                #[pyo3(signature = (**fields))]
                fn new(fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
                    merged(serde_json::Map::new(), fields)
                }
            }
        )*};
    }
    command_methods!(
        Start => Command::Start,
        Load => |c| Command::Load(Box::new(c)),
        Activate => Command::Activate,
        PrepareRequest => Command::PrepareRequest,
        Invoke => |c| Command::Invoke(Box::new(c)),
        Map => Command::Map,
        Unload => Command::Unload,
        Probe => Command::Probe,
        Fork => Command::Fork,
        Bearer => Command::Bearer
    );
}
