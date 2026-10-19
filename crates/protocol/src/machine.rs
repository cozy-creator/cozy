//! The machine API (design/machine-http-api.md): HTTP + JSON on the machine's one pinned-TLS port,
//! every request carrying `authorization: Cozy-Cap <token>`. Live state and run events are SSE;
//! files move by ranged `PUT`/`HEAD`/`GET`. Refusals are [`crate::error::Envelope`]. Unknown fields
//! are ignored, and clients skip SSE events they do not know.
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

pub mod api;
pub mod client;

/// Routes, as axum templates.
pub mod path {
    pub const HEALTH: &str = "/v1/health";
    /// `GET`: one JSON [`super::StatusFrame`]; SSE `status` frames with `Accept: text/event-stream`.
    pub const STATUS: &str = "/v1/status";
    /// `POST`: renews a rental's idle deadline; answers the frame.
    pub const KEEPALIVE: &str = "/v1/keepalive";
    /// `POST` a [`super::RunRequest`].
    pub const RUNS: &str = "/v1/runs";
    /// `GET`: SSE of the run's [`super::RunEvent`]s after `Last-Event-ID`.
    pub const EVENTS: &str = "/v1/runs/{id}/events";
    pub const CANCEL: &str = "/v1/runs/{id}/cancel";
    /// `GET` with `Range` and `If-Match`; `?index=&member=`.
    pub const OUTPUT: &str = "/v1/runs/{id}/outputs/{output}";
    pub const TRIAGE: &str = "/v1/runs/{id}/triage";
    /// `GET` with `?tail=`.
    pub const LOG: &str = "/v1/logs/{name}";
    /// `PUT` with `Content-Range`, `HEAD`; `{digest}` is `sha256:<hex>`.
    pub const OBJECT: &str = "/v1/objects/{digest}";
    /// `POST` a [`super::HeldRequest`]: what the machine holds of many objects, in one request.
    pub const OBJECTS_HELD: &str = "/v1/objects/held";
}

/// The header naming the byte ranges of an object the machine holds: `0-1048575,2097152-4194303`
/// (inclusive, as `Content-Range` counts); empty when it holds none.
pub const HELD: &str = "cozy-held";

/// Byte ranges `[start, end)`, sorted and apart.
pub type Ranges = Vec<(u64, u64)>;

/// `ranges` as [`HELD`] spells them.
pub fn spell_held(ranges: &[(u64, u64)]) -> String {
    let spelled: Vec<_> = ranges
        .iter()
        .filter(|(start, end)| end > start)
        .map(|(start, end)| format!("{start}-{}", end - 1))
        .collect();
    spelled.join(",")
}

/// A [`HELD`] value as ranges, or None when it is malformed.
pub fn parse_held(value: &str) -> Option<Ranges> {
    let mut ranges: Ranges = vec![];
    for range in value.split(',').map(str::trim).filter(|r| !r.is_empty()) {
        let (first, last) = range.split_once('-')?;
        let (first, last): (u64, u64) = (first.parse().ok()?, last.parse().ok()?);
        if last < first || ranges.last().is_some_and(|(_, end)| first <= *end) {
            return None;
        }
        ranges.push((first, last + 1));
    }
    Some(ranges)
}

/// The bytes `ranges` cover.
pub fn held_bytes(ranges: &[(u64, u64)]) -> u64 {
    ranges
        .iter()
        .map(|(start, end)| end.saturating_sub(*start))
        .sum()
}

/// What of `[0, length)` `ranges` does not cover.
pub fn missing(ranges: &[(u64, u64)], length: u64) -> Ranges {
    let mut gaps = vec![];
    let mut at = 0;
    for &(start, end) in ranges {
        if start > at {
            gaps.push((at, start.min(length)));
        }
        at = at.max(end);
    }
    if at < length {
        gaps.push((at, length));
    }
    gaps.retain(|(start, end)| end > start);
    gaps
}

/// One output of a run, as `GET /v1/runs/{id}/outputs/{output}?index=&member=` names it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct OutputTarget {
    pub run: String,
    pub output: String,
    /// A list item's 1-based index; 0 for a single output.
    pub index: u32,
    /// One file of a tree output, by its manifest path; empty: the manifest.
    pub member: String,
}

/// `POST /v1/objects/held`: which of these objects (`sha256:<hex>`) the machine holds for this
/// signer.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct HeldRequest {
    pub digests: Vec<String>,
}

/// What the machine holds of each object asked, in the order asked; a stored object is held
/// whole, and asking renews it, as a `HEAD` does.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct HeldAnswer {
    pub held: Vec<HeldObject>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct HeldObject {
    pub digest: String,
    /// Byte ranges `[start, end)`; none: nothing held.
    #[schema(value_type = Vec<Vec<u64>>)]
    pub ranges: Ranges,
}

/// What an upload holds, as `GET /v1/objects/sha256:<hex>?length=N` streams it (`event: held`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct HeldBytes {
    /// Bytes landed, synced or not; the last event says all N.
    pub held: u64,
}

/// Whole lines of a machine log, as `GET /v1/logs/{name}` streams them (`event: lines`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct LogLines {
    /// The place after them, also the event's id: a follow resumed there goes on with the next line.
    pub after: String,
    pub text: String,
}

/// Raw JSON, kept as its exact text: canonical bytes and 64-bit integers survive.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Json(pub Box<serde_json::value::RawValue>);

impl Json {
    /// `text` if it is JSON.
    pub fn parse(text: impl Into<String>) -> Result<Self, serde_json::Error> {
        serde_json::value::RawValue::from_string(text.into()).map(Self)
    }
    pub fn get(&self) -> &str {
        self.0.get()
    }
}

impl PartialEq for Json {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl utoipa::PartialSchema for Json {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        utoipa::openapi::schema::Object::new().into()
    }
}
impl ToSchema for Json {}

// ---- Run ------------------------------------------------------------------------------------

/// `POST /v1/runs`: submits when `id` is new (idempotent on id per signer); the same id with
/// another spec is `run_id_conflict`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct RunRequest {
    /// The caller's run id, 1-256 bytes.
    pub id: String,
    pub spec: RunSpec,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum RunKind {
    /// Every function is a call.
    #[default]
    Call,
    /// Prepare only: install and download, then succeed.
    Warm,
    /// Stage, verify and activate a software cohort at idle.
    Update,
    /// `remove/1`: the caller's installations of the source leave the machine: an installation by
    /// its name, or a release's package (its release, or every release when empty) from the run's
    /// Hub (local/ packages from none). Their environments are deleted once no process and no
    /// other installation holds them.
    Remove,
}

/// How far a function is kept ready; each level includes the ones before it.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum WarmLevel {
    /// Not kept: a warm set's item at this level leaves it.
    #[default]
    Off,
    /// The package environment.
    Installed,
    /// The function's bound weights in the store.
    Downloaded,
    /// A process with its imports done: no device, no VRAM.
    Imported,
    /// An executor with the model constructed, weights in the host tier.
    Host,
    /// Its weights on the GPU.
    Gpu,
}

/// What a run runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Release(Release),
    /// An installation this machine already holds for the signer.
    Installation(String),
    Local(LocalSource),
}

/// What a warm set's member keeps ready.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum WarmSource {
    Release(Release),
    Installation(String),
}

/// One member of a caller's warm set: a function the machine keeps ready up to `level`. It is
/// the last idle thing to give its memory to a request, and comes back when room returns.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct WarmItem {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<WarmSource>,
    pub entrypoint: String,
    /// None: the slots' defaults, as for a run.
    pub models: Vec<ModelChoice>,
    pub level: WarmLevel,
    /// Status only: the level it holds now (installed, downloaded, imported, host, gpu).
    pub holds: String,
    /// Status only: why that is lower than `level`.
    pub held_back: String,
    /// As [`RunSpec::bits`].
    pub bits: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct WarmSet {
    pub items: Vec<WarmItem>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct RunSpec {
    pub kind: RunKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
    pub entrypoint: String,
    /// The function's parameters, a JSON object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<Json>,
    /// Objects uploaded with `PUT /v1/objects`, bound to payload fields.
    pub inputs: Vec<InputFile>,
    pub models: Vec<ModelChoice>,
    /// The catalog revision the caller made its choices under (th-245): a held resolution is
    /// reused only under the same one, so a publish, yank, bind or retarget the caller knows of
    /// resolves once more at the Hub.
    pub catalog_revision: String,
    pub attention_kernel: String,
    /// org/name: the requester's repository for the call's weights outputs, or a warm run's
    /// source model.
    pub weights_destination: String,
    /// Where the run reads, and its capability there.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hub: Option<HubAccess>,
    /// The account that org-relative references of local code name.
    pub owner: String,
    /// Provider tokens for source models, memory only like `hub`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub providers: Option<ProviderAccess>,
    /// `warm/2`: the caller's whole warm set, replacing its previous one; no items clears it.
    /// The run installs and downloads every member, and its result lists each one's `level` and
    /// `held_back`. Absent: the set is left as it is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub set: Option<WarmSet>,
    /// 0: widest group every model slot supports on this machine. Positive: at most this many
    /// GPUs, for the run and its children.
    pub gpus: u32,
    /// The run's weight precision: 16, 8 or 4. 0: the machine's rules choose each slot's lane.
    /// Children inherit it.
    pub bits: u32,
}

/// Dev builds only (xs-045 U12): the profile that makes a warm run's model choice a migration
/// of checkpoints from before TensorFS's format cut, which every Calcifer passes through. With a
/// repository and manifest: that old checkpoint at the run's Hub, rewritten on the machine and
/// put in the run's weights destination. With a `source`: every `local/` model of the pre-cut
/// store at that path on the machine, into the machine's own store.
#[cfg(feature = "migrate")]
pub const MIGRATE: &str = "migrate";

/// The caller's provider tokens, for models a run makes from a provider source.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct ProviderAccess {
    pub huggingface: String,
    pub civitai: String,
}

/// A published release by name (th-245): a machine that holds no installation of it reads its
/// card and locked requirements at the Hub by name.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Release {
    /// org/name
    pub package: String,
    /// Empty: the Hub's newest.
    pub release: String,
}

/// Local code: `manifest` names a JSON object written with `PUT /v1/objects`:
/// `{package: "local/<dist>", release, python_requires, python_version, callees: {distribution:
/// package}}` and its code. A live project is installed editable from `files: [{name, digest,
/// length, executable}]`: the project and the path dependencies its lock names, each file under
/// their common directory, with `project` the project's directory there. The machine keeps one
/// such tree and one environment per signer and package. `path` names that common directory on
/// this computer's machine, which reads it in place. `wheels: [{name, digest, length}]` are locked
/// dependencies the caller carries because the machine cannot fetch them. Members are objects the
/// same signer wrote.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct LocalSource {
    /// `sha256:<hex>` of the manifest object.
    pub manifest: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct InputFile {
    /// The payload field path, list index included.
    pub field: String,
    /// `sha256:<hex>`
    pub digest: String,
    pub length: u64,
    pub media_type: String,
    pub order: u32,
}

/// One model slot's choice, by name (th-245: the machine resolves it at the Hub). A Hub model is
/// `repository` at `release` (empty: the newest) in `lane`, else in the lane the machine's rules
/// pick (xs-042). `manifest` names a checkpoint by hash only: the owner's private one at the Hub
/// (with `repository`; the run's capability names it) or one this machine holds (without). A slot
/// with no choice takes its declared default release, a callee's included (`<package>/<path>`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct ModelChoice {
    /// A root slot parameter or full slot path.
    pub parameter: String,
    pub repository: String,
    pub release: String,
    pub lane: String,
    /// `sha256:<hex>`
    pub manifest: String,
    pub manifest_length: u64,
    /// Each by name, a hash of the owner's, or a provider source.
    pub adapters: Vec<Adapter>,
    /// A provider source (`hf://<org>/<repo>@<rev>[/member]`, `civitai://<version>[/file]`) the
    /// machine makes the slot's model from, under these reviewed `profiles` (none: the one its
    /// headers select); never with repository or manifest.
    pub source: String,
    pub profiles: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Adapter {
    pub component: String,
    pub model: String,
    pub release: String,
    pub lane: String,
    /// `sha256:<hex>`, the owner's checkpoint by hash; else `model` at `release`, `lane`.
    pub manifest: String,
    /// Canonical decimal; empty means 1.
    pub scale: String,
    /// A provider source (`civitai://<version>`, `hf://…`) made here and normalized at ingest,
    /// under these reviewed `profiles` (none: the one its headers select); never with `model`.
    pub source: String,
    pub profiles: Vec<String>,
}

/// A Hub a run reads. Public content is read anonymously.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct HubAccess {
    pub origin: String,
    /// An origin's private CA (DER, base64), when it has one.
    #[serde(with = "base64_bytes", skip_serializing_if = "Vec::is_empty")]
    #[schema(value_type = String, format = Byte)]
    pub ca_der: Vec<u8>,
    /// The Hub's object-storage hosts downloads may follow.
    pub object_hosts: Vec<String>,
    /// The owner's capability for this run's private operations (th-241): a compact JWS its
    /// device key signed for the per-run key (`cnf.jkt`) and the Hub's resource (`aud`).
    /// The Runtime (or TensorD for code-less runs) trades it once on first use.
    pub capability: String,
    /// The private P-256 JWK this request proves with. Empty for public-only runs.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub key: String,
    /// A token already issued for this request's key by its metadata helper.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub access_token: String,
    /// The Hub's authorization server's, as its metadata names it.
    pub token_endpoint: String,
}

/// One entry of a run's log, as `GET /v1/runs/{id}/events` sends it: SSE `event: <variant>`,
/// `id: <sequence>` for durable entries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct RunEvent {
    /// 0: a frame that is not in the log (the snapshot, progress before acceptance, live state).
    #[serde(default)]
    pub sequence: u64,
    #[serde(default)]
    pub at_ms: i64,
    pub event: Event,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    State(RunState),
    Progress(Progress),
    Product(Product),
    Log(LogLine),
    Outcome(Outcome),
    Call(Call),
    Gpu(GpuAllocation),
}

impl Event {
    /// Its SSE event name.
    pub fn name(&self) -> &'static str {
        match self {
            Event::State(_) => "state",
            Event::Progress(_) => "progress",
            Event::Product(_) => "product",
            Event::Log(_) => "log",
            Event::Outcome(_) => "outcome",
            Event::Call(_) => "call",
            Event::Gpu(_) => "gpu",
        }
    }
    /// The SSE event names a run's stream carries.
    pub const NAMES: [&'static str; 7] = [
        "state", "progress", "product", "log", "outcome", "call", "gpu",
    ];
}

/// A root run's GPU allocation changed: `grant` binds its lane, `wait` queues it behind `behind`
/// (a run id), `release` ends its lane (`Context.release_gpus()` or its exit).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct GpuAllocation {
    pub state: String,
    /// The lane, as machine GPU indices.
    pub gpus: Vec<u32>,
    pub width: u32,
    pub behind: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct RunState {
    pub id: String,
    /// This machine's run number.
    pub number: u64,
    /// queued, preparing, running, succeeded, failed, canceled
    pub state: String,
    /// The log's head.
    pub sequence: u64,
    pub attempt: u32,
    /// Why a queued run waits, for people.
    pub waiting: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Progress {
    pub stage: String,
    /// -1: unknown.
    pub fraction: f64,
    pub completed: u64,
    pub total: u64,
    pub bytes_done: u64,
    pub bytes_total: u64,
    /// The latest step's time, while a stage steps.
    pub step_ms: f64,
    /// How far this stage is, when it says.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_fraction: Option<f64>,
}

/// One revision of an output: a single output is replaced by each, a list output grows. Its bytes
/// are `GET /v1/runs/{id}/outputs/{output}`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Product {
    pub output: String,
    /// A list item's 1-based index; 0 for a single output.
    pub index: u32,
    pub rev: u64,
    pub length: u64,
    /// `sha256:<hex>` of these bytes.
    pub digest: String,
    pub media_type: String,
    pub label: String,
    /// Media time, for a growing video.
    pub duration_us: u64,
    /// These bytes extend the previous revision's, from here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub appended_from: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct LogLine {
    pub level: String,
    pub text: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Outcome {
    /// succeeded, failed, canceled
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<Reason>,
    /// The result, canonical JSON, assets as digests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Json>,
    pub outputs: Vec<Product>,
    /// `GET /v1/runs/{id}/triage` answers its bundle.
    pub triage: bool,
    /// What its executor measured, canonical JSON: `attribution` (stage and step tracks),
    /// `execution` (degree, ranks with the GPU and attention each served, executor boot,
    /// construction load and warm) and `observations` (attention rows). None for a run that ran
    /// on no device.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measurements: Option<Json>,
    /// Its own time running its callable, every attempt summed; 0: unknown.
    pub execution_ms: u64,
    /// A memoizable callable's computation (`sha256:<hex>`), whether it ran or was answered.
    pub computation_digest: String,
    /// Answered from a result this machine held: nothing ran.
    pub memoized: bool,
    /// Why it was neither looked up nor held (its author's condition unmet, e.g. "seed not given").
    pub unmemoized: String,
    /// How its executor served the request other than as sent, in the executor's order. Advisory.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub adjustments: Vec<Adjustment>,
}

/// One request field served other than as sent.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Adjustment {
    /// Its path in the request: `steps`, `assets.2.asset`.
    pub field: String,
    /// `adjusted`, `clamp` or `warn`.
    pub kind: String,
    pub requested: String,
    pub applied: String,
    pub reason: String,
    /// `handler`, or what the Runtime applied: `policy`, `checkpoint:…`, `adapter:…`.
    pub source: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Reason {
    pub code: String,
    pub message: String,
    /// request, runtime, author, machine
    pub origin: String,
    /// A failed call's traceback, else its executor's stderr: the last lines, secrets masked.
    /// It explains the failure after its machine is gone.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub detail: String,
}

/// One settled call the run made: a child run, or an effect such as its weights publication.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Call {
    /// The child run's id.
    pub run: String,
    /// Its index among the run's calls.
    pub index: u32,
    pub function: String,
    /// The author's label for it.
    pub label: String,
    /// succeeded, failed, canceled
    pub status: String,
    pub called_at_ms: i64,
    pub finished_at_ms: i64,
    /// Its executor's, as [`Outcome::measurements`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measurements: Option<Json>,
    /// Why it did not succeed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<Reason>,
    /// Answered from a result this machine held; nothing ran.
    pub memoized: bool,
    /// A memoized call's computation (`sha256:<hex>`), ran or answered.
    pub computation_digest: String,
    /// Why a memoizable call was neither looked up nor held ("seed not given").
    pub unmemoized: String,
}

// ---- Status ---------------------------------------------------------------------------------

/// The machine's picture. Without a machine-scope cap, `GET /v1/status` answers identity,
/// version, capabilities and the sealed readiness `receipt` only. With one it answers all of it,
/// and as SSE (`Accept: text/event-stream`) sends the whole frame again whenever it changes. An
/// open stream is not activity.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct StatusFrame {
    pub worker_id: String,
    pub boot_id: String,
    /// This machine's release.
    pub version: String,
    /// The commit its `cozy` binary was built from.
    pub commit: String,
    /// Its binary arrived by a dev update (`cozy machine update --tensord`), not a release.
    pub dev: bool,
    pub capabilities: Vec<String>,
    /// The sealed readiness envelope (base64); empty until sealed.
    #[serde(with = "base64_bytes")]
    #[schema(value_type = String, format = Byte)]
    pub receipt: Vec<u8>,
    /// An idle rental's release, 15 min after its last job; 0: none (a job is queued or running,
    /// or it never releases).
    pub idle_deadline_unix_ms: i64,
    /// The caller's live runs.
    pub runs: Vec<RunState>,
    /// booting, ready, releasing
    pub phase: String,
    /// os/arch
    pub platform: String,
    pub started_at_unix_ms: i64,
    /// Measured at boot.
    pub gpus: Vec<Gpu>,
    pub hubs: Vec<Hub>,
    /// The executors' cozy-runtime.
    pub runtime: String,
    pub tensorfs: String,
    /// The caller's held package installations.
    pub environments: Vec<Environment>,
    /// The filesystem holding the machine's store.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk: Option<Disk>,
    /// The models the machine's store holds.
    pub models: Vec<Model>,
    /// Their bytes together, shared content counted once.
    pub models_bytes: u64,
    /// The direct player endpoint, when the machine listens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webrtc: Option<WebRtc>,
    /// The caller's warm set, each with what it holds now.
    pub warm: Vec<WarmItem>,
    /// The oldest NVIDIA driver this machine's kernels run on; empty: no GPU stack.
    pub driver_floor: String,
    /// What builds kernels here when no prebuilt one serves the card.
    pub toolchain: Toolchain,
    /// Each native kernel's verdict on this machine's cards, as the Runtime keeps it.
    pub kernels: Vec<KernelVerdict>,
}

/// The machine's build toolchain: CUDA's compiler and a host C/C++ compiler.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Toolchain {
    /// ready, absent, installing, failed
    pub state: String,
    /// `nvcc --version`'s release line; empty: none found.
    pub nvcc: String,
    /// The host compiler's first version line; empty: none found.
    pub host_compiler: String,
    /// What it lacks: `nvcc`, `cc`, `ninja`.
    pub missing: Vec<String>,
    /// The OS's own command for a missing host compiler; empty when none is missing or known.
    pub install: String,
    /// Why the last install failed.
    pub detail: String,
}

/// One kernel's verification on this machine (the Runtime's `verify_op`), keyed by its
/// artifact, the card's architecture and the driver.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct KernelVerdict {
    pub kernel: String,
    pub artifact: String,
    pub sm: u32,
    pub driver: String,
    pub ok: bool,
    /// What was measured, or why it failed.
    pub detail: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct WebRtc {
    /// The listener; a persistent machine keeps it across restarts.
    pub port: u32,
    /// host:port to dial: own interfaces, or the provider's mapping.
    pub addresses: Vec<String>,
    /// SHA-256 hex of the certificate its DTLS presents.
    pub fingerprint: String,
    /// Why no address is reported (a malformed mapping).
    pub unavailable_reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Model {
    /// org/name; org "local" is a local alias.
    pub repository: String,
    /// Everything this repository reaches.
    pub total_bytes: u64,
    /// Reached by no other held repository.
    pub unique_bytes: u64,
    pub checkpoints: Vec<Checkpoint>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Checkpoint {
    /// Empty for a local alias.
    pub release: String,
    pub lane: String,
    /// `sha256:<hex>`
    pub manifest: String,
    pub yanked: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Gpu {
    pub index: u32,
    pub name: String,
    pub uuid: String,
    pub pci_bus_id: String,
    pub memory_bytes: u64,
    pub driver: String,
    /// The tensor-core row by compute capability ("sm90"); empty: unread.
    pub family: String,
    /// discrete, coherent (GH200, GB200), unified (GB10); empty: unread.
    pub memory_class: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Hub {
    pub origin: String,
    /// This machine's worker id there.
    pub machine_id: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Environment {
    /// [`Source::Installation`] names it.
    pub installation: String,
    pub package: String,
    pub release: String,
    pub entrypoints: Vec<String>,
    /// The cozy-runtime installed in it.
    pub runtime: String,
    /// What it holds now (as [`WarmItem::holds`]), in a warm set or not.
    pub level: String,
    /// Why it does not run the machine's own Runtime; empty when it does.
    pub warning: String,
    /// The Hub origin a published installation came from; empty for local code.
    pub hub: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct Disk {
    pub total_bytes: u64,
    pub free_bytes: u64,
}

/// Bytes as base64 text.
pub(crate) mod base64_bytes {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use serde::{Deserialize, Deserializer, Serializer, de::Error as _};

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        STANDARD
            .decode(String::deserialize(deserializer)?)
            .map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_ranges_spell_and_parse_back() {
        let ranges = vec![(0, 1 << 20), (2 << 20, 4 << 20)];
        assert_eq!(spell_held(&ranges), "0-1048575,2097152-4194303");
        assert_eq!(parse_held(&spell_held(&ranges)), Some(ranges.clone()));
        assert_eq!(parse_held(""), Some(vec![]));
        assert_eq!(parse_held("5-4"), None);
        assert_eq!(parse_held("0-9,5-20"), None, "overlapping");
        assert_eq!(parse_held("x"), None);
        assert_eq!(held_bytes(&ranges), 3 << 20);
        assert_eq!(
            missing(&ranges, 5 << 20),
            [(1 << 20, 2 << 20), (4 << 20, 5 << 20)]
        );
        assert_eq!(missing(&[], 7), [(0, 7)]);
        assert_eq!(missing(&[(0, 7)], 7), []);
    }

    /// A spec's payload keeps its exact text, unknown fields are ignored, and an event is named
    /// by its variant.
    #[test]
    fn the_wire_keeps_exact_json_ignores_unknown_fields_and_names_events() {
        let payload = r#"{"seed":9007199254740993,"nested":{"b":2,"a":1}}"#;
        let request = RunRequest {
            id: "r1".into(),
            spec: RunSpec {
                kind: RunKind::Warm,
                source: Some(Source::Release(Release {
                    package: "a/b".into(),
                    release: String::new(),
                })),
                payload: Some(Json::parse(payload).unwrap()),
                hub: Some(HubAccess {
                    ca_der: vec![1, 2, 3],
                    ..Default::default()
                }),
                set: Some(WarmSet {
                    items: vec![WarmItem {
                        level: WarmLevel::Gpu,
                        ..Default::default()
                    }],
                }),
                ..Default::default()
            },
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains(payload), "{json}");
        assert!(json.contains(r#""kind":"warm""#), "{json}");
        assert!(
            json.contains(r#""source":{"release":{"package":"a/b","release":""}}"#),
            "{json}"
        );
        assert!(json.contains(r#""ca_der":"AQID""#), "{json}");
        assert!(json.contains(r#""level":"gpu""#), "{json}");
        assert_eq!(serde_json::from_str::<RunRequest>(&json).unwrap(), request);

        let newer = r#"{"id":"r1","spec":{"kind":"call","future":{"x":1}},"also":true}"#;
        let read: RunRequest = serde_json::from_str(newer).unwrap();
        assert_eq!((read.id.as_str(), read.spec.kind), ("r1", RunKind::Call));

        let event = RunEvent {
            sequence: 4,
            at_ms: 9,
            event: Event::Product(Product {
                output: "image".into(),
                rev: 2,
                appended_from: Some(7),
                ..Default::default()
            }),
        };
        assert_eq!(event.event.name(), "product");
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["event"]["product"]["appended_from"], 7);
        assert_eq!(serde_json::from_value::<RunEvent>(json).unwrap(), event);

        let frame: StatusFrame =
            serde_json::from_str(r#"{"worker_id":"w","receipt":"AQID"}"#).unwrap();
        assert_eq!(
            (frame.worker_id.as_str(), frame.receipt),
            ("w", vec![1, 2, 3])
        );
    }
}
