//! Calcifer's local API: what the CLI sends Calcifer over loopback HTTP, the records it answers,
//! and the Responses-style events of one run (SSE, and `--json` NDJSON on the CLI).
//!
//! Durable events carry a `sequence_number`, per run and monotonic; a reconnect with
//! `starting_after=N` gets the run's current lifecycle event, each output item changed after N at
//! its newest revision, and every log line after N. Live events (`run.progress`, `run.waiting`)
//! carry none and are never replayed, but the latest of each is sent on connect.
//!
//! An output's bytes land on this computer after its revision is recorded: `output_item.landed`
//! carries the item with its `path` once its file holds the item's current revision (an item
//! that will not land is `output_item.done` again, carrying `export_error`). A reconnect replays
//! a landed item as `output_item.landed`, the whole item. The terminal events `run.completed`,
//! `run.failed` and `run.canceled` end the run; its stream ends after the terminal once each of
//! its outputs has landed or carries an `export_error`.
use std::{fmt, net::SocketAddr, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::ids::{Digest, RunId};

/// What a run does on its machine (`RunSpec.kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Call,
    Warm,
    Update,
    Remove,
}

/// Where a run goes: `local`, a machine's name, `any`, or `new[:sku]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum On {
    Machine(String),
    Any,
    New(Option<String>),
}

/// `POST /v1/runs`, with an `Idempotency-Key` header: one key names one submission forever.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubmitRun {
    pub on: On,
    /// The Tensorhub origin the run belongs to.
    pub hub: String,
    pub kind: Kind,
    /// What it runs, for people (`org/pkg/fn@release`).
    pub target: String,
    /// The spec as it is sent to the machine, frozen at submit.
    pub spec: crate::machine::RunSpec,
    /// Local files the run reads, copied once at submit and written to its machine before the
    /// send: what `spec.inputs` names and any other object the spec refers to
    /// (`object://sha256:<hex>/<file>`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<StagedInput>,
    /// Where outputs land; absent: `~/.cozy/outputs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out: Option<PathBuf>,
    /// What the CLI noticed while binding the run (fields it dropped, …), kept with the run.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Warning {
    pub code: String,
    pub message: String,
}

/// A local file a run reads.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagedInput {
    pub path: PathBuf,
    pub digest: Digest,
    pub length: u64,
}

/// The answer to a submit: 202 when this call recorded the run, 200 (`replay`) when the key had.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Submitted {
    pub run: Run,
    pub replay: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Not yet accepted by a machine (still in the outbox, or sent and unconfirmed).
    Queued,
    InProgress,
    Completed,
    Failed,
    Canceled,
}

/// One run as Calcifer holds it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Run {
    pub id: RunId,
    pub number: u64,
    pub kind: Kind,
    pub target: String,
    pub hub: String,
    pub machine: Option<String>,
    pub status: Status,
    #[serde(default)]
    pub cancel_requested: bool,
    /// The machine's own word for the run: `preparing`, `queued`, `running`, ….
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub machine_state: Option<String>,
    /// Why a run that did not succeed ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<Reason>,
    #[serde(default)]
    pub output: Vec<OutputItem>,
    pub created_at_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted_at_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settled_at_ms: Option<i64>,
    /// Live: why a queued run waits, from Calcifer's memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub waiting: Option<String>,
    /// Live: the machine's latest progress.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<Progress>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    /// How its machine served the request other than as sent, once settled.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adjustments: Vec<Adjustment>,
    /// The result's canonical JSON, once settled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_ms: Option<u64>,
    /// With `?detail=full`: what the machine measured (stages, steps, GPUs, attention).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurements: Option<serde_json::Value>,
    /// With `?detail=full`: the run's settled calls, in the order they ended.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<Call>,
}

/// One request field a run's executor served other than as sent (`Outcome.adjustments`).
pub use crate::machine::Adjustment;

/// One settled call of a run (the run's own function or a child call).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Call {
    pub run: String,
    pub index: u32,
    pub function: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
    pub status: String,
    pub called_at_ms: i64,
    pub finished_at_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurements: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<Reason>,
    #[serde(default)]
    pub memoized: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reason {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemStatus {
    InProgress,
    Completed,
    Incomplete,
}

/// One output of a run at its newest revision, rewritten in place.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputItem {
    /// `<number>/<name>`, or `<number>/<name>/<i>` for a list's item `i`.
    pub id: String,
    pub name: String,
    /// A list item's 1-based index; 0 for a single output.
    pub index: u32,
    pub rev: u64,
    pub length: u64,
    pub media_type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
    pub status: ItemStatus,
    /// The final bytes' digest, once done.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<Digest>,
    /// The file on this computer that holds this revision's bytes, once they landed there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// Why this revision's bytes will not land (its machine ended, or holds them no more).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export_error: Option<String>,
}

/// A weights output's media type: its bytes are a checkpoint's manifest.
pub const CHECKPOINT: &str = "application/vnd.cozy.model-manifest";

impl OutputItem {
    /// A checkpoint (a weights output) stays on the machine that wrote it, which uploads it to
    /// the Hub; it is never saved to this computer, so it is final as soon as it is done.
    pub fn is_checkpoint(&self) -> bool {
        self.media_type == CHECKPOINT
    }
}

/// The machine's progress report, as it last sent it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Progress {
    pub stage: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fraction: Option<f64>,
    #[serde(default)]
    pub completed: u64,
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub bytes_done: u64,
    #[serde(default)]
    pub bytes_total: u64,
    #[serde(default)]
    pub step_ms: f64,
}

/// One event of a run's stream.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Event {
    #[serde(rename = "run.created")]
    Created { sequence_number: u64, run: Run },
    #[serde(rename = "run.queued")]
    Queued { sequence_number: u64, run: Run },
    #[serde(rename = "run.in_progress")]
    InProgress { sequence_number: u64, run: Run },
    #[serde(rename = "run.completed")]
    Completed { sequence_number: u64, run: Run },
    #[serde(rename = "run.failed")]
    Failed { sequence_number: u64, run: Run },
    #[serde(rename = "run.canceled")]
    Canceled { sequence_number: u64, run: Run },
    #[serde(rename = "output_item.added")]
    OutputAdded {
        sequence_number: u64,
        output_index: u32,
        item: OutputItem,
    },
    #[serde(rename = "output_item.delta")]
    OutputDelta {
        sequence_number: u64,
        output_index: u32,
        item_id: String,
        rev: u64,
        length: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        appended_from: Option<u64>,
    },
    #[serde(rename = "output_item.done")]
    OutputDone {
        sequence_number: u64,
        output_index: u32,
        item: OutputItem,
    },
    /// The item's file on this computer holds its current revision (`item.path`).
    #[serde(rename = "output_item.landed")]
    OutputLanded {
        sequence_number: u64,
        output_index: u32,
        item: OutputItem,
    },
    #[serde(rename = "run.log")]
    Log {
        sequence_number: u64,
        at_ms: i64,
        level: String,
        text: String,
    },
    /// A settled call of the run (its own function's or a child's).
    #[serde(rename = "run.call")]
    Call { sequence_number: u64, call: Call },
    #[serde(rename = "run.progress")]
    Progress { progress: Progress },
    /// Why the run waits now; an empty reason: it waits no longer.
    #[serde(rename = "run.waiting")]
    Waiting { reason: String },
}

impl Event {
    /// The lifecycle event that says `run` as it is now; `created` for its first.
    pub fn lifecycle(sequence_number: u64, run: Run, created: bool) -> Self {
        match run.status {
            Status::Queued if created => Event::Created {
                sequence_number,
                run,
            },
            Status::Queued => Event::Queued {
                sequence_number,
                run,
            },
            Status::InProgress => Event::InProgress {
                sequence_number,
                run,
            },
            Status::Completed => Event::Completed {
                sequence_number,
                run,
            },
            Status::Failed => Event::Failed {
                sequence_number,
                run,
            },
            Status::Canceled => Event::Canceled {
                sequence_number,
                run,
            },
        }
    }

    /// The event's place in its run's durable stream; `None` for a live event.
    pub fn sequence_number(&self) -> Option<u64> {
        match self {
            Event::Created {
                sequence_number, ..
            }
            | Event::Queued {
                sequence_number, ..
            }
            | Event::InProgress {
                sequence_number, ..
            }
            | Event::Completed {
                sequence_number, ..
            }
            | Event::Failed {
                sequence_number, ..
            }
            | Event::Canceled {
                sequence_number, ..
            }
            | Event::OutputAdded {
                sequence_number, ..
            }
            | Event::OutputDelta {
                sequence_number, ..
            }
            | Event::OutputDone {
                sequence_number, ..
            }
            | Event::OutputLanded {
                sequence_number, ..
            }
            | Event::Log {
                sequence_number, ..
            }
            | Event::Call {
                sequence_number, ..
            } => Some(*sequence_number),
            Event::Progress { .. } | Event::Waiting { .. } => None,
        }
    }

    /// Whether the run ends here: nothing about it changes after it but its outputs landing.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Event::Completed { .. } | Event::Failed { .. } | Event::Canceled { .. }
        )
    }
}

/// `GET /v1/runs`, newest first.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunList {
    pub runs: Vec<Run>,
    /// How many more runs match beyond the limit.
    #[serde(default)]
    pub older: u64,
}

/// `GET /v1/runs/events`: every run's changes, each run whole as `GET /v1/runs/{id}` answers it. No
/// backlog: a client subscribes, then lists. A stream that falls behind ends; its client lists again.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RunsEvent {
    #[serde(rename = "run.updated")]
    Updated { run: Run },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MachineKind {
    /// This computer's machine.
    Local,
    /// A machine rented through a Hub.
    Rental,
    /// Another computer's machine, added by address.
    Added,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MachineState {
    /// No address yet: runs to it wait in the outbox.
    Booting,
    Ready,
    /// Its end is asked of its Hub, which has not yet affirmed it.
    Ending,
    Ended,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Machine {
    pub name: String,
    pub kind: MachineKind,
    pub state: MachineState,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub hub: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rental_id: Option<String>,
    /// The name it had before its Hub named it; it keeps resolving.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
    /// The sha256 of its pinned TLS leaf: what `cozy machine add … --fingerprint` takes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<Digest>,
    /// Why it ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The Hub's word, for a rental.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rental: Option<Rental>,
    /// The machine's own latest Status, once it answered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<MachineStatus>,
    /// Why its Status cannot be read now (`status` is then the last it read).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unreachable: Option<String>,
    /// A booting rental's phase, as its Hub says it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boot: Option<String>,
    /// Calcifer's runs on it.
    #[serde(default)]
    pub running: u32,
    #[serde(default)]
    pub queued: u32,
    /// False: a live rental of this account that this controller did not rent.
    #[serde(default = "yes")]
    pub recorded: bool,
    /// Its Hub did not answer the last listing.
    #[serde(default)]
    pub unverified: bool,
}

fn yes() -> bool {
    true
}

/// A rental as its Hub says it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rental {
    /// The product asked for, or the card when another client rented it.
    pub sku: String,
    pub accelerator: String,
    pub gpus: u32,
    pub usd_micros_per_hour: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spend_usd_micros: Option<i64>,
    /// `charged`, `unmetered` or `estimate`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spend_basis: Option<String>,
    pub rented_at_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at_ms: Option<i64>,
    pub disk_gb: u32,
    /// The Hub's failure code, for a rental that failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
    /// Why it runs off the requester's continent: nothing nearer fit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement_note: Option<String>,
}

/// A machine's `Status`, as it last sent it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MachineStatus {
    pub worker_id: String,
    pub version: String,
    pub platform: String,
    /// `booting`, `ready` or `releasing`.
    pub phase: String,
    #[serde(default)]
    pub runtime: String,
    #[serde(default)]
    pub tensorfs: String,
    pub started_at_ms: i64,
    /// When a rental releases itself if it stays idle; none while it works or for a machine
    /// that never releases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_deadline_ms: Option<i64>,
    #[serde(default)]
    pub gpus: Vec<Gpu>,
    #[serde(default)]
    pub disk_total_bytes: u64,
    #[serde(default)]
    pub disk_free_bytes: u64,
    /// This controller's queued, preparing or running runs there.
    #[serde(default)]
    pub live_runs: u32,
    #[serde(default)]
    pub models: Vec<Model>,
    #[serde(default)]
    pub models_bytes: u64,
    #[serde(default)]
    pub environments: Vec<Environment>,
    #[serde(default)]
    pub warm: Vec<WarmItem>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gpu {
    pub index: u32,
    pub name: String,
    pub memory_bytes: u64,
    #[serde(default)]
    pub driver: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_class: Option<String>,
}

/// A model repository the machine holds.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model {
    pub repository: String,
    pub total_bytes: u64,
    pub unique_bytes: u64,
    #[serde(default)]
    pub checkpoints: Vec<Checkpoint>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    #[serde(default)]
    pub release: String,
    pub lane: String,
    pub manifest: String,
    #[serde(default)]
    pub yanked: bool,
}

/// A package environment the machine holds for this controller.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Environment {
    pub installation: String,
    pub package: String,
    #[serde(default)]
    pub release: String,
    #[serde(default)]
    pub hub: String,
    /// What it holds now: `installed`, `downloaded`, `imported`, `host` or `gpu`.
    #[serde(default)]
    pub level: String,
    /// Why it does not run the machine's own Runtime; empty when it does.
    #[serde(default)]
    pub warning: String,
}

/// One member of this controller's warm set on the machine.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WarmItem {
    /// `org/name[@release]`, or an installation id.
    pub source: String,
    #[serde(default)]
    pub entrypoint: String,
    /// The level asked for, and the one it holds now.
    pub level: String,
    #[serde(default)]
    pub holds: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub held_back: String,
}

/// `GET /v1/machines` (`?refresh=true` lists every Hub again first).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MachineList {
    pub machines: Vec<Machine>,
    /// When each Hub last answered a listing.
    #[serde(default)]
    pub hubs: Vec<HubListing>,
    /// Hubs whose last listing failed: their rows are `unverified`.
    #[serde(default)]
    pub unanswered: Vec<HubProblem>,
    /// Hubs this computer holds no login for.
    #[serde(default)]
    pub signed_out: Vec<String>,
}

/// `GET /v1/machines/events`: a machine as `GET /v1/machines` lists it, each time its row changes,
/// and the name of one that left the list. No backlog: a client subscribes, then lists.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MachinesEvent {
    #[serde(rename = "machine.updated")]
    Updated { machine: Box<Machine> },
    #[serde(rename = "machine.removed")]
    Removed { name: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HubListing {
    pub hub: String,
    pub listed_at_ms: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HubProblem {
    pub hub: String,
    pub code: String,
    pub message: String,
}

/// `POST /v1/machines` with `Rent`: 202, or 200 when the key had asked already (`replay`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rented {
    pub machine: Machine,
    /// The price the rental was asked at: an acceptance above it is released.
    pub quote: Quote,
    pub replay: bool,
}

/// The Hub's price for the exact rental asked: the machine and its disk an hour, traffic a GB, and
/// what an hour plus its download is expected to cost.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quote {
    pub gpu_usd_micros_per_hour: i64,
    pub storage_usd_micros_per_hour: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bandwidth_usd_micros_per_gb: Option<i64>,
    pub disk_gb: u32,
    pub expected_hours: f64,
    pub expected_download_gb: u32,
    pub expected_total_usd_micros: i64,
}

/// `DELETE /v1/machines/{name}[?force=true]`: `changed` is false when it had ended already. While
/// its runs' outputs have not landed it is refused `outputs_exporting`, unless `force`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ended {
    pub machine: Machine,
    pub changed: bool,
    /// Outputs of its runs that had not landed when `force` ended it, given up: `run 12 big`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub abandoned: Vec<String>,
}

/// `GET /v1/runs/{id}/play?output=<name[/i]>&expires_s=N`: how a browser plays one run output
/// straight from its machine over WebRTC.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Play {
    /// `host:port` the browser dials.
    pub address: String,
    /// SHA-256 hex of the certificate the machine's DTLS presents.
    pub fingerprint: String,
    /// A Cozy-Cap for that run output alone.
    pub token: String,
}

/// `POST /v1/machines/{name}/keepalive`, once the machine acknowledged it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Keepalive {
    pub acknowledged_at_ms: i64,
    /// None while a job is queued or running there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_deadline_ms: Option<i64>,
}

/// `POST /v1/machines`: rent one (with an `Idempotency-Key`) or add another computer's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NewMachine {
    Rent {
        sku: String,
        hub: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gpus: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disk_gb: Option<u32>,
    },
    Add {
        name: String,
        address: String,
        /// `sha256:<hex>` of the machine's TLS leaf.
        fingerprint: Digest,
    },
}

/// `GET /v1/machines/{name}/logs/{log}`: the logs a machine keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MachineLog {
    /// TensorD's own log.
    Tensord,
    Tensorfs,
    Memory,
}

/// `POST /v1/down`: Calcifer stops; machines keep running what they accepted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Down {
    pub stopping: bool,
    /// What Calcifer still owes, which the next Calcifer resumes.
    pub owed: Vec<String>,
}

/// The file in Calcifer's directory (`~/.cozy/calcifer`) that says, as an `error::Envelope`, why
/// the last start failed; a successful start deletes it.
pub const REFUSED: &str = "refused.json";

/// What a serving Calcifer publishes beside its lock (`calcifer.lock.holder`, 0600, one JSON
/// line): where its API listens and the bearer it wants, fresh at every start. Absent while the
/// lock is held: Calcifer is still opening its store.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Holder {
    pub address: SocketAddr,
    pub token: String,
    pub pid: u32,
    pub version: String,
    pub started_at_ms: i64,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Call => "call",
            Kind::Warm => "warm",
            Kind::Update => "update",
            Kind::Remove => "remove",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        [Kind::Call, Kind::Warm, Kind::Update, Kind::Remove]
            .into_iter()
            .find(|kind| kind.as_str() == text)
    }
}

impl From<String> for On {
    fn from(text: String) -> Self {
        match text.as_str() {
            "any" => On::Any,
            "new" => On::New(None),
            _ => match text.strip_prefix("new:") {
                Some(sku) => On::New(Some(sku.to_owned())),
                None => On::Machine(text),
            },
        }
    }
}

impl From<On> for String {
    fn from(on: On) -> Self {
        on.to_string()
    }
}

impl fmt::Display for On {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            On::Machine(name) => f.write_str(name),
            On::Any => f.write_str("any"),
            On::New(None) => f.write_str("new"),
            On::New(Some(sku)) => write!(f, "new:{sku}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(status: Status) -> Run {
        Run {
            id: RunId::mint([1; 12]),
            number: 3,
            kind: Kind::Call,
            target: "fidika/sdxl/generate".into(),
            hub: "https://tensorhub.com".into(),
            machine: Some("local".into()),
            status,
            cancel_requested: false,
            machine_state: None,
            reason: None,
            output: vec![],
            created_at_ms: 1,
            accepted_at_ms: None,
            settled_at_ms: None,
            waiting: None,
            progress: None,
            warnings: vec![],
            adjustments: vec![],
            result: None,
            execution_ms: None,
            measurements: None,
            calls: vec![],
        }
    }

    #[test]
    fn events_carry_responses_names_and_sequence_numbers() {
        let created = Event::lifecycle(1, run(Status::Queued), true);
        let json = serde_json::to_value(&created).unwrap();
        assert_eq!(json["type"], "run.created");
        assert_eq!(json["sequence_number"], 1);
        assert_eq!(serde_json::from_value::<Event>(json).unwrap(), created);
        let done = Event::lifecycle(9, run(Status::Canceled), false);
        assert!(done.is_terminal());
        assert_eq!(serde_json::to_value(&done).unwrap()["type"], "run.canceled");
        let live = Event::Waiting {
            reason: "booting".into(),
        };
        assert_eq!(live.sequence_number(), None);
        assert_eq!(
            serde_json::to_value(&live).unwrap(),
            serde_json::json!({"type": "run.waiting", "reason": "booting"})
        );
    }

    #[test]
    fn a_submission_round_trips_with_its_spec_and_unknown_fields_ignored() {
        let submit = SubmitRun {
            on: On::New(Some("rtx-4090".into())),
            hub: "https://tensorhub.com".into(),
            kind: Kind::Warm,
            target: "fidika/sdxl".into(),
            spec: crate::machine::RunSpec {
                kind: crate::machine::RunKind::Warm,
                ..Default::default()
            },
            inputs: vec![],
            out: None,
            warnings: vec![],
        };
        let mut json = serde_json::to_value(&submit).unwrap();
        assert_eq!(json["on"], "new:rtx-4090");
        assert_eq!(json["spec"]["kind"], "warm");
        json["from_a_newer_cli"] = serde_json::json!(true);
        assert_eq!(serde_json::from_value::<SubmitRun>(json).unwrap(), submit);
        for on in ["local", "any", "new", "new:h100", "kiyui"] {
            assert_eq!(On::from(on.to_owned()).to_string(), on);
        }
    }
}
