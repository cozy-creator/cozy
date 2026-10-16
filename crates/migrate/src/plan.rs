//! The Go records as this layout holds them: Calcifer's machine rows and runs, and the key, session
//! and config files. Nothing here writes.
use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
};

use calcifer::store::{Endpoint, ImportedOutput, ImportedRun, MachineRow, State};
use cozy_platform::paths::Home;
use cozy_proto::{
    calcifer::{Kind, MachineKind, MachineState, Reason, Status},
    config::{self, Hub},
    ids::RunId,
    machine as v1,
};
use ed25519_dalek::{SigningKey, pkcs8::DecodePrivateKey as _};
use prost::Message as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::go::{self, Records, Request};

/// A file this layout gets: a key, a session or the config, written owner-only.
pub struct File {
    pub path: PathBuf,
    pub bytes: Vec<u8>,
}

#[derive(Default, Serialize)]
pub struct Counts {
    pub hubs: usize,
    pub current_hub: Option<String>,
    pub device_keys: usize,
    pub sessions: usize,
    pub owner_key: bool,
    pub rental_keys: usize,
    pub local_machine: bool,
    pub rentals_ready: usize,
    pub rentals_booting: usize,
    pub history_machines: usize,
    pub runs: usize,
    pub history: usize,
    pub completed: usize,
    pub failed: usize,
    pub canceled: usize,
    pub unsettled_accepted: usize,
    pub unsettled_sent: usize,
    pub outputs: usize,
    pub outputs_without_path: usize,
}

/// A record left behind, and why.
#[derive(Serialize)]
pub struct Skip {
    pub what: String,
    pub reason: String,
}

#[derive(Default)]
pub struct Plan {
    pub files: Vec<File>,
    pub machines: Vec<MachineRow>,
    pub runs: Vec<ImportedRun>,
    pub counts: Counts,
    pub skipped: Vec<Skip>,
}

impl Plan {
    fn skip(&mut self, what: impl Into<String>, reason: impl Into<String>) {
        self.skipped.push(Skip {
            what: what.into(),
            reason: reason.into(),
        });
    }
}

pub fn plan(from: &Path, into: &Home, records: Records) -> Result<Plan, String> {
    let mut plan = Plan::default();
    config(from, into, &mut plan)?;
    keys(from, into, &mut plan)?;
    let live = machines(from, into, &records, &mut plan)?;
    runs(into, records, &live, &mut plan);
    Ok(plan)
}

/// `config.yaml` as the Go CLI read it; any other setting has no counterpart here.
#[derive(Deserialize)]
struct GoConfig {
    tensorhub_url: Option<String>,
    #[serde(default)]
    hubs: BTreeMap<String, String>,
    huggingface_token: Option<String>,
    civitai_token: Option<String>,
    port: Option<u16>,
    #[serde(flatten)]
    other: BTreeMap<String, serde_yaml::Value>,
}

/// The current Hub, the named Hubs, the model-source tokens and Calcifer's port.
fn config(from: &Path, into: &Home, plan: &mut Plan) -> Result<(), String> {
    let path = from.join("config.yaml");
    if into.config().exists() && path.exists() {
        plan.skip(
            "config.yaml",
            format!("{} exists and is kept", into.config().display()),
        );
        return Ok(());
    }
    let go: GoConfig = match std::fs::read_to_string(&path) {
        Ok(text) => serde_yaml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let mut doc = toml_edit::DocumentMut::new();
    for (name, url) in &go.hubs {
        let name = name.to_ascii_lowercase();
        match config::origin(url) {
            Ok(origin) if config::valid_name(&name) => {
                doc["hubs"][&name] = toml_edit::value(origin);
                plan.counts.hubs += 1;
            }
            _ => plan.skip(
                format!("config.yaml hubs.{name}"),
                "not a Hub name and an http(s) URL",
            ),
        }
    }
    for (key, token) in [
        ("huggingface_token", go.huggingface_token),
        ("civitai_token", go.civitai_token),
    ] {
        if let Some(token) = token.filter(|t| !t.trim().is_empty()) {
            doc[key] = toml_edit::value(token.trim());
        }
    }
    if let Some(port) = go.port {
        doc["calcifer"]["port"] = toml_edit::value(i64::from(port));
    }
    for key in go.other.keys() {
        plan.skip(format!("config.yaml {key}"), "no such setting in this cozy");
    }
    if let Some(hub) = go
        .tensorhub_url
        .map(|h| h.trim().to_owned())
        .filter(|h| !h.is_empty())
    {
        doc["hub"] = toml_edit::value(config::origin(&hub).unwrap_or(hub.to_ascii_lowercase()));
        let (parsed, _) =
            config::Config::parse(&doc.to_string(), "config.toml").map_err(|e| e.message)?;
        match parsed.current() {
            Ok(current) => plan.counts.current_hub = Some(current.label().to_owned()),
            Err(_) => {
                doc.remove("hub");
                plan.skip(
                    format!("config.yaml tensorhub_url {hub:?}"),
                    "names no Hub this config knows",
                );
            }
        }
    }
    plan.files.push(File {
        path: into.config(),
        bytes: doc.to_string().into_bytes(),
    });
    Ok(())
}

/// Device keys and their unexpired sessions from `auth/`, and the owner key.
fn keys(from: &Path, into: &Home, plan: &mut Plan) -> Result<(), String> {
    let auth = from.join("auth");
    let mut names: Vec<_> = std::fs::read_dir(&auth)
        .map(|dir| {
            dir.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    let mut kept = vec![];
    for name in &names {
        let path = auth.join(name);
        let stem = name
            .strip_suffix(".json")
            .filter(|s| !s.ends_with(".session"));
        let key = stem.and_then(|_| hub_client::DeviceKey::load(&path).ok());
        let origin = key.as_ref().and_then(|k| config::origin(&k.hub).ok());
        let (Some(stem), Some(key), Some(origin)) = (stem, key, origin) else {
            if path.is_dir() {
                plan.skip(
                    format!("auth/{name}/"),
                    "a cache this cozy reads from the Hub again",
                );
            } else if !name.ends_with(".session.json") {
                plan.skip(
                    format!("auth/{name}"),
                    "not a device key this cozy can load",
                );
            }
            continue;
        };
        let hub = Hub { name: None, origin };
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        plan.files.push(File {
            path: hub.device_key(into),
            bytes,
        });
        plan.counts.device_keys += 1;
        let session = format!("{stem}.session.json");
        let Some(session) = go::json::<go::Session>(&auth.join(&session))
            .ok()
            .flatten()
            .filter(|s| {
                s.device_key_id == key.id
                    && s.expires_at
                        .parse::<jiff::Timestamp>()
                        .is_ok_and(|t| t > jiff::Timestamp::now())
            })
        else {
            continue;
        };
        let bearer = hub_client::Bearer {
            token: session.access_token,
            expires: session
                .expires_at
                .parse::<jiff::Timestamp>()
                .map_or(0, |t| t.as_second() as u64),
            device_key_id: session.device_key_id,
        };
        plan.files.push(File {
            path: hub.bearer(into),
            bytes: serde_json::to_vec(&bearer).expect("serializes"),
        });
        plan.counts.sessions += 1;
        kept.push(format!("{stem}.session.json"));
    }
    for name in names
        .iter()
        .filter(|n| n.ends_with(".session.json") && !kept.contains(n))
    {
        plan.skip(
            format!("auth/{name}"),
            "expired, or its device key was not imported; the key signs in again",
        );
    }
    let owner = from.join("machine/owner.pem");
    match signing_key(&owner) {
        Some(bytes) => {
            plan.files.push(File {
                path: owner_key(into),
                bytes,
            });
            plan.counts.owner_key = true;
        }
        None if owner.exists() => plan.skip("machine/owner.pem", "not a PKCS#8 Ed25519 key"),
        None => {}
    }
    Ok(())
}

pub fn owner_key(home: &Home) -> PathBuf {
    home.keys().join("owner.pem")
}

fn rental_key(home: &Home, name: &str) -> PathBuf {
    home.keys().join(format!("rental-{name}.pem"))
}

/// The key file's bytes, when it holds a PKCS#8 Ed25519 key.
fn signing_key(path: &Path) -> Option<Vec<u8>> {
    let pem = std::fs::read_to_string(path).ok()?;
    SigningKey::from_pkcs8_pem(&pem).ok()?;
    Some(pem.into_bytes())
}

/// The machines runs can still be followed on, by Go machine id → name here: the local machine
/// and every rental not ended.
fn machines(
    from: &Path,
    into: &Home,
    records: &Records,
    plan: &mut Plan,
) -> Result<HashMap<String, String>, String> {
    let mut live = HashMap::new();
    let leaf = go::certificate(&from.join("machine/leaf.pem"));
    match (
        go::json::<go::Agent>(&from.join("machine/agent.json"))?,
        leaf,
        plan.counts.owner_key,
    ) {
        (Some(agent), Some(der), true) => {
            plan.machines.push(MachineRow {
                name: "local".into(),
                kind: MachineKind::Local,
                hub: String::new(),
                rental_id: None,
                endpoint: Some(Endpoint {
                    worker_id: agent.worker_id,
                    address: format!("127.0.0.1:{}", agent.worker_port),
                    pin: Sha256::digest(&der).into(),
                }),
                key: owner_key(into),
                state: MachineState::Ready,
                served: true,
                reason: None,
            });
            live.insert("local".to_owned(), "local".to_owned());
            plan.counts.local_machine = true;
        }
        (None, ..) => {}
        _ => plan.skip(
            "machine/agent.json",
            "the local machine's leaf or owner key is missing",
        ),
    }
    for rental in &records.rentals {
        let what = format!("rental {}", rental.id);
        if matches!(rental.state.as_str(), "failed" | "released") {
            plan.skip(what, format!("it is {}", rental.state));
            continue;
        }
        let Some(key) = signing_key(&rental.key) else {
            plan.skip(
                what,
                "its key is missing or unreadable, so its runs cannot be followed",
            );
            continue;
        };
        let name = machine_name(if rental.hub_name.is_empty() {
            &rental.asked
        } else {
            &rental.hub_name
        });
        let endpoint = match go::certificate(&rental.cert) {
            Some(der) if !rental.address.is_empty() && !rental.worker_id.is_empty() => {
                Some(Endpoint {
                    worker_id: rental.worker_id.clone(),
                    address: rental.address.clone(),
                    pin: Sha256::digest(&der).into(),
                })
            }
            _ => None,
        };
        if endpoint.is_some() {
            plan.counts.rentals_ready += 1;
        } else {
            plan.counts.rentals_booting += 1;
        }
        plan.files.push(File {
            path: rental_key(into, &name),
            bytes: key,
        });
        plan.counts.rental_keys += 1;
        plan.machines.push(MachineRow {
            name: name.clone(),
            kind: MachineKind::Rental,
            hub: config::origin(&rental.hub).unwrap_or_else(|_| rental.hub.clone()),
            rental_id: Some(rental.id.clone()),
            state: if endpoint.is_some() {
                MachineState::Ready
            } else {
                MachineState::Booting
            },
            served: endpoint.is_some(),
            endpoint,
            key: rental_key(into, &name),
            reason: None,
        });
        live.insert(rental.id.clone(), name);
    }
    Ok(live)
}

/// A machine name here: 1–32 of `[a-z0-9-]`, a letter first.
fn machine_name(text: &str) -> String {
    let mut name: String = text
        .to_ascii_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_lowercase() || c.is_ascii_digit() {
                c
            } else {
                '-'
            }
        })
        .collect();
    if !name.starts_with(|c: char| c.is_ascii_lowercase()) {
        name.insert_str(0, "m-");
    }
    name.truncate(32);
    name
}

/// Settled history, and the runs still owed on a live machine: sent or accepted, by id and cursor.
fn runs(into: &Home, records: Records, live: &HashMap<String, String>, plan: &mut Plan) {
    let Records {
        requests,
        mut products,
        ..
    } = records;
    let mut ended: BTreeMap<String, MachineRow> = BTreeMap::new();
    for (index, request) in requests.into_iter().enumerate() {
        let number = index as u64 + 1;
        let Ok(id) = RunId::try_from(request.id.clone()) else {
            plan.skip(format!("run {number}"), "its id is not a run id");
            continue;
        };
        let created_at_ms = ms(&request.created_at).unwrap_or(0);
        let settled = matches!(
            request.state.as_str(),
            "succeeded" | "failed" | "canceled" | "refused" | "abandoned"
        );
        let owed = request.execution.as_ref().filter(|e| {
            !settled
                && e.sent
                && !e.dropped
                && e.outcome.is_empty()
                && live.contains_key(&e.machine)
        });
        let mut run = ImportedRun {
            id,
            number,
            kind: kind(&request.kind),
            target: target(&request),
            hub: config::origin(&request.hub).unwrap_or(request.hub.clone()),
            machine: None,
            out_dir: request
                .out_dir
                .as_deref()
                .map_or_else(|| into.outputs(), PathBuf::from),
            state: State::Settled,
            cancel_requested: false,
            cursor: 0,
            machine_number: None,
            machine_state: None,
            status: None,
            reason: None,
            outcome: None,
            created_at_ms,
            accepted_at_ms: request.accepted_at.as_deref().and_then(ms),
            settled_at_ms: None,
            outputs: vec![],
        };
        let items = products.remove(&request.id).unwrap_or_default();
        if let Some(execution) = owed {
            let receipt = v1::RunState::decode(execution.receipt.as_slice())
                .ok()
                .filter(|_| execution.accepted);
            run.state = if receipt.is_some() {
                State::Accepted
            } else {
                State::Sent
            };
            run.machine = live.get(&execution.machine).cloned();
            run.cancel_requested = execution.cancel_requested;
            run.cursor = if receipt.is_some() {
                execution.cursor
            } else {
                0
            };
            run.machine_number = receipt.as_ref().map(|r| r.number);
            run.machine_state = receipt.map(|r| r.state);
            if run.state == State::Accepted {
                plan.counts.unsettled_accepted += 1;
            } else {
                plan.counts.unsettled_sent += 1;
            }
        } else {
            let (status, reason) = ending(&request, settled, live);
            if reason.as_ref().is_some_and(|r| r.code == "not_migrated") {
                let message = reason
                    .as_ref()
                    .map(|r| r.message.clone())
                    .unwrap_or_default();
                plan.skip(
                    format!("run {number} ({})", request.id),
                    format!("imported as failed: {message}"),
                );
            }
            match status {
                Status::Completed => plan.counts.completed += 1,
                Status::Canceled => plan.counts.canceled += 1,
                _ => plan.counts.failed += 1,
            }
            run.status = Some(status);
            run.reason = reason;
            run.settled_at_ms = request.settled_at.as_deref().and_then(ms);
            run.outcome = request
                .execution
                .as_ref()
                .filter(|e| e.accepted)
                .and_then(|e| {
                    v1::Outcome::decode(e.outcome.as_slice())
                        .ok()
                        .filter(|_| !e.outcome.is_empty())
                });
            let name = machine_name(&history_machine(&request));
            ended
                .entry(name.clone())
                .or_insert_with(|| ended_row(into, &name, &run.hub));
            run.machine = Some(name);
            plan.counts.history += 1;
        }
        run.outputs = outputs(&request, items, run.state == State::Settled);
        plan.counts.outputs += run.outputs.len();
        plan.counts.outputs_without_path += run.outputs.iter().filter(|o| o.path.is_none()).count();
        plan.runs.push(run);
    }
    plan.counts.runs = plan.runs.len();
    let named: Vec<String> = plan.machines.iter().map(|m| m.name.clone()).collect();
    ended.retain(|name, _| !named.contains(name));
    plan.counts.history_machines = ended.len();
    plan.machines.extend(ended.into_values());
}

/// A settled Go request's status, and the reason one that did not complete gives.
fn ending(
    request: &Request,
    settled: bool,
    live: &HashMap<String, String>,
) -> (Status, Option<Reason>) {
    #[derive(Default, Deserialize)]
    struct Payload {
        #[serde(default)]
        error: String,
        #[serde(default)]
        error_code: String,
        #[serde(default)]
        error_type: String,
    }
    let payload: Payload = request
        .ending
        .as_deref()
        .and_then(|p| serde_json::from_str(p).ok())
        .unwrap_or_default();
    let reason = |code: &str, message: &str| Reason {
        code: [
            payload.error_code.as_str(),
            payload.error_type.as_str(),
            code,
        ]
        .into_iter()
        .find(|c| !c.is_empty())
        .unwrap_or(code)
        .to_owned(),
        message: if payload.error.is_empty() {
            message.to_owned()
        } else {
            payload.error.clone()
        },
    };
    match request.state.as_str() {
        "succeeded" => (Status::Completed, None),
        "canceled" => (Status::Canceled, Some(reason("canceled", "canceled"))),
        "abandoned" => (
            Status::Failed,
            Some(reason("abandoned", "the Go CLI stopped following it")),
        ),
        "refused" => (Status::Failed, Some(reason("refused", "refused"))),
        _ if settled => (Status::Failed, Some(reason("failed", "failed"))),
        _ => {
            let message = match &request.execution {
                Some(e) if e.dropped => "the Go CLI had stopped following it".to_owned(),
                Some(e) if e.sent && !live.contains_key(&e.machine) => {
                    format!(
                        "its machine {} had ended before the move from the Go CLI",
                        e.machine
                    )
                }
                _ => "it was not sent before the move from the Go CLI; run it again".to_owned(),
            };
            (
                Status::Failed,
                Some(Reason {
                    code: "not_migrated".into(),
                    message,
                }),
            )
        }
    }
}

/// The machine word a history run names: its rental's human name, else its machine id.
fn history_machine(request: &Request) -> String {
    let id = request
        .execution
        .as_ref()
        .map(|e| e.machine.as_str())
        .unwrap_or_default();
    [
        request.machine.as_str(),
        id,
        request.worker.as_str(),
        "local",
    ]
    .into_iter()
    .find(|s| !s.is_empty())
    .unwrap_or("local")
    .to_owned()
}

fn ended_row(into: &Home, name: &str, hub: &str) -> MachineRow {
    let kind = match name {
        "local" => MachineKind::Local,
        _ if name.starts_with("endpoint-") => MachineKind::Added,
        _ => MachineKind::Rental,
    };
    MachineRow {
        name: name.to_owned(),
        hub: if kind == MachineKind::Rental {
            hub.to_owned()
        } else {
            String::new()
        },
        rental_id: None,
        endpoint: None,
        key: if kind == MachineKind::Rental {
            rental_key(into, name)
        } else {
            owner_key(into)
        },
        kind,
        state: MachineState::Ended,
        served: true,
        reason: Some("ended before the move from the Go CLI".into()),
    }
}

/// Output items at their newest revision, or, for a run from before items, its exported files.
fn outputs(request: &Request, items: Vec<go::Product>, settled: bool) -> Vec<ImportedOutput> {
    if !items.is_empty() {
        return items
            .into_iter()
            .map(|p| ImportedOutput {
                path: p
                    .path
                    .as_deref()
                    .filter(|p| !p.is_empty())
                    .map(PathBuf::from),
                name: p.output,
                index: p.index,
                rev: p.rev.max(1),
                length: p.length,
                digest: p.digest,
                media_type: p.media_type,
                label: p.label.unwrap_or_default(),
                duration_us: p.duration_us,
                done: settled,
            })
            .collect();
    }
    let mut seen: HashMap<String, u32> = HashMap::new();
    request
        .published
        .iter()
        .map(|path| {
            let path = PathBuf::from(path);
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let index = seen.entry(name.clone()).or_default();
            *index += 1;
            ImportedOutput {
                media_type: "application/octet-stream".into(),
                name,
                index: *index - 1,
                rev: 1,
                length: 0,
                digest: String::new(),
                label: String::new(),
                duration_us: 0,
                done: true,
                path: Some(path),
            }
        })
        .collect()
}

/// Go's request kinds as this cozy's: a function's run is a call, installs and downloads warm.
fn kind(go: &str) -> Kind {
    match go {
        "install" | "warm" | "download" => Kind::Warm,
        "update" => Kind::Update,
        "remove" => Kind::Remove,
        _ => Kind::Call,
    }
}

/// What the run ran, as `cozy run list` showed it.
fn target(request: &Request) -> String {
    #[derive(Default, Deserialize)]
    struct Operation {
        #[serde(default)]
        models: Vec<Model>,
    }
    #[derive(Deserialize)]
    struct Model {
        #[serde(default)]
        model: String,
    }
    let at = |name: String| match request.release.as_str() {
        "" => name,
        release => format!("{name}@{release}"),
    };
    match (request.package.as_str(), request.entrypoint.as_str()) {
        ("", _) if request.kind == "update" => "software".into(),
        ("", _) => {
            let op: Operation = serde_json::from_slice(&request.operation).unwrap_or_default();
            op.models
                .into_iter()
                .map(|m| m.model)
                .collect::<Vec<_>>()
                .join(", ")
        }
        (package, "") => at(package.to_owned()),
        (package, function) => at(format!("{package}/{function}")),
    }
}

fn ms(text: &str) -> Option<i64> {
    text.parse::<jiff::Timestamp>()
        .ok()
        .map(|t| t.as_millisecond())
}
