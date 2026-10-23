//! Runs (G/API.md): submit, follow and the translation of the engine's event pages into a run's
//! log. Status and software-update runs are `machine_status`'s and `machine_update`'s.
use super::{
    auth::VerifiedActor,
    backend::MachineBackend,
    capability::Grant,
    domain,
    http::blocking,
    v1, refused,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use crate::gpu_service::Level;
use protocol::error::Error;
use serde_json::Value;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::mpsc;

/// The outcome document's adjustment rows, as TensorD kept them.
fn adjustments(rows: &Value) -> Vec<v1::Adjustment> {
    let rows: Vec<crate::journal::Adjustment> = serde_json::from_value(rows.clone()).unwrap_or_default();
    rows.into_iter()
        .map(|row| v1::Adjustment {
            field: row.field,
            kind: row.kind.as_str().into(),
            requested: row.requested,
            applied: row.applied,
            reason: row.reason,
            source: row.source,
        })
        .collect()
}

/// What an output-limited grant sees of an event: its outputs' products, and the outcome's
/// status and inventory of them. Results, reasons, triage, measurements, adjustments, waiting
/// details, progress and logs stay the run owner's.
fn visible(grant: &Grant, event: &mut v1::RunEvent) -> bool {
    let allows = |p: &v1::Product| grant.allows(&grant.run, &p.output, (p.index > 0).then_some(p.index));
    match &mut event.event {
        v1::Event::Product(product) => allows(&*product),
        v1::Event::Outcome(outcome) => {
            outcome.outputs.retain(allows);
            outcome.result = None;
            outcome.reason = None;
            outcome.triage = false;
            outcome.measurements = None;
            outcome.adjustments.clear();
            true
        }
        v1::Event::State(state) => {
            state.waiting.clear();
            true
        }
        _ => false,
    }
}

pub(super) fn query(
    backend: &impl MachineBackend,
    actor: VerifiedActor,
    id: &str,
) -> Result<domain::MachineExecutionQuery, Error> {
    let workspace = backend.workspace(actor, domain::MachineExecutionWorkspaceQuery::default())?;
    Ok(domain::MachineExecutionQuery {
        request_id: id.into(),
        expected_execution_workspace_id: workspace.execution_workspace_id,
    })
}

/// The run's state as the log's first frame (sequence 0: a snapshot, not a log entry).
pub(super) fn state(id: &str, state: &domain::MachineExecutionState) -> v1::RunState {
    v1::RunState {
        id: id.into(),
        number: state.number,
        state: match state.state.as_str() {
            "starting" => "queued".into(),
            other => other.into(),
        },
        sequence: state.sequence,
        attempt: state.attempt_ordinal as u32,
        waiting: String::new(),
        started_at_ms: state.started_at_ms,
    }
}

/// A run's model choices as the preparation path reads them.
fn choices(sent: Vec<v1::ModelChoice>) -> Result<Vec<domain::ModelChoice>, Error> {
    sent.into_iter()
        .map(|choice| -> Result<domain::ModelChoice, Error> {
            Ok(domain::ModelChoice {
                parameter: choice.parameter,
                repository: choice.repository,
                release: choice.release,
                lane: choice.lane,
                manifest: if choice.manifest.is_empty() {
                    None
                } else {
                    Some(domain::Ref {
                        digest: digest(&choice.manifest)?,
                        length: choice.manifest_length,
                    })
                },
                source: choice.source,
                profiles: choice.profiles,
                adapters: choice
                    .adapters
                    .into_iter()
                    .map(|a| domain::DownloadAdapterRef {
                        component: a.component,
                        model: a.model,
                        release: a.release,
                        lane: a.lane,
                        manifest: a.manifest,
                        scale: a.scale,
                        source: a.source,
                        profiles: a.profiles,
                        ..Default::default()
                    })
                    .collect(),
            })
        })
        .collect()
}

/// A published release by name (th-245): empty is the Hub's newest.
fn release_source(release: v1::Release) -> crate::runs::Source {
    crate::runs::Source::Release { package: release.package, release: release.release }
}

/// One warm set member, kept as it was sent for Status; None for one at `off`, which leaves
/// the set.
fn member(mut item: v1::WarmItem) -> Result<Option<crate::runs::SetItem>, Error> {
    (item.holds, item.held_back) = Default::default();
    let level = match item.level {
        v1::WarmLevel::Off => return Ok(None),
        v1::WarmLevel::Installed => Level::Installed,
        v1::WarmLevel::Downloaded => Level::Downloaded,
        v1::WarmLevel::Imported => Level::Imported,
        v1::WarmLevel::Host => Level::Host,
        v1::WarmLevel::Gpu => Level::Gpu,
    };
    let sent = serde_json::to_vec(&item).expect("a warm item serializes");
    let source = match item.source {
        Some(v1::WarmSource::Release(release)) => release_source(release),
        Some(v1::WarmSource::Installation(alias)) => crate::runs::Source::Installation(alias),
        None => {
            return Err(Error::validation(
                "invalid_warm_set",
                "a warm set member names a release or an installation",
            ))
        }
    };
    Ok(Some(crate::runs::SetItem {
        source,
        entrypoint: item.entrypoint,
        models: choices(item.models)?,
        bits: item.bits,
        level,
        sent,
    }))
}

/// A Run spec as this machine's run sources (`runs`), and the Hub it reads. Its digest (the
/// Hub's origin alone) makes the id idempotent: the same id with another spec is refused.
fn spec_of(mut spec: v1::RunSpec) -> Result<(crate::runs::Spec, Option<v1::HubAccess>), Error> {
    let (warm, remove) = match spec.kind {
        v1::RunKind::Call => (false, false),
        v1::RunKind::Warm => (true, false),
        v1::RunKind::Remove => (false, true),
        v1::RunKind::Update => return Err(Error::validation("invalid_kind", "an update is not a run of code")),
    };
    // Kept while it prepares, for a restart to prepare it again; provider tokens stay in memory.
    let tokens = spec.providers.as_ref().is_some_and(|p| !p.huggingface.is_empty() || !p.civitai.is_empty())
        || spec.tensorfs.as_ref().is_some_and(|options| !options.token.is_empty());
    let journaled = if tokens { vec![] } else { serde_json::to_vec(&spec).unwrap_or_default() };
    let input: Value = match &spec.payload {
        None => serde_json::json!({}),
        Some(payload) => crate::boundary_json::parse(payload.get().as_bytes())
            .map_err(|_| Error::validation("invalid_payload", "the payload is not JSON"))?,
    };
    let hub = spec.hub.take();
    let providers = spec.providers.take();
    let tensorfs = spec.tensorfs.take();
    // What the run is, not how it reaches its sources: a spec re-sent with other access,
    // another catalog revision or a reformatted payload names the same run.
    let identity_digest = {
        let mut identity = spec.clone();
        identity.catalog_revision.clear();
        identity.payload = None;
        identity.hub = hub.as_ref().map(|hub| v1::HubAccess {
            origin: hub.origin.trim_end_matches('/').into(),
            ..Default::default()
        });
        identity.tensorfs = tensorfs.as_ref().map(|options| v1::TensorFSOptions {
            endpoint: options.endpoint.trim_end_matches('/').into(),
            ..Default::default()
        });
        // Auto transport for the existing Hub is the same run as the older caller's None.
        // A genuinely distinct endpoint remains part of the identity.
        if identity.tensorfs.as_ref().is_some_and(|options| {
            identity.hub.as_ref().is_some_and(|hub| hub.origin == options.endpoint)
        }) {
            identity.tensorfs = None;
        }
        // A JSON value's objects are sorted maps and its numbers exact: one spelling per spec.
        let mut identity = serde_json::to_value(&identity).expect("a run spec serializes");
        identity["payload"] = input.clone();
        let encoded = crate::boundary_json::exact(&identity);
        format!("sha256:{}", tensorfs::sha256::hex_digest(&encoded))
    };
    if hub.as_ref().is_some_and(|hub| !crate::hub::valid_origin(&hub.origin)) {
        return Err(Error::validation("invalid_hub_origin", "the run's Hub origin is invalid"));
    }
    if tensorfs.as_ref().is_some_and(|options| !crate::hub::valid_origin(&options.endpoint)) {
        return Err(Error::validation("invalid_tensorfs_endpoint", "the TensorFS endpoint is invalid"));
    }
    let source = match spec.source {
        Some(v1::Source::Release(release)) => release_source(release),
        Some(v1::Source::Installation(alias)) => crate::runs::Source::Installation(alias),
        Some(v1::Source::Local(local)) => crate::runs::Source::Local(local.manifest),
        // A warm run of model choices alone makes them (and uploads to its destination).
        None if warm => crate::runs::Source::Models,
        None => return Err(Error::validation("invalid_source", "a run spec names its source")),
    };
    // The caller's whole warm set, each member kept as it was sent for Status.
    let set = match spec.set.take().filter(|_| warm) {
        None => None,
        Some(set) => Some(set.items.into_iter().filter_map(|item| member(item).transpose()).collect::<Result<_, _>>()?),
    };
    let models = choices(spec.models)?;
    if remove && !matches!(source, crate::runs::Source::Release { .. } | crate::runs::Source::Installation(_)) {
        return Err(Error::validation("invalid_source", "a removal names a release or an installation"));
    }
    Ok((crate::runs::Spec {
        warm,
        remove,
        set,
        parent: String::new(),
        source,
        entrypoint: spec.entrypoint,
        input,
        inputs: spec
            .inputs
            .into_iter()
            .map(|file| crate::journal::InputFile {
                input_id: file.field,
                digest: file.digest,
                length: file.length,
                media_type: file.media_type,
                order: file.order,
            })
            .collect(),
        models,
        attention_kernel: spec.attention_kernel,
        gpus: spec.gpus,
        bits: spec.bits,
        hub: None,
        tensorfs,
        providers: providers
            .map(|p| crate::published::Providers {
                huggingface: p.huggingface,
                civitai: p.civitai,
            })
            .unwrap_or_default(),
        weights_destination: spec.weights_destination,
        owner: spec.owner,
        catalog_revision: spec.catalog_revision,
        application: String::new(),
        digest: identity_digest,
        journaled,
    }, hub))
}

fn digest(text: &str) -> Result<Vec<u8>, Error> {
    let invalid = || Error::validation("invalid_digest", "a digest is sha256:<64 hex>");
    let hex = text
        .strip_prefix("sha256:")
        .filter(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or_else(invalid)?;
    (0..64)
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| invalid()))
        .collect()
}

fn spell(bytes: &[u8]) -> String {
    format!("sha256:{}", tensorfs::sha256::hex(bytes))
}

/// Canonical JSON bytes as the wire's raw JSON; none when empty or not JSON.
fn raw(bytes: Vec<u8>) -> Option<v1::Json> {
    String::from_utf8(bytes).ok().and_then(|text| v1::Json::parse(text).ok())
}

/// The current outcome carries a stable code and a plain detail separately.
fn reason(body: &Value) -> v1::Reason {
    let origin = match body["cause"]["origin"].as_u64() {
        Some(1) => "author",
        Some(2) => "runtime",
        Some(6) => "request",
        _ => "machine",
    };
    let mut reason = message_reason(
        body["error_code"].as_str().unwrap_or_default(),
        body["safe_message"].as_str().unwrap_or_default(),
        origin,
    );
    reason.detail = body["failure_detail"].as_str().unwrap_or_default().into();
    reason
}

fn message_reason(code: &str, message: &str, origin: &str) -> v1::Reason {
    if code.is_empty() {
        return v1::Reason {
            code: "failure_record_invalid".into(),
            message: "the recorded failure has no structured error code".into(),
            origin: "machine".into(),
            ..Default::default()
        };
    }
    v1::Reason {
        code: code.into(),
        message: message.into(),
        origin: origin.into(),
        ..Default::default()
    }
}

/// One revision's parts: (digest, length) each.
type Parts = Vec<(Vec<u8>, u64)>;

/// Translates the engine's event pages into the run's log, numbering each output's revisions.
#[derive(Default)]
struct Log {
    revisions: HashMap<(String, u32), u64>,
    products: Vec<v1::Product>,
    /// Each item's latest revision as parts (digest, length): a revision whose parts extend
    /// them appends.
    parts: HashMap<(String, u32), Parts>,
}
impl Log {
    fn event(&mut self, event: domain::MachineExecutionEvent) -> Option<v1::RunEvent> {
        let body: Value =
            serde_json::from_slice(&event.body_canonical_bytes).unwrap_or(Value::Null);
        let kind = match event.kind.as_str() {
            "state" | "running" => {
                let state = body["state"].as_str().unwrap_or("running").replace("starting", "queued");
                v1::Event::State(v1::RunState {
                    // Running from this event: its time counts from here.
                    started_at_ms: if state == "running" { event.at_ms } else { 0 },
                    state,
                    sequence: event.sequence,
                    attempt: event.attempt_ordinal as u32,
                    waiting: body["waiting_reason"].as_str().unwrap_or_default().into(),
                    ..Default::default()
                })
            }
            "progress" => {
                let payload = &body["payload"];
                v1::Event::Progress(v1::Progress {
                    step_ms: payload["step_ms"].as_f64().unwrap_or(0.0),
                    step_avg_ms: payload["step_avg_ms"].as_f64().unwrap_or(0.0),
                    stage: payload["stage"].as_str().unwrap_or_default().into(),
                    fraction: payload["overall_fraction"].as_f64().unwrap_or(-1.0),
                    completed: payload["position"].as_u64().unwrap_or(0),
                    total: payload["total"].as_u64().unwrap_or(0),
                    bytes_done: payload["bytes_done"].as_u64().unwrap_or(0),
                    bytes_total: payload["bytes_total"].as_u64().unwrap_or(0),
                    stage_fraction: payload["stage_fraction"].as_f64(),
                })
            }
            "call" => v1::Event::Call(v1::Call {
                run: body["request"].as_str().unwrap_or_default().into(),
                index: body["index"].as_u64().unwrap_or_default() as u32,
                function: body["export"].as_str().unwrap_or_default().into(),
                label: body["label"].as_str().unwrap_or_default().into(),
                status: body["status"].as_str().unwrap_or_default().into(),
                called_at_ms: body["called_unix_ms"].as_i64().unwrap_or_default(),
                finished_at_ms: body["finished_unix_ms"].as_i64().unwrap_or_default(),
                measurements: match &body["measurements"] {
                    Value::Null => None,
                    measured => raw(crate::boundary_json::exact(measured)),
                },
                memoized: body["memoized"].as_bool().unwrap_or_default(),
                computation_digest: body["computation_digest"].as_str().unwrap_or_default().into(),
                unmemoized: body["unmemoized"].as_str().unwrap_or_default().into(),
                reason: (!body["error_code"].as_str().unwrap_or_default().is_empty()
                    || matches!(
                        body["status"].as_str(),
                        Some("failed" | "canceled" | "refused")
                    ))
                .then(|| {
                    message_reason(
                        body["error_code"].as_str().unwrap_or_default(),
                        body["error"].as_str().unwrap_or_default(),
                        "runtime",
                    )
                }),
            }),
            "log" => match body.get("gpu") {
                Some(gpu) => v1::Event::Gpu(v1::GpuAllocation {
                    state: gpu["state"].as_str().unwrap_or_default().into(),
                    gpus: gpu["gpus"].as_array().into_iter().flatten().filter_map(|g| g.as_u64()).map(|g| g as u32).collect(),
                    width: gpu["width"].as_u64().unwrap_or_default() as u32,
                    behind: gpu["behind"].as_str().unwrap_or_default().into(),
                }),
                None => v1::Event::Log(v1::LogLine {
                    level: body["level"].as_str().unwrap_or("info").into(),
                    text: body["text"].as_str().unwrap_or_default().into(),
                }),
            },
            "product" => {
                let product = event.product?;
                let list = product.op == domain::RunProductOp::Append as i32;
                let index = if list { product.index + 1 } else { 0 };
                let rev = self
                    .revisions
                    .entry((product.output.clone(), index))
                    .or_default();
                *rev += 1;
                let (length, digest) = product.content.as_ref()
                    .map(|content| (content.length, spell(&content.digest)))
                    .unwrap_or_default();
                let parts: Parts = product
                    .parts
                    .iter()
                    .filter_map(|p| p.content.as_ref().map(|c| (c.digest.clone(), c.length)))
                    .collect();
                let previous = self
                    .parts
                    .insert((product.output.clone(), index), parts.clone());
                let appended_from = previous
                    .filter(|before| {
                        !before.is_empty()
                            && before.len() <= parts.len()
                            && parts.starts_with(before)
                    })
                    .map(|before| before.iter().map(|(_, length)| length).sum());
                let converted = v1::Product {
                    output: product.output,
                    checkpoint: product.checkpoint,
                    index,
                    rev: *rev,
                    length,
                    digest,
                    media_type: product.media_type,
                    label: product.label,
                    duration_us: product.parts.iter().map(|p| p.duration_us).sum(),
                    appended_from,
                };
                self.products.retain(|p| {
                    (p.output.as_str(), p.index) != (converted.output.as_str(), converted.index)
                });
                self.products.push(converted.clone());
                v1::Event::Product(converted)
            }
            "outcome" => {
                let outcome = event.outcome?;
                let body: Value =
                    serde_json::from_slice(&outcome.outcome_canonical_bytes).unwrap_or(Value::Null);
                let status = match body["status"].as_u64() {
                    Some(1) => "succeeded",
                    Some(4) => "canceled",
                    _ => "failed",
                };
                let result = body["result"]["inline_result"]
                    .as_str()
                    .and_then(|encoded| STANDARD.decode(encoded).ok())
                    .and_then(raw);
                v1::Event::Outcome(v1::Outcome {
                    status: status.into(),
                    reason: (status != "succeeded").then(|| reason(&body)),
                    result,
                    outputs: self.products.clone(),
                    triage: body.get("triage_bundle").is_some(),
                    adjustments: adjustments(&body["adjustments"]),
                    // `log` adds the measurements, the execution time and the memo.
                    ..Default::default()
                })
            }
            _ => return None,
        };
        Some(v1::RunEvent {
            sequence: event.sequence,
            at_ms: event.at_ms as i64,
            event: kind,
        })
    }
}

/// Accepts a run, or answers the one a re-sent spec already names (false). The run reads its
/// Hub's public content anonymously, and private content under its owner's newest bearer, which
/// a re-sent spec's renews too.
fn accept(runs: &Arc<crate::runs::Runs>, actor: &str, id: &str, spec: crate::runs::Spec, hub: Option<v1::HubAccess>) -> Result<bool, Error> {
    let spec = reach(actor, spec, hub)?;
    if runs.existing(actor, id, &spec.digest)?.is_some() {
        return Ok(false);
    }
    runs.submit(actor, id, spec)?;
    Ok(true)
}

/// `spec` reading `hub`, under `actor`'s newest bearer there when its TensorFS options carry one
/// for that origin (a token goes only to the origin it was given for).
fn reach(actor: &str, mut spec: crate::runs::Spec, hub: Option<v1::HubAccess>) -> Result<crate::runs::Spec, Error> {
    if let Some(hub) = hub {
        let origin = crate::hub::origin_key(&hub.origin);
        let bearer = (spec.tensorfs.as_ref())
            .filter(|options| origin.is_some() && crate::hub::origin_key(&options.endpoint) == origin)
            .and_then(|options| crate::hub::Bearer::offer(actor, &hub.origin, &options.token, options.expires_unix_ms));
        let ca = Some(hub.ca_der).filter(|der| !der.is_empty());
        let source = crate::hub::Source::new(&hub.origin, ca, bearer).map_err(|e| refused("hub_access_invalid", &e.0))?;
        spec.hub = Some(source);
    }
    Ok(spec)
}

/// The runs a restart interrupted while they prepared: each prepares again from its journaled
/// spec, or ends FAILED with why it cannot.
pub(crate) fn resume(runs: &Arc<crate::runs::Runs>) -> std::io::Result<()> {
    for (run, actor, sent) in runs.service.engine.interrupted()? {
        let spec = serde_json::from_slice(&sent)
            .map_err(|e| refused("preparation_interrupted", format!("its kept spec is unreadable: {e}")))
            .and_then(spec_of)
            .and_then(|(spec, hub)| reach(&actor, spec, hub));
        let failure = match spec {
            Ok(spec) => {
                runs.prepare_later(&actor, &run, spec)?;
                continue;
            }
            Err(error) => crate::journal::Failure::machine(&error.code, error.message),
        };
        runs.service.engine.end_preparation(&run, crate::journal::Outcome::Failed(failure))?;
    }
    Ok(())
}

/// Accepts the run `request` names: whether it is new, and its state.
pub(super) async fn submit<B: MachineBackend>(
    backend: &Arc<B>,
    actor: VerifiedActor,
    request: v1::RunRequest,
) -> Result<(bool, v1::RunState), Error> {
    let runs = backend
        .runs()
        .ok_or_else(|| Error::not_found("capability_unavailable", "this machine takes no runs"))?;
    let (spec, hub) = spec_of(request.spec)?;
    let (id, signer) = (request.id, crate::machine_api::actor_id(actor));
    let accepted = id.clone();
    let created = tokio::task::spawn_blocking(move || accept(&runs, &signer, &accepted, spec, hub))
        .await
        .map_err(|_| Error::internal("machine_operation_stopped", "machine operation stopped"))??;
    let run = id.clone();
    let current = blocking(backend, move |backend| backend.get(actor, query(backend, actor, &run)?)).await?;
    Ok((created, state(&id, &current)))
}

/// A run's log after `cursor`, its current state first, until its outcome or the caller leaves;
/// it holds no admission. An output-limited grant (`limit`) sees only what [`visible`] keeps.
pub(super) async fn follow<B: MachineBackend>(
    backend: Arc<B>,
    actor: VerifiedActor,
    id: String,
    cursor: u64,
    limit: Option<Grant>,
) -> Result<mpsc::Receiver<Result<v1::RunEvent, Error>>, Error> {
    let run = id.clone();
    let current = blocking(&backend, move |backend| backend.get(actor, query(backend, actor, &run)?)).await?;
    let (sender, receiver) = mpsc::channel(256);
    let snapshot = v1::RunEvent { sequence: 0, at_ms: 0, event: v1::Event::State(state(&id, &current)) };
    let _ = sender.try_send(Ok(snapshot));
    tokio::spawn(async move {
        if let Err(error) = log(backend, actor, id, cursor, limit, &sender).await {
            let _ = sender.send(Err(error)).await;
        }
    });
    Ok(receiver)
}

async fn log<B: MachineBackend>(
    backend: Arc<B>,
    actor: VerifiedActor,
    id: String,
    cursor: u64,
    limit: Option<Grant>,
    sender: &mpsc::Sender<Result<v1::RunEvent, Error>>,
) -> Result<(), Error> {
    // The log is read from its start, so a reattach numbers each output's revisions and lists
    // the outcome's outputs as the run made them; only entries past the cursor are sent.
    let (mut log, mut after) = (Log::default(), 0);
    loop {
        let run = id.clone();
        let page = blocking(&backend, move |backend| {
            let execution = Some(query(backend, actor, &run)?);
            backend.events(actor, domain::MachineExecutionEventsQuery { execution, after, limit: 256, wait: true })
        })
        .await?;
        let mut ended = false;
        for event in page.events {
            after = after.max(event.sequence);
            let Some(mut converted) = log.event(event) else { continue };
            ended |= matches!(converted.event, v1::Event::Outcome(_));
            let hidden = limit.as_ref().is_some_and(|grant| !visible(grant, &mut converted));
            if converted.sequence <= cursor || hidden {
                continue;
            }
            if let (v1::Event::Outcome(outcome), None) = (&mut converted.event, &limit) {
                let run = id.clone();
                let (measurements, execution_ms, memo) = blocking(&backend, move |backend| {
                    let run = query(backend, actor, &run)?;
                    let measured = backend.measurements(actor, run.clone())?;
                    let executed = backend.execution_ms(actor, run.clone())?;
                    Ok((measured, executed, backend.memo(actor, run)?))
                })
                .await?;
                outcome.measurements = measurements.and_then(raw);
                outcome.execution_ms = execution_ms;
                (outcome.computation_digest, outcome.memoized, outcome.unmemoized) =
                    (memo.computation, memo.memoized, memo.unmemoized);
            }
            if sender.send(Ok(converted)).await.is_err() {
                return Ok(());
            }
        }
        if ended || sender.is_closed() {
            return Ok(());
        }
        after = after.max(page.next_after);
    }
}
