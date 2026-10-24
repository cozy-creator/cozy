//! Child calls. Every call an attempt makes to a callable of
//! its environment arrives on the seam (`child_call`) and becomes a run of its own,
//! `<parent>/<call_index>`, under the parent's signer: same records, dispatched like any run,
//! canceled with the parent. A child's result and files reach the parent through the seam
//! (`state`), its files copied into the parent's spool. A memoized child's success is held when
//! it completes, whatever its parent then does: a parent canceled or failed and run again finds
//! its finished calls answered.
use crate::{
    catalog::HeldGeneration,
    device_executor::{Answer, ByteGrant, CallInterface, CallProgress, Frame, Kind},
    execution::Engine,
    journal::{Execution, Failure, InputFile, State},
    launch_identity::LaunchIdentity,
    runs::Runs,
    service::Service,
};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Weak},
};

pub struct Calls {
    identity: Option<LaunchIdentity>,
    service: Weak<Service>,
    /// Children prepare as runs do, with their parent's context.
    runs: Weak<Runs>,
    /// Running attempts that may call, by execution id.
    parents: Mutex<HashMap<String, Arc<Parent>>>,
    machine: Arc<crate::origin::Machine>,
    /// Each live call's fetches, by run: its cancel ends them at once.
    fetches: Mutex<HashMap<String, Vec<Weak<crate::access::Fetch>>>>,
}

/// A running attempt that calls: its signer, spool, callables and calls.
struct Parent {
    id: String,
    request: String,
    actor: String,
    spool: PathBuf,
    /// `(module, export)` of each invocable it may call: the App it belongs to (empty: its own
    /// package's; else a callee's in its environment) and the entrypoint registered for it.
    callables: Callables,
    /// It is itself memoized: each call it makes must be too, or replaying its result would
    /// skip that call's effects.
    pure: bool,
    /// The callables declared `memoize=True` (with or without an operation identity).
    pure_callables: std::collections::HashSet<(String, String)>,
    /// The parent's own file inputs: a child may be handed any of them.
    inputs: Vec<InputFile>,
    calls: Mutex<CallTable>,
}

impl Parent {
    /// What the child call `request` measured of where it is now: the bytes done of the download
    /// it is in, of their total, and its stage's steady step time.
    fn child_measured(&self, engine: &Engine, request: &str) -> Option<Value> {
        let child = self.calls.lock().unwrap().by_index.values().find(|c| c.request == request)?.child.clone();
        let progress: Value = serde_json::from_str(&engine.get(&child).ok()?.progress?).ok()?;
        let total = progress["bytes_total"].as_u64().filter(|total| *total > 0);
        let mut measured = json!({});
        if let Some(total) = total {
            (measured["bytes_done"], measured["bytes_total"]) = (progress["bytes_done"].as_u64()?.min(total).into(), total.into());
        }
        if let Some(steady) = progress.get("step_avg_ms") {
            measured["step_avg_ms"] = steady.clone();
        }
        Some(measured)
    }

    fn callable(&self, frame: &Frame) -> Result<&(String, String), (&'static str, String)> {
        self.callables
            .get(&(frame.module.clone(), frame.export.clone()))
            .ok_or((
                "child_undeclared",
                format!(
                    "{}.{} is not an invocable of this package or its environment's",
                    frame.module, frame.export
                ),
            ))?
            .as_ref()
            .ok_or((
                "child_ambiguous",
                format!(
                    "{}.{} is registered by more than one App in this environment",
                    frame.module, frame.export
                ),
            ))
    }

    /// A call of this attempt has not ended: its executor is watched, and the parent waits on it.
    fn unfinished(&self, engine: &Engine) -> bool {
        let calls = self.calls.lock().unwrap();
        calls.by_index.values().any(|call| {
            engine
                .get(&call.child)
                .is_ok_and(|record| !record.state.terminal())
        })
    }
}

/// An attempt that may call: its calls' record, kept for the nudges while it lives.
pub(crate) struct Caller {
    calls: Arc<Calls>,
    parent: Arc<Parent>,
}
impl Caller {
    /// One of its attempt's call frames: a child call, poll, cancel, prefetch or GPU release.
    pub(crate) fn seam(&self, frame: &Frame) -> io::Result<(Answer, Option<File>)> {
        self.calls.seam(&self.parent, frame)
    }
    /// Its executor is still while a call it waits on runs: that call's executor is watched.
    pub(crate) fn waits(&self, engine: &Arc<Engine>) -> crate::process::Waits {
        let (parent, engine) = (self.parent.clone(), engine.clone());
        Arc::new(move || parent.unfinished(&engine))
    }
}
impl Drop for Caller {
    fn drop(&mut self) {
        self.calls.parents.lock().unwrap().remove(&self.parent.id);
    }
}

/// What a package's executor may call, sent on its Start (`interface`: its App's interface
/// beside the executor).
pub(crate) fn call_interfaces(held: &HeldGeneration, interface: &Path) -> Vec<CallInterface> {
    invocables(&held.record, &held.record.application, &held.record.interface, interface).0
}

#[derive(Default)]
struct CallTable {
    /// The parent's nudge socket (`child_events`): a byte when one of its calls may have moved.
    watcher: Option<platform::ipc::LocalStream>,
    by_index: HashMap<u64, ChildCall>,
    /// Child files granted to the parent, by digest: later children may be handed them.
    received: HashMap<String, Received>,
}

struct ChildCall {
    child: String,
    /// The callee and the author's label for it, for the parent's `call` event.
    module: String,
    function: String,
    label: String,
    /// Its `call` event is in the parent's log.
    recorded: bool,
    request: String,
    /// The answered result and grants, once the child succeeded (materialized once).
    settled: Option<(String, Vec<ByteGrant>)>,
    progress: (u64, Option<String>),
}

#[derive(Clone)]
struct Received {
    length: u64,
    media_type: String,
}

impl Calls {
    /// The machine's call broker; `runs` prepares children (none: no children).
    pub fn configure(
        service: &Arc<Service>,
        machine: Arc<crate::origin::Machine>,
        runs: Option<&Arc<Runs>>,
    ) -> io::Result<Arc<Self>> {
        let calls = Arc::new(Self {
            identity: service.pool().and_then(|pool| pool.config().identity),
            service: Arc::downgrade(service),
            runs: runs.map_or_else(Weak::new, Arc::downgrade),
            parents: Mutex::new(HashMap::new()),
            machine,
            fetches: Mutex::default(),
        });
        let (watching, engine) = (Arc::downgrade(&calls), service.engine.clone());
        std::thread::Builder::new()
            .name("call-nudges".into())
            .spawn(move || nudge(watching, engine))?;
        service.configure_calls(calls.clone());
        Ok(calls)
    }

    /// `id`'s attempt waits on a call it made.
    pub(crate) fn waiting(&self, engine: &Engine, id: &str) -> bool {
        self.parents.lock().unwrap().get(id).is_some_and(|parent| parent.unfinished(engine))
    }

    pub(crate) fn tensorfs(&self, id: &str, line: crate::device_executor::Line) -> Option<protocol::machine::TensorFSOptions> {
        self.runs.upgrade().and_then(|runs| runs.tensorfs(id, line))
    }

    pub(crate) fn destination(&self, id: &str) -> String {
        self.runs.upgrade().map(|runs| runs.destination(id)).unwrap_or_default()
    }

    /// One call's fetches, under its run's Hub access and ordered by its root's acceptance;
    /// they end with the call, or at once with its run's cancel.
    pub(crate) fn fetch(&self, record: &Execution) -> Arc<crate::access::Fetch> {
        let runs = self.runs.upgrade();
        let hub = runs.as_ref().and_then(|runs| runs.hub(&record.id));
        let fetch = crate::access::Fetch::new(self.machine.clone(), hub, record.root_seq());
        fetch.replaces(runs.and_then(|runs| runs.ahead(&record.id)));
        let mut fetches = self.fetches.lock().unwrap();
        let live = fetches.entry(record.id.clone()).or_default();
        live.retain(|fetch| fetch.strong_count() > 0);
        live.push(Arc::downgrade(&fetch));
        fetch
    }

    /// `record`'s attempt as a caller of its environment's callables (`document`: its App's
    /// interface, at `interface` beside its executor), and the call interfaces it is sent.
    pub(crate) fn caller(
        self: &Arc<Self>,
        record: &Execution,
        held: &HeldGeneration,
        document: &Value,
        interface: &Path,
        spool: &Path,
    ) -> (Caller, Vec<CallInterface>) {
        let (rows, callables) = invocables(&held.record, &record.invocation.module, document, interface);
        let submission = record.submission.as_ref();
        let parent = Arc::new(Parent {
            id: record.id.clone(),
            request: submission.map_or_else(|| record.id.clone(), |s| s.request_id.clone()),
            actor: submission.map(|s| s.actor.clone()).unwrap_or_default(),
            spool: spool.into(),
            callables,
            pure: declarations(document).any(|(declared, entry)| {
                entry["name"].as_str() == Some(record.invocation.entrypoint.as_str())
                    && declared["memoize"].as_bool() == Some(true)
            }),
            pure_callables: pure_callables(&held.record),
            inputs: record.invocation.inputs.clone(),
            calls: Mutex::new(CallTable::default()),
        });
        self.parents.lock().unwrap().insert(record.id.clone(), parent.clone());
        (Caller { calls: self.clone(), parent }, rows)
    }

    /// Write roots no unfinished run needs (`objects::Objects::sweep`).
    pub fn sweep_objects(&self) -> io::Result<usize> {
        match self.runs.upgrade() {
            Some(runs) => runs.objects.sweep(),
            None => Ok(0),
        }
    }

    /// Every unfinished child of `parent` is canceled with it.
    pub(crate) fn stop_children(&self, parent: &str) {
        let Some(service) = self.service.upgrade() else {
            return;
        };
        let Ok(children) = service.engine.children(parent) else {
            return;
        };
        for child in &children {
            cancel(&service.engine, child);
        }
    }

    /// A run's attempt has stopped: its children follow and prepare no more. A run that is
    /// queued again (deferred before it started) runs again and nothing ends.
    pub(crate) fn ended(&self, id: &str) {
        let Some(service) = self.service.upgrade() else {
            return;
        };
        if service.engine.get(id).is_ok_and(|r| r.state == State::Queued) {
            return;
        }
        self.fetches.lock().unwrap().remove(id);
        self.stop_children(id);
        if let Some(runs) = self.runs.upgrade() {
            runs.end(id);
        }
    }

    /// A cancel reached a run: its fetches end at once, and when no attempt runs (it was
    /// queued) its children end too.
    pub fn canceled(&self, record: &Execution) {
        if record.cancel_actor.is_none() {
            return;
        }
        for fetch in self.fetches.lock().unwrap().remove(&record.id).into_iter().flatten() {
            if let Some(fetch) = fetch.upgrade() {
                fetch.end();
            }
        }
        if record.state.terminal() {
            self.ended(&record.id);
        }
    }

    /// `child_call`: the call's run, accepted once per index (its intent must not change).
    fn child_call(&self, parent: &Parent, frame: &Frame) -> Result<Answer, (&'static str, String)> {
        let service = self
            .service
            .upgrade()
            .ok_or(("child_call_refused", "machine is stopping".into()))?;
        let (application, entrypoint) = parent.callable(frame)?;
        let callable = (frame.module.clone(), frame.export.clone());
        if parent.pure && !parent.pure_callables.contains(&callable) {
            return Err((
                "child_impure",
                format!(
                    "a memoized function cannot call {}.{}, which is not memoized",
                    frame.module, frame.export
                ),
            ));
        }
        let input: Value = crate::boundary_json::parse(frame.payload.as_bytes()).map_err(|e| {
            (
                "child_call_refused",
                format!("call request is not JSON: {e}"),
            )
        })?;
        // Every call is a serving call: its request, and the models it passes by value to the
        // callee's slots (null leaves a slot to the choice its parent carries for it).
        let (request_input, choices) = serving_call(&input, entrypoint)?;
        let runs = self
            .runs
            .upgrade()
            .ok_or(("child_call_refused", "machine is stopping".into()))?;
        let mut calls = parent.calls.lock().unwrap();
        let inputs = child_inputs(&request_input, parent, &calls.received);
        let request = format!("{}/{}", parent.request, frame.call_index);
        let intent = child_intent(&frame.module, &frame.export, &input);
        let caller = service
            .engine
            .get(&parent.id)
            .map_err(|e| ("child_call_refused", e.to_string()))?;
        let record = runs
            .child(&caller, &request, &intent, application, entrypoint, request_input, inputs, choices)
            .map_err(|refusal| match refusal.code {
                "run_id_conflict" => (
                    "child_call_refused",
                    "an existing call index changed its exact intent".to_string(),
                ),
                code => (code, refusal.message),
            })?;
        calls.by_index.entry(frame.call_index).or_insert(ChildCall {
            child: record.id.clone(),
            module: frame.module.clone(),
            function: frame.export.clone(),
            label: frame.progress_label.clone(),
            recorded: false,
            request: request.clone(),
            settled: None,
            progress: (0, None),
        });
        drop(calls);
        self.state(parent, frame)
    }

    /// `child_poll`: the call's state; a success carries its result and the parent's grants.
    fn state(&self, parent: &Parent, frame: &Frame) -> Result<Answer, (&'static str, String)> {
        let service = self
            .service
            .upgrade()
            .ok_or(("child_call_refused", "machine is stopping".into()))?;
        let mut calls = parent.calls.lock().unwrap();
        let CallTable {
            by_index, received, ..
        } = &mut *calls;
        let call = by_index.get_mut(&frame.call_index).ok_or((
            "child_call_refused",
            "parent call has not been activated".into(),
        ))?;
        let record = service
            .engine
            .get(&call.child)
            .map_err(|e| ("child_call_refused", e.to_string()))?;
        let mut answer = Answer::ok(frame.seq);
        answer.child_request_id = call.request.clone();
        if record.progress != call.progress.1 && record.progress.is_some() {
            call.progress = (call.progress.0 + 1, record.progress.clone());
        }
        if let Some(detail) = &call.progress.1 {
            let mut payload = serde_json::from_str::<Value>(detail).unwrap_or_else(|_| json!({}));
            if let Some(fields) = payload.as_object_mut() {
                fields.insert("call_request".into(), call.request.clone().into());
                fields.insert("call_attempt".into(), record.attempt.into());
            }
            answer.progress = Some(CallProgress { sequence: call.progress.0, payload });
        }
        match record.state {
            State::Completed => {
                if call.settled.is_none() {
                    let result = record
                        .result
                        .as_ref()
                        .ok_or(("child_failed", "child result is absent".into()))?;
                    let directory = parent
                        .spool
                        .join("child-results")
                        .join(frame.call_index.to_string());
                    let (text, rows, held) = receive(&service.engine.root, result, &parent.spool, &directory, self.identity)
                        .map_err(|e| ("child_result_unavailable", e.to_string()))?;
                    let settled = (text, rows);
                    // Each file is the signer's object too: a later child may be handed it.
                    let runs = self
                        .runs
                        .upgrade()
                        .ok_or(("child_call_refused", "machine is stopping".into()))?;
                    for (local, object) in &held {
                        runs.objects
                            .adopt(&parent.actor, &parent.id, local, object)
                            .map_err(|e| ("child_result_unavailable", e.to_string()))?;
                    }
                    for row in &settled.1 {
                        received.insert(
                            row.digest.clone(),
                            Received { length: row.length, media_type: row.media_type.clone() },
                        );
                    }
                    call.settled = Some(settled);
                }
                let (result, grants) = call.settled.clone().expect("settled above");
                answer.state = "succeeded".into();
                answer.result = result;
                answer.byte_grants = grants;
            }
            State::Failed => {
                let failure = Failure::decode(record.failure.as_deref().unwrap_or_default())
                    .map_err(|error| ("failure_record_invalid", error.to_string()))?;
                answer.ok = false;
                answer.code = failure.code;
                answer.detail = failure.message;
                answer.traceback = child_traceback(&service.engine, &record.id);
            }
            State::Canceled => {
                answer.ok = false;
                answer.code = "child_canceled".into();
                answer.detail = "the child run was canceled".into();
            }
            _ => answer.state = "pending".into(),
        }
        if record.state.terminal() && !call.recorded {
            call.recorded = record_call(&service.engine, parent, frame.call_index, call, &record);
        }
        Ok(answer)
    }

    fn seam(&self, parent: &Parent, frame: &Frame) -> io::Result<(Answer, Option<File>)> {
        let answered = match frame.kind {
            Kind::ChildCall => self.child_call(parent, frame),
            Kind::ChildPoll => self.state(parent, frame),
            Kind::ChildCancel => {
                let call = parent
                    .calls
                    .lock()
                    .unwrap()
                    .by_index
                    .get(&frame.call_index)
                    .map(|c| c.child.clone());
                if let (Some(child), Some(service)) = (call, self.service.upgrade()) {
                    if let Ok(record) = service.engine.get(&child) {
                        cancel(&service.engine, &record);
                    }
                }
                let mut answer = Answer::ok(frame.seq);
                answer.state = "pending".into();
                Ok(answer)
            }
            Kind::ChildForget => {
                let mut calls = parent.calls.lock().unwrap();
                let settled = calls.by_index.get(&frame.call_index).and_then(|c| {
                    let record = self.service.upgrade()?.engine.get(&c.child).ok()?;
                    Some(record.state.terminal())
                });
                if settled == Some(true) {
                    calls.by_index.remove(&frame.call_index);
                }
                Ok(Answer::ok(frame.seq))
            }
            Kind::ChildEvents => {
                let (ours, theirs) = platform::ipc::pair()?;
                ours.set_nonblocking(true)?;
                parent.calls.lock().unwrap().watcher = Some(ours);
                return Ok((
                    Answer::ok(frame.seq),
                    Some(File::from(platform::handles::Owned::from(theirs))),
                ));
            }
            // A hint about a future call: TensorD may prepare its weights and ask Runtime
            // to build the construction. This answer does not promise completed loading.
            Kind::ModelPrefetch => self.model_prefetch(parent, frame),
            Kind::GpuRelease => {
                let service = self.service.upgrade()
                    .ok_or_else(|| io::Error::other("the machine is stopping"))?;
                let record = service.engine.get(&parent.id)?;
                // Only a deviceless root ends its lane: a call on a GPU cannot give up its own
                // device safely, and a child cannot end its root's future GPU work.
                if let (true, Some(pool)) = (record.invocation.parent.is_empty() && record.invocation.degree == 0, service.pool()) {
                    pool.release(&service.engine, record.root_seq());
                }
                Ok(Answer::ok(frame.seq))
            }
            _ => return Ok((Answer::unavailable(frame.seq), None)),
        };
        Ok((
            answered.unwrap_or_else(|(code, detail)| crate::device_executor::refused(frame.seq, code, detail)),
            None,
        ))
    }

    fn model_prefetch(&self, parent: &Parent, frame: &Frame) -> Result<Answer, (&'static str, String)> {
        let (application, entrypoint) = parent.callable(frame)?;
        // Explicit choices use the ordinary child-call decoder and selection path; a hint
        // cannot turn a model name into byte authority.
        let raw = if frame.payload.is_empty() { "{}" } else { &frame.payload };
        let models = crate::boundary_json::parse(raw.as_bytes())
            .map_err(|error| ("child_call_refused", format!("prefetch choices are not JSON: {error}")))?;
        let choices = model_choices(&models, entrypoint)?;
        let runs = self.runs.upgrade().ok_or((
            "model_prefetch_unsupported", "machine does not prepare child models".into(),
        ))?;
        let service = self.service.upgrade().ok_or((
            "model_prefetch_unsupported", "machine is stopping".into(),
        ))?;
        let caller = service.engine.get(&parent.id)
            .map_err(|error| ("child_call_refused", error.to_string()))?;
        runs.prefetch(&caller, application, entrypoint, choices)
            .map_err(|error| (error.code, error.message))?;
        Ok(Answer::ok(frame.seq))
    }
}

/// A child follows its parent's stop: canceled.
fn cancel(engine: &Engine, child: &Execution) {
    let actor = child.submission.as_ref().map(|s| s.actor.as_str()).unwrap_or_default();
    if let Err(error) = engine.cancel(&child.id, actor) {
        tracing::warn!("child run {} of {} remains: {error}", child.id, child.invocation.parent);
    }
}

/// What an attempt's progress frames told so far: its completed units, and its stage's steps after
/// the first (which carries the stage's setup) and their time.
#[derive(Default)]
pub(crate) struct Seen {
    pub completed: u64,
    steps: (String, u32, f64),
}

/// One progress frame of run `id`'s executor: a package log line, or its stage, with its steady
/// step time (the child call's it relays, when it is one, and that call's download bytes).
pub(crate) fn observe(engine: &Engine, id: &str, seen: &mut Seen, frame: &Frame, caller: Option<&Caller>) {
    if !(frame.request_id.is_empty() || frame.request_id == id) {
        return;
    }
    if let Some((level, text)) = frame.log() {
        let _ = engine.append_package_log(id, level, &text);
        return;
    }
    if frame.stage.is_empty() {
        return;
    }
    // `advance` is the attempt's monotone position in completed units (Runtime
    // `Telemetry`); a zero one is telemetry, not work.
    seen.completed = seen.completed.max(frame.advance);
    let mut payload = json!({"stage": frame.stage.chars().take(120).collect::<String>(), "step_ms": frame.step_ms.unwrap_or(0.0)});
    for (name, value) in [
        ("stage_fraction", frame.stage_fraction.map(Value::from)),
        ("overall_fraction", frame.overall_fraction.map(Value::from)),
        ("position", frame.position.map(Value::from)),
        ("total", frame.total.map(Value::from)),
    ] {
        if let Some(value) = value {
            payload[name] = value;
        }
    }
    let call = frame.call_request.as_deref().filter(|call| !call.is_empty());
    match (call, frame.position, frame.step_ms.filter(|ms| *ms > 0.0)) {
        (Some(call), ..) => {
            let measured = caller.and_then(|caller| caller.parent.child_measured(engine, call));
            for (name, value) in measured.as_ref().and_then(Value::as_object).into_iter().flatten() {
                payload[name] = value.clone();
            }
        }
        (None, Some(position), Some(ms)) => {
            if seen.steps.0 != frame.stage {
                seen.steps = (frame.stage.clone(), 0, 0.0);
            }
            if position > 1 {
                (seen.steps.1, seen.steps.2) = (seen.steps.1 + 1, seen.steps.2 + ms);
            }
            if seen.steps.1 > 0 {
                payload["step_avg_ms"] = (seen.steps.2 / f64::from(seen.steps.1)).into();
            }
        }
        _ => {}
    }
    let _ = engine.observe_progress(id, seen.completed, payload.to_string());
}

/// `(module, export)` of every invocable of the environment declared `memoize=True`.
fn pure_callables(generation: &crate::catalog::Generation) -> std::collections::HashSet<(String, String)> {
    std::iter::once(&generation.interface)
        .chain(generation.callees.iter().map(|c| &c.interface))
        .flat_map(declarations)
        .filter(|(declared, _)| declared["memoize"].as_bool() == Some(true))
        .filter_map(|(declared, _)| {
            Some((declared["module"].as_str()?.into(), declared["export"].as_str()?.into()))
        })
        .collect()
}

/// Each declared invocable of an interface: its declaration and its entry.
fn declarations(interface: &Value) -> impl Iterator<Item = (&Value, &Value)> {
    interface["entrypoints"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|entry| (&entry["invocable"], entry))
}

/// Every App this environment holds: its own exports, including internal ones, and the
/// other Apps' public exports. A callee can call its own App or another dependency too.
/// None names an export owned by different Apps: only that call is refused.
type Callables = HashMap<(String, String), Option<(String, String)>>;
fn invocables(
    generation: &crate::catalog::Generation,
    application: &str,
    own: &Value,
    path: &Path,
) -> (Vec<CallInterface>, Callables) {
    let mut rows = vec![];
    let mut callables: Callables = HashMap::new();
    // A callee module's typed callers need only its own callables: its first row carries that
    // part of its App's interface, so the whole App travels once, not once per export.
    let mut documented = std::collections::HashSet::new();
    let apps = std::iter::once((application, own, false))
        .chain(std::iter::once((generation.application.as_str(), &generation.interface, false)))
        .chain(generation.callees.iter().map(|c| {
            (c.application.as_str(), &c.interface, c.package == crate::catalog::BUILTIN_PACKAGE)
        }));
    for (owner, interface, builtin) in apps {
        let own = owner == application;
        for (declared, entry) in declarations(interface) {
            let (Some(module), Some(export), Some(name)) = (
                declared["module"].as_str(),
                declared["export"].as_str(),
                entry["name"].as_str(),
            ) else {
                continue;
            };
            // Another package's internal callable is its own business.
            if !own && entry["internal"].as_bool() == Some(true) {
                continue;
            }
            let key = (module.to_string(), export.to_string());
            if let Some(existing) = callables.get_mut(&key) {
                if existing.as_ref().is_some_and(|(app, _)| app != owner) {
                    *existing = None;
                }
                continue;
            }
            callables.insert(key, Some((owner.to_string(), name.to_string())));
            rows.push(CallInterface {
                module: module.into(),
                export: export.into(),
                interface_path: if own { path.into() } else { PathBuf::new() },
                self_call: own,
                // The Runtime binds its own operations from its own code, not a document.
                interface_document: (!own && !builtin && documented.insert(module.to_string()))
                    .then(|| module_interface(interface, module)),
                builtin: (builtin && !own).then(|| "operations".into()),
            });
        }
    }
    (rows, callables)
}

/// An App's interface narrowed to one module's exported callables (the Runtime generates that
/// module's callers from it).
fn module_interface(interface: &Value, module: &str) -> Value {
    let mut narrowed = interface.clone();
    if let Some(Value::Array(entries)) = narrowed.get_mut("entrypoints") {
        entries.retain(|entry| {
            entry["invocable"]["module"].as_str() == Some(module)
                && entry["internal"].as_bool() != Some(true)
        });
    }
    narrowed
}

/// A serving call's request and the model choices it carries: each `models` entry that is a
/// ModelArtifact becomes the exact checkpoint of the callee's slot `<entrypoint>.models.<name>`.
fn serving_call(
    call: &Value,
    entrypoint: &str,
) -> Result<(Value, Vec<crate::api::domain::ModelChoice>), (&'static str, String)> {
    let refuse = |why: &str| ("child_call_refused", format!("serving call {why}"));
    let request = call.get("payload").filter(|p| p.is_object()).ok_or_else(|| refuse("has no request object"))?;
    Ok((request.clone(), model_choices(&call["models"], entrypoint)?))
}

fn model_choices(
    models: &Value,
    entrypoint: &str,
) -> Result<Vec<crate::api::domain::ModelChoice>, (&'static str, String)> {
    let refuse = |why: &str| ("child_call_refused", format!("model choices {why}"));
    if !models.is_null() && !models.is_object() {
        return Err(refuse("must be an object"));
    }
    let mut choices = vec![];
    for (name, artifact) in models.as_object().into_iter().flatten() {
        if artifact.is_null() {
            continue;
        }
        let digest = artifact["checkpoint"].as_str().and_then(|d| d.strip_prefix("sha256:"))
            .filter(|d| d.len() == 64 && d.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
            .ok_or_else(|| refuse(&format!("passes {name} without an exact checkpoint")))?;
        let digest = (0..64).step_by(2).map(|i| u8::from_str_radix(&digest[i..i + 2], 16).unwrap()).collect();
        choices.push(crate::api::domain::ModelChoice {
            parameter: format!("{entrypoint}.models.{name}"),
            manifest: Some(crate::api::domain::Ref {
                digest,
                // Local model resolution obtains the header's length from the shared store.
                length: 0,
            }),
            ..Default::default()
        });
    }
    Ok(choices)
}

/// The child's file inputs: every string of its request naming a file the parent holds (its
/// own inputs, or files its earlier children returned), by field path, as the SDK names them.
fn child_inputs(
    input: &Value,
    parent: &Parent,
    received: &HashMap<String, Received>,
) -> Vec<InputFile> {
    fn walk(
        value: &Value,
        path: &mut Vec<String>,
        order: u32,
        found: &mut Vec<(String, u32, String)>,
    ) {
        match value {
            Value::String(text) if text.starts_with("sha256:") => {
                found.push((path.join("."), order, text.clone()))
            }
            Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    path.push(index.to_string());
                    walk(item, path, index as u32, found);
                    path.pop();
                }
            }
            Value::Object(fields) => {
                for (name, item) in fields {
                    path.push(name.clone());
                    walk(item, path, order, found);
                    path.pop();
                }
            }
            _ => (),
        }
    }
    let mut found = vec![];
    walk(input, &mut vec![], 0, &mut found);
    found
        .into_iter()
        .filter_map(|(input_id, order, digest)| {
            let (length, media_type) = match parent.inputs.iter().find(|i| i.digest == digest) {
                Some(own) => (own.length, own.media_type.clone()),
                None => received
                    .get(&digest)
                    .map(|r| (r.length, r.media_type.clone()))?,
            };
            Some(InputFile {
                input_id,
                digest,
                length,
                media_type,
                order,
            })
        })
        .collect()
}

/// A completed child's result as its parent receives it: each file leaf named by its bytes
/// (`asset_ref` = `digest` = sha256) and granted at a copy in the parent's spool.
/// The files a parent now holds from a child's result, to adopt as its own objects.
type Held = Vec<(PathBuf, tensorfs::format::ObjectRef)>;

fn receive(
    root: &Path,
    result: &crate::journal::ResultRecord,
    spool: &Path,
    directory: &Path,
    identity: Option<LaunchIdentity>,
) -> io::Result<(String, Vec<ByteGrant>, Held)> {
    fn leaves(value: &Value, path: &mut Vec<String>, found: &mut Vec<String>) {
        match value {
            Value::Object(fields) if fields.contains_key("asset_ref") => found.push(path.join(".")),
            Value::Object(fields) => {
                for (name, item) in fields {
                    path.push(name.clone());
                    leaves(item, path, found);
                    path.pop();
                }
            }
            Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    path.push(index.to_string());
                    leaves(item, path, found);
                    path.pop();
                }
            }
            _ => (),
        }
    }
    let mut value = result.value.clone();
    let mut paths = vec![];
    leaves(&value, &mut vec![], &mut paths);
    let (mut grants, mut held) = (vec![], vec![]);
    // Each file the parent now holds, also in its spool as `sha256-<hex>`: it may return it
    // as its own output by the digest it names it by (`device_executor::tree_binding`).
    let mut hold = |local: &Path, sha: &str, length: u64| -> io::Result<()> {
        match fs::hard_link(local, spool.join(format!("sha256-{sha}"))) {
            Err(error) if error.kind() != io::ErrorKind::AlreadyExists => return Err(error),
            _ => (),
        }
        let object = tensorfs::format::ObjectRef { sha256: sha.into(), length };
        held.push((local.to_path_buf(), object));
        Ok(())
    };
    for output_id in paths {
        let pointer = format!("/{}", output_id.replace('.', "/"));
        let node = value
            .pointer_mut(if output_id.is_empty() { "" } else { &pointer })
            .ok_or_else(|| io::Error::other("result leaf vanished"))?;
        let reference = node["asset_ref"].as_str().unwrap_or_default().to_string();
        let binding = result
            .asset_bindings
            .iter()
            .find(|b| b.asset_ref == reference)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "child result file has no binding",
                )
            })?;
        let artifact = result
            .artifacts
            .iter()
            .find(|a| a.name == binding.relative_path)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "child result file is not held")
            })?;
        fs::create_dir_all(directory)?;
        let local = directory.join(format!("{}-{}", grants.len(), artifact.sha256));
        fs::copy(root.join(&artifact.path), &local)?;
        if let Some(identity) = identity {
            identity.readable(&local)?;
        }
        hold(&local, &artifact.sha256, artifact.length)?;
        let digest = format!("sha256:{}", artifact.sha256);
        if binding.media_type == crate::gpu_service::TREE_MEDIA {
            // A tree: its manifest's members materialized as the directory the parent reads.
            let tree = directory.join(format!("{}-tree-{}", grants.len(), artifact.sha256));
            let mut content = 0;
            for (path, object) in crate::trees::parse(&fs::read(&local)?)? {
                let (sha, length) = (object.sha256.as_str(), object.length);
                let relative = Path::new(&path);
                let member = result.artifacts.iter().find(|a| a.sha256 == sha && a.length == length)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "child tree member is not held"))?;
                let target = tree.join(relative);
                fs::create_dir_all(target.parent().unwrap_or(&tree))?;
                fs::copy(root.join(&member.path), &target)?;
                if let Some(identity) = identity {
                    identity.readable(&target)?;
                }
                hold(&target, sha, length)?;
                content += length;
            }
            if let Some(identity) = identity {
                for entry in crate::gpu_service::walk_dirs(&tree)? {
                    identity.readable(&entry)?;
                }
            }
            node["asset_ref"] = digest.clone().into();
            node["digest"] = digest.clone().into();
            node["size_bytes"] = content.into();
            grants.push(ByteGrant {
                output_id: output_id.clone(),
                kind: "tree".into(),
                digest,
                local: tree,
                length: artifact.length,
                content_bytes: content,
                media_type: binding.media_type.clone(),
            });
            continue;
        }
        node["asset_ref"] = digest.clone().into();
        node["digest"] = digest.clone().into();
        node["size_bytes"] = artifact.length.into();
        node["media_type"] = binding.media_type.clone().into();
        grants.push(ByteGrant {
            output_id: output_id.clone(),
            kind: node["kind"].as_str().unwrap_or_default().into(),
            digest,
            local,
            length: artifact.length,
            content_bytes: artifact.length,
            media_type: binding.media_type.clone(),
        });
    }
    let exact = String::from_utf8(crate::boundary_json::exact(&value)).map_err(io::Error::other)?;
    Ok((exact, grants, held))
}

/// A child call's intent: its callable and its input, numbers exact.
fn child_intent(module: &str, export: &str, input: &Value) -> String {
    let semantic = crate::boundary_json::exact(&json!([module, export, input]));
    format!("sha256:{}", tensorfs::sha256::hex_digest(&semantic))
}


/// One byte to every calling attempt's nudge socket each time anything moved: its calls re-poll.
fn nudge(calls: Weak<Calls>, engine: Arc<Engine>) {
    let mut seen = engine.activity_epoch();
    loop {
        seen = engine.wait_activity(seen, None);
        let Some(calls) = calls.upgrade() else {
            return;
        };
        if seen == u64::MAX {
            return;
        }
        for parent in calls.parents.lock().unwrap().values() {
            let calls = parent.calls.lock().unwrap();
            if let (Some(mut watcher), false) = (calls.watcher.as_ref(), calls.by_index.is_empty())
            {
                // A full socket already holds a nudge; one is enough.
                let _ = watcher.write(&[0]);
            }
        }
    }
}

/// Journals a settled child as a `call` event on its parent: the callee, the author's label,
/// how it ended and when, and what its executor measured (`run show` lists each call). True
/// once it is in the parent's log, or the parent has ended and takes no more.
fn record_call(
    engine: &Engine,
    parent: &Parent,
    index: u64,
    call: &ChildCall,
    record: &Execution,
) -> bool {
    let (status, error_code, error) = match record.state {
        State::Completed => ("succeeded", String::new(), String::new()),
        State::Canceled => ("canceled", "child_canceled".into(), "the child run was canceled".into()),
        _ => match Failure::decode(record.failure.as_deref().unwrap_or_default()) {
            Ok(failure) => ("failed", failure.code, failure.message),
            Err(error) => {
                tracing::warn!("run {}: invalid child failure record: {error}", parent.id);
                return false;
            }
        },
    };
    // A failed child's whole cause chain is the caller's too: in its log, as `run show` prints it.
    if status == "failed" {
        let traceback = child_traceback(engine, &record.id);
        let head = format!("call {index} ({}.{}) failed: {error_code}: {error}", call.module, call.function);
        for line in std::iter::once(head.as_str()).chain(traceback.lines()) {
            if let Err(failed) = engine.append_log(&parent.id, "error", line) {
                tracing::warn!("run {}: call {index}'s failure is not in its log: {failed}", parent.id);
                break;
            }
        }
    }
    let measurements = engine
        .with_journal(|journal| journal.measurements(&record.id))
        .ok()
        .flatten()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .unwrap_or(Value::Null);
    let body = json!({
        "request": call.request, "parent": parent.request, "index": index,
        "attempt": record.attempt.max(1), "module": call.module, "export": call.function,
        "label": call.label, "status": status, "error_code": error_code, "error": error,
        "called_unix_ms": record.accepted_at_ms, "finished_unix_ms": record.finished_at_ms,
        "measurements": measurements, "computation_digest": record.memo.computation,
        "memoized": record.memo.memoized, "unmemoized": record.memo.unmemoized,
    });
    match engine.append_call(&parent.id, &record.id, &crate::boundary_json::exact(&body)) {
        Ok(_) => true,
        Err(error) => {
            tracing::warn!("run {}: call {index}: {error}", parent.id);
            false
        }
    }
}

/// A failed child's whole traceback, from its triage bundle; empty when it kept none.
fn child_traceback(engine: &Engine, child: &str) -> String {
    let bundle = engine.triage(child).ok().flatten();
    bundle.and_then(|(_, bundle)| crate::triage::traceback(&bundle)).unwrap_or_default()
}
