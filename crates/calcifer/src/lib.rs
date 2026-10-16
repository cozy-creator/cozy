//! Calcifer, the controller daemon's core: the outbox of runs and their followers over Calcifer's
//! store.
//!
//! Every unsettled run has one driver task. It waits for a machine with an endpoint, writes the
//! run's inputs, crosses the send boundary (one transaction against a cancel) and sends the run
//! under its id, or attaches first when it may already be there. The machine's first frame is
//! its acceptance. From then on, the task follows the run from the durable cursor to its outcome.
//! A Calcifer that stops leaves accepted work running; the next one resumes each run from the
//! cursor.
pub mod api;
pub mod link;
pub mod machines;
pub mod outputs;
mod rentals;
pub mod serve;
pub mod store;

use std::{
    collections::{HashMap, HashSet},
    io::{Read as _, Write as _},
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};

use cozy_proto::{
    calcifer::{Event, On, Progress, Reason, Run, StagedInput, SubmitRun, Submitted},
    ids::{Digest, RunId},
    machine as v1,
};
use ed25519_dalek::{SigningKey, pkcs8::DecodePrivateKey as _};
use prost::Message as _;
use sha2::{Digest as _, Sha256};
use tokio::{
    sync::{Notify, broadcast, watch},
    task::JoinSet,
};
use tokio_stream::{Stream, StreamExt as _};
use tracing::warn;

pub use link::{Link, LinkError};
use store::{Acceptance, Batch, MachineRow, NewRun, RunRow, State, Store, StoreError};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("{0}")]
    Invalid(String),
    #[error("cannot stage input {path}: {message}")]
    Input { path: PathBuf, message: String },
}

pub struct Calcifer {
    store: Store,
    inputs: PathBuf,
    outputs: PathBuf,
    events: broadcast::Sender<(RunId, Event)>,
    live: Mutex<HashMap<RunId, Live>>,
    drivers: Mutex<HashMap<RunId, Arc<Notify>>>,
    machines: Mutex<HashMap<String, Machine>>,
    tasks: Mutex<JoinSet<()>>,
    /// Orders staging a run's inputs against deleting a settled run's.
    staging: Mutex<()>,
    /// Told each time a run is settled: what Calcifer owes may now be nothing.
    owed: watch::Sender<()>,
    exports: outputs::Exports,
    /// Each machine's warm set as it last reported it, and as this Calcifer since sent it.
    warm: Mutex<HashMap<String, Vec<v1::WarmItem>>>,
}

/// What a run is doing now, never stored: the latest of each live event.
#[derive(Default)]
struct Live {
    waiting: Option<String>,
    progress: Option<Progress>,
}

/// A machine's connection, shared by its runs' drivers.
struct Machine {
    changed: watch::Sender<()>,
    link: Option<(store::Endpoint, Link)>,
    /// Runs are sent to one machine one at a time, in the order their drivers ask.
    sending: Arc<tokio::sync::Mutex<()>>,
}

/// Why a driver's step stopped short of its run's end.
enum Stop {
    Unreachable(String),
    Refused(Reason),
    Released(String),
}

impl From<LinkError> for Stop {
    fn from(e: LinkError) -> Self {
        match e {
            LinkError::Refused { code, message } => Stop::Refused(Reason { code, message }),
            LinkError::Released(message) => Stop::Released(message),
            LinkError::NotFound => Stop::Unreachable("the machine does not know the run".into()),
            LinkError::Unreachable(message) | LinkError::Unauthorized(message) => {
                Stop::Unreachable(message)
            }
        }
    }
}

impl From<StoreError> for Stop {
    fn from(e: StoreError) -> Self {
        Stop::Unreachable(e.to_string())
    }
}

impl Calcifer {
    /// Opens Calcifer's home (`~/.cozy/calcifer`) and resumes every run it owes. `outputs` is
    /// where a run's outputs land unless it names its own.
    pub fn open(home: &Path, outputs: &Path) -> Result<Arc<Self>, Error> {
        let store = Store::open(&home.join("state.db"))?;
        let inputs = home.join("inputs");
        std::fs::create_dir_all(&inputs).map_err(|e| Error::Input {
            path: inputs.clone(),
            message: e.to_string(),
        })?;
        let calcifer = Arc::new(Self {
            store,
            inputs,
            outputs: outputs.to_owned(),
            events: broadcast::channel(4096).0,
            live: Mutex::default(),
            drivers: Mutex::default(),
            machines: Mutex::default(),
            tasks: Mutex::default(),
            staging: Mutex::default(),
            owed: watch::channel(()).0,
            exports: outputs::Exports::default(),
            warm: Mutex::default(),
        });
        calcifer.sweep_inputs()?;
        for id in calcifer.store.unsettled()? {
            calcifer.drive(id);
        }
        calcifer.export_outputs()?;
        Ok(calcifer)
    }

    /// Stops every driver where it stands, as if the process ended. Machines keep their runs.
    pub fn close(&self) {
        lock(&self.tasks).abort_all();
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    /// What Calcifer still owes, which a next Calcifer would resume: every unsettled run's id,
    /// each run with an output still to land, and every ask of a Hub not yet answered. A store it
    /// cannot read owes its error.
    pub fn owed(&self) -> Vec<String> {
        let owed = self.store.unsettled().and_then(|ids| {
            let outputs = self.store.exports_owed()?.into_iter();
            let outputs = outputs.map(|(_, number)| format!("outputs of run {number}"));
            let asks = self.store.pending_asks()?.into_iter();
            let asks = asks.map(|ask| format!("the Hub ask {}", ask.key));
            Ok(ids
                .into_iter()
                .map(String::from)
                .chain(outputs)
                .chain(asks)
                .collect())
        });
        owed.unwrap_or_else(|e| vec![e.to_string()])
    }

    /// Tells the exit watch that what Calcifer owes may have shrunk.
    pub(crate) fn owe_less(&self) {
        self.owed.send_replace(());
    }

    /// Changes each time a run is settled, its outputs have all landed, or a Hub ask is answered.
    pub fn owed_changed(&self) -> watch::Receiver<()> {
        self.owed.subscribe()
    }

    /// Every event of every run, as it happens.
    pub fn subscribe(&self) -> broadcast::Receiver<(RunId, Event)> {
        self.events.subscribe()
    }

    /// Records a run under `key` and starts it; the same key again answers the recorded run.
    pub async fn submit(
        self: &Arc<Self>,
        key: &str,
        submit: SubmitRun,
    ) -> Result<Submitted, Error> {
        let spec = v1::RunSpec::decode(submit.spec.as_slice())
            .map_err(|e| Error::Invalid(format!("the run's spec is not a RunSpec: {e}")))?;
        let digest = format!(
            "sha256:{}",
            hex::encode(Sha256::digest(
                serde_json::to_vec(&submit).expect("serializes")
            ))
        );
        let (machine, pinned, wants) = match &submit.on {
            On::Machine(name) => (Some(name.clone()), true, None),
            on => (None, false, Some(on.to_string())),
        };
        let new = NewRun {
            id: RunId::mint(random()),
            kind: submit.kind,
            target: submit.target,
            hub: submit.hub,
            machine,
            pinned,
            spec: submit.spec,
            out_dir: submit.out.unwrap_or_else(|| self.outputs.clone()),
            warnings: submit.warnings,
            inputs: vec![],
            wants,
        };
        let (me, key, given) = (self.clone(), key.to_owned(), submit.inputs);
        let (row, fresh) = tokio::task::spawn_blocking(move || {
            // Copied first; then, against a settled run's deletion of the same objects, checked
            // again and recorded under one guard.
            stage(&me.inputs, &spec, &given)?;
            let _staging = lock(&me.staging);
            let inputs = stage(&me.inputs, &spec, &given)?;
            Ok::<_, Error>(me.store.submit(&key, &digest, NewRun { inputs, ..new })?)
        })
        .await
        .map_err(|e| Error::Invalid(e.to_string()))??;
        if fresh {
            let created = self.store.events_after(row.id.as_str(), 0)?;
            self.publish(&row.id, created);
            self.drive(row.id.clone());
        }
        Ok(Submitted {
            run: self.record(&row)?,
            replay: !fresh,
        })
    }

    /// Asks for the run's cancellation: unsent, it ends here; sent, its machine ends it.
    pub fn cancel(&self, id: &RunId) -> Result<Option<Run>, Error> {
        let events = self.store.request_cancel(id.as_str())?;
        self.publish(id, events);
        self.wake(id);
        Ok(match self.store.run(id.as_str())? {
            Some(row) => Some(self.record(&row)?),
            None => None,
        })
    }

    /// The run as the API answers it, with what it is doing now.
    pub fn record(&self, row: &RunRow) -> Result<Run, Error> {
        let mut run = self.store.record(row)?;
        if let Some(live) = lock(&self.live).get(&row.id) {
            run.waiting.clone_from(&live.waiting);
            run.progress.clone_from(&live.progress);
        }
        Ok(run)
    }

    /// The live events a stream opening now starts with.
    pub fn live(&self, id: &RunId) -> Vec<Event> {
        let live = lock(&self.live);
        let Some(live) = live.get(id) else {
            return vec![];
        };
        let waiting = live.waiting.clone().map(|reason| Event::Waiting { reason });
        let progress = live
            .progress
            .clone()
            .map(|progress| Event::Progress { progress });
        waiting.into_iter().chain(progress).collect()
    }

    /// Records or updates a machine and wakes the runs waiting on it.
    pub fn put_machine(&self, machine: &MachineRow) -> Result<(), Error> {
        self.store.put_machine(machine)?;
        if let Some(entry) = lock(&self.machines).get(&machine.name) {
            entry.changed.send_replace(());
        }
        Ok(())
    }

    /// Records the machine as gone for good: its unconfirmed runs return to the outbox or fail,
    /// its accepted ones fail.
    pub fn machine_ended(&self, name: &str, reason: &str) -> Result<(), Error> {
        let events = self.store.machine_ended(name, reason)?;
        let mut woken = HashSet::new();
        for (id, event) in events {
            self.publish(&id, vec![event]);
            woken.insert(id);
        }
        for id in &woken {
            self.wake(id);
        }
        if let Some(entry) = lock(&self.machines).get_mut(name) {
            entry.link = None;
            entry.changed.send_replace(());
        }
        Ok(())
    }

    /// Renames a machine that has not served (its Hub named it); its runs move with it, and the
    /// old name stays its alias.
    pub fn rename_machine(&self, old: &str, new: &str) -> Result<bool, Error> {
        let renamed = self.store.rename_machine(old, new)?;
        if renamed {
            let mut machines = lock(&self.machines);
            for name in [old, new] {
                if let Some(entry) = machines.remove(name) {
                    entry.changed.send_replace(());
                }
            }
        }
        Ok(renamed)
    }

    /// Ends a run no machine took, with `reason`.
    pub fn fail(&self, id: &RunId, reason: Reason) -> Result<(), Error> {
        let events = self.store.fail(id.as_str(), reason)?;
        self.publish(id, events);
        self.wake(id);
        Ok(())
    }

    /// The machine's connection, shared with its runs' drivers; none while it has no endpoint.
    pub fn link(&self, name: &str) -> Result<Option<Link>, String> {
        self.machine(name).map(|(_, link, _, _)| link)
    }

    /// Records the warm set the machine reported, which a warm run's item is merged into.
    pub fn observe_warm(&self, machine: &str, items: Vec<v1::WarmItem>) {
        lock(&self.warm).insert(machine.to_owned(), items);
    }

    /// Places an unplaced run on `machine`.
    pub fn place(&self, id: &RunId, machine: &str) -> Result<bool, Error> {
        let placed = self.store.place(id.as_str(), machine)?;
        self.wake(id);
        Ok(placed)
    }

    fn publish(&self, id: &RunId, events: Vec<Event>) {
        for event in events {
            let _ = self.events.send((id.clone(), event));
        }
    }

    fn not_waiting(&self, id: &RunId) {
        if let Some(live) = lock(&self.live).get_mut(id) {
            live.waiting = None;
        }
    }

    fn waiting(&self, id: &RunId, reason: String) {
        let mut live = lock(&self.live);
        let entry = live.entry(id.clone()).or_default();
        if entry.waiting.as_ref() != Some(&reason) {
            entry.waiting = Some(reason.clone());
            let _ = self.events.send((id.clone(), Event::Waiting { reason }));
        }
    }

    fn progress(&self, id: &RunId, progress: Progress) {
        lock(&self.live).entry(id.clone()).or_default().progress = Some(progress.clone());
        let _ = self.events.send((id.clone(), Event::Progress { progress }));
    }

    fn wake(&self, id: &RunId) {
        if let Some(wake) = lock(&self.drivers).get(id) {
            wake.notify_one();
        }
    }

    /// Starts the run's driver unless it has one.
    fn drive(self: &Arc<Self>, id: RunId) {
        let wake = {
            let mut drivers = lock(&self.drivers);
            if drivers.contains_key(&id) {
                return;
            }
            let wake = Arc::new(Notify::new());
            drivers.insert(id.clone(), wake.clone());
            wake
        };
        let me = self.clone();
        let mut tasks = lock(&self.tasks);
        while tasks.try_join_next().is_some() {}
        tasks.spawn(async move {
            me.follow_to_end(&id, &wake).await;
            lock(&me.drivers).remove(&id);
            lock(&me.live).remove(&id);
            me.owed.send_replace(());
        });
    }

    async fn follow_to_end(self: &Arc<Self>, id: &RunId, wake: &Notify) {
        let mut pause = Pause::default();
        loop {
            let run = match self.store.run(id.as_str()) {
                Ok(Some(run)) => run,
                Ok(None) => return,
                Err(e) => {
                    warn!(run = %id, "{e}");
                    pause.wait(wake, None).await;
                    continue;
                }
            };
            if run.state == State::Settled {
                self.release_inputs(&run);
                return;
            }
            let Some(name) = run.machine.clone() else {
                self.waiting(id, "waiting for a machine to be chosen".into());
                wake.notified().await;
                continue;
            };
            let (machine, link, mut changed, sending) = match self.machine(&name) {
                Ok(found) => found,
                Err(message) => {
                    self.waiting(id, message);
                    pause.wait(wake, None).await;
                    continue;
                }
            };
            if machine.state == cozy_proto::calcifer::MachineState::Ended {
                let reason = machine.reason.unwrap_or_else(|| "it ended".into());
                if let Err(e) = self.machine_ended(&name, &reason) {
                    warn!(run = %id, "{e}");
                    pause.wait(wake, None).await;
                }
                continue;
            }
            let Some(link) = link else {
                self.waiting(id, format!("waiting for {name} to start"));
                tokio::select! {
                    _ = changed.changed() => {}
                    _ = wake.notified() => {}
                }
                continue;
            };
            match self
                .step(&run, &name, &link, &sending, wake, &mut pause)
                .await
            {
                Ok(()) => {}
                Err(Stop::Unreachable(message)) => {
                    self.waiting(id, format!("{name} is unreachable: {message}"));
                    pause.wait(wake, Some(&mut changed)).await;
                }
                // A run its machine accepted ends only with the machine's outcome.
                Err(Stop::Refused(reason)) if run.state == State::Accepted => {
                    self.waiting(
                        id,
                        format!(
                            "{name} refused to attach ({}): {}",
                            reason.code, reason.message
                        ),
                    );
                    pause.wait(wake, Some(&mut changed)).await;
                }
                Err(Stop::Refused(reason)) => match self.store.fail(id.as_str(), reason) {
                    Ok(events) => self.publish(id, events),
                    Err(e) => warn!(run = %id, "{e}"),
                },
                Err(Stop::Released(message)) => {
                    if let Err(e) = self.machine_ended(&name, &message) {
                        warn!(run = %id, "{e}");
                    }
                }
            }
        }
    }

    /// The machine's row and connection, and a receiver told when the machine changes.
    #[allow(clippy::type_complexity)]
    fn machine(
        &self,
        name: &str,
    ) -> Result<
        (
            MachineRow,
            Option<Link>,
            watch::Receiver<()>,
            Arc<tokio::sync::Mutex<()>>,
        ),
        String,
    > {
        let mut machines = lock(&self.machines);
        let entry = machines.entry(name.to_owned()).or_insert_with(|| Machine {
            changed: watch::channel(()).0,
            link: None,
            sending: Arc::default(),
        });
        let changed = entry.changed.subscribe();
        let row = match self.store.machine(name) {
            Ok(Some(row)) => row,
            Ok(None) => return Err(format!("no machine named {name}")),
            Err(e) => return Err(e.to_string()),
        };
        let link = match &row.endpoint {
            None => None,
            Some(endpoint) => match &entry.link {
                Some((linked, link)) if linked == endpoint => Some(link.clone()),
                _ => {
                    let key = std::fs::read_to_string(&row.key)
                        .ok()
                        .and_then(|pem| SigningKey::from_pkcs8_pem(&pem).ok())
                        .ok_or_else(|| {
                            format!("cannot read the signing key {}", row.key.display())
                        })?;
                    let link = Link::new(endpoint, key).map_err(|e| e.to_string())?;
                    entry.link = Some((endpoint.clone(), link.clone()));
                    Some(link)
                }
            },
        };
        Ok((row, link, changed, entry.sending.clone()))
    }

    /// Sends the run or attaches to it, then follows it until its outcome or the stream ends.
    async fn step(
        &self,
        run: &RunRow,
        machine: &str,
        link: &Link,
        sending: &tokio::sync::Mutex<()>,
        wake: &Notify,
        pause: &mut Pause,
    ) -> Result<(), Stop> {
        let id = run.id.as_str();
        // The spec as sent: finished under the machine's send lock, so a warm set merges into what
        // the machine reported after the runs sent before it.
        let spec = || {
            let mut spec = v1::RunSpec::decode(run.spec.as_slice()).map_err(|e| {
                Stop::Refused(Reason {
                    code: "spec_unreadable".into(),
                    message: format!("the stored spec: {e}"),
                })
            })?;
            self.finish_spec(run, &mut spec);
            Ok::<_, Stop>(spec)
        };
        let mut canceled = false;
        let (state, stream) = match run.state {
            State::Settled => return Ok(()),
            State::Queued => {
                self.write_inputs(link, run).await?;
                let _turn = sending.lock().await;
                self.known_warm(run, machine, link).await?;
                let spec = spec()?;
                if !self.store.mark_sent(id, machine)? {
                    return Ok(());
                }
                link.run(id, 0, Some(spec)).await?
            }
            State::Sent | State::Accepted => {
                if run.cancel_requested {
                    link.cancel(id).await?;
                    canceled = true;
                }
                // It may be there: attach first, and send again only to a machine that never got it.
                match link.run(id, run.cursor, None).await {
                    // Imported from the Go CLI, which did not keep what it sent.
                    Err(LinkError::NotFound) if run.state == State::Sent && run.spec.is_empty() => {
                        return Err(Stop::Refused(Reason {
                            code: "spec_not_kept".into(),
                            message: "its machine never confirmed the run, and its spec was not kept across \
                                      the move from the Go CLI; run it again"
                                .into(),
                        }));
                    }
                    Err(LinkError::NotFound) if run.state == State::Sent => {
                        self.write_inputs(link, run).await?;
                        let _turn = sending.lock().await;
                        self.known_warm(run, machine, link).await?;
                        link.run(id, 0, Some(spec()?)).await?
                    }
                    attached => attached?,
                }
            }
        };
        pause.reset();
        match self.store.accept(id, machine, &state)? {
            Acceptance::Fresh(events) => {
                self.not_waiting(&run.id);
                self.publish(&run.id, events);
            }
            Acceptance::Again => {}
            Acceptance::Stale => {
                link.cancel(id).await?;
                return Ok(());
            }
        }
        self.follow(&run.id, stream, link, wake, canceled).await
    }

    /// Records the run's frames until its outcome. Log lines, calls and products are durable,
    /// committed in one transaction per run of frames already received; progress, state and GPU
    /// frames are live and never move the cursor past what is recorded.
    async fn follow(
        &self,
        id: &RunId,
        mut stream: impl Stream<Item = Result<v1::RunEvent, tonic::Status>> + Unpin,
        link: &Link,
        wake: &Notify,
        mut canceled: bool,
    ) -> Result<(), Stop> {
        use v1::run_event::Event as Frame;
        let (mut batch, mut frames) = (Batch::default(), 0u64);
        loop {
            let next = if batch.is_empty() {
                tokio::select! {
                    next = stream.next() => next,
                    _ = wake.notified() => {
                        let Some(run) = self.store.run(id.as_str())? else { return Ok(()) };
                        if run.state == State::Settled {
                            return Ok(()); // ended here: its machine was lost
                        }
                        if run.cancel_requested && !canceled {
                            link.cancel(id.as_str()).await?;
                            canceled = true;
                        }
                        continue;
                    }
                }
            } else {
                match ready(&mut stream) {
                    Some(next) => next,
                    None => {
                        self.record_batch(id, &mut batch)?;
                        continue;
                    }
                }
            };
            let frame = match next {
                Some(Ok(frame)) => frame,
                Some(Err(status)) => {
                    self.record_batch(id, &mut batch)?;
                    return Err(link::classify(status).into());
                }
                // The machine ends a stream it holds open only to start another: attach again.
                None if frames > 0 => return self.record_batch(id, &mut batch),
                None => {
                    return Err(Stop::Unreachable(
                        "the machine closed the run's stream".into(),
                    ));
                }
            };
            frames += 1;
            let durable = frame.sequence;
            match frame.event {
                Some(Frame::Log(line)) => batch.logs.push((frame.at_ms, line.level, line.text)),
                Some(Frame::Call(call)) => {
                    let call = cozy_proto::calcifer::Call {
                        run: call.run,
                        index: call.index,
                        function: call.function,
                        label: call.label,
                        status: call.status,
                        called_at_ms: call.called_at_ms,
                        finished_at_ms: call.finished_at_ms,
                        measurements: serde_json::from_slice(&call.measurements).ok(),
                        reason: call.reason.map(|r| Reason {
                            code: r.code,
                            message: r.message,
                        }),
                        memoized: call.memoized,
                    };
                    let text = serde_json::to_string(&call).expect("serializes");
                    batch.logs.push((frame.at_ms, "call".into(), text));
                }
                Some(Frame::Product(product)) => batch.products.push(product),
                Some(Frame::Gpu(gpu)) => {
                    if gpu.state == "wait" {
                        self.waiting(
                            id,
                            format!("waiting for {} GPUs behind run {}", gpu.width, gpu.behind),
                        );
                    }
                }
                Some(Frame::Progress(p)) => {
                    self.progress(
                        id,
                        Progress {
                            stage: p.stage,
                            fraction: (p.fraction >= 0.0).then_some(p.fraction),
                            completed: p.completed,
                            total: p.total,
                            bytes_done: p.bytes_done,
                            bytes_total: p.bytes_total,
                            step_ms: p.step_ms,
                        },
                    );
                    continue;
                }
                Some(Frame::State(state)) => {
                    match state.waiting.as_str() {
                        "" => self.not_waiting(id),
                        waiting => self.waiting(id, format!("{}: {waiting}", state.state)),
                    }
                    batch.machine_state = Some(state.state);
                    continue;
                }
                Some(Frame::Outcome(outcome)) => {
                    self.record_batch(id, &mut batch)?;
                    let events = self.store.settle(id.as_str(), &outcome, durable)?;
                    self.publish(id, events);
                    return Ok(());
                }
                None => continue,
            }
            batch.cursor = batch.cursor.max(durable);
        }
    }

    fn record_batch(&self, id: &RunId, batch: &mut Batch) -> Result<(), Stop> {
        if !batch.is_empty() {
            let events = self.store.observe(id.as_str(), std::mem::take(batch))?;
            self.publish(id, events);
        }
        Ok(())
    }

    async fn write_inputs(&self, link: &Link, run: &RunRow) -> Result<(), Stop> {
        for (digest, length) in &run.inputs {
            let path = self.inputs.join(hex::encode(digest.bytes()));
            let file = tokio::fs::File::open(&path).await.map_err(|e| {
                Stop::Refused(Reason {
                    code: "input_missing".into(),
                    message: format!(
                        "input {digest} is no longer staged at {}: {e}",
                        path.display()
                    ),
                })
            })?;
            link.write(digest, *length, file).await?;
        }
        Ok(())
    }

    /// Reads the machine's warm set before a warm run's set is merged into it, if none was seen
    /// yet.
    async fn known_warm(&self, run: &RunRow, machine: &str, link: &Link) -> Result<(), Stop> {
        let merges = v1::RunSpec::decode(run.spec.as_slice())
            .is_ok_and(|spec| spec.kind == v1::RunKind::Warm as i32 && spec.set.is_some());
        if !merges || lock(&self.warm).contains_key(machine) {
            return Ok(());
        }
        let frame = link
            .status(false)
            .await?
            .message()
            .await
            .map_err(link::classify)?;
        self.observe_warm(machine, frame.map(|f| f.warm).unwrap_or_default());
        Ok(())
    }

    /// The spec as the machine is sent it: the run's Hub origin, unless the spec names one.
    fn finish_spec(&self, run: &RunRow, spec: &mut v1::RunSpec) {
        if spec.hub.is_none() && !run.hub.is_empty() {
            spec.hub = Some(v1::HubAccess {
                origin: run.hub.clone(),
                ..Default::default()
            });
        }
        if let Some(machine) = &run.machine
            && let Some(warm) = lock(&self.warm).get_mut(machine)
        {
            machines::merge_warm(warm, spec);
        }
    }

    /// Deletes the staged inputs of a settled run that no unsettled run names.
    fn release_inputs(&self, run: &RunRow) {
        let _staging = lock(&self.staging);
        for (digest, _) in &run.inputs {
            if let Ok(false) = self.store.input_in_use(digest) {
                let _ = std::fs::remove_file(self.inputs.join(hex::encode(digest.bytes())));
            }
        }
    }

    /// Deletes staged inputs no unsettled run names (a Calcifer that stopped between a run's end
    /// and their deletion).
    fn sweep_inputs(&self) -> Result<(), Error> {
        let entries = std::fs::read_dir(&self.inputs).map_err(|e| Error::Input {
            path: self.inputs.clone(),
            message: e.to_string(),
        })?;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            match format!("sha256:{name}").parse::<Digest>() {
                Ok(digest) if self.store.input_in_use(&digest)? => {}
                _ => {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
        Ok(())
    }
}

/// Copies each given file into `inputs/<sha256 hex>`, checking its digest and length, unless a
/// copy is there, and answers the objects the run's machine is written: every given file and every
/// object `spec.inputs` names, each of which must be staged.
fn stage(
    inputs: &Path,
    spec: &v1::RunSpec,
    given: &[StagedInput],
) -> Result<Vec<(Digest, u64)>, Error> {
    let mut objects = vec![];
    for source in given {
        let target = inputs.join(hex::encode(source.digest.bytes()));
        if !target.exists() {
            copy_checked(inputs, source, &target)?;
        }
        objects.push((source.digest, source.length));
    }
    for input in &spec.inputs {
        let invalid = |message: String| Error::Invalid(format!("input {}: {message}", input.field));
        let digest: Digest = input.digest.parse().map_err(|e: &str| invalid(e.into()))?;
        if !inputs.join(hex::encode(digest.bytes())).exists() {
            return Err(invalid(format!("no file was given for {digest}")));
        }
        objects.push((digest, input.length));
    }
    objects.sort();
    objects.dedup_by_key(|(digest, _)| *digest);
    Ok(objects)
}

fn copy_checked(inputs: &Path, source: &StagedInput, target: &Path) -> Result<(), Error> {
    let failed = |message: String| Error::Input {
        path: source.path.clone(),
        message,
    };
    let mut from = std::fs::File::open(&source.path).map_err(|e| failed(e.to_string()))?;
    let mut part = tempfile::NamedTempFile::new_in(inputs).map_err(|e| failed(e.to_string()))?;
    let (mut hasher, mut length, mut buffer) = (Sha256::new(), 0u64, vec![0; 1 << 20]);
    loop {
        let read = from.read(&mut buffer).map_err(|e| failed(e.to_string()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        part.write_all(&buffer[..read])
            .map_err(|e| failed(e.to_string()))?;
        length += read as u64;
    }
    let got = Digest::from_bytes(hasher.finalize().into());
    if got != source.digest || length != source.length {
        return Err(failed(format!(
            "it holds {length} bytes with digest {got}, not {} with {}",
            source.length, source.digest
        )));
    }
    part.as_file()
        .sync_all()
        .map_err(|e| failed(e.to_string()))?;
    part.persist(target).map_err(|e| failed(e.to_string()))?;
    Ok(())
}

/// The stream's next item if it has one now, without waiting.
fn ready<S: Stream + Unpin>(stream: &mut S) -> Option<Option<S::Item>> {
    match Pin::new(stream).poll_next(&mut Context::from_waker(std::task::Waker::noop())) {
        Poll::Ready(item) => Some(item),
        Poll::Pending => None,
    }
}

/// The pause between attempts to reach a machine: it doubles to 10 s while nothing moves and
/// starts over after any progress. It paces retries; nothing is given up.
#[derive(Default)]
struct Pause(Duration);

impl Pause {
    fn reset(&mut self) {
        self.0 = Duration::ZERO;
    }

    async fn wait(&mut self, wake: &Notify, changed: Option<&mut watch::Receiver<()>>) {
        self.0 = (self.0 * 2).clamp(Duration::from_millis(500), Duration::from_secs(10));
        let changed = async {
            match changed {
                Some(changed) => changed.changed().await.unwrap_or(()),
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            _ = tokio::time::sleep(self.0) => {}
            _ = wake.notified() => {}
            _ = changed => {}
        }
    }
}

fn random() -> [u8; 12] {
    let mut bytes = [0; 12];
    getrandom::fill(&mut bytes).expect("the OS provides randomness");
    bytes
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}
