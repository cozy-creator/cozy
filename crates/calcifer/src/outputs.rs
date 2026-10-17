//! Output export: each output item of a run lands at `<out_dir>/<number>-<name>[-<i>]` with its
//! media type's extension, rewritten in place as revisions arrive. A revision's bytes are read
//! from the run's machine into a `.part` beside the file (ranged and side by side when big) and
//! replace the file whole once they match the revision's digest; a tree output is a directory
//! replaced the same way. One exporter works per run with something to export, started by the
//! run's output events or, at startup, from the store. `serve` answers
//! `GET /v1/runs/{id}/outputs/{name}` from the exported file.
use std::{
    collections::{HashMap, HashSet},
    io::{Read as _, Seek as _, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use axum::{
    body::Body,
    extract::{Path as Route, Query, State, rejection::QueryRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::Response,
};
use cozy_proto::{
    calcifer::{CHECKPOINT, Event, MachineState},
    error::Kind,
    ids::{Digest, RunId},
    machine::{
        self as v1,
        client::{self, Bulk},
    },
};
use futures_util::StreamExt as _;
use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use tokio::{
    io::{AsyncReadExt as _, AsyncSeekExt as _, AsyncWriteExt as _},
    sync::{Notify, broadcast::error::RecvError, watch},
};
use tracing::warn;

use crate::{
    Calcifer, Error, Link, Pause,
    api::{self, Api, Refusal},
    lanes::{self, Ranged},
    link::LANES,
    lock,
    store::{OutputRow, RunRow, State as RunState, StoreError},
};

/// A tree output's media type: its bytes are the manifest of its files.
const TREE: &str = "application/vnd.cozy.tree-manifest";
/// Bytes are read in ranges of `clamp(length / (RANGES_PER_LANE * LANES), 1 MiB, RANGE)`, at
/// most `LANES` side by side on their own connections. More ranges than lanes, as an S3 client
/// cuts parts: a lane that finishes early takes the next from the queue, so no lane trails alone.
const RANGE: u64 = 8 << 20;
const RANGES_PER_LANE: usize = 4;

/// An output Calcifer still owes this computer: its run's number, its name and index, its length
/// and the bytes of it in place.
pub(crate) struct Owed {
    pub number: u64,
    pub name: String,
    pub index: u32,
    pub length: u64,
    pub moved: u64,
}

impl std::fmt::Display for Owed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let length = crate::bytes(self.length as f64);
        match self.index {
            0 => write!(f, "run {} {} ({length})", self.number, self.name),
            i => write!(f, "run {} {}/{i} ({length})", self.number, self.name),
        }
    }
}

/// The exporters at work, and what they share.
#[derive(Default)]
pub(crate) struct Exports {
    /// One per run, woken by its output events.
    runs: Mutex<HashMap<RunId, Arc<Notify>>>,
    /// Where a revision extends the one before it, by run, item id and revision.
    appended: Mutex<HashMap<(RunId, String, u64), u64>>,
    /// Orders replacing a file and recording it against `serve` reading both.
    placing: Mutex<()>,
    /// The bytes in place of each export in flight, by run, output and index.
    moving: Mutex<HashMap<(RunId, String, u32), Arc<AtomicU64>>>,
}

/// Why an export stopped short.
#[derive(Clone)]
enum Failure {
    /// A newer revision is recorded: that one is exported instead.
    Superseded,
    /// This revision's bytes cannot be had: recorded with the item and not tried again.
    Lost(String),
    /// Tried again after a pause, or once the machine changes.
    Again(String),
    /// The machine's definitive refusal: final once the run has ended (no newer revision can
    /// come to read instead), tried again before.
    Refused(String),
}

impl From<StoreError> for Failure {
    fn from(e: StoreError) -> Self {
        Failure::Again(e.to_string())
    }
}

impl Calcifer {
    /// Starts an exporter for each run that owes one, then for each run whose outputs change.
    pub(crate) fn export_outputs(self: &Arc<Self>) -> Result<(), Error> {
        let mut events = self.subscribe();
        for (id, _) in self.store.exports_owed()? {
            self.export(&id);
        }
        let me = Arc::downgrade(self);
        self.spawn_task(async move {
            loop {
                let received = events.recv().await;
                let Some(me) = me.upgrade() else { return };
                match received {
                    Ok((
                        id,
                        Event::OutputDelta {
                            item_id,
                            rev,
                            appended_from: Some(from),
                            ..
                        },
                    )) => {
                        lock(&me.exports.appended).insert((id.clone(), item_id, rev), from);
                        me.export(&id);
                    }
                    Ok((
                        id,
                        Event::OutputAdded { .. }
                        | Event::OutputDelta { .. }
                        | Event::OutputDone { .. },
                    )) => me.export(&id),
                    Ok(_) => {}
                    Err(RecvError::Lagged(_)) => match me.store.exports_owed() {
                        Ok(owed) => owed.iter().for_each(|(id, _)| me.export(id)),
                        Err(e) => warn!("{e}"),
                    },
                    Err(RecvError::Closed) => return,
                }
            }
        });
        Ok(())
    }

    fn spawn_task(&self, task: impl Future<Output = ()> + Send + 'static) {
        let mut tasks = lock(&self.tasks);
        while tasks.try_join_next().is_some() {}
        tasks.spawn(task);
    }

    /// Starts the run's exporter, or wakes it.
    fn export(self: &Arc<Self>, id: &RunId) {
        let wake = {
            let mut runs = lock(&self.exports.runs);
            if let Some(wake) = runs.get(id) {
                wake.notify_one();
                return;
            }
            let wake = Arc::new(Notify::new());
            runs.insert(id.clone(), wake.clone());
            wake
        };
        let (me, id) = (self.clone(), id.clone());
        self.spawn_task(async move { me.export_run(&id, &wake).await });
    }

    /// Exports the run's outputs, each at its newest revision, until none is left: up to `LANES`
    /// items side by side, so a run's many small outputs cost one round trip together, not one
    /// each. Paces retries while nothing moves; never gives up without recording why.
    async fn export_run(&self, id: &RunId, wake: &Notify) {
        let mut pause = Pause::default();
        loop {
            let (run, items) = match self.next_exports(id) {
                Ok(Some(next)) => next,
                Ok(None) => return,
                Err(e) => {
                    warn!(run = %id, "{e}");
                    pause.wait(wake, None).await;
                    continue;
                }
            };
            let mut changed = None;
            let exported = match self.reach(&run, &mut changed) {
                Ok(link) => {
                    let exports = (items.iter())
                        .map(|item| self.export_item(&run, item, &link, changed.clone()));
                    futures_util::future::join_all(exports).await
                }
                Err(failure) => items.iter().map(|_| Err(failure.clone())).collect(),
            };
            let mut again = false;
            for (item, exported) in items.iter().zip(exported) {
                let (name, index, rev) = (&item.name, item.index, item.rev);
                let exported = match exported {
                    Err(Failure::Refused(message)) if run.state == RunState::Settled => {
                        Err(Failure::Lost(format!("the machine refused it ({message})")))
                    }
                    exported => exported,
                };
                match exported {
                    Ok(()) | Err(Failure::Superseded) => {}
                    Err(Failure::Lost(reason)) => {
                        warn!(run = %id, output = %name, rev, "not exported: {reason}");
                        let dest = (run.out_dir).join(file_name(
                            run.number,
                            name,
                            index,
                            &item.media_type,
                        ));
                        remove(&suffixed(&dest, ".part"));
                        remove(&suffixed(&dest, ".manifest.part"));
                        match (self.store).export_failed(id.as_str(), name, index, rev, &reason) {
                            Ok(events) => self.publish(id, events),
                            Err(e) => {
                                warn!(run = %id, "{e}");
                                again = true;
                            }
                        }
                    }
                    Err(Failure::Again(message) | Failure::Refused(message)) => {
                        warn!(run = %id, output = %name, rev, "export waits: {message}");
                        again = true;
                    }
                }
            }
            match again {
                true => pause.wait(wake, changed.as_mut()).await,
                false => pause.reset(),
            }
        }
    }

    /// The run and its first `LANES` outputs to export; none ends the exporter, under the lock
    /// `export` takes, so an event after this starts another.
    fn next_exports(&self, id: &RunId) -> Result<Option<(RunRow, Vec<OutputRow>)>, StoreError> {
        let mut runs = lock(&self.exports.runs);
        let next = match self.store.run(id.as_str())? {
            Some(run) => {
                let mut items = self.store.outputs_to_export(id.as_str())?;
                items.truncate(LANES);
                Some((run, items)).filter(|(_, items)| !items.is_empty())
            }
            None => None,
        };
        if next.is_none() {
            runs.remove(id);
            lock(&self.exports.appended).retain(|(run, ..), _| run != id);
        }
        Ok(next)
    }

    /// The run's machine's connection, whose bulk lanes read; `changed` is told when the
    /// machine changes.
    fn reach(
        &self,
        run: &RunRow,
        changed: &mut Option<watch::Receiver<()>>,
    ) -> Result<Link, Failure> {
        let Some(name) = &run.machine else {
            return Err(Failure::Lost("the run never reached a machine".into()));
        };
        let (machine, link, watch, _) = self.machine(name).map_err(Failure::Again)?;
        *changed = Some(watch);
        if machine.state == MachineState::Ended {
            let reason = machine.reason.as_deref().unwrap_or("no reason was given");
            return Err(Failure::Lost(format!("its machine {name} ended: {reason}")));
        }
        link.ok_or_else(|| Failure::Again(format!("waiting for {name} to start")))
    }

    /// Makes the item's file (a directory for a tree) this revision's bytes. A newer revision
    /// stops it; so does its machine's end (`changed` tells of it), which loses it.
    async fn export_item(
        &self,
        run: &RunRow,
        item: &OutputRow,
        link: &Link,
        mut changed: Option<watch::Receiver<()>>,
    ) -> Result<(), Failure> {
        let mut events = self.subscribe();
        let superseded = || -> Result<bool, Failure> {
            let now = self.store.output(run.id.as_str(), &item.name, item.index)?;
            Ok(now.is_none_or(|now| now.rev != item.rev))
        };
        if superseded()? {
            return Err(Failure::Superseded);
        }
        let name = file_name(run.number, &item.name, item.index, &item.media_type);
        let dest = run.out_dir.join(name);
        let target = v1::OutputTarget {
            run: run.id.to_string(),
            output: item.name.clone(),
            index: item.index,
            member: String::new(),
        };
        let moved = Arc::new(AtomicU64::new(0));
        let key = (run.id.clone(), item.name.clone(), item.index);
        lock(&self.exports.moving).insert(key.clone(), moved.clone());
        let work = async {
            tokio::fs::create_dir_all(&run.out_dir).await.map_err(|e| {
                Failure::Again(format!("cannot create {}: {e}", run.out_dir.display()))
            })?;
            let staged = if item.media_type == TREE {
                fetch_tree(link, &target, item, &dest, &moved).await?
            } else {
                let appended = (run.id.clone(), item_id(run.number, item), item.rev);
                let appended = lock(&self.exports.appended).get(&appended).copied();
                fetch_file(link, &target, item, &dest, appended, &moved).await?
            };
            self.land(run, item, &staged, &dest)
        };
        tokio::pin!(work);
        let done = loop {
            let told = tokio::select! {
                done = &mut work => break done,
                told = events.recv() => told,
                () = machine_changed(&mut changed) => {
                    if let Some(name) = &run.machine
                        && let Ok(Some(machine)) = self.store.machine(name)
                        && machine.state == MachineState::Ended
                    {
                        let reason = machine.reason.as_deref().unwrap_or("no reason was given");
                        break Err(Failure::Lost(format!("its machine {name} ended: {reason}")));
                    }
                    continue;
                }
            };
            let newer = match told {
                Ok((id, event)) => {
                    id == run.id
                        && matches!(
                            event,
                            Event::OutputAdded { .. }
                                | Event::OutputDelta { .. }
                                | Event::OutputDone { .. }
                        )
                }
                Err(RecvError::Lagged(_)) => true,
                Err(RecvError::Closed) => break (&mut work).await,
            };
            match newer.then(superseded) {
                Some(Ok(true)) => break Err(Failure::Superseded),
                Some(Err(e)) => break Err(e),
                _ => {}
            }
        };
        lock(&self.exports.moving).remove(&key);
        done
    }

    /// The outputs of runs on `machine` that have not landed and are not given up, each with the
    /// bytes its export has in place.
    pub(crate) fn owed_on(&self, machine: &str) -> Result<Vec<Owed>, StoreError> {
        let owed = self.store.exports_owed_on(machine)?;
        let moving = lock(&self.exports.moving);
        let owed = owed.into_iter().map(|owed| {
            let key = (owed.run, owed.name, owed.index);
            let moved = (moving.get(&key)).map_or(0, |moved| moved.load(Ordering::Relaxed));
            let (_, name, index) = key;
            Owed {
                number: owed.number,
                name,
                index,
                length: owed.length,
                moved,
            }
        });
        Ok(owed.collect())
    }

    /// Puts the staged bytes in the item's place and records them, one step as `serve` sees it.
    fn land(
        &self,
        run: &RunRow,
        item: &OutputRow,
        staged: &Path,
        dest: &Path,
    ) -> Result<(), Failure> {
        let old = suffixed(dest, ".old");
        let unplaced =
            |e: std::io::Error| Failure::Again(format!("cannot place {}: {e}", dest.display()));
        {
            let _placing = lock(&self.exports.placing);
            if let Err(e) = std::fs::rename(staged, dest) {
                // A directory is not replaced by a rename: the old one moves aside first.
                if !(dest.is_dir() || staged.is_dir() && dest.symlink_metadata().is_ok()) {
                    return Err(unplaced(e));
                }
                remove(&old);
                std::fs::rename(dest, &old).map_err(unplaced)?;
                if let Err(e) = std::fs::rename(staged, dest) {
                    let _ = std::fs::rename(&old, dest);
                    return Err(unplaced(e));
                }
            }
            if let Some(directory) = dest.parent() {
                let _ = cozy_platform::durable::sync_dir(directory);
            }
            let (id, name, index, rev) = (run.id.as_str(), &item.name, item.index, item.rev);
            self.publish(&run.id, self.store.exported(id, name, index, rev, dest)?);
        }
        remove(&old);
        Ok(())
    }
}

/// Which item of a list output; 0 (the default) for a single output.
#[derive(Deserialize)]
pub struct Item {
    #[serde(default)]
    index: u32,
}

/// `GET /v1/runs/{id|number}/outputs/{name}[?index=i]`: the output's exported file, whole or one
/// `Range` of it, with `ETag: "r<rev>"` naming the revision it holds; otherwise why not.
pub async fn serve(
    State(api): State<Arc<Api>>,
    Route((run, name)): Route<(String, String)>,
    query: Result<Query<Item>, QueryRejection>,
    headers: HeaderMap,
) -> Result<Response, Refusal> {
    let index = query?.0.index;
    let row = api::find(&api, &run)?;
    let number = row.number;
    let (item, file) = {
        let _placing = lock(&api.calcifer.exports.placing);
        let item = api.calcifer.store().output(row.id.as_str(), &name, index)?;
        let item = item.ok_or_else(|| {
            let message = format!("run {number} has no output {name} at index {index}");
            api::refuse(Kind::NotFound, "output_not_found", message)
        })?;
        let file = match (&item.path, item.exported_rev) {
            (Some(path), Some(_)) => Some(std::fs::File::open(path).map_err(|e| {
                let message = format!("its file {} is gone: {e}", path.display());
                api::refuse(Kind::NotFound, "output_file_missing", message)
            })?),
            _ => None,
        };
        (item, file)
    };
    if item.media_type == TREE {
        let message = format!("output {name} of run {number} is a folder: open it on disk");
        return Err(api::refuse(Kind::Conflict, "output_is_a_tree", message));
    }
    if item.media_type == CHECKPOINT {
        let machine = row.machine.as_deref().unwrap_or("its machine");
        let message = format!(
            "output {name} of run {number} is a checkpoint: it stays on {machine}, not this computer"
        );
        let refusal = api::refuse(Kind::Conflict, "output_is_a_checkpoint", message);
        let next = format!("cozy run upload {number}#{name} <org/model>");
        return Err(Refusal(refusal.0.next(next)));
    }
    let (Some(mut file), Some(rev)) = (file, item.exported_rev) else {
        if let Some(reason) = item.export_error {
            let message = format!("output {name} of run {number} will not land: {reason}");
            return Err(api::refuse(
                Kind::Unavailable,
                "output_not_exported",
                message,
            ));
        }
        let state = match row.state {
            RunState::Settled => "has ended",
            RunState::Accepted => "is running",
            _ => "is queued",
        };
        let message = format!("output {name} of run {number} has not landed yet; the run {state}");
        let refusal = api::refuse(Kind::Conflict, "output_not_ready", message);
        return Err(Refusal(refusal.0.next(format!("cozy run watch {number}"))));
    };
    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let octets = HeaderValue::from_static("application/octet-stream");
    let media_type = HeaderValue::from_str(&item.media_type)
        .ok()
        .filter(|v| !v.is_empty());
    let mut response = Response::builder()
        .header(header::CONTENT_TYPE, media_type.unwrap_or(octets))
        .header(header::ETAG, format!("\"r{rev}\""))
        .header(header::ACCEPT_RANGES, "bytes");
    if item.done && rev == item.rev {
        response = response.header("x-cozy-digest", &item.digest);
    }
    let (status, first, count) = match ranged(headers.get(header::RANGE), len) {
        Ok(None) => (StatusCode::OK, 0, len),
        Ok(Some((first, last))) => {
            let range = format!("bytes {first}-{last}/{len}");
            response = response.header(header::CONTENT_RANGE, range);
            (StatusCode::PARTIAL_CONTENT, first, last - first + 1)
        }
        Err(()) => {
            return Ok(response
                .status(StatusCode::RANGE_NOT_SATISFIABLE)
                .header(header::CONTENT_RANGE, format!("bytes */{len}"))
                .body(Body::empty())
                .expect("a response"));
        }
    };
    file.seek(SeekFrom::Start(first))
        .map_err(|e| api::refuse(Kind::Internal, "output_unreadable", e.to_string()))?;
    let bytes = tokio::fs::File::from_std(file).take(count);
    Ok(response
        .status(status)
        .header(header::CONTENT_LENGTH, count)
        .body(Body::from_stream(tokio_util::io::ReaderStream::new(bytes)))
        .expect("a response"))
}

/// The one range a `Range: bytes=…` header asks of `len` bytes, as (first, last): none for the
/// whole (no header, or another unit), `Err` for several ranges or one not satisfiable.
fn ranged(header: Option<&HeaderValue>, len: u64) -> Result<Option<(u64, u64)>, ()> {
    let asked = header.and_then(|h| h.to_str().ok());
    let Some(spec) = asked.and_then(|h| h.trim().strip_prefix("bytes=")) else {
        return Ok(None);
    };
    let (first, last) = spec.split_once('-').ok_or(())?;
    let number = |text: &str| text.trim().parse::<u64>().map_err(|_| ());
    let end = len.checked_sub(1).ok_or(())?;
    let (first, last) = match (first.trim(), last.trim()) {
        ("", suffix) => (len.saturating_sub(number(suffix)?), end),
        (first, "") => (number(first)?, end),
        (first, last) => (number(first)?, number(last)?.min(end)),
    };
    if first > last {
        return Err(());
    }
    Ok(Some((first, last)))
}

/// Where a run that names no directory puts its outputs: its package's folder, `<org>-<name>` as
/// the Go CLI named it (`fidika-minimax-h3`, `local-cozy-script-<hex>`), from a target
/// `<org>/<name>/<function>[@release]`; the outputs folder itself for any other target.
pub fn package_outputs(outputs: &Path, target: &str) -> PathBuf {
    let named = |part: &&str| {
        !part.is_empty()
            && !part.starts_with('.')
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    };
    match target.split('/').collect::<Vec<_>>().as_slice() {
        [org, name, function] if named(org) && named(name) && !function.is_empty() => {
            outputs.join(format!("{org}-{name}"))
        }
        _ => outputs.to_owned(),
    }
}

/// The item's name in its run's output directory: `<number>-<name>[-<i>]` and its media type's
/// extension (none for a tree, a directory).
pub fn file_name(number: u64, name: &str, index: u32, media_type: &str) -> String {
    let name: String = (name.chars().take(128))
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '_' | '-' => c,
            _ => '_',
        })
        .collect();
    let stem = match index {
        0 => format!("{number}-{name}"),
        i => format!("{number}-{name}-{i}"),
    };
    let media_type = media_type.split(';').next().unwrap_or("").trim();
    let extension = match media_type.to_ascii_lowercase().as_str() {
        "image/png" => ".png",
        "image/jpeg" | "image/jpg" => ".jpg",
        "image/webp" => ".webp",
        "image/gif" => ".gif",
        "image/avif" => ".avif",
        "video/mp4" => ".mp4",
        "video/webm" => ".webm",
        "audio/wav" | "audio/x-wav" | "audio/wave" => ".wav",
        "audio/mpeg" => ".mp3",
        "audio/flac" => ".flac",
        "audio/ogg" => ".ogg",
        "application/json" => ".json",
        "text/plain" => ".txt",
        _ => "",
    };
    stem + extension
}

fn item_id(number: u64, item: &OutputRow) -> String {
    match item.index {
        0 => format!("{number}/{}", item.name),
        i => format!("{number}/{}/{i}", item.name),
    }
}

/// Resolves when the machine changes; never without a receiver or once its sender is gone.
async fn machine_changed(changed: &mut Option<watch::Receiver<()>>) {
    let changed = match changed {
        Some(changed) => changed.changed().await.is_ok(),
        None => false,
    };
    if !changed {
        std::future::pending::<()>().await;
    }
}

fn suffixed(path: &Path, suffix: &str) -> PathBuf {
    let mut path = path.as_os_str().to_owned();
    path.push(suffix);
    path.into()
}

fn remove(path: &Path) {
    let _ = std::fs::remove_dir_all(path).or_else(|_| std::fs::remove_file(path));
}

/// This revision's bytes in `<file>.part`, checked against its digest. A revision that extends
/// the one the file holds reads only its new tail.
async fn fetch_file(
    link: &Link,
    target: &v1::OutputTarget,
    item: &OutputRow,
    dest: &Path,
    appended: Option<u64>,
    moved: &Arc<AtomicU64>,
) -> Result<PathBuf, Failure> {
    let part = suffixed(dest, ".part");
    let follows = item.exported_rev.is_some_and(|held| held + 1 == item.rev);
    let mut from = appended.filter(|_| follows).unwrap_or(0);
    loop {
        let prepared = blocking({
            let (part, dest, length) = (part.clone(), dest.to_owned(), item.length);
            move || prepare(&part, &dest, from, length)
        })
        .await;
        if let Err(e) = prepared {
            if from == 0 {
                return Err(Failure::Again(format!(
                    "cannot write {}: {e}",
                    part.display()
                )));
            }
            from = 0;
            continue;
        }
        moved.store(from, Ordering::Relaxed);
        fetch(link, target, item.rev, from, item.length, &part, moved).await?;
        match check(&part, item.length, &item.digest).await {
            Ok(()) => return Ok(part),
            // The file did not hold the revision this one extends.
            Err(_) if from > 0 => from = 0,
            Err(message) => return Err(Failure::Again(message)),
        }
    }
}

/// This revision's tree in `<dir>.part/`: its manifest, then each file it names, each checked
/// against its digest; a file the tree names twice is read once.
async fn fetch_tree(
    link: &Link,
    target: &v1::OutputTarget,
    item: &OutputRow,
    dest: &Path,
    moved: &Arc<AtomicU64>,
) -> Result<PathBuf, Failure> {
    let (stage, manifest) = (suffixed(dest, ".part"), suffixed(dest, ".manifest.part"));
    let unwritable =
        |e: std::io::Error| Failure::Again(format!("cannot write {}: {e}", stage.display()));
    let prepared = blocking({
        let (stage, manifest, length) = (stage.clone(), manifest.clone(), item.length);
        move || {
            remove(&stage);
            std::fs::create_dir_all(&stage)?;
            prepare(&manifest, &stage, 0, length)
        }
    });
    prepared.await.map_err(unwritable)?;
    fetch(link, target, item.rev, 0, item.length, &manifest, moved).await?;
    check(&manifest, item.length, &item.digest)
        .await
        .map_err(Failure::Again)?;
    let listed = std::fs::read(&manifest).map_err(unwritable)?;
    let _ = std::fs::remove_file(&manifest);
    let mut read: HashMap<String, PathBuf> = HashMap::new();
    for (path, digest, length) in members(&listed).map_err(Failure::Lost)? {
        let file = stage.join(&path);
        if let Some(parent) = file.parent() {
            std::fs::create_dir_all(parent).map_err(unwritable)?;
        }
        if let Some(first) = read.get(&digest) {
            std::fs::copy(first, &file).map_err(unwritable)?;
            continue;
        }
        let empty = std::fs::File::create(&file).and_then(|f| f.set_len(length));
        empty.map_err(unwritable)?;
        let member = v1::OutputTarget {
            member: path,
            ..target.clone()
        };
        fetch(link, &member, item.rev, 0, length, &file, moved).await?;
        check(&file, length, &digest)
            .await
            .map_err(Failure::Again)?;
        read.insert(digest, file);
    }
    Ok(stage)
}

#[derive(Deserialize)]
struct Manifest {
    entries: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    kind: String,
    path: String,
    blob: Option<Blob>,
}

#[derive(Deserialize)]
struct Blob {
    sha256: String,
    length: u64,
}

/// The files a tree's manifest names, as (path, digest, length): relative paths that stay inside
/// the tree, none named twice or inside another.
fn members(manifest: &[u8]) -> Result<Vec<(String, String, u64)>, String> {
    let manifest: Manifest =
        serde_json::from_slice(manifest).map_err(|e| format!("the tree's manifest: {e}"))?;
    let mut files = vec![];
    for Entry { kind, path, blob } in manifest.entries {
        if kind != "file" {
            warn!("a tree entry of kind {kind:?} is skipped");
            continue;
        }
        let inside = (path.split('/'))
            .all(|part| !matches!(part, "" | "." | "..") && !part.contains(['\\', ':', '\0']));
        if !inside {
            return Err(format!("the tree names the path {path:?}"));
        }
        let blob = blob.ok_or_else(|| format!("the tree's {path} names no bytes"))?;
        let digest: Digest = format!("sha256:{}", blob.sha256)
            .parse()
            .map_err(|e| format!("the tree's {path}: {e}"))?;
        files.push((path, digest.to_string(), blob.length));
    }
    let paths: HashSet<&str> = files.iter().map(|(path, ..)| path.as_str()).collect();
    if paths.len() != files.len() {
        return Err("the tree names a path twice".into());
    }
    let nested = paths.iter().find(|path| {
        path.match_indices('/')
            .any(|(i, _)| paths.contains(&path[..i]))
    });
    if let Some(path) = nested {
        return Err(format!("the tree names {path} inside another of its files"));
    }
    Ok(files)
}

async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> std::io::Result<T> + Send + 'static,
) -> std::io::Result<T> {
    tokio::task::spawn_blocking(work)
        .await
        .unwrap_or_else(|e| Err(std::io::Error::other(e)))
}

/// A fresh `part` of `length` bytes, its first `from` copied from `dest`.
fn prepare(part: &Path, dest: &Path, from: u64, length: u64) -> std::io::Result<()> {
    let _ = std::fs::remove_file(part);
    let mut file = std::fs::File::create(part)?;
    if from > 0 {
        let copied = std::io::copy(&mut std::fs::File::open(dest)?.take(from), &mut file)?;
        if copied != from {
            return Err(std::io::Error::other(
                "the file is shorter than what extends it",
            ));
        }
    }
    file.set_len(length)
}

/// Whether `path` holds exactly `length` bytes with `digest`; they are synced to disk if so.
async fn check(path: &Path, length: u64, digest: &str) -> Result<(), String> {
    let (path, digest) = (path.to_owned(), digest.to_owned());
    let checked = blocking(move || {
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;
        let (mut hasher, mut held, mut buffer) = (Sha256::new(), 0u64, vec![0; 1 << 20]);
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            held += read as u64;
        }
        let got = Digest::from_bytes(hasher.finalize().into()).to_string();
        if held != length || got != digest {
            return Ok(Err(format!(
                "read {held} bytes with digest {got}, not {length} with {digest}"
            )));
        }
        file.sync_all()?;
        Ok(Ok(()))
    });
    checked.await.unwrap_or_else(|e| Err(e.to_string()))
}

/// Reads bytes `[from, length)` of `target` at revision `rev` into `part` at their offsets: ranges
/// cut as `lanes::parts` cuts them (at most `RANGE`), `LANES` side by side on the machine's bulk
/// lanes (`lanes::transfer`).
async fn fetch(
    link: &Link,
    target: &v1::OutputTarget,
    rev: u64,
    from: u64,
    length: u64,
    part: &Path,
    moved: &Arc<AtomicU64>,
) -> Result<(), Failure> {
    let ranges = lanes::parts(&[(0, from)], length, RANGES_PER_LANE * LANES, RANGE);
    let read = lanes::transfer(ranges, LANES, |leg| {
        let lane = link.lane(leg.lane, leg.fresh);
        let (target, part, moved) = (target.clone(), part.to_owned(), moved.clone());
        async move {
            let lane = lane.ok_or_else(|| Ranged::Again("no connection to the machine".into()))?;
            let range = (leg.first, leg.end, length);
            read_range(lane, target, rev, range, part, leg.at, &moved).await
        }
    });
    match read.await {
        Ok(()) => Ok(()),
        Err(Ranged::Stop(failure)) => Err(failure),
        Err(Ranged::Again(message)) => Err(Failure::Again(message)),
    }
}

/// Reads `[first, end)` of an output `length` bytes long into `part`, moving `at` past each byte
/// in place.
async fn read_range(
    lane: Bulk,
    target: v1::OutputTarget,
    rev: u64,
    (first, end, length): (u64, u64, u64),
    part: PathBuf,
    at: Arc<AtomicU64>,
    moved: &AtomicU64,
) -> Result<(), Ranged<Failure>> {
    let mut output = lane
        .get(&target, (first, end), rev)
        .await
        .map_err(refused)?;
    if output.length != length {
        let held = format!(
            "the machine holds {} bytes of revision {rev}, not {length}",
            output.length
        );
        return Err(Ranged::Stop(Failure::Lost(held)));
    }
    if output.range.0 != first {
        return Err(Ranged::Again(format!(
            "the machine answered bytes from {}, not {first}",
            output.range.0
        )));
    }
    let unwritable = |e: std::io::Error| {
        Ranged::Stop(Failure::Again(format!(
            "cannot write {}: {e}",
            part.display()
        )))
    };
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .open(&part)
        .await
        .map_err(unwritable)?;
    file.seek(SeekFrom::Start(first))
        .await
        .map_err(unwritable)?;
    let mut done = first;
    while done < end {
        let data = match output.body.next().await {
            Some(Ok(data)) => data,
            Some(Err(e)) => return Err(Ranged::Again(e.to_string())),
            None => return Err(Ranged::Again("the machine ended the range early".into())),
        };
        let data = &data[..data.len().min((end - done) as usize)];
        file.write_all(data).await.map_err(unwritable)?;
        file.flush().await.map_err(unwritable)?;
        done += data.len() as u64;
        at.store(done, Ordering::Relaxed);
        // Only bytes in place count, each once: a range asked again starts where these stopped.
        moved.fetch_add(data.len() as u64, Ordering::Relaxed);
    }
    Ok(())
}

/// A machine's refusal of a read, as the read acts on it.
fn refused(e: client::Error) -> Ranged<Failure> {
    match e {
        client::Error::Refused { status: 404, error } => Ranged::Stop(Failure::Lost(format!(
            "the machine no longer holds it ({}: {})",
            error.code, error.message
        ))),
        // Definitive: the machine will not serve this read (403, 400, 416, 501) or holds the
        // output at another revision (412). While the run goes on, its events bring the newer
        // one; once it ended, none comes.
        client::Error::Refused {
            status: 400 | 403 | 412 | 416 | 501,
            error,
        } => Ranged::Stop(Failure::Refused(format!(
            "{}: {}",
            error.code, error.message
        ))),
        // Anything else may pass (its key not admitted yet, a refusal while it starts): asked
        // again, paced, until it serves or the machine ends, which loses the item.
        client::Error::Refused { status, error } if status < 500 => {
            Ranged::Stop(Failure::Again(format!("{}: {}", error.code, error.message)))
        }
        client::Error::NotMachineApi(message) => Ranged::Stop(Failure::Again(message)),
        e => Ranged::Again(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_runs_outputs_default_to_its_packages_folder_as_go_named_it() {
        let outputs = Path::new("/home/me/.cozy/outputs");
        for (target, folder) in [
            ("fidika/minimax-h3/long_form@1.39.0", "fidika-minimax-h3"),
            ("paul/sdxl/generate", "paul-sdxl"),
            ("local/h3-tools/caption", "local-h3-tools"),
            (
                "local/cozy-script-0123456789abcdef01234567/main",
                "local-cozy-script-0123456789abcdef01234567",
            ),
        ] {
            assert_eq!(
                package_outputs(outputs, target),
                outputs.join(folder),
                "{target}"
            );
        }
        for target in [
            "hf://org/model → acme/model",
            "../x/y",
            "a/b",
            "a/../b/c",
            "",
        ] {
            assert_eq!(package_outputs(outputs, target), outputs, "{target}");
        }
    }
}
