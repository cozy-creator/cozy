//! Executor process policy: the progress meter's pace, wedge watches and reaping. The OS half
//! (births, exact handles, groups, meters) is `platform::process`. Nothing here kills
//! because time passed. Callers kill on a measured wedge (`Pace`), an owner restart, or a launch
//! that never reached authored code.
use platform::process::{Birth, Exact, State, gone, group_members, meter, written_bytes};
use std::{
    fs::File,
    io,
    process::{Child, ExitStatus},
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

/// Whether a member of `leader`'s group born before `before` (start ticks) still lives: a
/// previous machine's follower, killed with its leader, until its exit is observed.
pub fn group_outlives(leader: &Birth, before: u64) -> bool {
    group_members(leader)
        .iter()
        .any(|member| member.start_ticks < before && member.state().is_ok_and(|state| state != State::Gone))
}

/// The leader's meter plus its group members': a follower's imports, fills and collectives
/// are the executor's work. `None` only when the leader itself is unreadable.
pub fn group_burn(leader: &Birth) -> Option<u64> {
    group_meter(leader).map(|(cpu, moved)| cpu.saturating_add(moved))
}

/// The bytes the leader and its group members moved: what a call's work is measured in besides
/// the progress it reports (`Meter::Work`).
pub fn group_moved(leader: &Birth) -> Option<u64> {
    group_meter(leader).map(|(_, moved)| moved)
}

/// `meter` (CPU ns, bytes moved) summed over the leader and its group members.
fn group_meter(leader: &Birth) -> Option<(u64, u64)> {
    let sum = |(cpu, moved): (u64, u64), (c, m): (u64, u64)| (cpu.saturating_add(c), moved.saturating_add(m));
    Some(group_members(leader).iter().filter_map(|member| meter(member.pid)).fold(meter(leader.pid)?, sum))
}

/// Sampling resolution and noise floor of the wedge rule (Runtime `liveness.py`). A sample
/// period is how often the meter is read, never a deadline.
#[derive(Clone, Copy, Debug)]
pub struct Liveness {
    pub sample: Duration,
}
impl Default for Liveness {
    fn default() -> Self {
        Self {
            sample: Duration::from_secs(5),
        }
    }
}
impl Liveness {
    const STILL_SAMPLES: u32 = 6;
    pub fn floor(self) -> Duration {
        self.sample * Self::STILL_SAMPLES
    }
}

/// One subject's observed pace. Wedged means still for longer than eight times the longest
/// pause it has already shown, and longer than the floor.
#[derive(Clone, Debug, Default)]
pub struct Pace {
    reading: Option<u64>,
    moved_at: Option<Instant>,
    pub worst_pause: Duration,
    changes: u64,
}
impl Pace {
    const STILL_FACTOR: u32 = 8;
    pub fn seeded(worst_pause: Duration) -> Self {
        Self {
            worst_pause,
            ..Self::default()
        }
    }
    pub fn observe(&mut self, reading: Option<u64>, now: Instant) {
        let Some(reading) = reading else { return };
        match (self.reading, self.moved_at) {
            (Some(previous), _) if previous == reading => {}
            (Some(_), Some(moved_at)) => {
                self.worst_pause = self.worst_pause.max(now - moved_at);
                self.changes += 1;
                self.reading = Some(reading);
                self.moved_at = Some(now);
            }
            _ => {
                self.reading = Some(reading);
                self.moved_at = Some(now);
            }
        }
    }
    /// Time spent waiting on someone else (the machine answering a request) is not stillness.
    pub fn excuse(&mut self, now: Instant) {
        if self.moved_at.is_some() {
            self.moved_at = Some(now);
        }
    }
    pub fn still_since(&self) -> Option<Instant> {
        self.moved_at
    }
    pub fn patience(&self, floor: Duration) -> Duration {
        (self.worst_pause * Self::STILL_FACTOR).max(floor)
    }
    /// Why a canceled subject was killed: it did not stop within the patience its own pace
    /// earned before the cancel.
    pub fn ignored(&self, since: Duration, floor: Duration) -> String {
        format!(
            "it did not stop for its cancel in {:.1} s; its longest pause before the cancel was \
             {:.1} s over {} advance(s), so it was given {:.1} s",
            since.as_secs_f64(),
            self.worst_pause.as_secs_f64(),
            self.changes,
            self.patience(floor).as_secs_f64()
        )
    }
    pub fn verdict(&self, still: Duration, floor: Duration) -> String {
        format!(
            "no measurable progress for {:.1} s; its longest earlier pause was {:.1} s over {} \
             advance(s), so it was given {:.1} s",
            still.as_secs_f64(),
            self.worst_pause.as_secs_f64(),
            self.changes,
            self.patience(floor).as_secs_f64()
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Meter {
    /// CPU plus bytes moved by the process (Runtime `ExecutorChild.watched`).
    Burn,
    /// An invocation's work toward its result, judged while it runs, on every executor and every
    /// lane: progress it reports (a step or stage advanced, an output, weights kept, a child call
    /// started; never a repeated frame, telemetry or a log line), bytes moved, or GPU memory
    /// traffic. Never its CPU: a hung call's background threads compute too. A CPU-bound phase
    /// (compile, hash, quantize) advances by the progress it reports. Its call tree is its own
    /// (`Waits`: a child call is watched by its own executor).
    Work,
}

/// Whether a call's GPUs moved device memory (kernels' traffic, copies, peer writes) since the
/// last reading; None while they cannot be read, which judges nothing. An executor with no GPUs
/// has no reader: its GPUs read idle.
pub type Activity = Arc<dyn Fn() -> Option<bool> + Send + Sync>;

/// A measured wedge's verdict, as the error of the command it ended.
#[derive(Debug)]
pub struct Wedged(pub String);
impl std::fmt::Display for Wedged {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl std::error::Error for Wedged {}

#[derive(Default)]
struct WatchState {
    done: bool,
    serving: bool,
    frames: u64,
    /// The work this subject waits on (another call loading a layout it needs), whose progress
    /// is its own while it waits: its re-asks are not.
    lender: Option<Lender>,
    /// The last stage the subject reported (`progress`), where a wedge's verdict says it stopped.
    at: String,
    place: Option<Place>,
    canceled_at: Option<Instant>,
    killed: Option<String>,
    worst_gap: Duration,
}

/// Whether the subject is waiting on work watched elsewhere (a job's unfinished child runs).
pub type Waits = Arc<dyn Fn() -> bool + Send + Sync>;

/// The progress meter of work a subject waits on (the bytes the call loading a layout it asked
/// for moved); None once that work is gone, which lends nothing more.
pub type Lender = Arc<dyn Fn() -> Option<u64> + Send + Sync>;

/// What the subject waits on now (GPU memory asked of other processes), for a wedge's verdict;
/// empty when it names nothing.
pub type Place = Arc<dyn Fn() -> String + Send + Sync>;

/// One exchange's progress observer, shared with whoever may cancel it. It kills the exact
/// process (and its group) only on a measured wedge; the kill closes the channel, so the
/// blocked exchange returns.
pub struct Watch {
    meter: Meter,
    waits: Option<Waits>,
    activity: Option<Activity>,
    state: Mutex<WatchState>,
    changed: Condvar,
}
impl Watch {
    fn run(&self, exact: Exact, liveness: Liveness, worst_gap: Duration, what: &str) {
        let floor = liveness.floor();
        // An invocation's gaps teach the next invocation; CPU-plus-bytes starts its own pace.
        let mut pace = match self.meter {
            Meter::Work => Pace::seeded(worst_gap),
            Meter::Burn => Pace::default(),
        };
        // A GPU call's samples with memory traffic.
        let mut moves = 0;
        // What the work it waits on did so far, kept once that work is gone.
        let mut lent = 0;
        let mut state = self.state.lock().unwrap();
        while !state.done {
            let now = Instant::now();
            // Canceled, the subject must stop: frames it still sends, the machine serving it and
            // its waits on other work no longer count as progress (run 11093 stayed canceling
            // for two hours on them). Its patience is what its pace earned before the cancel.
            if let Some(canceled) = state.canceled_at {
                let since = now - canceled;
                if since > pace.patience(floor) {
                    let verdict = format!("canceled during {what}: {}", pace.ignored(since, floor));
                    match exact.kill() {
                        Ok(()) => state.killed = Some(verdict),
                        Err(error) => tracing::warn!("{verdict}; kill failed: {error}"),
                    }
                    return;
                }
                state = self.changed.wait_timeout(state, liveness.sample).unwrap().0;
                continue;
            }
            lent = state.lender.as_ref().and_then(|lender| lender()).unwrap_or(lent);
            let reading = match self.meter {
                Meter::Burn => group_burn(exact.birth()).map(|burn| burn + lent),
                Meter::Work => self.activity.as_ref().map_or(Some(false), |gpus| gpus()).map(|moved_memory| {
                    moves += u64::from(moved_memory);
                    state.frames + group_moved(exact.birth()).unwrap_or_default() + moves + lent
                }),
            };
            pace.observe(reading, now);
            let unread = self.meter == Meter::Work && reading.is_none();
            if unread || state.serving || self.waits.as_ref().is_some_and(|waits| waits()) {
                pace.excuse(now);
            }
            state.worst_gap = pace.worst_pause;
            if let Some(since) = pace.still_since() {
                let still = now - since;
                if still > pace.patience(floor) {
                    let measures = match self.meter {
                        Meter::Work => " (steps and stages advanced, outputs, bytes moved, GPU memory traffic; never CPU)",
                        Meter::Burn => "",
                    };
                    let at = match state.at.as_str() {
                        "" => String::new(),
                        stage => format!(" at {stage:?}"),
                    };
                    let place = state.place.as_ref().map(|place| place()).filter(|place| !place.is_empty());
                    let place = place.map(|place| format!("; it waits on {place}")).unwrap_or_default();
                    let verdict = format!("wedged during {what}{at}: {}{measures}{place}", pace.verdict(still, floor));
                    match exact.kill() {
                        Ok(()) => state.killed = Some(verdict),
                        Err(error) => tracing::warn!("{verdict}; kill failed: {error}"),
                    }
                    return;
                }
            }
            state = self.changed.wait_timeout(state, liveness.sample).unwrap().0;
        }
    }
    /// The stage the subject reported last.
    pub fn at(&self, stage: &str) {
        let mut state = self.state.lock().unwrap();
        if state.at != stage {
            state.at = stage.to_string();
        }
    }
    /// What it waits on, read when a wedge is judged.
    pub fn place(&self, place: Option<Place>) {
        self.state.lock().unwrap().place = place;
    }
    /// The subject made real progress (`device_executor::advances`).
    pub fn frame(&self) {
        self.state.lock().unwrap().frames += 1;
    }
    /// The subject waits on `lender`'s work (None: it no longer waits).
    pub fn waits_on(&self, lender: Option<Lender>) {
        self.state.lock().unwrap().lender = lender;
    }
    /// Time the machine spends answering this process is not the process's stillness.
    pub fn serving(&self, serving: bool) {
        self.state.lock().unwrap().serving = serving;
    }
    pub fn canceled(&self) {
        let mut state = self.state.lock().unwrap();
        state.canceled_at.get_or_insert_with(Instant::now);
        self.changed.notify_all();
    }
}

/// A running watch; dropping it stops the observer on every path.
pub struct Watching {
    watch: Arc<Watch>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Watching {
    pub fn start(
        exact: Exact,
        meter: Meter,
        liveness: Liveness,
        worst_gap: Duration,
        waits: Option<Waits>,
        activity: Option<Activity>,
        what: &'static str,
    ) -> io::Result<Self> {
        let watch = Arc::new(Watch {
            meter,
            waits,
            activity,
            state: Mutex::new(WatchState::default()),
            changed: Condvar::new(),
        });
        let observer = watch.clone();
        let thread = std::thread::Builder::new()
            .name(format!("watch-{what}"))
            .spawn(move || observer.run(exact, liveness, worst_gap, what))?;
        Ok(Self {
            watch,
            thread: Some(thread),
        })
    }
    pub fn watch(&self) -> Arc<Watch> {
        self.watch.clone()
    }
    /// The kill's measurement, if one was needed, and the longest frame gap seen.
    pub fn finish(mut self) -> (Option<String>, Duration) {
        self.stop();
        let state = self.watch.state.lock().unwrap();
        (state.killed.clone(), state.worst_gap)
    }
    fn stop(&mut self) {
        if let Some(thread) = self.thread.take() {
            self.watch.state.lock().unwrap().done = true;
            self.watch.changed.notify_all();
            let _ = thread.join();
        }
    }
}
impl Drop for Watching {
    fn drop(&mut self) {
        self.stop();
    }
}

/// After its channel closed: wait for this exact process to exit, killing it only on a
/// measured wedge, then reap it. The sample period is the meter's cadence, not a deadline.
/// Then its scope: whatever outlived it (a `setsid` daemon) is killed and counted, and the
/// scope is removed.
pub fn reap(
    exact: &Exact,
    child: Option<&mut Child>,
    liveness: Liveness,
) -> io::Result<Reaped> {
    let floor = liveness.floor();
    let mut pace = Pace::default();
    let mut killed = None;
    while !exact.wait_for(liveness.sample)? {
        let now = Instant::now();
        pace.observe(group_burn(exact.birth()), now);
        if killed.is_some() {
            continue;
        }
        if let Some(since) = pace.still_since() {
            let still = now - since;
            if still > pace.patience(floor) {
                exact.kill()?;
                killed = Some(format!(
                    "wedged while ending: {}",
                    pace.verdict(still, floor)
                ));
            }
        }
    }
    let status = match child {
        Some(child) => child.wait()?,
        None => ExitStatus::default(),
    };
    let stragglers = match exact.scope() {
        Some(scope) => scope.end()?,
        None => 0,
    };
    Ok(Reaped {
        status,
        killed,
        stragglers,
    })
}

/// How a process ended: its status, the measurement behind a kill, and how many of its
/// descendants were still alive in its scope and were killed after it.
#[derive(Debug)]
pub struct Reaped {
    pub status: ExitStatus,
    pub killed: Option<String>,
    pub stragglers: usize,
}

/// `reap` the leader, then every member left in its group (followers die with their leader
/// by parent-death signal; their device teardown is waited for, a wedged one killed).
pub fn reap_group(
    exact: &Exact,
    child: Option<&mut Child>,
    liveness: Liveness,
) -> io::Result<Reaped> {
    // With a scope the leader's reap already ended its followers (they share it).
    let mut reaped = reap(exact, child, liveness)?;
    for member in group_members(exact.birth()) {
        let Some(member) = Exact::open(&member)? else {
            continue;
        };
        if let Some(verdict) = reap(&member, None, liveness)?.killed {
            reaped.killed = Some(match reaped.killed.take() {
                Some(first) => format!("{first}; process {}: {verdict}", member.birth().pid),
                None => format!("process {}: {verdict}", member.birth().pid),
            });
        }
    }
    Ok(reaped)
}

/// How a supervised process ended (`supervise`).
#[derive(Debug)]
pub enum Supervised {
    Exited(ExitStatus),
    /// Its run was canceled: its group was killed.
    Canceled,
    /// Its group's CPU and bytes stopped moving past its measured patience: killed.
    Wedged(String),
}

/// Waits for `child`, the leader of its own process group, reading it once a second: `wrote`
/// sees the bytes its group has written (downloads landing, wheels unpacked; a network read or a
/// linked file counts no byte), and its group is killed when `stop` is canceled or when the
/// group's CPU and bytes stop moving for longer than its measured patience (`Pace`, floored by
/// `Liveness`). The second is resolution, never a verdict.
pub fn supervise(
    child: &mut Child,
    stop: Option<&crate::stop::Stop>,
    what: &str,
    wrote: &dyn Fn(u64),
) -> io::Result<Supervised> {
    let exact = match Birth::of(child.id()).map_err(io::Error::from) {
        Ok(birth) => Exact::open(&birth)?,
        Err(error) if gone(&error) => None,
        Err(error) => return Err(error),
    };
    let Some(exact) = exact else {
        return child.wait().map(Supervised::Exited);
    };
    let (floor, tick) = (Liveness::default().floor(), Duration::from_secs(1));
    let (mut pace, mut ended) = (Pace::default(), None);
    let (mut last, mut advances) = (None, 0u64);
    // What each process of the group wrote so far: a gone member keeps what it wrote.
    let mut writes = std::collections::HashMap::new();
    while !exact.wait_for(tick)? {
        if ended.is_some() {
            continue;
        }
        let now = Instant::now();
        // Bytes moved are progress, and CPU used or owed at a tenth of a core or more (`meter`).
        // An idle process's timer wakeups are not: uv waiting on its build spends one 10 ms
        // tick in about 10 s.
        let group: Vec<u32> = std::iter::once(exact.birth().pid).chain(group_members(exact.birth()).into_iter().map(|m| m.pid)).collect();
        let reading = group.iter().copied().filter_map(meter)
            .fold((0u64, 0u64), |(cpu, moved), (c, m)| (cpu + c, moved + m));
        if last.is_some_and(|(cpu, moved): (u64, u64)| reading.1 != moved || reading.0.saturating_sub(cpu) >= tick.as_nanos() as u64 / 10) {
            advances += 1;
            stop.inspect(|stop| stop.moved());
        }
        last = Some(reading);
        pace.observe(Some(advances), now);
        for (pid, bytes) in group.iter().filter_map(|pid| Some((*pid, written_bytes(*pid)?))) {
            let held: &mut u64 = writes.entry(pid).or_default();
            *held = (*held).max(bytes);
        }
        wrote(writes.values().sum());
        if stop.is_some_and(|stop| stop.is_cancelled()) {
            exact.kill()?;
            ended = Some(Supervised::Canceled);
        } else if let Some(since) = pace.still_since() {
            let still = now - since;
            if still > pace.patience(floor) {
                exact.kill()?;
                ended = Some(Supervised::Wedged(format!("{what} wedged: {}", pace.verdict(still, floor))));
            }
        }
    }
    let status = child.wait()?;
    Ok(ended.unwrap_or(Supervised::Exited(status)))
}

/// A preparation's verdict, on its measured progress (`Stop::progress`: bytes told as they move,
/// stages, its installer's meter) as an invocation's is: still past eight times the longest pause
/// its own pace has shown, measured from `start` (its progress when it began), with no floor. One
/// that never advances is judged too (a job waiting on a silent Hub moved nothing for 20 min under
/// #1875's `changes > 0`); a transfer told as its bytes land advances, however slowly. None once
/// `ended` says it finished. Waiting on work watched elsewhere (another run's install of the same
/// code) is not stillness, nor is `thread` waiting on a TensorFS exchange, which the transport
/// judges by what it moves (a provider's first byte, run 15327).
pub fn judge_preparation(
    stop: &crate::stop::Stop,
    ended: &std::sync::mpsc::Receiver<()>,
    waits: impl Fn() -> bool,
    liveness: Liveness,
    start: (u64, Instant),
    thread: std::thread::ThreadId,
) -> Option<String> {
    let mut pace = Pace::default();
    pace.observe(Some(start.0), start.1);
    while let Err(std::sync::mpsc::RecvTimeoutError::Timeout) = ended.recv_timeout(liveness.sample) {
        let now = Instant::now();
        pace.observe(Some(stop.progress()), now);
        if waits() || tensorfs::fetch::asking::waits(thread) {
            pace.excuse(now);
        }
        let still = pace.still_since().map(|since| now - since).unwrap_or_default();
        if still > pace.patience(Duration::ZERO) {
            return Some(format!("preparing: {}", pace.verdict(still, Duration::ZERO)));
        }
    }
    None
}

/// The last bytes of a log, for a failure reason.
pub fn tail(path: &std::path::Path) -> String {
    tail_from(path, 0, MESSAGE_TAIL)
}

/// The stderr a failure message quotes; a triage bundle keeps [`crate::triage::MAX_STDERR`].
pub const MESSAGE_TAIL: u64 = 2 << 10;

/// The last `limit` bytes written at or after byte `from`.
pub fn tail_from(path: &std::path::Path, from: u64, limit: u64) -> String {
    let Ok(mut file) = File::open(path) else {
        return String::new();
    };
    use std::io::{Read, Seek};
    let length = file.metadata().map(|m| m.len()).unwrap_or(0);
    let _ = file.seek(io::SeekFrom::Start(length.saturating_sub(limit).max(from)));
    let mut bytes = Vec::new();
    let _ = file.read_to_end(&mut bytes);
    String::from_utf8_lossy(&bytes).trim().to_string()
}
