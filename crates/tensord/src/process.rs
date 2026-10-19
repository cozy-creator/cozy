//! Executor process policy: the progress meter's pace, wedge watches and reaping. The OS half
//! (births, exact handles, groups, meters) is `platform::process`. Nothing here kills
//! because time passed. Callers kill on a measured wedge (`Pace`), an owner restart, or a launch
//! that never reached authored code.
use platform::process::{Birth, Exact, State, gone, group_members, meter, read_bytes};
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
    /// Frames of the running invocation; a canceled one must stop within its pace's patience.
    Frames,
    /// A GPU invocation's work, judged while it runs: its frames, the bytes its group moved and
    /// its GPUs' kernels and copies. Never CPU, which a spin burns (runs 11818, 11822).
    Work,
}

/// GPU work seen so far (a count of busy readings); None while it cannot be read, which judges
/// nothing.
pub type Activity = Arc<dyn Fn() -> Option<u64> + Send + Sync>;

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
    canceled_at: Option<Instant>,
    killed: Option<String>,
    worst_gap: Duration,
}

/// Whether the subject is waiting on work watched elsewhere (a job's unfinished child runs).
pub type Waits = Arc<dyn Fn() -> bool + Send + Sync>;

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
        // Frame gaps teach only the frame meter; CPU-plus-bytes starts its own pace.
        let mut pace = match self.meter {
            Meter::Frames | Meter::Work => Pace::seeded(worst_gap),
            Meter::Burn => Pace::default(),
        };
        // A GPU call's samples whose group CPU ran above one core, and its last CPU reading.
        let (mut cores, mut last_cpu) = (0, None);
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
            let reading = match self.meter {
                Meter::Burn => group_burn(exact.birth()),
                Meter::Frames => Some(state.frames),
                Meter::Work => self.activity.as_ref().and_then(|activity| activity()).map(|gpu| {
                    let (cpu, moved) = group_meter(exact.birth()).unwrap_or_default();
                    // CPU on more than one core is work (1.5 cores leaves room for sampling skew):
                    // real CPU phases (torch intra-op, ffmpeg, BLAS) run many threads, a spin one,
                    // and a missed kill beats killing a real run.
                    if let Some((before, at)) = last_cpu.replace((cpu, now)) {
                        cores += u64::from(u128::from(cpu.saturating_sub(before)) * 2 > (now - at).as_nanos() * 3);
                    }
                    gpu + state.frames + moved + cores
                }),
            };
            pace.observe(reading, now);
            let unread = self.meter == Meter::Work && reading.is_none();
            if unread || state.serving || self.waits.as_ref().is_some_and(|waits| waits()) {
                pace.excuse(now);
            }
            state.worst_gap = pace.worst_pause;
            // Frames alone are judged only after a cancel (above); CPU and bytes, and a GPU
            // call's work, always.
            if let (Meter::Burn | Meter::Work, Some(since)) = (self.meter, pace.still_since()) {
                let still = now - since;
                if still > pace.patience(floor) {
                    let measures = match self.meter {
                        Meter::Work => " (frames, bytes moved, GPU kernels and copies, CPU beyond one core)",
                        _ => "",
                    };
                    let verdict = format!("wedged during {what}: {}{measures}", pace.verdict(still, floor));
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
    pub fn frame(&self) {
        self.state.lock().unwrap().frames += 1;
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

/// Waits for `child`, the leader of its own process group, reading it once a second: `read`
/// sees the bytes it has read, and its group is killed when `stop` is canceled or when the
/// group's CPU and bytes stop moving for longer than its measured patience (`Pace`, floored by
/// `Liveness`). The second is resolution, never a verdict.
pub fn supervise(
    child: &mut Child,
    stop: Option<&crate::stop::Stop>,
    what: &str,
    read: &dyn Fn(u64),
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
    while !exact.wait_for(tick)? {
        if ended.is_some() {
            continue;
        }
        let now = Instant::now();
        // Bytes moved are progress, and CPU used or owed at a tenth of a core or more (`meter`).
        // An idle process's timer wakeups are not: uv waiting on its build spends one 10 ms
        // tick in about 10 s.
        let members = group_members(exact.birth()).into_iter().map(|member| member.pid);
        let reading = std::iter::once(exact.birth().pid).chain(members).filter_map(meter)
            .fold((0u64, 0u64), |(cpu, moved), (c, m)| (cpu + c, moved + m));
        if last.is_some_and(|(cpu, moved): (u64, u64)| reading.1 != moved || reading.0.saturating_sub(cpu) >= tick.as_nanos() as u64 / 10) {
            advances += 1;
        }
        last = Some(reading);
        pace.observe(Some(advances), now);
        if let Some(bytes) = read_bytes(exact.birth().pid) {
            read(bytes);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::process::CommandExt;
    #[cfg(unix)]
    use std::process::{Command, Stdio};

    #[test]
    fn pace_learns_its_own_pauses_and_judges_against_the_floor() {
        let start = Instant::now();
        let floor = Duration::from_secs(30);
        let mut pace = Pace::default();
        pace.observe(Some(1), start);
        pace.observe(Some(2), start + Duration::from_secs(10));
        assert_eq!(pace.worst_pause, Duration::from_secs(10));
        assert_eq!(pace.patience(floor), Duration::from_secs(80));
        pace.observe(None, start + Duration::from_secs(500)); // unreadable decides nothing
        pace.observe(Some(2), start + Duration::from_secs(500));
        assert_eq!(pace.still_since(), Some(start + Duration::from_secs(10)));
        assert_eq!(Pace::default().patience(floor), floor);
    }

    #[cfg(unix)]
    #[test]
    fn reap_group_waits_for_members_left_behind_and_ends_a_still_one() {
        // The leader leaves two members in its group; killing only the leader (no group
        // signal) orphans them, as a follower can outlive rank 0 for a moment.
        let mut child = Command::new("/bin/sh")
            .args(["-c", "sleep 1000 & sleep 1000 & wait"])
            .process_group(0)
            .stdin(Stdio::null())
            .spawn()
            .unwrap();
        let birth = Birth::of(child.id()).unwrap();
        let members = loop {
            let members = group_members(&birth);
            if members.len() == 2 {
                break members;
            }
            std::thread::yield_now();
        };
        let exact = Exact::open(&birth).unwrap().unwrap();
        unsafe { libc::kill(birth.pid as i32, libc::SIGKILL) };
        let liveness = Liveness {
            sample: Duration::from_millis(50),
        };
        let reaped = reap_group(&exact, Some(&mut child), liveness).unwrap();
        let (status, killed) = (reaped.status, reaped.killed);
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(status.signal(), Some(libc::SIGKILL));
        // A sleeping member shows no progress: it is ended on that measurement, then waited.
        let killed = killed.expect("still members are ended on their measurement");
        assert!(killed.contains("no measurable progress"), "{killed}");
        for member in members {
            assert!(member.ended().unwrap(), "{member:?}");
        }
        assert!(group_members(&birth).is_empty());
    }

    /// A GPU call is judged while it runs on its work: one spinning on one core with no frames,
    /// no bytes and idle GPUs is killed (runs 11818, 11822); one that moves bytes, whose GPUs
    /// work, whose GPUs cannot be read or that runs on more than one core is not.
    #[cfg(unix)]
    #[test]
    fn a_gpu_call_spinning_without_work_is_killed_and_working_ones_are_not() {
        let cases: [(&str, &str, Option<Activity>, bool); 5] = [
            ("spins", "while :; do :; done", Some(Arc::new(|| Some(0))), true),
            ("moves bytes", "while :; do echo work; done > /dev/null", Some(Arc::new(|| Some(0))), false),
            ("GPU works", "while :; do :; done", {
                let busy = std::sync::atomic::AtomicU64::new(0);
                Some(Arc::new(move || Some(busy.fetch_add(1, std::sync::atomic::Ordering::Relaxed))))
            }, false),
            ("GPU unread", "while :; do :; done", Some(Arc::new(|| None)), false),
            ("spins on two cores", "while :; do :; done & while :; do :; done", Some(Arc::new(|| Some(0))), false),
        ];
        for (case, script, activity, killed) in cases {
            let mut child = Command::new("/bin/sh").args(["-c", script]).process_group(0).stdin(Stdio::null()).spawn().unwrap();
            let birth = Birth::of(child.id()).unwrap();
            let exact = Exact::open(&birth).unwrap().unwrap();
            let liveness = Liveness { sample: Duration::from_millis(20) };
            let watching = Watching::start(exact, Meter::Work, liveness, Duration::ZERO, None, activity, "invoke").unwrap();
            // Long past the floor (0.12 s): only measured stillness ends it.
            std::thread::sleep(liveness.floor() * 25);
            assert_eq!(birth.ended().unwrap(), killed, "{case}");
            let verdict = watching.finish().0;
            assert_eq!(verdict.is_some(), killed, "{case}: {verdict:?}");
            if killed {
                let verdict = verdict.unwrap();
                assert!(verdict.contains("wedged during invoke: no measurable progress"), "{case}");
                assert!(verdict.contains("(frames, bytes moved, GPU kernels and copies, CPU beyond one core)"), "{verdict}");
            } else {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_subject_waiting_on_watched_work_is_still_only_once_that_work_ends() {
        // A job root sleeps while its child runs on the GPU (run 3858 was ended there).
        let mut child = Command::new("/bin/sleep")
            .arg("1000")
            .process_group(0)
            .stdin(Stdio::null())
            .spawn()
            .unwrap();
        let birth = Birth::of(child.id()).unwrap();
        let exact = Exact::open(&birth).unwrap().unwrap();
        let liveness = Liveness {
            sample: Duration::from_millis(20),
        };
        let waiting = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let waits = waiting.clone();
        let watching = Watching::start(
            exact,
            Meter::Burn,
            liveness,
            Duration::ZERO,
            Some(Arc::new(move || {
                waits.load(std::sync::atomic::Ordering::Acquire)
            })),
            None,
            "run_job",
        )
        .unwrap();
        std::thread::sleep(liveness.floor() * 4);
        assert!(!birth.ended().unwrap());
        waiting.store(false, std::sync::atomic::Ordering::Release);
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(child.wait().unwrap().signal(), Some(libc::SIGKILL));
        let killed = watching.finish().0.unwrap();
        assert!(killed.contains("wedged during run_job"), "{killed}");
    }


    /// A canceled call that keeps sending frames (or that the machine serves, or that waits on
    /// other work) still has to stop: it is killed once its pace's patience after the cancel
    /// is spent, with a verdict naming the cancel (run 11093).
    #[cfg(unix)]
    #[test]
    fn a_canceled_subject_is_killed_however_busy_it_looks() {
        for excuse in ["frames", "serving", "waits"] {
            let mut child = Command::new("/bin/sleep")
                .arg("1000")
                .process_group(0)
                .stdin(Stdio::null())
                .spawn()
                .unwrap();
            let birth = Birth::of(child.id()).unwrap();
            let exact = Exact::open(&birth).unwrap().unwrap();
            let liveness = Liveness { sample: Duration::from_millis(20) };
            let waits: Option<Waits> = (excuse == "waits").then(|| Arc::new(|| true) as Waits);
            let meter = if excuse == "waits" { Meter::Burn } else { Meter::Frames };
            let watching = Watching::start(exact, meter, liveness, Duration::ZERO, waits, None, "invoke").unwrap();
            let watch = watching.watch();
            watch.serving(excuse == "serving");
            let busy = Arc::new(std::sync::atomic::AtomicBool::new(true));
            let frames = {
                let (watch, busy) = (watch.clone(), busy.clone());
                std::thread::spawn(move || {
                    while busy.load(std::sync::atomic::Ordering::Acquire) {
                        watch.frame();
                        std::thread::sleep(Duration::from_millis(5));
                    }
                })
            };
            std::thread::sleep(liveness.floor() * 2);
            assert!(!birth.ended().unwrap(), "{excuse}: killed before any cancel");
            watch.canceled();
            assert_eq!(std::os::unix::process::ExitStatusExt::signal(&child.wait().unwrap()), Some(libc::SIGKILL), "{excuse}");
            busy.store(false, std::sync::atomic::Ordering::Release);
            frames.join().unwrap();
            let killed = watching.finish().0.unwrap();
            assert!(killed.contains("canceled during invoke: it did not stop for its cancel"), "{excuse}: {killed}");
        }
    }

}
