//! The run's watch over its worker: a verdict from OBSERVATION, never from a clock.
//!
//! `tfs ingest run` used to wrap the worker in `ulimit -t 3600`. Ingest is CPU-bound at
//! ~114 MB/s, so that was a 400 GB ceiling on any input, SIGKILL mid-write, and the retry
//! hit the same wall (xs-007 row 4). No duration here is a verdict. The worker announces
//! every chunk the store moves on its stdout (`progress\t<moved>\t<admitted>`, see
//! `ingest::Reporter`), and on Linux its CPU clock and scheduler state are readable in
//! `/proc/<pid>/stat`; the two windows below are derived from those observations.
//!
//! **patience** — how long the progress channel may be silent before the parent looks
//! harder: `PATIENCE_FACTOR` × the longest silence between two records this run has itself
//! survived (the preamble before the first byte counts), floored at the catalog's declared
//! lock wait, the one wait a store call can legitimately spend asleep. Silence alone never
//! kills. At each expiry the parent reads the worker's CPU clock and state; a worker that has
//! moved no byte, printed nothing, burned no CPU tick and been seen neither running nor in
//! I/O across one FULL window between two readings is idle. That is observed inactivity.
//!
//! **runaway** — CPU burning with no byte moved is the other failure the ruling names, lack
//! of progress. Once the run has shown a rate, silence on the channel is granted
//! `PATIENCE_FACTOR` × the wall time the whole declared input takes at that rate (declared
//! bytes ÷ observed throughput), never less than patience: no phase of the worker that does
//! not stream legitimately costs more than streaming its entire input. Before the first byte
//! there is no rate and no fence — the worker's preamble repeats the parent's own `prepare`
//! on the same bytes and a deterministic golden suite, so a spin there would have spun the
//! parent first.

use std::io::{BufRead, BufReader};
use std::process::{Child, ChildStdout, ExitStatus};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

/// How many times its own worst silence a worker is granted before the parent looks for
/// inactivity, and how many times its own projected run it may spin without progress.
/// Three doublings past the demonstrated maximum: the worst of n observed gaps
/// underestimates the true tail, each doubling of the margin covers another binary order
/// of it, and at three the cost of the wait (a dead worker holds its session that much
/// longer) is still negligible against the cost of a false kill (the run so far — hours on
/// a real ingest). It is the margin `cozy-runtime`'s pull ledger gives a stream
/// (`STALL_FACTOR`), so the two stall rules on the ingest path agree.
const PATIENCE_FACTOR: u32 = 8;

/// The record the worker prints per chunk moved. A tab, so no human line can spell it.
pub const PROGRESS_PREFIX: &str = "progress\t";

#[derive(Debug)]
pub enum Verdict {
    Exited(ExitStatus),
    /// No byte, no line, no CPU tick, never seen running or in I/O, for a full window.
    Stalled {
        silent: Duration,
        patience: Duration,
    },
    /// Alive and burning CPU, but silent past what its own rate says the whole input costs.
    Runaway {
        silent: Duration,
        allowed: Duration,
    },
    Lost(std::io::Error),
}

#[derive(Debug)]
pub struct Report {
    pub verdict: Verdict,
    pub moved: u64,
    pub admitted: u64,
    pub elapsed: Duration,
    pub worst_gap: Duration,
    pub patience: Duration,
}

enum Event {
    Progress(u64, u64),
    Line(String),
}

/// One look at `/proc/<pid>/stat`: CPU ticks consumed and the scheduler state letter.
struct Reading {
    ticks: u64,
    state: char,
}

fn read_proc(pid: u32) -> Option<Reading> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // Everything after the parenthesised comm, which may itself hold spaces.
    let rest = &stat[stat.rfind(')')? + 1..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let state = fields.first()?.chars().next()?;
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some(Reading {
        ticks: utime + stime,
        state,
    })
}

fn parse_progress(line: &str) -> Option<(u64, u64)> {
    let body = line.strip_prefix(PROGRESS_PREFIX)?;
    let (moved, admitted) = body.split_once('\t')?;
    Some((moved.parse().ok()?, admitted.parse().ok()?))
}

/// Patience before a worker has shown any gap to measure against.
const FIRST_PATIENCE: Duration = Duration::from_secs(5);

fn patience(worst_gap: Duration) -> Duration {
    (worst_gap * PATIENCE_FACTOR).max(FIRST_PATIENCE)
}

/// What a silent-but-active worker is allowed: the whole declared input at the rate this
/// run has shown, times the factor — and never less than patience.
fn runaway_allowance(declared: u64, moved: u64, elapsed: Duration, patience: Duration) -> Duration {
    let whole = elapsed.mul_f64(declared.max(moved) as f64 / moved as f64);
    (whole * PATIENCE_FACTOR).max(patience)
}

/// Relay the worker's lines, consume its progress records, and end it only on observed
/// inactivity or observed lack of progress. `declared` is the plan's byte total.
pub fn supervise(child: &mut Child, stdout: ChildStdout, declared: u64) -> Report {
    let pid = child.id();
    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            let event = match parse_progress(&line) {
                Some((moved, admitted)) => Event::Progress(moved, admitted),
                None => Event::Line(line),
            };
            if tx.send(event).is_err() {
                break;
            }
        }
    });

    let spawned = Instant::now();
    let mut last_record = spawned;
    let mut last_active = spawned;
    let mut worst_gap = Duration::ZERO;
    let (mut moved, mut admitted) = (0u64, 0u64);
    // The reading a silence is judged against. None until the first look into a silence,
    // so the first expiry only takes a baseline and the verdict needs a second, one full
    // window later, that agrees.
    let mut baseline: Option<Reading> = None;
    let verdict = loop {
        let patience = patience(worst_gap);
        match rx.recv_timeout(patience) {
            Ok(Event::Progress(m, a)) => {
                let now = Instant::now();
                worst_gap = worst_gap.max(now - last_record);
                last_record = now;
                last_active = now;
                moved = m;
                admitted = a;
                baseline = None;
            }
            Ok(Event::Line(line)) => {
                println!("{line}");
                last_active = Instant::now();
                baseline = None;
            }
            Err(RecvTimeoutError::Disconnected) => break wait(child),
            Err(RecvTimeoutError::Timeout) => {
                let now = Instant::now();
                let reading = read_proc(pid);
                // `observed` is false on the first look into a silence: that reading is
                // the baseline, and a verdict needs the next one.
                let (observed, active) = match (&baseline, &reading) {
                    (None, _) => (false, true),
                    (Some(b), Some(r)) => (true, r.ticks > b.ticks || matches!(r.state, 'R' | 'D')),
                    (Some(_), None) => (true, false),
                };
                if active {
                    last_active = now;
                }
                baseline = Some(reading.unwrap_or(Reading {
                    ticks: 0,
                    state: '?',
                }));
                if now.duration_since(last_active) >= patience {
                    break kill(
                        child,
                        Verdict::Stalled {
                            silent: now - last_record,
                            patience,
                        },
                    );
                }
                if observed && moved > 0 {
                    let allowed =
                        runaway_allowance(declared, moved, last_record - spawned, patience);
                    if now.duration_since(last_record) >= allowed {
                        break kill(
                            child,
                            Verdict::Runaway {
                                silent: now - last_record,
                                allowed,
                            },
                        );
                    }
                }
            }
        }
    };
    let _ = reader.join();
    Report {
        verdict,
        moved,
        admitted,
        elapsed: spawned.elapsed(),
        worst_gap,
        patience: patience(worst_gap),
    }
}

fn wait(child: &mut Child) -> Verdict {
    match child.wait() {
        Ok(status) => Verdict::Exited(status),
        Err(e) => Verdict::Lost(e),
    }
}

fn kill(child: &mut Child, verdict: Verdict) -> Verdict {
    if let Err(e) = child.kill() {
        return Verdict::Lost(e);
    }
    match child.wait() {
        Ok(_) => verdict,
        Err(e) => Verdict::Lost(e),
    }
}
