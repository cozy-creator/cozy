//! The pull's liveness ledger and the caller's deadline — cozy-runtime `pull.py`'s
//! `Deadline`/`_Streams`, ported with their evidence (tfs-049).
//!
//! **No magic timeouts, and no unbounded silence either.** A deadline arrives from the
//! caller or there is none; nothing here invents a number of seconds to fail on. But a
//! presigned GET that accepts a connection and then stops sending held a pool slot until
//! the deadline expired, so one black-holed stream cost the whole pull. This ledger is the
//! answer, and its rule is MEASURED rather than invented: every chunk is counted where it
//! crosses into this process, the longest gap the pull has ACTUALLY shown — the origin's
//! own time-to-answer included — is remembered, and a stream silent for far longer than
//! that is stalled, closed, and its object re-asked, which the plan already made cost one
//! object. A slow link teaches the ledger its own pace and is left alone. The same rule,
//! lifted out for processes, is cozy-runtime's `internal/liveness.py`, and the constants
//! below are shared with it so a stream and a process are judged at the same resolution.
//!
//! The Python original watched from a sampling thread and woke blocked reads by shutting
//! the socket down. Here each stream samples ITSELF: its socket read timeout is the
//! sampling resolution — resolution, not a verdict — and on each empty tick it consults
//! the deadline and this ledger. Same rule, same granularity, no watchdog thread.

use crate::err::{refuse, Code, Result};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How many objects are in flight by default. A MEASURED number, not a taste: #566b swept
/// the same pods, the same bucket and the same presigned GETs and found the per-stream
/// rate flat and the aggregate saturating by 16 (49.8 MB/s at 1, 117.6 MB/s at 16, against
/// the 141 MB/s ceiling the pod's own transport probe reached). It is a parameter because
/// the right number is a property of the LINK, and this module cannot measure the caller's.
pub const STREAMS: usize = 16;

/// How often a waiting stream SAMPLES — the runtime's one sampling cadence, shared with
/// cozy-runtime `internal/liveness.py::SAMPLE_SECONDS`. Resolution, not a verdict: nothing
/// is ended because this many seconds passed.
pub const SAMPLE_SECONDS: f64 = 5.0;

/// The NOISE FLOOR, in samples, below which silence is not evidence of anything — shared
/// with `liveness.py::STILL_SAMPLES`. A pull whose chunks came back in milliseconds must
/// not condemn a stream that is merely a second slow.
pub const STILL_SAMPLES: u32 = 6;

/// How many times the pull's OWN worst measured gap a stream may be silent for before it
/// is stalled. This is the term that carries the verdict, and it is MEASURED: a link that
/// has shown 30 s between chunks — or an origin that took 30 s to start answering — earns
/// 240 s of patience, and one that has never paused earns the floor.
pub const STALL_FACTOR: f64 = 8.0;

/// How many times one object may be re-asked for. A grant is finite and the plan is
/// per-object resumable, so a retry costs one object, not the pull. A stall or a transient
/// object-store answer that survives this many attempts is the link, not a blip.
pub const FETCH_ATTEMPTS: u32 = 4;

/// A caller-supplied wall deadline, or none at all. `remaining()` is what every socket
/// operation is bounded by, so a slow hub and a slow object share one budget instead of
/// each inventing its own. `None` means the caller set no deadline and the pull blocks
/// until the transport itself gives up — which is honest: a module that substituted a
/// number here would be deciding, on the caller's behalf, that a large artifact on a slow
/// link is a failure.
#[derive(Debug, Clone, Copy)]
pub struct Deadline {
    at: Option<Instant>,
}

impl Deadline {
    pub fn none() -> Deadline {
        Deadline { at: None }
    }

    pub fn after_seconds(seconds: Option<f64>) -> Deadline {
        Deadline {
            at: seconds.map(|s| Instant::now() + Duration::from_secs_f64(s.clamp(0.0, 1e9))),
        }
    }

    /// What is LEFT, or the typed refusal once nothing is.
    pub fn remaining(&self) -> Result<Option<Duration>> {
        match self.at {
            None => Ok(None),
            Some(at) => {
                let now = Instant::now();
                if now >= at {
                    refuse(
                        Code::DEADLINE_EXCEEDED,
                        "the caller's deadline passed with objects still to move — re-run: \
                         a pull resumes per object and re-fetches only what is missing",
                    )
                } else {
                    Ok(Some(at - now))
                }
            }
        }
    }
}

/// The one liveness ledger a pull shares across its streams.
pub struct Ledger {
    sample: Duration,
    worst_gap: Mutex<Duration>,
}

impl Ledger {
    pub fn new() -> Ledger {
        Ledger::with_resolution(Duration::from_secs_f64(SAMPLE_SECONDS))
    }

    /// The cadence is configuration — resolution, never a verdict — so a test may sample
    /// in milliseconds and judge by exactly the production rule.
    pub fn with_resolution(sample: Duration) -> Ledger {
        Ledger {
            sample: sample.max(Duration::from_millis(1)),
            worst_gap: Mutex::new(Duration::ZERO),
        }
    }

    pub fn sample(&self) -> Duration {
        self.sample
    }

    /// The smallest patience ever granted: `STILL_SAMPLES` intervals of the resolution.
    pub fn floor(&self) -> Duration {
        self.sample * STILL_SAMPLES
    }

    /// One observed gap between a stream's chunks — or an origin's time-to-first-answer,
    /// which is the first gap a stream has shown. A stream a caller has already judged
    /// stalled must not report here: counting post-verdict silence as the link's honest
    /// pace would let every stall raise the patience the next stall is judged by.
    pub fn taught(&self, gap: Duration) {
        let mut worst = self.worst_gap.lock().unwrap();
        if gap > *worst {
            *worst = gap;
        }
    }

    /// The verdict term: silence beyond BOTH the measured term and the floor is a stall.
    pub fn patience(&self) -> Duration {
        let worst = *self.worst_gap.lock().unwrap();
        worst.mul_f64(STALL_FACTOR).max(self.floor())
    }
}

impl Default for Ledger {
    fn default() -> Ledger {
        Ledger::new()
    }
}
