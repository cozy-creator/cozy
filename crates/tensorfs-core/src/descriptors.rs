//! File descriptors as a budget: raised to what the process may have, measured before a
//! transfer sizes itself, and an exhausted table reported as backpressure.
//!
//! Run 1297 (2026-09-27) grew a pull toward 128 streams under a 1,024 soft RLIMIT_NOFILE;
//! each object in flight holds a socket, a temp file, writer markers and a catalog
//! connection, and the table ran out mid-pull (`name resolution failed (Too many open
//! files)`). A Go supervisor raises its own soft limit but hands its children the original.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Once;

/// Descriptors one object stream may hold at once, and what it is spent on: the object
/// socket and one pooled idle socket (2); the temp file and the two writer-marker files of
/// its admission, nested once more by the catalog record (5); the catalog connection and its
/// journal (2); and the transient DNS socket or contender of a retry (3).
pub const PER_STREAM: u64 = 12;

/// Descriptors left for everything that is not a stream: stdio, the Store's own locks, the
/// hub calls, the progress channel and whatever the caller opens next.
pub const MARGIN: u64 = 64;

static RAISE: Once = Once::new();
static EXHAUSTED: AtomicU64 = AtomicU64::new(0);

/// Raise the soft RLIMIT_NOFILE to the hard limit, once per process. Idempotent; a refused
/// raise leaves the limit where it was and the budget below measures that.
pub fn raise() {
    RAISE.call_once(|| {
        use rustix::process::{getrlimit, setrlimit, Resource, Rlimit};
        let limit = getrlimit(Resource::Nofile);
        if let (Some(soft), Some(hard)) = (limit.current, limit.maximum) {
            if soft < hard {
                let _ = setrlimit(
                    Resource::Nofile,
                    Rlimit {
                        current: Some(hard),
                        maximum: Some(hard),
                    },
                );
            }
        }
    });
}

/// Descriptors this process can still open now: soft limit less those open and `MARGIN`.
/// `None` when the platform cannot say (no `/proc`, or no soft limit).
pub fn available() -> Option<u64> {
    let soft = rustix::process::getrlimit(rustix::process::Resource::Nofile).current?;
    let open = std::fs::read_dir("/proc/self/fd").ok()?.count() as u64;
    Some(soft.saturating_sub(open).saturating_sub(MARGIN))
}

/// Object streams the descriptor table allows now: at most `wanted`, at least one.
pub fn streams_within(wanted: usize) -> usize {
    streams_for(wanted, available())
}

/// `streams_within` against a stated number of free descriptors.
pub fn streams_for(wanted: usize, available: Option<u64>) -> usize {
    let fits = available.map_or(usize::MAX, |free| {
        usize::try_from(free / PER_STREAM).unwrap_or(usize::MAX)
    });
    wanted.min(fits).max(1)
}

/// Whether `error` is the descriptor table being full, counting it when it is. Call it
/// where an open, socket or resolution fails, before the error loses its errno.
pub fn exhausted(error: &std::io::Error) -> bool {
    let full = matches!(
        error.raw_os_error(),
        Some(code) if code == rustix::io::Errno::MFILE.raw_os_error()
            || code == rustix::io::Errno::NFILE.raw_os_error()
    );
    if full {
        EXHAUSTED.fetch_add(1, Ordering::Relaxed);
    }
    full
}

/// How many times this process has found its descriptor table full. A caller diffs two
/// reads around its own work.
pub fn exhaustions() -> u64 {
    EXHAUSTED.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_budget_is_descriptors_over_per_stream_cost() {
        assert_eq!(streams_for(128, Some(PER_STREAM * 10)), 10);
        assert_eq!(streams_for(8, Some(PER_STREAM * 10)), 8);
        assert_eq!(streams_for(128, Some(0)), 1, "one stream is always allowed");
        assert_eq!(
            streams_for(128, None),
            128,
            "an unmeasurable table is not a shortage"
        );
    }

    #[test]
    fn only_a_full_table_counts_as_exhaustion() {
        let before = exhaustions();
        assert!(exhausted(&std::io::Error::from_raw_os_error(
            rustix::io::Errno::MFILE.raw_os_error()
        )));
        assert!(!exhausted(&std::io::Error::from_raw_os_error(
            rustix::io::Errno::CONNREFUSED.raw_os_error()
        )));
        assert!(exhaustions() > before);
    }
}
