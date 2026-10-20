//! File descriptors: raised to what the process may have, and an exhausted table reported as
//! backpressure.
//!
//! Run 1297 (2026-09-27) grew a pull toward 128 streams under a 1,024 soft RLIMIT_NOFILE;
//! each object in flight holds a socket, a temp file, writer markers and a catalog
//! connection, and the table ran out mid-pull (`name resolution failed (Too many open
//! files)`). A Go supervisor raises its own soft limit but hands its children the original.
use std::sync::Once;

/// Descriptors one request in flight may hold: its socket and an idle one, its object's temp
/// file, and that object's admission.
pub const PER_REQUEST: u64 = 4;

/// Descriptors kept back for what is not a transfer: stdio, the Store's own locks, the hub
/// calls, the progress channel and whatever the caller opens next.
pub const MARGIN: u64 = 64;

static RAISE: Once = Once::new();

/// Raise the soft RLIMIT_NOFILE to the hard limit, once per process. Idempotent; a refused
/// raise leaves the limit where it was and the budget below measures that.
pub fn raise() {
    RAISE.call_once(platform::process::raise_fd_limit);
}

fn full(error: &std::io::Error) -> bool {
    platform::process::fd_table_full(error)
}

/// Requests in flight the raised descriptor limit holds, at most `wanted`: a budget sized once
/// from the limit, not from what happens to be free.
pub fn requests(wanted: usize) -> usize {
    raise();
    platform::process::fd_limit().map_or(wanted, |soft| {
        usize::try_from(soft.saturating_sub(MARGIN) / PER_REQUEST).unwrap_or(usize::MAX)
    })
    .clamp(1, wanted)
}

/// Whether `error` is the descriptor table being full. Call it
/// where an open, socket or resolution fails, before the error loses its errno.
pub fn exhausted(error: &std::io::Error) -> bool {
    full(error)
}

