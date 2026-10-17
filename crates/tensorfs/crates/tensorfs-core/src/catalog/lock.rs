//! Waiting for another process's catalog lock by whether its holder can still release it.
//!
//! SQLite's locks are POSIX locks: they die with the process that holds them. A holder
//! that is alive is committing or about to, so a writer waits for it for as long as it
//! takes; a flat deadline refused a busy, progressing writer as if it were dead. The only
//! holder that cannot release is one that is stopped (a debugger or SIGSTOP), and only
//! then does a statement give up. Dead processes have already released their locks.
//!
//! Holders are named by a lock test on the catalog and its WAL `-shm` (`cozy_platform::lock`), which
//! keeps its probe descriptor open: closing a second descriptor on the catalog would drop every
//! POSIX lock this process holds on it. Where the OS names no holder (Windows, which also has no
//! stopped processes), a statement waits.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::Connection;

use crate::err::Result;
use cozy_platform as platform;

thread_local! {
    /// The catalog this thread's statements wait on. A connection is opened for each
    /// operation just before it is used, on the thread that uses it, so the one opened last
    /// here is the one a busy statement belongs to.
    static WAITING_ON: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// Install the waiting handler on one connection.
pub(crate) fn wait_for_live_holders(connection: &Connection, path: &Path) -> Result<()> {
    WAITING_ON.with(|waiting| *waiting.borrow_mut() = Some(path.to_path_buf()));
    connection
        .busy_handler(Some(busy))
        .map_err(|error| super::sql("install catalog lock wait", error))
}

fn busy(attempt: i32) -> bool {
    let path = WAITING_ON.with(|waiting| waiting.borrow().clone());
    if path.is_some_and(|path| holders_cannot_release(&path)) {
        return false;
    }
    std::thread::sleep(Duration::from_millis(attempt.clamp(1, 64) as u64));
    true
}

/// Every currently held write lock belongs to a stopped process. Without a current
/// holder to name (the lock just went, or the OS cannot say), keep asking.
fn holders_cannot_release(path: &Path) -> bool {
    let writers = || -> Option<Vec<u32>> {
        let mut pids = platform::lock::posix_writers(path).ok()?;
        pids.extend(platform::lock::posix_writers(&shm(path)).ok()?);
        Some(pids)
    };
    let Some(observed) = writers() else {
        return false;
    };
    snapshots_cannot_release(observed, stopped, writers)
}

fn snapshots_cannot_release(
    holders: Vec<u32>,
    is_stopped: impl Fn(u32) -> bool,
    current: impl FnOnce() -> Option<Vec<u32>>,
) -> bool {
    if holders.is_empty() || !holders.iter().all(|pid| is_stopped(*pid)) {
        return false;
    }
    // A holder may unlock or exit between the lock snapshot and its process-state
    // read. Refuse only if those stopped processes still hold the actual locks.
    let Some(current) = current() else {
        return false;
    };
    !current.is_empty()
        && current
            .iter()
            .all(|pid| holders.contains(pid) && is_stopped(*pid))
}

fn shm(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push("-shm");
    PathBuf::from(name)
}

/// Only stopped and traced-stopped processes keep locks they cannot release.
fn stopped(pid: u32) -> bool {
    platform::process::Birth::of(pid)
        .and_then(|birth| birth.state())
        .is_ok_and(|state| state == platform::process::State::Stopped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopped_holder_must_still_own_the_lock_before_refusal() {
        let held = || Some(vec![4242]);
        // Stable SIGSTOP remains a typed refusal, while snapshot -> unlock ->
        // SIGSTOP must let SQLite retry rather than refuse the old snapshot.
        assert!(snapshots_cannot_release(vec![4242], |_| true, held));
        assert!(!snapshots_cannot_release(vec![4242], |_| true, || Some(vec![])));
        assert!(!snapshots_cannot_release(vec![4242], |_| true, || None));
        assert!(!snapshots_cannot_release(vec![4242], |_| true, || Some(vec![6161])));
        assert!(!snapshots_cannot_release(vec![4242], |_| false, || panic!(
            "a live holder must retry"
        )));
        assert!(!snapshots_cannot_release(vec![], |_| true, held));
    }
}
