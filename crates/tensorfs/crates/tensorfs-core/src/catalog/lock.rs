//! Waiting for another process's catalog lock by whether its holder can still release it.
//!
//! SQLite's locks are POSIX locks: they die with the process that holds them. A holder
//! that is alive is committing or about to, so a writer waits for it for as long as it
//! takes; a flat deadline refused a busy, progressing writer as if it were dead. The only
//! holder that cannot release is one that is stopped (a debugger or SIGSTOP), and only
//! then does a statement give up. Dead processes have already released their locks.
//!
//! Holders come from `/proc/locks`, matched by the catalog's inode (and its WAL `-shm`).
//! The catalog file itself is never opened here: closing a second descriptor on it would
//! drop every POSIX lock this process holds on it.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::Connection;

use crate::err::Result;

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
/// holder to name (the lock just went, or no `/proc`), keep asking.
fn holders_cannot_release(path: &Path) -> bool {
    let files: Vec<(u64, u64)> = [path.to_path_buf(), shm(path)]
        .iter()
        .filter_map(|file| std::fs::metadata(file).ok())
        .map(|metadata| {
            use std::os::unix::fs::MetadataExt;
            (metadata.dev(), metadata.ino())
        })
        .collect();
    let Ok(locks) = std::fs::read_to_string("/proc/locks") else {
        return false;
    };
    snapshots_cannot_release(&locks, &files, std::process::id(), stopped, || {
        std::fs::read_to_string("/proc/locks").ok()
    })
}

fn snapshots_cannot_release(
    observed: &str,
    files: &[(u64, u64)],
    own: u32,
    is_stopped: impl Fn(u32) -> bool,
    current: impl FnOnce() -> Option<String>,
) -> bool {
    let holders = write_holders(observed, files, own);
    if holders.is_empty() || !holders.iter().all(|pid| is_stopped(*pid)) {
        return false;
    }
    // A holder may unlock or exit between the lock snapshot and its process-state
    // read. Refuse only if those stopped processes still hold the actual locks.
    let Some(current) = current() else {
        return false;
    };
    let current = write_holders(&current, files, own);
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

/// Other processes' POSIX write locks on these inodes, from `/proc/locks` lines such as
/// `3: POSIX  ADVISORY  WRITE 167671 103:02:204736108 1073741824 1073742335`.
fn write_holders(locks: &str, files: &[(u64, u64)], own: u32) -> Vec<u32> {
    locks
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let at = fields.iter().position(|field| *field == "POSIX")?;
            if fields[..at].contains(&"->") || fields.get(at + 2) != Some(&"WRITE") {
                return None;
            }
            let pid: u32 = fields.get(at + 3)?.parse().ok()?;
            let mut file = fields.get(at + 4)?.split(':');
            let major = u32::from_str_radix(file.next()?, 16).ok()?;
            let minor = u32::from_str_radix(file.next()?, 16).ok()?;
            let inode: u64 = file.next()?.parse().ok()?;
            let device = rustix::fs::makedev(major, minor) as u64;
            (file.next().is_none() && pid != own && files.contains(&(device, inode))).then_some(pid)
        })
        .collect()
}

/// Only stopped and traced-stopped processes keep locks they cannot release.
fn stopped(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .is_some_and(|stat| process_is_stopped(&stat))
}

fn process_is_stopped(stat: &str) -> bool {
    stat.rsplit_once(')')
        .and_then(|(_, fields)| fields.split_whitespace().next())
        .is_some_and(|state| matches!(state, "T" | "t"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_other_processes_write_locks_on_the_catalog_are_holders() {
        let locks = "\
1: POSIX  ADVISORY  WRITE 4242 103:02:77 1073741824 1073742335
2: POSIX  ADVISORY  READ  5151 103:02:77 1073741826 1073742335
3: POSIX  ADVISORY  WRITE 6161 103:02:99 120 120
4: FLOCK  ADVISORY  WRITE 7171 103:02:77 0 EOF
5: POSIX  ADVISORY  WRITE 1000 103:02:77 0 EOF
6: -> POSIX  ADVISORY  WRITE 8181 103:02:77 0 EOF
7: POSIX  ADVISORY  WRITE 9191 104:02:77 0 EOF
";
        let device = rustix::fs::makedev(0x103, 2) as u64;
        assert_eq!(write_holders(locks, &[(device, 77)], 1000), vec![4242]);
        assert_eq!(
            write_holders(locks, &[(device, 77), (device, 99)], 1000),
            vec![4242, 6161]
        );
        assert!(write_holders(locks, &[(device, 5)], 1000).is_empty());
    }

    #[test]
    fn exited_holder_is_progress_and_never_a_stopped_writer() {
        for state in ["Z", "X", "R", "S", "D"] {
            assert!(!process_is_stopped(&format!("4242 (writer) {state} 1 2 3")));
        }
        for state in ["T", "t"] {
            assert!(process_is_stopped(&format!("4242 (writer) {state} 1 2 3")));
        }
        assert!(!process_is_stopped("unavailable"));
    }

    #[test]
    fn stopped_holder_must_still_own_the_lock_before_refusal() {
        let held = "1: POSIX ADVISORY WRITE 4242 103:02:77 0 EOF\n";
        let files = [(rustix::fs::makedev(0x103, 2) as u64, 77)];
        // Stable SIGSTOP remains a typed refusal, while snapshot -> unlock ->
        // SIGSTOP must let SQLite retry rather than refuse the old snapshot.
        assert!(snapshots_cannot_release(
            held,
            &files,
            1000,
            |_| true,
            || Some(held.into())
        ));
        assert!(!snapshots_cannot_release(
            held,
            &files,
            1000,
            |_| true,
            || Some(String::new())
        ));
        assert!(!snapshots_cannot_release(
            held,
            &files,
            1000,
            |_| true,
            || None
        ));
        let replacement = held.replace("4242", "6161");
        assert!(!snapshots_cannot_release(
            held,
            &files,
            1000,
            |_| true,
            || Some(replacement)
        ));
        assert!(!snapshots_cannot_release(
            held,
            &files,
            1000,
            |_| false,
            || panic!("a live holder must retry")
        ));
    }
}
