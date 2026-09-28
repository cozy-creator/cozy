//! Waiting for another process's catalog lock by whether its holder can still release it.
//!
//! SQLite's locks are POSIX locks: they die with the process that holds them. A holder
//! that is alive is committing or about to, so a writer waits for it for as long as it
//! takes; a flat deadline refused a busy, progressing writer as if it were dead. The only
//! holder that cannot release is one that is stopped (a debugger, SIGSTOP) or a zombie,
//! and only then does a statement give up.
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

/// Every process holding a write lock on the catalog is stopped or a zombie. Without a
/// holder to name (the lock just went, or no `/proc`), keep asking.
fn holders_cannot_release(path: &Path) -> bool {
    let inodes: Vec<u64> = [path.to_path_buf(), shm(path)]
        .iter()
        .filter_map(|file| std::fs::metadata(file).ok())
        .map(|metadata| std::os::unix::fs::MetadataExt::ino(&metadata))
        .collect();
    let Ok(locks) = std::fs::read_to_string("/proc/locks") else {
        return false;
    };
    let holders = write_holders(&locks, &inodes, std::process::id());
    !holders.is_empty() && holders.iter().all(|pid| stopped(*pid))
}

fn shm(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push("-shm");
    PathBuf::from(name)
}

/// Other processes' POSIX write locks on these inodes, from `/proc/locks` lines such as
/// `3: POSIX  ADVISORY  WRITE 167671 103:02:204736108 1073741824 1073742335`.
fn write_holders(locks: &str, inodes: &[u64], own: u32) -> Vec<u32> {
    locks
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let at = fields.iter().position(|field| *field == "POSIX")?;
            if fields.get(at + 2) != Some(&"WRITE") {
                return None;
            }
            let pid: u32 = fields.get(at + 3)?.parse().ok()?;
            let inode: u64 = fields.get(at + 4)?.rsplit(':').next()?.parse().ok()?;
            (pid != own && inodes.contains(&inode)).then_some(pid)
        })
        .collect()
}

/// Stopped, traced-stopped, zombie or dead: a process that cannot release its lock.
fn stopped(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|stat| {
            let state = stat
                .rsplit_once(')')?
                .1
                .split_whitespace()
                .next()?
                .to_string();
            Some(matches!(state.as_str(), "T" | "t" | "Z" | "X"))
        })
        .unwrap_or(false)
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
";
        assert_eq!(write_holders(locks, &[77], 1000), vec![4242, 8181]);
        assert_eq!(
            write_holders(locks, &[77, 99], 1000),
            vec![4242, 6161, 8181]
        );
        assert!(write_holders(locks, &[5], 1000).is_empty());
    }
}
