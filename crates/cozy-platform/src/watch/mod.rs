//! Waking when a directory's files change: inotify on Linux, kqueue on macOS, change
//! notifications on Windows. A wake says only that something changed; the caller rereads.
//!
//! Every OS wakes on a file created, renamed in or out, removed, closed after a write, or written
//! and synced. Linux and macOS also wake on a plain write; Windows reports one only once the
//! cache flushes (measured: not within 10 s), so a follower there rereads on `most` too.
use crate::Result;
use std::{path::Path, time::Duration};

#[cfg_attr(target_os = "linux", path = "linux.rs")]
#[cfg_attr(target_os = "macos", path = "macos.rs")]
#[cfg_attr(windows, path = "windows.rs")]
mod imp;

/// A directory whose files' changes (written, created, renamed in, removed) wake [`DirWatch::wait`].
pub struct DirWatch(imp::Watch);

impl DirWatch {
    pub fn new(dir: &Path) -> Result<Self> {
        imp::Watch::new(dir).map(Self)
    }

    /// Waits until a file changes or `most` passes; true when one changed since the last wait.
    pub fn wait(&mut self, most: Duration) -> Result<bool> {
        self.0.wait(most)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, io::Write};

    /// Long enough for any OS to report a change; a missed wake fails, a prompt one costs nothing.
    const WAKE: Duration = Duration::from_secs(10);
    const QUIET: Duration = Duration::from_millis(50);

    fn watched() -> (tempfile::TempDir, DirWatch) {
        let dir = tempfile::tempdir().unwrap();
        let watch = DirWatch::new(dir.path()).unwrap();
        (dir, watch)
    }

    #[test]
    fn a_write_wakes_the_watch_and_quiet_lets_it_time_out() {
        let (dir, mut watch) = watched();
        assert!(!watch.wait(QUIET).unwrap());
        fs::write(dir.path().join("a.log"), b"one\n").unwrap();
        assert!(watch.wait(WAKE).unwrap());
        assert!(!watch.wait(QUIET).unwrap(), "the queue was drained");
    }

    #[test]
    fn a_synced_append_to_a_file_held_open_wakes_the_watch() {
        let (dir, mut watch) = watched();
        let mut log = fs::File::create(dir.path().join("a.log")).unwrap();
        assert!(watch.wait(WAKE).unwrap(), "the create");
        while watch.wait(QUIET).unwrap() {}
        for line in 0..3 {
            writeln!(log, "line {line}").unwrap();
            log.sync_data().unwrap();
            assert!(watch.wait(WAKE).unwrap(), "append {line}");
            while watch.wait(QUIET).unwrap() {}
        }
    }

    #[test]
    fn a_rename_in_a_remove_and_a_rotated_in_file_each_wake_the_watch() {
        let (dir, mut watch) = watched();
        let outside = tempfile::tempdir().unwrap();
        let (log, rotated) = (dir.path().join("a.log"), dir.path().join("a.log.1"));
        fs::write(outside.path().join("b.log"), b"b\n").unwrap();
        fs::rename(outside.path().join("b.log"), dir.path().join("b.log")).unwrap();
        assert!(watch.wait(WAKE).unwrap(), "a rename in");
        while watch.wait(QUIET).unwrap() {}
        fs::remove_file(dir.path().join("b.log")).unwrap();
        assert!(watch.wait(WAKE).unwrap(), "a remove");
        while watch.wait(QUIET).unwrap() {}
        // Rotation: the live file is renamed aside and a new one takes its name.
        fs::write(&log, b"old\n").unwrap();
        assert!(watch.wait(WAKE).unwrap());
        while watch.wait(QUIET).unwrap() {}
        fs::rename(&log, &rotated).unwrap();
        let mut fresh = fs::File::create(&log).unwrap();
        assert!(watch.wait(WAKE).unwrap(), "the rotation");
        while watch.wait(QUIET).unwrap() {}
        writeln!(fresh, "new").unwrap();
        fresh.sync_data().unwrap();
        assert!(watch.wait(WAKE).unwrap(), "a write to the rotated-in file");
        // And the file rotated aside is no longer this directory's live log, but still its file.
        while watch.wait(QUIET).unwrap() {}
        fs::OpenOptions::new()
            .append(true)
            .open(&rotated)
            .unwrap()
            .write_all(b"late\n")
            .unwrap();
        assert!(watch.wait(WAKE).unwrap(), "a write to the rotated file");
    }
}
