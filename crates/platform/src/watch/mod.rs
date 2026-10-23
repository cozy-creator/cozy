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
