//! Processes named by their birth, never by a bare PID, and process trees ended whole.
//! Nothing here kills because time passed: callers decide from what they measured.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    process::{Child, ChildStdout, Command, ExitStatus},
    time::Duration,
};

#[cfg_attr(target_os = "linux", path = "linux.rs")]
#[cfg_attr(target_os = "macos", path = "macos.rs")]
#[cfg_attr(windows, path = "windows.rs")]
mod sys;
#[cfg(unix)]
mod unix;
#[cfg(windows)]
use sys as tree;
#[cfg(unix)]
use unix as tree;

/// One process incarnation: no other process ever has the same birth. The field names are
/// TensorD's journal `ProcessBirth`, so a machine reads its journal across an update.
/// `start_ticks` is the OS's start stamp: clock ticks since boot (Linux), microseconds since the
/// epoch (macOS) or a FILETIME (Windows). Only Linux's restarts at boot, so only Linux needs
/// `boot_id`; it is empty elsewhere.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Birth {
    pub pid: u32,
    pub boot_id: String,
    pub start_ticks: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// Exited (a zombie included) or never this birth.
    Gone,
    Live,
    /// Stopped or traced-stopped: it cannot release what it holds until continued.
    Stopped,
    /// Exiting or with SIGKILL pending: it still holds its files a moment longer.
    Dying,
}

impl Birth {
    pub fn of(pid: u32) -> Result<Birth> {
        sys::birth(pid).map_err(Error::os("read a process's birth"))
    }

    pub fn own() -> Result<Birth> {
        Self::of(std::process::id())
    }

    pub fn state(&self) -> Result<State> {
        sys::state(self).map_err(Error::os("read a process's state"))
    }
}

/// A handle naming exactly one birth (a pidfd, a kqueue on its exit, a process HANDLE): its kill
/// and its exit never reach a process that reused the PID.
#[derive(Debug)]
pub struct Exact {
    birth: Birth,
    handle: sys::Handle,
}

impl Exact {
    /// `None` when that birth has been reaped or its PID names another process now.
    pub fn open(birth: &Birth) -> Result<Option<Exact>> {
        let handle = sys::open(birth).map_err(Error::os("open a process handle"))?;
        Ok(handle.map(|handle| Exact {
            birth: birth.clone(),
            handle,
        }))
    }

    pub fn birth(&self) -> &Birth {
        &self.birth
    }

    /// SIGKILL this birth and the process group it still leads (TerminateProcess on Windows).
    pub fn kill(&self) -> Result<()> {
        sys::kill(&self.handle, &self.birth).map_err(Error::os("kill a process"))
    }

    pub fn ended(&self) -> bool {
        matches!(sys::wait(&self.handle, Some(Duration::ZERO)), Ok(true))
    }

    /// Blocks until this process has exited. No deadline: exit is the only answer.
    pub fn wait(&self) -> Result<()> {
        sys::wait(&self.handle, None)
            .map(drop)
            .map_err(Error::os("wait for a process"))
    }

    /// Whether it exited within `sample`: a sampling period, never a verdict.
    pub fn wait_for(&self, sample: Duration) -> Result<bool> {
        sys::wait(&self.handle, Some(sample)).map_err(Error::os("wait for a process"))
    }
}

#[cfg(unix)]
impl std::os::fd::AsFd for Exact {
    /// Readable once the process has exited, for a caller's own poll set.
    fn as_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        std::os::fd::AsFd::as_fd(&self.handle.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifetime {
    /// The spawner's: dropping the `Tree` ends it, and on Windows so does the spawner's death.
    Owned,
    /// Outlives the spawner: its own session, no console, out of the spawner's job when allowed.
    Detached,
}

/// A process and everything it starts: its own process group (Unix) or Job Object (Windows,
/// created suspended and resumed only once the job holds it, so nothing ever runs outside it).
/// The seam sets the command's process group and, on Windows, its creation flags.
#[derive(Debug)]
pub struct Tree {
    child: Child,
    group: tree::Group,
    lifetime: Lifetime,
    reaped: bool,
}

pub fn spawn(mut command: Command, lifetime: Lifetime) -> Result<Tree> {
    let (child, group) = tree::spawn(&mut command, lifetime).map_err(Error::os("spawn"))?;
    Ok(Tree {
        child,
        group,
        lifetime,
        reaped: false,
    })
}

impl Tree {
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn stdout(&mut self) -> Option<ChildStdout> {
        self.child.stdout.take()
    }

    /// Ends every process in the tree. On Unix the group is reached while its leader is unreaped
    /// (its zombie keeps the group's number); after `wait`, members that outlived the leader are
    /// for a containment scope to end.
    pub fn kill(&self) -> Result<()> {
        if cfg!(unix) && self.reaped {
            return Ok(());
        }
        tree::kill_group(&self.group, &self.child).map_err(Error::os("kill a process tree"))
    }

    /// The leader's exit.
    pub fn wait(&mut self) -> Result<ExitStatus> {
        let status = self.child.wait().map_err(Error::os("wait for a process"))?;
        self.reaped = true;
        Ok(status)
    }

    pub fn try_wait(&mut self) -> Result<Option<ExitStatus>> {
        let status = self
            .child
            .try_wait()
            .map_err(Error::os("wait for a process"))?;
        self.reaped |= status.is_some();
        Ok(status)
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        if self.lifetime == Lifetime::Owned {
            let _ = self.kill();
        }
    }
}
