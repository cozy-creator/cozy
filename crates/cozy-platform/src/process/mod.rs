//! Processes named by their birth, never by a bare PID, and process trees ended whole.
//! Nothing here kills because time passed: callers decide from what they measured.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    io,
    process::{Child, ChildStdout, Command, ExitStatus},
    sync::Arc,
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

    /// Exited (a zombie included) or never this birth: `state() == Gone`.
    pub fn ended(&self) -> Result<bool> {
        Ok(self.state()? == State::Gone)
    }
}

/// What a process is doing: CPU it has used (monotonic, in the OS's own unit) and whether a
/// thread of it is running or waiting on a disk now (Linux `R`/`D`, macOS running threads;
/// Windows cannot say, so false).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usage {
    pub cpu: u64,
    pub running: bool,
}

pub fn usage(pid: u32) -> Result<Usage> {
    sys::usage(pid).map_err(Error::os("read a process's usage"))
}

/// This kernel's boot. Page cache and process starts belong to a boot.
pub fn boot_id() -> Result<String> {
    sys::boot_id().map_err(Error::os("read the boot id"))
}

/// This process's peak resident memory, in bytes.
pub fn peak_resident() -> Result<u64> {
    sys::peak_resident().map_err(Error::os("read peak resident memory"))
}

/// Raises the soft descriptor limit as far as the OS lets a process; once is enough.
pub fn raise_fd_limit() {
    #[cfg(unix)]
    unix::raise_fd_limit();
}

/// The soft descriptor limit, or `None` where the table has no practical limit (Windows'
/// handle table) or the limit is infinite.
pub fn fd_limit() -> Option<u64> {
    #[cfg(unix)]
    return unix::fd_limit();
    #[cfg(windows)]
    None
}

/// Descriptors this process has open now.
pub fn open_fds() -> Result<u64> {
    let dir = if cfg!(target_os = "linux") {
        "/proc/self/fd"
    } else if cfg!(target_vendor = "apple") {
        "/dev/fd"
    } else {
        return Err(Error::unsupported(crate::Seam::Process));
    };
    Ok(std::fs::read_dir(dir)
        .map_err(Error::io("list descriptors", dir))?
        .count() as u64)
}

/// The process's or the system's descriptor table is full.
pub fn fd_table_full(error: &io::Error) -> bool {
    #[cfg(unix)]
    let full = [libc::EMFILE, libc::ENFILE];
    // ERROR_TOO_MANY_OPEN_FILES, WSAEMFILE.
    #[cfg(windows)]
    let full = [4, 10024];
    error
        .raw_os_error()
        .is_some_and(|code| full.contains(&code))
}

/// Names this process for `ps` and `top` by its role (`calcifer`, `tensord`). It names the calling
/// thread, so call it from the main thread before others start; Linux keeps 15 bytes. macOS and
/// Windows can rename only threads, not a process: `Unsupported`.
pub fn set_name(name: &str) -> Result<()> {
    sys::set_name(name)
}

/// A handle naming exactly one birth (a pidfd, a kqueue on its exit, a process HANDLE): its kill
/// and its exit never reach a process that reused the PID.
#[derive(Debug)]
pub struct Exact {
    birth: Birth,
    handle: sys::Handle,
    /// The process's own containment scope: a kill reaches every descendant.
    scope: Option<Arc<crate::contain::Scope>>,
}

impl Exact {
    /// `None` when that birth has been reaped (on Windows: has exited) or its PID names another
    /// process now.
    pub fn open(birth: &Birth) -> Result<Option<Exact>> {
        let handle = sys::open(birth).map_err(Error::os("open a process handle"))?;
        Ok(handle.map(|handle| Exact {
            birth: birth.clone(),
            handle,
            scope: None,
        }))
    }

    pub fn birth(&self) -> &Birth {
        &self.birth
    }

    pub fn scope(&self) -> Option<&Arc<crate::contain::Scope>> {
        self.scope.as_ref()
    }

    pub fn with_scope(mut self, scope: Option<Arc<crate::contain::Scope>>) -> Self {
        self.scope = scope;
        self
    }

    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            birth: self.birth.clone(),
            handle: sys::Handle(
                self.handle
                    .0
                    .try_clone()
                    .map_err(Error::os("clone a process handle"))?,
            ),
            scope: self.scope.clone(),
        })
    }

    /// The handle as a file a caller keeps: readable (Unix) or signaled (Windows) once it exits.
    #[cfg(unix)]
    pub fn into_file(self) -> std::fs::File {
        self.handle.0.into()
    }

    /// SIGKILL this birth, its scope when it has one, and the process group it still leads
    /// (TerminateProcess on Windows).
    pub fn kill(&self) -> Result<()> {
        if let Some(scope) = &self.scope {
            scope.kill()?;
        }
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

/// Every process in `leader`'s process group born with or after it: a leader's followers. A
/// group number is not reused while a member lives. Empty where it cannot be read (only Linux
/// reads groups so far).
pub fn group_members(leader: &Birth) -> Vec<Birth> {
    sys::group_members(leader)
}

/// CPU nanoseconds used or owed (time runnable but waiting for a CPU) and bytes moved, the
/// measure of a process's progress. `None`: unreadable, which decides nothing (Linux only so far).
pub fn meter(pid: u32) -> Option<(u64, u64)> {
    sys::meter(pid)
}

/// `meter`'s two parts summed: Runtime's `proctree.progress_burn`.
pub fn burn(pid: u32) -> Option<u64> {
    meter(pid).map(|(cpu, moved)| cpu.saturating_add(moved))
}

/// The bytes a process has read. `None`: unreadable.
pub fn read_bytes(pid: u32) -> Option<u64> {
    sys::read_bytes(pid)
}

/// A process this error was about is gone: its record vanished, or it was reaped mid-read.
pub fn gone(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::NotFound || sys::gone(error)
}

/// Orphans of this process's descendants reparent to it, which must then reap them (a pod's
/// stable parent, PID 1 or not). Linux only.
pub fn become_subreaper() -> Result<()> {
    sys::become_subreaper()
}

/// In a child just forked from `parent`: SIGKILL arrives when the parent dies. `false` when the
/// parent was already gone before this call armed it (the child then has no parent to serve).
/// Linux only.
pub fn die_with_parent(parent: u32) -> Result<bool> {
    sys::die_with_parent(parent)
}

/// An OS shortage that may clear by itself; nothing else justifies requeueing a launch.
pub fn transient(error: &io::Error) -> bool {
    sys::transient(error)
}

#[cfg(unix)]
pub use unix::{cloexec_from, inherit_nothing};

/// Whether a process handle kept as a file (a pidfd, a kqueue on its exit) shows its exit now.
#[cfg(unix)]
pub fn ended(handle: &impl std::os::fd::AsFd) -> bool {
    matches!(
        unix::readable(handle.as_fd(), Some(Duration::ZERO)),
        Ok(true)
    )
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
