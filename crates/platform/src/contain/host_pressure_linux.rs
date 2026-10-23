//! The Linux trigger: PSI on the system or this cgroup, inotify on each limited cgroup v2's
//! `memory.events`, and an eventfd TensorD's own events write.
#![allow(unsafe_code)]
use super::{HostMode, WINDOW};
use std::{
    ffi::CString,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::{ffi::OsStrExt, fs::OpenOptionsExt},
    },
    path::{Path, PathBuf},
    sync::Arc,
};

/// What wakes the watcher: a PSI trigger, if this process may arm one, inotify on each limited
/// cgroup v2's `memory.events` it may watch, and its [`Nudge`].
pub struct Pressure {
    psi: Option<File>,
    events: Option<OwnedFd>,
    nudge: Nudge,
    how: String,
}

/// TensorD's own events wake the watcher for a look: the only wakeups where the kernel allows
/// none, and ones between the kernel's (cgroup v1 is read on every look, as v2 is).
#[derive(Clone)]
pub struct Nudge(Arc<OwnedFd>);

impl Nudge {
    pub fn send(&self) {
        // SAFETY: an eventfd write of one u64 from our own stack; a full counter only coalesces.
        unsafe { libc::write(self.0.as_raw_fd(), (&1u64 as *const u64).cast(), 8) };
    }
}

impl Pressure {
    /// Armed on whatever this process may watch, each limit on its own; `how` says what and why
    /// not. An error only when not even an eventfd exists.
    pub fn arm(mode: HostMode) -> crate::Result<Self> {
        // SAFETY: plain flags; the descriptor is owned below.
        let raw = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        if raw < 0 {
            return Err(io::Error::last_os_error().into());
        }
        // SAFETY: a descriptor eventfd just returned and nothing else owns.
        let nudge = Nudge(Arc::new(unsafe { OwnedFd::from_raw_fd(raw) }));
        let cgroups = super::super::host_memory::cgroups();
        let (path, kind) = match (mode, own_cgroup().filter(|c| c.join("memory.pressure").exists())) {
            (HostMode::Dedicated, Some(cgroup)) => (cgroup.join("memory.pressure"), "full"),
            // System `full` can stay quiet while one container stalls on a busy host.
            _ => (PathBuf::from("/proc/pressure/memory"), "some"),
        };
        let limited: Vec<PathBuf> = match cgroups {
            Some((groups, limits, "memory.current")) => groups
                .into_iter()
                .filter(|g| limits.iter().any(|l| super::super::host_memory::number(&g.join(l)).is_some()))
                .map(|g| g.join("memory.events"))
                .collect(),
            _ => vec![],
        };
        let mut how = vec![];
        let psi = trigger(&path, kind)
            .inspect(|_| how.push(format!("psi {} {kind}", path.display())))
            .inspect_err(|error| how.push(format!("no psi trigger on {} ({error})", path.display())))
            .ok();
        let events = watch(&limited, &mut how);
        how.push("tensord events".into());
        Ok(Self { psi, events, nudge, how: how.join("; ") })
    }

    /// What it is armed on, for the record.
    pub fn how(&self) -> &str {
        &self.how
    }

    pub fn nudge(&self) -> Nudge {
        self.nudge.clone()
    }

    /// Blocks until the kernel reports a stall, a watched `memory.events` changes, or a nudge
    /// comes: no timeout, no work in between.
    pub fn wait(&mut self) -> crate::Result<()> {
        let fd = |file: Option<i32>, events| libc::pollfd { fd: file.unwrap_or(-1), events, revents: 0 };
        let mut fds = [
            fd(self.psi.as_ref().map(AsRawFd::as_raw_fd), libc::POLLPRI),
            fd(self.events.as_ref().map(AsRawFd::as_raw_fd), libc::POLLIN),
            fd(Some(self.nudge.0.as_raw_fd()), libc::POLLIN),
        ];
        loop {
            // SAFETY: live descriptors (-1 is ignored by poll).
            if unsafe { libc::poll(fds.as_mut_ptr(), 3, -1) } > 0 {
                if fds[0].revents & libc::POLLERR != 0 {
                    return Err(io::Error::other("the PSI trigger is gone").into());
                }
                let mut buf = [0u8; 4096];
                for drained in [self.events.as_ref(), Some(&*self.nudge.0)].into_iter().flatten() {
                    // SAFETY: reads into our own buffer from a nonblocking descriptor, until empty.
                    while unsafe { libc::read(drained.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len()) } > 0 {}
                }
                return Ok(());
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error.into());
            }
        }
    }
}

/// A PSI trigger on one pressure file (`memory.pressure` of a cgroup, or the system's).
fn trigger(path: &Path, kind: &str) -> io::Result<File> {
    let mut trigger = OpenOptions::new().read(true).write(true).custom_flags(libc::O_NONBLOCK).open(path)?;
    trigger.write_all(format!("{kind} 1 {}\0", WINDOW.as_micros()).as_bytes())?;
    Ok(trigger)
}

/// One inotify descriptor watching each file it can for IN_MODIFY, each noted in `how`; None
/// when it watches none.
fn watch(files: &[PathBuf], how: &mut Vec<String>) -> Option<OwnedFd> {
    if files.is_empty() {
        how.push("no limited cgroup v2".into());
        return None;
    }
    // SAFETY: plain flags; the descriptor is owned below.
    let raw = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
    if raw < 0 {
        how.push(format!("no inotify ({})", io::Error::last_os_error()));
        return None;
    }
    // SAFETY: a descriptor inotify_init1 just returned and nothing else owns.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    let mut watched = false;
    for file in files {
        let added = CString::new(file.as_os_str().as_bytes()).map_err(io::Error::other).and_then(|path| {
            // SAFETY: a live inotify descriptor and a NUL-terminated path.
            match unsafe { libc::inotify_add_watch(fd.as_raw_fd(), path.as_ptr(), libc::IN_MODIFY) } {
                ..0 => Err(io::Error::last_os_error()),
                _ => Ok(()),
            }
        });
        match added {
            Ok(()) => {
                how.push(format!("inotify {}", file.display()));
                watched = true;
            }
            Err(error) => how.push(format!("no inotify on {} ({error})", file.display())),
        }
    }
    watched.then_some(fd)
}

/// This process's cgroup v2 directory.
fn own_cgroup() -> Option<PathBuf> {
    let text = fs::read_to_string("/proc/self/cgroup").ok()?;
    let relative = text.lines().find_map(|line| line.strip_prefix("0::"))?;
    Some(PathBuf::from("/sys/fs/cgroup").join(relative.trim_start_matches('/')))
}
