//! What Linux and macOS share: process groups, sessions, signals and readiness.
use super::Lifetime;
use std::{
    io,
    os::{
        fd::{AsRawFd, BorrowedFd},
        unix::process::CommandExt,
    },
    process::{Child, Command},
    time::Duration,
};

/// The group is the leader's PID: nothing to hold.
#[derive(Debug)]
pub(super) struct Group;

pub(super) fn spawn(command: &mut Command, lifetime: Lifetime) -> io::Result<(Child, Group)> {
    match lifetime {
        Lifetime::Owned => {
            command.process_group(0);
        }
        // A session of its own: no terminal's hangup reaches it, and it leads its own group.
        // SAFETY: setsid is async-signal-safe and touches no memory.
        Lifetime::Detached => unsafe {
            command.pre_exec(|| match libc::setsid() {
                -1 => Err(io::Error::last_os_error()),
                _ => Ok(()),
            });
        },
    }
    Ok((command.spawn()?, Group))
}

pub(super) fn kill_group(_: &Group, leader: &Child) -> io::Result<()> {
    signal_group(leader.id())
}

pub(super) fn signal_group(group: u32) -> io::Result<()> {
    // SAFETY: plain syscall on a scalar.
    signaled(
        unsafe { libc::killpg(group as libc::pid_t, libc::SIGKILL) },
        group,
    )
}

/// A signal's result, where a target already gone is done. macOS answers EPERM, not ESRCH, for
/// a process (or a group led by one) that has exited and is not yet reaped.
pub(super) fn signaled(result: libc::c_int, pid: u32) -> io::Result<()> {
    if result >= 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(()),
        Some(libc::EPERM) if super::sys::exited(pid) => Ok(()),
        _ => Err(error),
    }
}

/// Whether `fd` turned readable within `period` (`None`: wait for it).
pub(super) fn readable(fd: BorrowedFd<'_>, period: Option<Duration>) -> io::Result<bool> {
    let milliseconds = match period {
        None => -1,
        Some(period) if period.is_zero() => 0,
        Some(period) => period.as_millis().clamp(1, i32::MAX as u128) as i32,
    };
    let mut poll = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    loop {
        // SAFETY: one live descriptor and its pollfd.
        let ready = unsafe { libc::poll(&mut poll, 1, milliseconds) };
        if ready < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        if ready > 0 && poll.revents & libc::POLLIN == 0 {
            return Err(io::Error::other("process exit is not observable"));
        }
        return Ok(ready > 0);
    }
}
