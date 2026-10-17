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

/// Marks every descriptor from `first` up close-on-exec in this process. In a child about to exec
/// it leaves the new program only what its spawner hands on afterwards; at a process's start it
/// keeps whatever it inherited from anything it starts. Async-signal-safe: `close_range` (Linux
/// 5.11), else each descriptor to the soft limit.
pub fn cloexec_from(first: u32) {
    // SAFETY: plain syscalls on this process's own descriptor table.
    unsafe {
        #[cfg(target_os = "linux")]
        if libc::syscall(
            libc::SYS_close_range,
            first,
            u32::MAX,
            libc::CLOSE_RANGE_CLOEXEC,
        ) == 0
        {
            return;
        }
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        let last = if libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) == 0 {
            limit.rlim_cur.min(1 << 20) as u32
        } else {
            1 << 16
        };
        for fd in first..last {
            libc::fcntl(fd as i32, libc::F_SETFD, libc::FD_CLOEXEC);
        }
    }
}

/// A child inherits no descriptor its spawner does not hand on explicitly: run before every other
/// `pre_exec` of `command`, which then clear close-on-exec on what they hand on.
pub fn inherit_nothing(command: &mut Command) {
    // SAFETY: cloexec_from is async-signal-safe.
    unsafe {
        command.pre_exec(|| {
            cloexec_from(3);
            Ok(())
        });
    }
}

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

pub(super) fn raise_fd_limit() {
    // SAFETY: plain-old-data in and out.
    let mut limit: libc::rlimit = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0 {
        return;
    }
    // macOS refuses a soft limit above OPEN_MAX (10240) even under an unlimited hard limit.
    let ceiling = if cfg!(target_vendor = "apple") {
        limit.rlim_max.min(10240)
    } else {
        limit.rlim_max
    };
    if limit.rlim_cur < ceiling {
        limit.rlim_cur = ceiling;
        // SAFETY: as above.
        unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) };
    }
}

pub(super) fn fd_limit() -> Option<u64> {
    // SAFETY: plain-old-data out-param.
    let mut limit: libc::rlimit = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0
        || limit.rlim_cur == libc::RLIM_INFINITY
    {
        return None;
    }
    #[allow(clippy::unnecessary_cast)]
    Some(limit.rlim_cur as u64)
}
