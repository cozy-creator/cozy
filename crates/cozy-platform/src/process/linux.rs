//! Linux: births from `/proc`, exact handles by pidfd (TensorD `process.rs`).
use super::{Birth, State, unix};
use std::{
    fs, io,
    os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd},
    time::Duration,
};

#[derive(Debug)]
pub(super) struct Handle(pub(super) OwnedFd);

pub(super) fn birth(pid: u32) -> io::Result<Birth> {
    Ok(Birth {
        pid,
        boot_id: boot_id()?,
        start_ticks: stat(pid)?.start_ticks,
    })
}

pub(super) fn state(birth: &Birth) -> io::Result<State> {
    const PF_EXITING: u64 = 0x4;
    if boot_id()? != birth.boot_id {
        return Ok(State::Gone);
    }
    let stat = match stat(birth.pid) {
        Ok(stat) => stat,
        Err(error) if gone(&error) => return Ok(State::Gone),
        Err(error) => return Err(error),
    };
    if stat.start_ticks != birth.start_ticks || matches!(stat.state, 'Z' | 'X') {
        return Ok(State::Gone);
    }
    if stat.flags & PF_EXITING != 0 || sigkill_pending(birth.pid)? {
        return Ok(State::Dying);
    }
    Ok(match stat.state {
        'T' | 't' => State::Stopped,
        _ => State::Live,
    })
}

pub(super) fn open(birth: &Birth) -> io::Result<Option<Handle>> {
    if boot_id()? != birth.boot_id {
        return Ok(None);
    }
    // SAFETY: pidfd_open takes a PID and flags, no memory.
    let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, birth.pid, 0) } as i32;
    if raw < 0 {
        let error = io::Error::last_os_error();
        return match error.raw_os_error() {
            // ESRCH: reaped. EINVAL: a leader already exiting (its threads tearing down).
            Some(libc::ESRCH | libc::EINVAL) => Ok(None),
            _ => Err(error),
        };
    }
    // SAFETY: pidfd_open returned a new descriptor that is now ours.
    let pidfd = unsafe { OwnedFd::from_raw_fd(raw) };
    // Read after opening: the pidfd now pins whichever task holds the PID.
    match stat(birth.pid) {
        Ok(stat) if stat.start_ticks == birth.start_ticks => Ok(Some(Handle(pidfd))),
        Ok(_) => Ok(None),
        Err(error) if gone(&error) => Ok(None),
        Err(error) => Err(error),
    }
}

pub(super) fn kill(handle: &Handle, birth: &Birth) -> io::Result<()> {
    // A group's number is not reused while a member lives, so a group this birth leads is its own.
    let leads_group = stat(birth.pid)
        .is_ok_and(|stat| stat.start_ticks == birth.start_ticks && stat.pgrp as u32 == birth.pid);
    // SAFETY: a live pidfd, a signal number and no siginfo.
    unix::signaled(
        unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                handle.0.as_raw_fd(),
                libc::SIGKILL,
                std::ptr::null::<libc::siginfo_t>(),
                0,
            )
        } as libc::c_int,
        birth.pid,
    )?;
    if leads_group {
        unix::signal_group(birth.pid)?;
    }
    Ok(())
}

pub(super) fn wait(handle: &Handle, period: Option<Duration>) -> io::Result<bool> {
    unix::readable(handle.0.as_fd(), period)
}

pub(super) fn set_name(name: &str) -> crate::Result<()> {
    let failed = crate::Error::os("name this process");
    let name = std::ffi::CString::new(name)
        .map_err(|error| failed(io::Error::new(io::ErrorKind::InvalidInput, error)))?;
    // SAFETY: a NUL-terminated name, of which the kernel copies at most 16 bytes.
    if unsafe { libc::prctl(libc::PR_SET_NAME, name.as_ptr()) } < 0 {
        return Err(crate::Error::os("name this process")(
            io::Error::last_os_error(),
        ));
    }
    Ok(())
}

/// Exited, a zombie included, or gone altogether.
pub(super) fn exited(pid: u32) -> bool {
    match stat(pid) {
        Ok(stat) => matches!(stat.state, 'Z' | 'X'),
        Err(error) => gone(&error),
    }
}

fn boot_id() -> io::Result<String> {
    Ok(fs::read_to_string("/proc/sys/kernel/random/boot_id")?
        .trim()
        .into())
}

/// `/proc/<pid>` vanished: ENOENT, or ESRCH when the process was reaped mid-read.
fn gone(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::NotFound || error.raw_os_error() == Some(libc::ESRCH)
}

struct Stat {
    state: char,
    pgrp: i32,
    /// The kernel's PF_* flags.
    flags: u64,
    start_ticks: u64,
}

fn stat(pid: u32) -> io::Result<Stat> {
    let record = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let (_, suffix) = record
        .rsplit_once(") ")
        .ok_or_else(|| io::Error::other("invalid process stat"))?;
    let fields: Vec<_> = suffix.split_whitespace().collect();
    let field = |index: usize| {
        fields
            .get(index)
            .copied()
            .ok_or_else(|| io::Error::other("process stat is truncated"))
    };
    Ok(Stat {
        state: field(0)?.chars().next().unwrap_or('?'),
        pgrp: field(2)?.parse().map_err(io::Error::other)?,
        flags: field(6)?.parse().map_err(io::Error::other)?,
        start_ticks: field(19)?.parse().map_err(io::Error::other)?,
    })
}

fn sigkill_pending(pid: u32) -> io::Result<bool> {
    let status = match fs::read_to_string(format!("/proc/{pid}/status")) {
        Ok(status) => status,
        Err(error) if gone(&error) => return Ok(true),
        Err(error) => return Err(error),
    };
    let kill = 1u64 << (libc::SIGKILL - 1);
    Ok(status
        .lines()
        .filter_map(|line| {
            line.strip_prefix("SigPnd:")
                .or_else(|| line.strip_prefix("ShdPnd:"))
        })
        .filter_map(|mask| u64::from_str_radix(mask.trim(), 16).ok())
        .any(|mask| mask & kill != 0))
}
