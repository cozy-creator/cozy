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

pub(super) fn parent_of(pid: u32) -> io::Result<u32> {
    Ok(stat(pid)?.ppid)
}

/// Every process with its parent.
pub(super) fn parents() -> Vec<(u32, u32)> {
    let Ok(entries) = fs::read_dir("/proc") else {
        return vec![];
    };
    entries
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
        .filter_map(|pid| Some((pid, stat(pid).ok()?.ppid)))
        .collect()
}

pub(super) fn group_members(leader: &Birth) -> Vec<Birth> {
    let Ok(boot) = boot_id() else {
        return vec![];
    };
    if boot != leader.boot_id {
        return vec![];
    }
    let Ok(entries) = fs::read_dir("/proc") else {
        return vec![];
    };
    entries
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
        .filter(|pid| *pid != leader.pid)
        .filter_map(|pid| {
            let stat = stat(pid).ok()?;
            (stat.pgrp as u32 == leader.pid && stat.start_ticks >= leader.start_ticks).then(|| {
                Birth {
                    pid,
                    boot_id: boot.clone(),
                    start_ticks: stat.start_ticks,
                }
            })
        })
        .collect()
}

/// Owed is the time its live threads waited runnable for a CPU (`schedstat` run delay): busy
/// cores or a container's spent CPU quota slow a working process, never still it.
pub(super) fn meter(pid: u32) -> Option<(u64, u64)> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let fields: Vec<_> = stat.rsplit_once(") ")?.1.split_whitespace().collect();
    let ticks = fields.get(11)?.parse::<u64>().ok()? + fields.get(12)?.parse::<u64>().ok()?;
    // SAFETY: sysconf takes a scalar key.
    let hertz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as u64;
    let waited: u64 = fs::read_dir(format!("/proc/{pid}/task"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|task| fs::read_to_string(task.path().join("schedstat")).ok())
        .filter_map(|line| line.split_whitespace().nth(1)?.parse::<u64>().ok())
        .sum();
    let cpu = ticks
        .saturating_mul(1_000_000_000 / hertz)
        .saturating_add(waited);
    let moved = fs::read_to_string(format!("/proc/{pid}/io"))
        .ok()
        .and_then(|io| {
            io.lines()
                .filter_map(|line| line.split_once(": "))
                .filter(|(name, _)| {
                    matches!(*name, "rchar" | "wchar" | "read_bytes" | "write_bytes")
                })
                .map(|(_, value)| value.trim().parse::<u64>().ok())
                .sum::<Option<u64>>()
        });
    Some((cpu, moved.unwrap_or(0)))
}

/// `rchar`.
pub(super) fn read_bytes(pid: u32) -> Option<u64> {
    fs::read_to_string(format!("/proc/{pid}/io"))
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("rchar: "))?
        .trim()
        .parse()
        .ok()
}

pub(super) fn become_subreaper() -> crate::Result<()> {
    // SAFETY: prctl with scalar arguments.
    if unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1) } < 0 {
        return Err(crate::Error::os("become a subreaper")(
            io::Error::last_os_error(),
        ));
    }
    Ok(())
}

pub(super) fn die_with_parent(parent: u32) -> crate::Result<bool> {
    // SAFETY: prctl with scalar arguments; async-signal-safe, as is getppid.
    if unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) } < 0 {
        return Err(crate::Error::os("die with the parent")(
            io::Error::last_os_error(),
        ));
    }
    // SAFETY: getppid has no arguments.
    Ok(unsafe { libc::getppid() } as u32 == parent)
}

pub(super) fn transient(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EAGAIN | libc::ENOMEM | libc::EMFILE | libc::ENFILE | libc::EINTR)
    )
}

/// Exited, a zombie included, or gone altogether.
pub(super) fn exited(pid: u32) -> bool {
    match stat(pid) {
        Ok(stat) => matches!(stat.state, 'Z' | 'X'),
        Err(error) => gone(&error),
    }
}

pub(super) fn usage(pid: u32) -> io::Result<super::Usage> {
    let stat = stat(pid)?;
    Ok(super::Usage {
        cpu: stat.cpu,
        running: matches!(stat.state, 'R' | 'D'),
    })
}

/// `VmHWM`, this image's own peak: `ru_maxrss` keeps the peak of the image this process exec'd
/// from.
pub(super) fn peak_resident() -> io::Result<u64> {
    let status = fs::read_to_string("/proc/self/status")?;
    let kib = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|value| {
            value
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<u64>()
                .ok()
        })
        .ok_or_else(|| io::Error::other("no VmHWM in /proc/self/status"))?;
    Ok(kib * 1024)
}

pub(super) fn boot_id() -> io::Result<String> {
    Ok(fs::read_to_string("/proc/sys/kernel/random/boot_id")?
        .trim()
        .into())
}

/// `/proc/<pid>` vanished: ENOENT, or ESRCH when the process was reaped mid-read.
pub(super) fn gone(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::NotFound || error.raw_os_error() == Some(libc::ESRCH)
}

struct Stat {
    state: char,
    ppid: u32,
    pgrp: i32,
    /// The kernel's PF_* flags.
    flags: u64,
    /// User plus system clock ticks.
    cpu: u64,
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
        ppid: field(1)?.parse().map_err(io::Error::other)?,
        pgrp: field(2)?.parse().map_err(io::Error::other)?,
        flags: field(6)?.parse().map_err(io::Error::other)?,
        cpu: field(11)?.parse::<u64>().map_err(io::Error::other)?
            + field(12)?.parse::<u64>().map_err(io::Error::other)?,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    #[test]
    fn exact_kill_takes_the_group_and_never_a_reused_birth() {
        let mut child = Command::new("/bin/sh")
            .args(["-c", "sleep 1000 & wait"])
            .process_group(0)
            .stdin(Stdio::null())
            .spawn()
            .unwrap();
        let birth = Birth::of(child.id()).unwrap();
        let mut stale = birth.clone();
        stale.start_ticks += 1;
        assert!(crate::process::Exact::open(&stale).unwrap().is_none());
        let exact = crate::process::Exact::open(&birth).unwrap().unwrap();
        // The grandchild is in the leader's group: wait until it exists.
        let grandchild = loop {
            let children =
                fs::read_to_string(format!("/proc/{0}/task/{0}/children", child.id())).unwrap();
            if let Some(pid) = children.split_whitespace().next() {
                break pid.parse::<u32>().unwrap();
            }
            std::thread::yield_now();
        };
        let grandchild = crate::process::Exact::open(&Birth::of(grandchild).unwrap())
            .unwrap()
            .unwrap();
        exact.kill().unwrap();
        exact.wait().unwrap();
        grandchild.wait().unwrap();
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(child.wait().unwrap().signal(), Some(libc::SIGKILL));
        assert!((birth.state().unwrap() == State::Gone));
    }

    #[test]
    fn burn_reads_this_process() {
        assert!(crate::process::burn(std::process::id()).is_some_and(|value| value > 0));
        assert!(crate::process::burn(u32::MAX).is_none());
    }
}
