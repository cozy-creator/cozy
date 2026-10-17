//! macOS: births from `proc_pidinfo`, exact handles as a kqueue watching the process's exit.
use super::{Birth, State, unix};
use std::{
    io, mem,
    os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd},
    ptr,
    time::Duration,
};

/// `proc_bsdinfo.pbi_flags`: the process is exiting.
const PROC_FLAG_INEXIT: u32 = 4;

#[derive(Debug)]
pub(super) struct Handle(pub(super) OwnedFd);

pub(super) fn birth(pid: u32) -> io::Result<Birth> {
    Ok(Birth {
        pid,
        boot_id: String::new(),
        start_ticks: start(&info(pid)?),
    })
}

pub(super) fn state(birth: &Birth) -> io::Result<State> {
    let info = match info(birth.pid) {
        Ok(info) => info,
        Err(error) if gone(&error) => return Ok(State::Gone),
        Err(error) => return Err(error),
    };
    Ok(
        if start(&info) != birth.start_ticks || info.pbi_status == libc::SZOMB {
            State::Gone
        } else if info.pbi_flags & PROC_FLAG_INEXIT != 0 {
            State::Dying
        } else if info.pbi_status == libc::SSTOP {
            State::Stopped
        } else {
            State::Live
        },
    )
}

pub(super) fn open(birth: &Birth) -> io::Result<Option<Handle>> {
    // SAFETY: kqueue takes nothing and returns a new descriptor or -1.
    let raw = unsafe { libc::kqueue() };
    if raw < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a new descriptor that is now ours.
    let queue = unsafe { OwnedFd::from_raw_fd(raw) };
    let exit = libc::kevent {
        ident: birth.pid as usize,
        filter: libc::EVFILT_PROC,
        flags: libc::EV_ADD | libc::EV_ONESHOT,
        fflags: libc::NOTE_EXIT,
        data: 0,
        udata: ptr::null_mut(),
    };
    // SAFETY: one change, no event buffer, no timeout.
    if unsafe { libc::kevent(queue.as_raw_fd(), &exit, 1, ptr::null_mut(), 0, ptr::null()) } < 0 {
        let error = io::Error::last_os_error();
        return match error.raw_os_error() {
            Some(libc::ESRCH) => Ok(None),
            _ => Err(error),
        };
    }
    // Read after registering: the queue now watches whichever process holds the PID.
    match info(birth.pid) {
        Ok(info) if start(&info) == birth.start_ticks => Ok(Some(Handle(queue))),
        Ok(_) => Ok(None),
        Err(error) if gone(&error) => Ok(None),
        Err(error) => Err(error),
    }
}

pub(super) fn kill(_: &Handle, birth: &Birth) -> io::Result<()> {
    // No pidfd_send_signal here: the birth is re-read, then the PID signalled. A reuse of the PID
    // inside that window is accepted.
    let info = match info(birth.pid) {
        Ok(info) if start(&info) == birth.start_ticks => info,
        Ok(_) => return Ok(()),
        Err(error) if gone(&error) => return Ok(()),
        Err(error) => return Err(error),
    };
    // SAFETY: plain syscall on scalars.
    unix::signaled(
        unsafe { libc::kill(birth.pid as libc::pid_t, libc::SIGKILL) },
        birth.pid,
    )?;
    if info.pbi_pgid == birth.pid {
        unix::signal_group(birth.pid)?;
    }
    Ok(())
}

pub(super) fn set_name(_: &str) -> crate::Result<()> {
    Err(crate::Error::unsupported(crate::Seam::Process))
}

pub(super) fn wait(handle: &Handle, period: Option<Duration>) -> io::Result<bool> {
    // The queue stays readable once the exit event is pending; it is never drained.
    unix::readable(handle.0.as_fd(), period)
}

pub(super) fn usage(pid: u32) -> io::Result<super::Usage> {
    // SAFETY: plain-old-data the call fills.
    let mut task: libc::proc_taskinfo = unsafe { mem::zeroed() };
    let size = mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
    // SAFETY: a writable buffer of exactly `size` bytes.
    let filled = unsafe {
        libc::proc_pidinfo(
            pid as libc::c_int,
            libc::PROC_PIDTASKINFO,
            0,
            (&mut task as *mut libc::proc_taskinfo).cast(),
            size,
        )
    };
    if filled != size {
        return Err(io::Error::last_os_error());
    }
    Ok(super::Usage {
        cpu: task.pti_total_user + task.pti_total_system,
        running: task.pti_numrunning > 0,
    })
}

/// `kern.bootsessionuuid`: a UUID the kernel draws at each boot.
pub(super) fn boot_id() -> io::Result<String> {
    let mut uuid = [0u8; 64];
    let mut len = uuid.len();
    // SAFETY: a NUL-terminated name and a writable buffer of `len` bytes.
    if unsafe {
        libc::sysctlbyname(
            c"kern.bootsessionuuid".as_ptr(),
            uuid.as_mut_ptr().cast(),
            &mut len,
            ptr::null_mut(),
            0,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(String::from_utf8_lossy(&uuid[..len])
        .trim_end_matches('\0')
        .into())
}

/// `ru_maxrss`, in bytes on macOS.
pub(super) fn peak_resident() -> io::Result<u64> {
    // SAFETY: plain-old-data the call fills.
    let mut usage: libc::rusage = unsafe { mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(usage.ru_maxrss as u64)
}

fn info(pid: u32) -> io::Result<libc::proc_bsdinfo> {
    // SAFETY: plain-old-data the call fills.
    let mut info: libc::proc_bsdinfo = unsafe { mem::zeroed() };
    let size = mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    // SAFETY: a writable buffer of exactly `size` bytes.
    let filled = unsafe {
        libc::proc_pidinfo(
            pid as libc::c_int,
            libc::PROC_PIDTBSDINFO,
            0,
            (&mut info as *mut libc::proc_bsdinfo).cast(),
            size,
        )
    };
    if filled == size {
        return Ok(info);
    }
    let error = io::Error::last_os_error();
    Err(match error.raw_os_error() {
        Some(0) | None => io::ErrorKind::NotFound.into(),
        _ => error,
    })
}

pub(super) fn parent_of(pid: u32) -> io::Result<u32> {
    Ok(info(pid)?.pbi_ppid)
}

/// Groups and meters are read only on Linux so far: unreadable here, which decides nothing.
pub(super) fn group_members(_: &Birth) -> Vec<Birth> {
    Vec::new()
}

pub(super) fn meter(_: u32) -> Option<(u64, u64)> {
    None
}

pub(super) fn read_bytes(_: u32) -> Option<u64> {
    None
}

pub(super) fn become_subreaper() -> crate::Result<()> {
    Err(crate::Error::unsupported(crate::Seam::Process))
}

pub(super) fn die_with_parent(_: u32) -> crate::Result<bool> {
    Err(crate::Error::unsupported(crate::Seam::Process))
}

pub(super) fn transient(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EAGAIN | libc::ENOMEM | libc::EMFILE | libc::ENFILE | libc::EINTR)
    )
}

/// Exited, a zombie included, or gone altogether.
pub(super) fn exited(pid: u32) -> bool {
    match info(pid) {
        Ok(info) => info.pbi_status == libc::SZOMB,
        Err(error) => gone(&error),
    }
}

fn start(info: &libc::proc_bsdinfo) -> u64 {
    info.pbi_start_tvsec * 1_000_000 + info.pbi_start_tvusec
}

pub(super) fn gone(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::NotFound || error.raw_os_error() == Some(libc::ESRCH)
}
