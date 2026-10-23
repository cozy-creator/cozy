//! macOS: births from `proc_pidinfo`, exact handles as a kqueue watching the process's exit.
#![allow(unsafe_code)]
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

/// Every process with its parent.
pub(super) fn parents() -> Vec<(u32, u32)> {
    pids()
        .into_iter()
        .filter_map(|pid| Some((pid, info(pid).ok()?.pbi_ppid)))
        .collect()
}

fn pids() -> Vec<u32> {
    // SAFETY: a null buffer asks for the count.
    let count = unsafe { libc::proc_listallpids(ptr::null_mut(), 0) };
    if count <= 0 {
        return vec![];
    }
    // Room for processes started since the count.
    let mut pids = vec![0 as libc::pid_t; count as usize + 64];
    let bytes = (pids.len() * mem::size_of::<libc::pid_t>()) as libc::c_int;
    // SAFETY: a writable buffer of the length given.
    let listed = unsafe { libc::proc_listallpids(pids.as_mut_ptr().cast(), bytes) };
    pids.truncate(listed.max(0) as usize);
    pids.into_iter().filter(|pid| *pid > 0).map(|pid| pid as u32).collect()
}

pub(super) fn group_members(leader: &Birth) -> Vec<Birth> {
    pids()
        .into_iter()
        .filter(|pid| *pid != leader.pid)
        .filter_map(|pid| {
            let info = info(pid).ok()?;
            (info.pbi_pgid == leader.pid && start(&info) >= leader.start_ticks).then(|| Birth {
                pid,
                boot_id: String::new(),
                start_ticks: start(&info),
            })
        })
        .collect()
}

/// CPU nanoseconds and disk bytes moved. macOS keeps no time a thread waited runnable, and
/// counts disk bytes, not every byte read and written as Linux does.
pub(super) fn meter(pid: u32) -> Option<(u64, u64)> {
    let usage = rusage(pid)?;
    let cpu = nanoseconds(usage.ri_user_time.saturating_add(usage.ri_system_time));
    Some((cpu, usage.ri_diskio_bytesread.saturating_add(usage.ri_diskio_byteswritten)))
}

/// Disk bytes written.
pub(super) fn written_bytes(pid: u32) -> Option<u64> {
    Some(rusage(pid)?.ri_diskio_byteswritten)
}

fn rusage(pid: u32) -> Option<libc::rusage_info_v2> {
    // SAFETY: plain-old-data the call fills.
    let mut usage: libc::rusage_info_v2 = unsafe { mem::zeroed() };
    let buffer = (&mut usage as *mut libc::rusage_info_v2).cast::<libc::rusage_info_t>();
    // SAFETY: a writable record of the flavor named.
    (unsafe { libc::proc_pid_rusage(pid as libc::c_int, libc::RUSAGE_INFO_V2, buffer) } == 0)
        .then_some(usage)
}

/// `rusage_info` times are Mach ticks (not nanoseconds on Apple silicon). libc marks the timebase
/// deprecated in favor of the mach2 crate; this one call is all it takes.
#[allow(deprecated)]
fn nanoseconds(ticks: u64) -> u64 {
    static TIMEBASE: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    let (numer, denom) = *TIMEBASE.get_or_init(|| {
        let mut base = libc::mach_timebase_info { numer: 0, denom: 0 };
        // SAFETY: a writable timebase record.
        unsafe { libc::mach_timebase_info(&mut base) };
        (u64::from(base.numer.max(1)), u64::from(base.denom.max(1)))
    });
    (u128::from(ticks) * u128::from(numer) / u128::from(denom)) as u64
}

pub(super) fn become_subreaper() -> crate::Result<()> {
    Err(crate::Error::unsupported(crate::Seam::Process))
}

/// No parent-death signal on macOS: a thread waits on the parent's exit (kqueue `NOTE_EXIT`)
/// and kills this process. The watch ends at an exec, which Linux's signal survives.
pub(super) fn die_with_parent(parent: u32) -> crate::Result<bool> {
    let born = match birth(parent) {
        Ok(born) => born,
        Err(error) if gone(&error) => return Ok(false),
        Err(error) => return Err(crate::Error::os("read the parent's birth")(error)),
    };
    // SAFETY: getppid has no arguments.
    if unsafe { libc::getppid() } as u32 != parent {
        return Ok(false);
    }
    let Some(watch) = open(&born).map_err(crate::Error::os("watch the parent"))? else {
        return Ok(false);
    };
    std::thread::Builder::new()
        .name("parent-watch".into())
        .spawn(move || {
            if wait(&watch, None).is_ok() {
                // SAFETY: a signal to this process.
                unsafe { libc::kill(libc::getpid(), libc::SIGKILL) };
            }
        })
        .map_err(crate::Error::os("start the parent watch"))?;
    Ok(true)
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
