//! Processes named by their birth, and this process's descriptor table.
use super::{Error, Result, Seam};
use std::io;

/// One process incarnation: no other process ever has the same birth. `start_ticks` is clock
/// ticks since boot (Linux) or microseconds since the epoch (macOS); only Linux's restart at
/// boot, so only Linux carries `boot_id`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
        sys::birth(pid)
    }

    pub fn state(&self) -> Result<State> {
        sys::state(self)
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
    sys::usage(pid)
}

/// This kernel's boot. Page cache and process starts belong to a boot.
pub fn boot_id() -> Result<String> {
    sys::boot_id()
}

/// This process's peak resident memory, in bytes.
pub fn peak_resident() -> Result<u64> {
    // Linux's `ru_maxrss` keeps the peak of the image this process exec'd from; `VmHWM` is
    // this image's own.
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status")?;
        let kib = status
            .lines()
            .find_map(|line| line.strip_prefix("VmHWM:"))
            .and_then(|value| value.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            .ok_or_else(|| io::Error::other("no VmHWM in /proc/self/status"))?;
        Ok(kib * 1024)
    }
    #[cfg(target_vendor = "apple")]
    {
        // SAFETY: plain-old-data the call fills.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        // In bytes on macOS.
        Ok(usage.ru_maxrss as u64)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: plain-old-data the call fills, its size passed.
        let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) } == 0 {
            return Err(io::Error::last_os_error().into());
        }
        Ok(counters.PeakWorkingSetSize as u64)
    }
}

/// Raise the soft descriptor limit as far as the OS lets a process, once is enough.
pub fn raise_fd_limit() {
    #[cfg(unix)]
    {
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
}

/// The soft descriptor limit, or `None` where the table has no practical limit (Windows'
/// handle table) or the limit is infinite.
pub fn fd_limit() -> Option<u64> {
    #[cfg(unix)]
    {
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
        return Err(Error::unsupported(Seam::Process));
    };
    Ok(std::fs::read_dir(dir)?.count() as u64)
}

/// The process's or the system's descriptor table is full.
pub fn fd_table_full(error: &io::Error) -> bool {
    #[cfg(unix)]
    let full = [libc::EMFILE, libc::ENFILE];
    // ERROR_TOO_MANY_OPEN_FILES, WSAEMFILE.
    #[cfg(windows)]
    let full = [4, 10024];
    error.raw_os_error().is_some_and(|code| full.contains(&code))
}

#[cfg(target_os = "linux")]
mod sys {
    use super::{Birth, Result, State};
    use std::{fs, io};

    pub(super) fn birth(pid: u32) -> Result<Birth> {
        Ok(Birth {
            pid,
            boot_id: boot_id()?,
            start_ticks: stat(pid)?.start_ticks,
        })
    }

    pub(super) fn state(birth: &Birth) -> Result<State> {
        const PF_EXITING: u64 = 0x4;
        if boot_id()? != birth.boot_id {
            return Ok(State::Gone);
        }
        let stat = match stat(birth.pid) {
            Ok(stat) => stat,
            Err(error) if gone(&error) => return Ok(State::Gone),
            Err(error) => return Err(error.into()),
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

    pub(super) fn usage(pid: u32) -> Result<super::Usage> {
        let stat = stat(pid)?;
        Ok(super::Usage {
            cpu: stat.cpu,
            running: matches!(stat.state, 'R' | 'D'),
        })
    }

    pub(super) fn boot_id() -> Result<String> {
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
}

#[cfg(target_vendor = "apple")]
mod sys {
    use super::{Birth, Result, State};
    use std::{io, mem};

    /// `proc_bsdinfo.pbi_flags`: the process is exiting.
    const PROC_FLAG_INEXIT: u32 = 4;

    pub(super) fn birth(pid: u32) -> Result<Birth> {
        Ok(Birth {
            pid,
            boot_id: String::new(),
            start_ticks: start(&info(pid)?),
        })
    }

    pub(super) fn state(birth: &Birth) -> Result<State> {
        let info = match info(birth.pid) {
            Ok(info) => info,
            Err(error) if gone(&error) => return Ok(State::Gone),
            Err(error) => return Err(error.into()),
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

    pub(super) fn usage(pid: u32) -> Result<super::Usage> {
        // SAFETY: plain-old-data the call fills.
        let mut task: libc::proc_taskinfo = unsafe { mem::zeroed() };
        let size = mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
        // SAFETY: a writable buffer of exactly `size` bytes.
        let filled = unsafe {
            libc::proc_pidinfo(pid as libc::c_int, libc::PROC_PIDTASKINFO, 0, (&mut task as *mut libc::proc_taskinfo).cast(), size)
        };
        if filled != size {
            return Err(io::Error::last_os_error().into());
        }
        Ok(super::Usage {
            cpu: task.pti_total_user + task.pti_total_system,
            running: task.pti_numrunning > 0,
        })
    }

    /// `kern.bootsessionuuid`: a UUID the kernel draws at each boot.
    pub(super) fn boot_id() -> Result<String> {
        let mut uuid = [0u8; 64];
        let mut len = uuid.len();
        // SAFETY: a NUL-terminated name and a writable buffer of `len` bytes.
        if unsafe {
            libc::sysctlbyname(c"kern.bootsessionuuid".as_ptr(), uuid.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0)
        } != 0
        {
            return Err(io::Error::last_os_error().into());
        }
        Ok(String::from_utf8_lossy(&uuid[..len]).trim_end_matches('\0').into())
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

    fn start(info: &libc::proc_bsdinfo) -> u64 {
        info.pbi_start_tvsec * 1_000_000 + info.pbi_start_tvusec
    }

    fn gone(error: &io::Error) -> bool {
        error.kind() == io::ErrorKind::NotFound || error.raw_os_error() == Some(libc::ESRCH)
    }
}

#[cfg(windows)]
mod sys {
    use super::{Birth, Error, Result, Seam, State};

    pub(super) fn birth(_: u32) -> Result<Birth> {
        Err(Error::unsupported(Seam::Process))
    }

    pub(super) fn usage(pid: u32) -> Result<super::Usage> {
        use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
        use windows_sys::Win32::System::Threading::{GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
        // SAFETY: a query-only handle, closed below.
        let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if process.is_null() {
            return Err(std::io::Error::last_os_error().into());
        }
        let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
        let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
        // SAFETY: out-params of the right type on a live handle.
        let read = unsafe { GetProcessTimes(process, &mut created, &mut exited, &mut kernel, &mut user) };
        let error = std::io::Error::last_os_error();
        // SAFETY: the handle opened above.
        unsafe { CloseHandle(process) };
        if read == 0 {
            return Err(error.into());
        }
        let ticks = |time: FILETIME| (time.dwHighDateTime as u64) << 32 | time.dwLowDateTime as u64;
        Ok(super::Usage {
            cpu: ticks(kernel) + ticks(user),
            running: false,
        })
    }

    pub(super) fn state(_: &Birth) -> Result<State> {
        Err(Error::unsupported(Seam::Process))
    }

    /// The kernel's boot time in seconds (100 ns ticks since 1601), which a clock step can
    /// move: then this boot reads as an earlier one, the safe side.
    pub(super) fn boot_id() -> Result<String> {
        use windows_sys::Wdk::System::SystemInformation::{NtQuerySystemInformation, SystemTimeOfDayInformation};
        // SYSTEM_TIMEOFDAY_INFORMATION: BootTime first, 48 bytes in all.
        let mut info = [0i64; 6];
        // SAFETY: a writable buffer of the size passed.
        let status = unsafe {
            NtQuerySystemInformation(SystemTimeOfDayInformation, info.as_mut_ptr().cast(), 48, std::ptr::null_mut())
        };
        if status < 0 {
            return Err(std::io::Error::other(format!("NtQuerySystemInformation: {status:#x}")).into());
        }
        Ok(format!("{}", info[0] / 10_000_000))
    }
}
