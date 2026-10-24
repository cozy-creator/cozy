//! What Linux and macOS share: process groups, sessions, signals and readiness.
#![allow(unsafe_code)]
use std::{
    ffi::OsString,
    fs::File,
    io,
    os::{
        fd::{AsRawFd, BorrowedFd, FromRawFd, RawFd},
        unix::process::CommandExt,
    },
    path::Path,
    process::{Child, Command, ExitStatus, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use super::HANDED;

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

/// [`super::inherit_nothing`] on Unix.
pub(super) fn inherit_nothing(command: &mut Command) {
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

pub(super) fn spawn(command: &mut Command) -> io::Result<(Child, Group)> {
    command.process_group(0);
    Ok((command.spawn()?, Group))
}

#[derive(Debug)]
pub(super) struct Detached(Child);

impl Detached {
    pub(super) fn pid(&self) -> u32 {
        self.0.id()
    }

    pub(super) fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.0.try_wait()
    }

    pub(super) fn wait(&mut self) -> io::Result<ExitStatus> {
        self.0.wait()
    }
}

/// A session of its own: no terminal's hangup reaches it, and it leads its own group.
pub(super) fn detach(program: &Path, args: &[OsString], log: Option<File>) -> io::Result<Detached> {
    let mut command = Command::new(program);
    command.args(args).stdin(Stdio::null());
    match log {
        Some(log) => command.stdout(log.try_clone()?).stderr(log),
        None => command.stdout(Stdio::null()).stderr(Stdio::null()),
    };
    // SAFETY: cloexec_from and setsid are async-signal-safe and touch no memory.
    unsafe {
        command.pre_exec(|| {
            cloexec_from(3);
            match libc::setsid() {
                -1 => Err(io::Error::last_os_error()),
                _ => Ok(()),
            }
        });
    }
    command.spawn().map(Detached)
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
    // SAFETY: a writable rlimit.
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
    // SAFETY: a writable rlimit.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0
        || limit.rlim_cur == libc::RLIM_INFINITY
    {
        return None;
    }
    #[allow(clippy::unnecessary_cast)]
    Some(limit.rlim_cur as u64)
}

/// [`super::keep_open`] on Unix.
pub(super) fn keep_open(command: &mut Command, fds: Vec<RawFd>) {
    // SAFETY: only async-signal-safe fcntl, in the child; a descriptor not open fails the spawn.
    unsafe {
        command.pre_exec(move || {
            for &fd in &fds {
                if libc::fcntl(fd, libc::F_SETFD, 0) < 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
}

/// Whether [`reserve_handed`] ran: until it has, nothing may be placed on a handed descriptor.
static RESERVED: AtomicBool = AtomicBool::new(false);

/// [`super::reserve_handed`].
pub(super) fn reserve_handed() -> io::Result<()> {
    // One thread: nothing can open a file between a slot's check and its fill.
    if threads()? != 1 {
        return Err(io::Error::other("this process has more than one thread"));
    }
    for fd in HANDED {
        // SAFETY: a query of a descriptor number; nothing is read or written.
        if unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0 {
            continue;
        }
        // `fd` is closed, so the lowest free descriptor a fresh open takes is at most `fd`.
        if let Ok(null) = File::open("/dev/null") {
            // SAFETY: `fd` is closed (checked above), so no owner loses its descriptor; `null`
            // is ours and stays open through the call.
            if null.as_raw_fd() != fd && unsafe { libc::dup2(null.as_raw_fd(), fd) } >= 0 {
                // SAFETY: as above, on the copy dup2 just made.
                unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
            } else if null.as_raw_fd() == fd {
                std::mem::forget(null);
            }
        }
    }
    RESERVED.store(true, Ordering::Release);
    Ok(())
}

/// [`super::take_handed`].
pub(super) fn take_handed(fd: RawFd) -> Option<File> {
    // SAFETY: duplicates `fd` if it is open; the copy is new and ours, `fd` stays as it was.
    let copy = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) };
    // SAFETY: a descriptor fcntl just created for this process alone.
    (copy >= 0).then(|| unsafe { File::from_raw_fd(copy) })
}

/// [`super::hand_on_exec`].
pub(super) fn hand_on_exec(fd: &impl AsRawFd, at: RawFd) -> io::Result<()> {
    if !HANDED.contains(&at) || !RESERVED.load(Ordering::Acquire) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "not a reserved handed descriptor"));
    }
    // SAFETY: `at` is a reserved slot, which no owner holds (reserve_handed); dup2 replaces
    // whatever unowned descriptor sits there and leaves the copy inheritable.
    if unsafe { libc::dup2(fd.as_raw_fd(), at) } < 0 {
        return Err(io::Error::last_os_error());
    }
    if fd.as_raw_fd() == at {
        // SAFETY: as above; dup2 onto itself keeps close-on-exec, so clear it.
        if unsafe { libc::fcntl(at, libc::F_SETFD, 0) } < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// [`super::fork`].
pub(super) fn fork() -> io::Result<super::Forked> {
    if threads()? != 1 {
        return Err(io::Error::other("this process has more than one thread"));
    }
    // SAFETY: this process has one thread (checked above, and only this thread could start
    // another), so the child inherits no lock another thread holds.
    match unsafe { libc::fork() } {
        -1 => Err(io::Error::last_os_error()),
        0 => Ok(super::Forked::Child),
        pid => Ok(super::Forked::Parent(pid as u32)),
    }
}

#[cfg(target_os = "linux")]
fn threads() -> io::Result<usize> {
    Ok(std::fs::read_dir("/proc/self/task")?.count())
}

#[cfg(target_os = "macos")]
fn threads() -> io::Result<usize> {
    // SAFETY: plain-old-data the call fills.
    let mut task: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
    // SAFETY: a writable buffer of exactly `size` bytes.
    let filled = unsafe {
        libc::proc_pidinfo(std::process::id() as libc::c_int, libc::PROC_PIDTASKINFO, 0, (&mut task as *mut libc::proc_taskinfo).cast(), size)
    };
    if filled != size {
        return Err(io::Error::last_os_error());
    }
    Ok(task.pti_threadnum as usize)
}

/// [`super::keep_sigchld`].
pub(super) fn keep_sigchld() -> io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        extern "C" fn noted(_: libc::c_int) {}
        // SAFETY: sigaction is plain old data; all-zero is a valid value.
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = noted as extern "C" fn(libc::c_int) as usize;
        // SAFETY: a handler that does nothing is async-signal-safe; `action` outlives the call.
        if unsafe { libc::sigaction(libc::SIGCHLD, &action, std::ptr::null_mut()) } != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// [`super::glibc_version`].
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub(super) fn glibc_version() -> String {
    // SAFETY: glibc returns a static NUL-terminated string.
    unsafe { std::ffi::CStr::from_ptr(libc::gnu_get_libc_version()) }.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};

    #[test]
    fn a_test_harness_has_threads_so_fork_and_reservation_are_refused() {
        let more = std::thread::spawn(|| std::thread::park_timeout(std::time::Duration::from_secs(5)));
        assert!(super::fork().is_err());
        assert!(super::reserve_handed().is_err());
        more.thread().unpark();
        more.join().unwrap();
    }

    #[test]
    fn a_handed_descriptor_is_copied_and_an_unreserved_slot_refused() {
        let (mut read, mut write) = std::io::pipe().unwrap();
        write.write_all(b"handed").unwrap();
        drop(write);
        let mut copy = super::take_handed(std::os::fd::AsRawFd::as_raw_fd(&read)).unwrap();
        let mut text = String::new();
        copy.read_to_string(&mut text).unwrap();
        assert_eq!(text, "handed");
        assert_eq!(read.read(&mut [0u8; 1]).unwrap(), 0, "the original stays open, at its end");
        assert!(super::take_handed(-1).is_none());
        assert!(super::hand_on_exec(&read, 7).is_err());
    }
}
