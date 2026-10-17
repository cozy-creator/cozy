//! The machine's stable parent, forked before any thread exists. It reaps orphans (a pod's
//! PID 1 must), forwards stop signals and starts the service again after an abnormal exit, as
//! the Go guardian restarts its request plane. A clean exit (a stop or an accepted release)
//! ends it, and so does a second consecutive exit before readiness: measured progress, no timer.
//! The service is always this executable, forked; an activated Runtime update is a new service
//! on the activated SDK, and a candidate that never proves readiness is rolled back by this same
//! parent.
use nix::{
    errno::Errno,
    sys::{
        signal::{kill, SigSet, Signal},
        wait::{waitpid, WaitPidFlag, WaitStatus},
    },
    unistd::{fork, ForkResult, Pid},
};
use std::{
    fs::File,
    io::{self, Read, Write},
    os::fd::AsRawFd,
};

/// Forks the service. Returns only in the service child, with the pipe it reports readiness
/// on; the parent exits with the service's final status.
pub fn supervise(paths: &super::update::Paths) -> io::Result<Ready> {
    let mut signals = SigSet::empty();
    for signal in [Signal::SIGTERM, Signal::SIGINT, Signal::SIGCHLD] {
        signals.add(signal);
    }
    signals.thread_block().map_err(io::Error::from)?;
    // Orphans of the service reparent here, not to a PID 1 that may not reap them. Elsewhere
    // (macOS) they reparent to launchd, which reaps them, and executor trees end by group.
    match cozy_platform::process::become_subreaper() {
        Err(error) if error.is_unsupported() => {}
        other => other?,
    }
    let parent = nix::unistd::getpid();
    let mut failures_before_ready = 0;
    loop {
        super::update::recover_activation(paths)?;
        let (read, write) = cloexec_pipe()?;
        // SAFETY: no thread exists yet in this process; the child continues single-threaded.
        let child = match unsafe { fork() }.map_err(io::Error::from)? {
            ForkResult::Child => {
                drop(read);
                // The service never outlives its supervisor.
                if !cozy_platform::process::die_with_parent(parent.as_raw() as u32)? {
                    std::process::exit(1);
                }
                SigSet::empty().thread_set_mask().map_err(io::Error::from)?;
                return Ok(Ready(Some(File::from(write))));
            }
            ForkResult::Parent { child } => child,
        };
        drop(write);
        let (status, stopping) = wait_service(child, &signals)?;
        let mut byte = [0u8; 1];
        let ready = File::from(read).read(&mut byte).unwrap_or(0) == 1;
        let code = match status {
            WaitStatus::Exited(_, code) => code,
            WaitStatus::Signaled(_, signal, _) => 128 + signal as i32,
            _ => 1,
        };
        if stopping || code == 0 {
            std::process::exit(code);
        }
        if code == super::update::REPLACE_EXIT {
            tracing::info!("starting the service on its updated software");
            failures_before_ready = 0;
            continue;
        }
        failures_before_ready = if ready { 0 } else { failures_before_ready + 1 };
        if failures_before_ready >= 2 {
            let cause = format!("the service exited twice before readiness ({code})");
            if super::update::rollback_pending(paths, &cause)? {
                tracing::warn!("{cause}; Runtime update rolled back");
                failures_before_ready = 0;
                continue;
            }
            tracing::error!("{cause}; stopping");
            std::process::exit(code);
        }
        tracing::warn!("the service exited ({code}); starting it again");
    }
}

/// Places `fd` at `at` for the re-executed machine to inherit (dup2 clears close-on-exec on the copy).
pub(super) fn inherit(fd: std::os::fd::OwnedFd, at: i32) -> bool {
    // SAFETY: plain fd calls in a single-threaded child.
    let placed = unsafe {
        if fd.as_raw_fd() == at {
            libc::fcntl(at, libc::F_SETFD, 0)
        } else {
            libc::dup2(fd.as_raw_fd(), at)
        }
    };
    if fd.as_raw_fd() == at {
        std::mem::forget(fd);
    }
    placed >= 0
}

/// Waits for the service to end, reaping every orphan meanwhile and forwarding stop signals.
fn wait_service(child: Pid, signals: &SigSet) -> io::Result<(WaitStatus, bool)> {
    let mut stopping = false;
    loop {
        match signals.wait().map_err(io::Error::from)? {
            signal @ (Signal::SIGTERM | Signal::SIGINT) => {
                stopping = true;
                let _ = kill(child, signal);
            }
            _ => loop {
                match waitpid(None, Some(WaitPidFlag::WNOHANG)) {
                    Ok(WaitStatus::StillAlive) | Err(Errno::ECHILD) => break,
                    Ok(status) if status.pid() == Some(child) => return Ok((status, stopping)),
                    Ok(_) | Err(Errno::EINTR) => continue,
                    Err(error) => return Err(error.into()),
                }
            },
        }
    }
}

/// The service's readiness report to its parent: written once, when this boot has proved itself.
pub struct Ready(Option<File>);
impl Ready {
    pub fn report(&mut self) {
        if let Some(mut pipe) = self.0.take() {
            let _ = pipe.write_all(b"r");
        }
    }
}

/// A pipe whose ends close on exec (std makes it so on every Unix).
pub(crate) fn cloexec_pipe() -> io::Result<(std::os::fd::OwnedFd, std::os::fd::OwnedFd)> {
    let (read, write) = std::io::pipe()?;
    Ok((read.into(), write.into()))
}
