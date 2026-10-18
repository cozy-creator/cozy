//! The machine's stable parent, forked before any thread exists. It reaps orphans (a pod's
//! PID 1 must), forwards stop signals and starts the service again after an abnormal exit, as
//! the Go guardian restarts its request plane. A clean exit (a stop or an accepted release)
//! ends it, and so does a second consecutive exit before readiness: measured progress, no timer.
//! The service is this executable, forked; an activated Runtime update is a new service on the
//! activated SDK. An activated binary (`super::binary`) is exec'd in place, same PID and arguments,
//! and its own parent forks the service. A candidate that never proves readiness is rolled back by
//! the parent then running: a binary by renaming the previous one back and exec'ing it.
use nix::{
    errno::Errno,
    fcntl::OFlag,
    sys::{
        signal::{kill, SigSet, Signal},
        wait::{waitpid, WaitPidFlag, WaitStatus},
    },
    unistd::{fork, pipe2, ForkResult, Pid},
};
use std::{
    fs::File,
    io::{self, Read, Write},
    os::fd::AsRawFd,
};

use super::{provider::ProviderSelf, update::Recovered};

/// Forks the service. Returns only in the service child, with the pipe it reports readiness
/// on; the parent exits with the service's final status, or execs an activated binary, handing
/// on a rental's `provider` credential.
pub fn supervise(paths: &super::update::Paths, provider: Option<ProviderSelf>) -> io::Result<Ready> {
    let mut signals = SigSet::empty();
    for signal in [Signal::SIGTERM, Signal::SIGINT, Signal::SIGCHLD] {
        signals.add(signal);
    }
    signals.thread_block().map_err(io::Error::from)?;
    // Orphans of the service reparent here, not to a PID 1 that may not reap them.
    cozy_platform::process::become_subreaper()?;
    let parent = nix::unistd::getpid();
    let mut failures_before_ready = 0;
    loop {
        if let Recovered::Exec(launch) = super::update::recover_activation(paths)? {
            exec(paths, &launch, provider.as_ref(), &signals)?;
        }
        let (read, write) = pipe2(OFlag::O_CLOEXEC).map_err(io::Error::from)?;
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
        // An activated binary is exec'd as the loop recovers its published activation.
        if code == super::update::REPLACE_EXIT || code == super::binary::EXEC_EXIT {
            tracing::info!("starting the service on its updated software");
            failures_before_ready = 0;
            continue;
        }
        failures_before_ready = if ready { 0 } else { failures_before_ready + 1 };
        if failures_before_ready >= 2 {
            let cause = format!("the service exited twice before readiness ({code})");
            match super::update::rollback_pending(paths, &cause)? {
                Recovered::Nothing => {
                    tracing::error!("{cause}; stopping");
                    std::process::exit(code);
                }
                Recovered::Restored => tracing::warn!("{cause}; update rolled back"),
                Recovered::Exec(launch) => {
                    tracing::warn!("{cause}; update rolled back to the previous binary");
                    exec(paths, &launch, provider.as_ref(), &signals)?;
                }
            }
            failures_before_ready = 0;
            continue;
        }
        tracing::warn!("the service exited ({code}); starting it again");
    }
}

/// Replaces this parent with the binary at `launch`. One that will not exec is rolled back, and
/// this image serves on.
fn exec(paths: &super::update::Paths, launch: &std::path::Path, provider: Option<&ProviderSelf>, signals: &SigSet) -> io::Result<()> {
    tracing::info!("running {}", launch.display());
    let error = super::binary::exec(launch, provider);
    signals.thread_block().map_err(io::Error::from)?;
    let cause = format!("cannot run {}: {error}", launch.display());
    tracing::error!("{cause}");
    super::update::rollback_pending(paths, &cause).map(drop)
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
