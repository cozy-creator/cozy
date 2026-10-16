//! A user service per OS, started at login and on demand: a systemd `--user` unit, a launchd
//! LaunchAgent, or the HKCU `Run` key (no admin; Calcifer is also started by any CLI call and
//! TensorD's stable parent supervises it, so Windows needs no restart policy). Whether a service
//! is alive is its own lock's answer, never the manager's. A service calls `announce` when it
//! starts, so `stop` finds it however it was started.
use crate::{Result, paths::Home};
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

#[cfg_attr(target_os = "linux", path = "systemd.rs")]
#[cfg_attr(target_os = "macos", path = "launchd.rs")]
#[cfg_attr(windows, path = "windows.rs")]
mod sys;

#[derive(Clone, Debug)]
pub struct ServiceSpec {
    /// `calcifer`, `tensord`.
    pub name: String,
    pub program: PathBuf,
    pub args: Vec<OsString>,
    /// Where its stdout and stderr append: what it says before its own log is open.
    pub log: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct Services {
    /// Where this OS's definitions (or, on Windows, the births of what it started) are kept.
    dir: PathBuf,
}

impl Services {
    /// `user_home` is the caller's config's answer; nothing here reads the environment.
    pub fn new(home: &Home, user_home: &Path) -> Self {
        Self {
            dir: sys::dir(home, user_home),
        }
    }

    /// Whether this session has a user service manager to talk to.
    pub fn available(&self) -> bool {
        sys::available()
    }

    /// Writes the definition and enables it at login, without starting it.
    pub fn install(&self, spec: &ServiceSpec) -> Result<()> {
        sys::install(&self.dir, spec)
    }

    /// Installs it, then starts it now. One already running is left as it is.
    pub fn start(&self, spec: &ServiceSpec) -> Result<()> {
        sys::install(&self.dir, spec)?;
        sys::start(&self.dir, spec)
    }

    /// Called by the running service itself, first thing: records its birth where `stop` looks
    /// for it. Windows' `Run` key has no manager to ask; systemd and launchd track it themselves.
    pub fn announce(&self, name: &str) -> Result<()> {
        sys::announce(&self.dir, name)
    }

    /// Ends it; nothing starts it again before `start` or the next login.
    pub fn stop(&self, name: &str) -> Result<()> {
        sys::stop(&self.dir, name)
    }

    /// Stops it and removes its definition.
    pub fn uninstall(&self, name: &str) -> Result<()> {
        sys::stop(&self.dir, name)?;
        sys::uninstall(&self.dir, name)
    }
}

/// Runs a service manager's command; its stderr is the error.
#[cfg(unix)]
fn run(manager: &'static str, command: &mut std::process::Command) -> Result<()> {
    let output = command.output().map_err(|error| crate::Error::Service {
        manager,
        detail: format!("{command:?}: {error}"),
    })?;
    if output.status.success() {
        return Ok(());
    }
    Err(crate::Error::Service {
        manager,
        detail: format!(
            "{command:?}: {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ),
    })
}
