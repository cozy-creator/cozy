//! Test children on every OS: an environment with nothing of the test's but what its OS needs,
//! and a process that lives until its stdin closes.
use std::{path::Path, process::Command};

#[allow(dead_code)]
pub trait BareEnv {
    /// No inherited variable but `PATH`, and on Windows `SystemRoot` (Winsock and Python's
    /// random source start from it) and the temporary directory.
    fn bare_env(&mut self) -> &mut Self;
    /// The user's home, by the name each OS reads it from (`HOME`, Windows `USERPROFILE`).
    fn user_home(&mut self, home: impl AsRef<Path>) -> &mut Self;
}

impl BareEnv for Command {
    fn bare_env(&mut self) -> &mut Self {
        self.env_clear();
        if cfg!(windows) {
            for name in ["PATH", "SYSTEMROOT", "TEMP", "TMP"] {
                if let Some(value) = std::env::var_os(name) {
                    self.env(name, value);
                }
            }
        } else {
            self.env("PATH", "/usr/bin:/bin");
        }
        self
    }
    fn user_home(&mut self, home: impl AsRef<Path>) -> &mut Self {
        let home = home.as_ref();
        self.env("HOME", home).env("USERPROFILE", home)
    }
}

/// A program that lives until its stdin closes: `cat`, or Windows' own `sort`.
#[allow(dead_code)]
pub fn until_stdin_closes() -> Command {
    Command::new(if cfg!(windows) { "sort" } else { "/bin/cat" })
}

/// Stops a machine as its service manager does: SIGTERM, which it ends cleanly on, or its end on
/// Windows (cozy_platform::service's stop).
#[allow(dead_code)]
pub fn stop(machine: &mut std::process::Child) {
    if cfg!(windows) {
        machine.kill().unwrap();
    } else {
        Command::new("kill").args(["-TERM", &machine.id().to_string()]).status().unwrap();
    }
    machine.wait().unwrap();
}
