//! A systemd `--user` unit kept in the home and linked into the user's manager, so no XDG path is
//! looked up. `Restart=on-failure`: a clean exit (a restart, an update) is not restarted.
use super::{ServiceSpec, run};
use crate::{Error, Result, durable, paths::Home};
use std::{
    ffi::OsString,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const MANAGER: &str = "systemd";

pub(super) fn dir(home: &Home, _: &Path) -> PathBuf {
    home.services()
}

pub(super) fn available() -> bool {
    systemctl()
        .arg("show-environment")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub(super) fn install(dir: &Path, spec: &ServiceSpec) -> Result<()> {
    let path =
        std::path::absolute(dir.join(unit(&spec.name))).map_err(Error::io("resolve", dir))?;
    fs::create_dir_all(dir).map_err(Error::io("create directory", dir))?;
    durable::replace(&path, render(spec).as_bytes(), 0o644)?;
    // `enable <path>` links the unit into the manager and wants it at login; the reload picks up a
    // changed definition of a unit that was already linked.
    run(MANAGER, systemctl().arg("enable").arg(&path))?;
    run(MANAGER, systemctl().arg("daemon-reload"))
}

/// Whether the unit kept here is `spec`'s, as `install` wrote it (and linked it).
pub(super) fn current(dir: &Path, spec: &ServiceSpec) -> bool {
    let kept =
        fs::read(dir.join(unit(&spec.name))).is_ok_and(|kept| kept == render(spec).as_bytes());
    // The manager's link too: a person's `systemctl --user disable` removes it, not the file.
    kept && systemctl()
        .args(["is-enabled", "--quiet", &unit(&spec.name)])
        .status()
        .is_ok_and(|status| status.success())
}

pub(super) fn start(_: &Path, spec: &ServiceSpec) -> Result<()> {
    run(MANAGER, systemctl().args(["start", &unit(&spec.name)]))
}

pub(super) fn announce(_: &Path, _: &str) -> Result<()> {
    Ok(())
}

pub(super) fn stop(dir: &Path, name: &str) -> Result<()> {
    if !dir.join(unit(name)).exists() {
        return Ok(());
    }
    run(MANAGER, systemctl().args(["stop", &unit(name)]))
}

pub(super) fn uninstall(dir: &Path, name: &str) -> Result<bool> {
    let path = dir.join(unit(name));
    if !path.exists() {
        return Ok(false);
    }
    // An unlinked unit (a removal cut short after its disable) refuses `disable`: the file goes
    // either way, and a link left to it would name nothing.
    let _ = run(MANAGER, systemctl().args(["disable", &unit(name)]));
    if let Err(error) = fs::remove_file(&path)
        && error.kind() != io::ErrorKind::NotFound
    {
        return Err(Error::io("remove", path)(error));
    }
    run(MANAGER, systemctl().arg("daemon-reload")).map(|()| true)
}

/// `spec` in a transient scope of its own (`systemd-run --scope` execs it in place), out of the
/// cgroup of the unit that starts it, which systemd ends whole when that unit stops.
pub(super) fn scope(spec: &ServiceSpec) -> Option<(PathBuf, Vec<OsString>)> {
    let run = ["--user", "--scope", "--quiet", "--collect", "--"].map(OsString::from);
    let command = std::iter::once(spec.program.clone().into_os_string());
    Some(("systemd-run".into(), run.into_iter().chain(command).chain(spec.args.iter().cloned()).collect()))
}

fn systemctl() -> Command {
    let mut command = Command::new("systemctl");
    command.arg("--user");
    command
}

fn unit(name: &str) -> String {
    format!("cozy-{name}.service")
}

fn render(spec: &ServiceSpec) -> String {
    let exec = std::iter::once(spec.program.as_os_str())
        .chain(spec.args.iter().map(|arg| arg.as_os_str()))
        .map(|word| quote(&word.to_string_lossy()))
        .collect::<Vec<_>>()
        .join(" ");
    let log = spec.log.as_ref().map_or(String::new(), |log| {
        let log = log.to_string_lossy().replace('%', "%%");
        format!("StandardOutput=append:{log}\nStandardError=append:{log}\n")
    });
    format!(
        "[Unit]\nDescription=cozy {}\n\n[Service]\nExecStart={exec}\nRestart=on-failure\n{log}\n\
         [Install]\nWantedBy=default.target\n",
        spec.name
    )
}

/// One ExecStart word: quoted, with systemd's specifier `%` and variable `$` made literal.
fn quote(word: &str) -> String {
    let mut quoted = String::from('"');
    for c in word.chars() {
        match c {
            '\\' | '"' => quoted.extend(['\\', c]),
            '%' => quoted.push_str("%%"),
            '$' => quoted.push_str("$$"),
            _ => quoted.push(c),
        }
    }
    quoted.push('"');
    quoted
}
