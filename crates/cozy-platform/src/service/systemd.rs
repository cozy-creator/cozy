//! A systemd `--user` unit kept in the home and linked into the user's manager, so no XDG path is
//! looked up. `Restart=on-failure`: a clean exit (`cozy down`, an update) is not restarted.
use super::{ServiceSpec, run};
use crate::{Error, Result, durable, paths::Home};
use std::{
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

/// Whether the service failed to start: the manager gave up on it, or restarted it.
pub(super) fn failed(name: &str) -> bool {
    let shown =
        output(systemctl().args(["show", &unit(name), "-p", "ActiveState", "-p", "NRestarts"]))
            .unwrap_or_default();
    shown.lines().any(|line| {
        line == "ActiveState=failed" || line.strip_prefix("NRestarts=").is_some_and(|n| n != "0")
    })
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

pub(super) fn uninstall(dir: &Path, name: &str) -> Result<()> {
    let path = dir.join(unit(name));
    if !path.exists() {
        return Ok(());
    }
    run(MANAGER, systemctl().args(["disable", &unit(name)]))?;
    if let Err(error) = fs::remove_file(&path)
        && error.kind() != io::ErrorKind::NotFound
    {
        return Err(Error::io("remove", path)(error));
    }
    run(MANAGER, systemctl().arg("daemon-reload"))
}

/// A `tensord-*` unit running a program under `under`: an older cozy's `systemd-run --user`.
pub(super) fn legacy(under: &Path) -> Result<Option<super::Legacy>> {
    let listed = output(systemctl().args([
        "list-units",
        "--type=service",
        "--all",
        "--plain",
        "--no-legend",
        "tensord-*",
    ]))?;
    for unit in listed
        .lines()
        .filter_map(|line| line.split_whitespace().next())
    {
        let shown = output(systemctl().args([
            "show",
            unit,
            "-p",
            "ExecStart",
            "-p",
            "Environment",
            "-p",
            "ActiveState",
            "-p",
            "WorkingDirectory",
        ]))?;
        let property = |name: &str| {
            shown
                .lines()
                .find_map(|line| line.strip_prefix(name)?.strip_prefix('='))
                .unwrap_or_default()
        };
        let program = property("ExecStart")
            .split("path=")
            .nth(1)
            .and_then(|rest| rest.split([' ', ';']).next())
            .map(PathBuf::from);
        let Some(program) = program.filter(|program| program.starts_with(under)) else {
            continue;
        };
        // `{ path=… ; argv[]=<program> <args…> ; ignore_errors=… }`
        let args = property("ExecStart")
            .split("argv[]=")
            .nth(1)
            .and_then(|rest| rest.split(" ;").next())
            .map(|argv| words(argv).into_iter().skip(1).collect())
            .unwrap_or_default();
        return Ok(Some(super::Legacy {
            unit: unit.to_owned(),
            program,
            args,
            working_directory: Some(property("WorkingDirectory"))
                .filter(|dir| !dir.is_empty())
                .map(PathBuf::from),
            active: property("ActiveState") == "active",
            environment: words(property("Environment"))
                .into_iter()
                .filter_map(|word| {
                    let (name, value) = word.split_once('=')?;
                    Some((name.to_owned(), value.to_owned()))
                })
                .collect(),
        }));
    }
    Ok(None)
}

/// Starts an older cozy's machine again as it ran: a transient unit of the same name, program,
/// directory and environment.
pub(super) fn revive(legacy: &super::Legacy) -> Result<()> {
    let mut command = Command::new("systemd-run");
    command.args([
        "--user",
        &format!("--unit={}", legacy.unit),
        "-p",
        "CollectMode=inactive-or-failed",
    ]);
    if let Some(dir) = &legacy.working_directory {
        command.arg(format!("--working-directory={}", dir.display()));
    }
    for (name, value) in &legacy.environment {
        command.arg(format!("--setenv={name}={value}"));
    }
    run(MANAGER, command.arg(&legacy.program).args(&legacy.args))
}

/// Stops it, and disables it where a unit file (not a transient one) would start it again.
pub(super) fn retire(unit: &str) -> Result<()> {
    run(MANAGER, systemctl().args(["stop", unit]))?;
    let shown = output(systemctl().args(["show", unit, "-p", "FragmentPath"]))?;
    let fragment = shown
        .trim()
        .strip_prefix("FragmentPath=")
        .unwrap_or_default();
    if fragment.is_empty() || fragment.starts_with("/run/") {
        return Ok(());
    }
    run(MANAGER, systemctl().args(["disable", unit]))?;
    if let Err(error) = fs::remove_file(fragment)
        && error.kind() != io::ErrorKind::NotFound
    {
        return Err(Error::io("remove", fragment)(error));
    }
    run(MANAGER, systemctl().arg("daemon-reload"))
}

/// A manager command's stdout; its stderr is the error.
fn output(command: &mut Command) -> Result<String> {
    let out = command.output().map_err(|error| Error::Service {
        manager: MANAGER,
        detail: format!("{command:?}: {error}"),
    })?;
    if !out.status.success() {
        return Err(Error::Service {
            manager: MANAGER,
            detail: format!(
                "{command:?}: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ),
        });
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// systemd's space-separated words, a double-quoted one kept whole.
fn words(text: &str) -> Vec<String> {
    let (mut words, mut word, mut quoted, mut escaped) = (vec![], String::new(), false, false);
    for c in text.chars() {
        match c {
            _ if escaped => {
                word.push(c);
                escaped = false;
            }
            '\\' => escaped = true,
            '"' => quoted = !quoted,
            ' ' if !quoted => {
                if !word.is_empty() {
                    words.push(std::mem::take(&mut word));
                }
            }
            _ => word.push(c),
        }
    }
    if !word.is_empty() {
        words.push(word);
    }
    words
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

#[cfg(test)]
mod tests {
    #[test]
    fn a_unit_keeps_every_word_literal() {
        let spec = super::ServiceSpec {
            name: "calcifer".into(),
            program: "/home/a b/.cozy/bin/cozy".into(),
            args: vec!["calcifer".into(), r#"say "100%" $HOME \n"#.into()],
            log: Some("/home/a b/100%.log".into()),
        };
        let unit = super::render(&spec);
        assert!(unit.contains(
            r#"ExecStart="/home/a b/.cozy/bin/cozy" "calcifer" "say \"100%%\" $$HOME \\n""#
        ));
        assert!(unit.contains("StandardError=append:/home/a b/100%%.log\n"));
        assert!(
            unit.contains("Restart=on-failure\n") && unit.ends_with("WantedBy=default.target\n")
        );
    }

    #[test]
    fn an_environment_reads_as_systemd_quotes_it() {
        let words = super::words(r#"A=1 "B=two words" C=x\"y LANG=en_US.UTF-8"#);
        assert_eq!(words, ["A=1", "B=two words", "C=x\"y", "LANG=en_US.UTF-8"]);
    }
}
