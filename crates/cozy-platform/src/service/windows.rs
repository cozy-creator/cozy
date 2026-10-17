//! The HKCU `Run` key: started at logon with no admin. Logon gives a console program a console
//! window; `announce` leaves it at once (a brief flash). `conhost --headless` was measured
//! unusable: it ends at its stdin's end, so a service started that way never runs. With no
//! manager to ask, the service's birth is kept beside the home (`start` records it, and so does
//! the service's own `announce`), and `stop` ends exactly that birth.
use super::ServiceSpec;
use crate::{
    Error, Result,
    paths::Home,
    process::{self, Birth, Exact, Lifetime},
};
use std::{
    fs::{self, OpenOptions},
    io, iter,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
use windows_sys::Win32::{
    Foundation::ERROR_FILE_NOT_FOUND,
    System::{
        Console::{FreeConsole, GetConsoleProcessList},
        Registry::{HKEY_CURRENT_USER, REG_SZ, RegDeleteKeyValueW, RegSetKeyValueW},
    },
};

const RUN: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";
const MANAGER: &str = "the HKCU Run key";

pub(super) fn dir(home: &Home, _: &Path) -> PathBuf {
    home.services()
}

pub(super) fn available() -> bool {
    true
}

pub(super) fn install(_: &Path, spec: &ServiceSpec) -> Result<()> {
    let data = wide(&command_line(spec));
    // SAFETY: NUL-terminated wide strings and the data's exact byte length.
    let status = unsafe {
        RegSetKeyValueW(
            HKEY_CURRENT_USER,
            wide(RUN).as_ptr(),
            wide(&value(&spec.name)).as_ptr(),
            REG_SZ,
            data.as_ptr().cast(),
            (data.len() * 2) as u32,
        )
    };
    registry("set", status)
}

/// The `Run` value is set on every install: one registry write, which flushes nothing.
pub(super) fn current(_: &Path, _: &ServiceSpec) -> bool {
    false
}

pub(super) fn start(dir: &Path, spec: &ServiceSpec) -> Result<()> {
    if running(dir, &spec.name)?.is_some() {
        return Ok(());
    }
    let (stdout, stderr) = match &spec.log {
        Some(log) => {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(log)
                .map_err(Error::io("open log", log))?;
            let copy = file.try_clone().map_err(Error::io("open log", log))?;
            (Stdio::from(file), Stdio::from(copy))
        }
        None => (Stdio::null(), Stdio::null()),
    };
    let mut command = Command::new(&spec.program);
    command
        .args(&spec.args)
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr);
    let tree = process::spawn(command, Lifetime::Detached)?;
    record(dir, &spec.name, &Birth::of(tree.pid())?)
}

pub(super) fn announce(dir: &Path, name: &str) -> Result<()> {
    // A console with no other process on it was opened for this service (a logon start): leave it
    // so its window closes. A terminal's console has the shell on it too, and stays.
    let mut processes = [0u32; 2];
    // SAFETY: a writable array of the stated length.
    if unsafe { GetConsoleProcessList(processes.as_mut_ptr(), processes.len() as u32) } == 1 {
        // SAFETY: detaching this process from its console; nothing here writes to it afterwards.
        unsafe { FreeConsole() };
    }
    record(dir, name, &Birth::own()?)
}

pub(super) fn stop(dir: &Path, name: &str) -> Result<()> {
    if let Some(exact) = running(dir, name)? {
        exact.kill()?;
        exact.wait()?;
    }
    let path = birth_path(dir, name);
    if let Err(error) = fs::remove_file(&path)
        && error.kind() != io::ErrorKind::NotFound
    {
        return Err(Error::io("remove", path)(error));
    }
    Ok(())
}

pub(super) fn uninstall(_: &Path, name: &str) -> Result<()> {
    // SAFETY: NUL-terminated wide strings.
    let status = unsafe {
        RegDeleteKeyValueW(
            HKEY_CURRENT_USER,
            wide(RUN).as_ptr(),
            wide(&value(name)).as_ptr(),
        )
    };
    if status == ERROR_FILE_NOT_FOUND {
        return Ok(());
    }
    registry("delete", status)
}

fn record(dir: &Path, name: &str, birth: &Birth) -> Result<()> {
    fs::create_dir_all(dir).map_err(Error::io("create directory", dir))?;
    let record = format!("{} {}", birth.pid, birth.start_ticks);
    crate::durable::replace(&birth_path(dir, name), record.as_bytes(), 0o600)
}

/// The recorded birth, while it still runs.
fn running(dir: &Path, name: &str) -> Result<Option<Exact>> {
    let path = birth_path(dir, name);
    let record = match fs::read_to_string(&path) {
        Ok(record) => record,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(Error::io("read", path)(error)),
    };
    let mut fields = record.split_whitespace().map(str::parse::<u64>);
    let (Some(Ok(pid)), Some(Ok(start_ticks))) = (fields.next(), fields.next()) else {
        return Ok(None);
    };
    let birth = Birth {
        pid: pid as u32,
        boot_id: String::new(),
        start_ticks,
    };
    Ok(Exact::open(&birth)?.filter(|exact| !exact.ended()))
}

fn birth_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.birth"))
}

fn value(name: &str) -> String {
    format!("cozy-{name}")
}

fn registry(op: &'static str, status: u32) -> Result<()> {
    match status {
        0 => Ok(()),
        _ => Err(Error::Service {
            manager: MANAGER,
            detail: format!("{op}: {}", io::Error::from_raw_os_error(status as i32)),
        }),
    }
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(iter::once(0)).collect()
}

fn command_line(spec: &ServiceSpec) -> String {
    iter::once(spec.program.as_os_str())
        .chain(spec.args.iter().map(|arg| arg.as_os_str()))
        .map(|word| quote(&word.to_string_lossy()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// One argument as `CommandLineToArgvW` reads it back: quoted when it must be, a quote escaped,
/// and the backslashes before a quote doubled.
fn quote(word: &str) -> String {
    if !word.is_empty() && !word.contains([' ', '\t', '"']) {
        return word.into();
    }
    let mut quoted = String::from('"');
    let mut backslashes = 0;
    for c in word.chars() {
        match c {
            '\\' => backslashes += 1,
            '"' => {
                quoted.extend(iter::repeat_n('\\', backslashes * 2 + 1));
                quoted.push('"');
                backslashes = 0;
            }
            _ => {
                quoted.extend(iter::repeat_n('\\', backslashes));
                quoted.push(c);
                backslashes = 0;
            }
        }
    }
    quoted.extend(iter::repeat_n('\\', backslashes * 2));
    quoted.push('"');
    quoted
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_command_line_reads_back_as_its_arguments() {
        assert_eq!(
            super::quote(r"C:\Users\a\.cozy\bin\cozy.exe"),
            r"C:\Users\a\.cozy\bin\cozy.exe"
        );
        assert_eq!(
            super::quote(r"C:\Users\a b\cozy.exe"),
            r#""C:\Users\a b\cozy.exe""#
        );
        assert_eq!(super::quote(r#"say "hi""#), r#""say \"hi\"""#);
        assert_eq!(
            super::quote(r"C:\dir with space\"),
            r#""C:\dir with space\\""#
        );
        assert_eq!(super::quote(""), r#""""#);
    }
}
