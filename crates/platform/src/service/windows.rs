//! The HKCU `Run` key: started at logon with no admin. Logon gives a console program a console
//! window; `announce` leaves it at once (a brief flash). `conhost --headless` was measured
//! unusable: it ends at its stdin's end, so a service started that way never runs. With no
//! manager to ask, the service's birth is kept beside the home (`start` records it, and so does
//! the service's own `announce`), and `stop` ends exactly that birth.
#![allow(unsafe_code)]
use super::ServiceSpec;
use crate::{
    Error, Result,
    paths::Home,
    process::{self, Birth, Exact},
};
use std::{
    fs, io, iter,
    path::{Path, PathBuf},
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
    let data = wide(&process::command_line(&spec.program, &spec.args));
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
    let child = process::detach(&spec.program, &spec.args, spec.log.as_deref())?;
    record(dir, &spec.name, &Birth::of(child.pid())?)
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

pub(super) fn uninstall(_: &Path, name: &str) -> Result<bool> {
    // SAFETY: NUL-terminated wide strings.
    let status = unsafe {
        RegDeleteKeyValueW(
            HKEY_CURRENT_USER,
            wide(RUN).as_ptr(),
            wide(&value(name)).as_ptr(),
        )
    };
    if status == ERROR_FILE_NOT_FOUND {
        return Ok(false);
    }
    registry("delete", status).map(|()| true)
}

/// A detached process leaves its starter's console and job: nothing more is needed.
pub(super) fn scope(_: &ServiceSpec) -> Option<(PathBuf, Vec<std::ffi::OsString>)> {
    None
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
