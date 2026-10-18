//! The `cozy` binary this machine runs from (design §5). `cozy update` renames a new binary over
//! the launch path and keeps the one it replaces beside it as `.cozy-previous`; an UPDATE run then
//! activates it: the service drains and exits `EXEC_EXIT`, and the stable parent execs the launch
//! path with the same arguments. A binary that never proves readiness is renamed back. Windows
//! has no exec: its stable parent starts each service from the launch path, so the next service
//! is the activated (or restored) binary.
use cozy_platform::durable::FileId;
#[cfg(unix)]
use std::os::unix::process::CommandExt as _;
use std::{
    io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::OnceLock,
};

/// The service's exit status asking its parent to exec the binary at the launch path.
pub const EXEC_EXIT: i32 = 76;

/// Where this process was started from, and that file's identity then: a swap renames a new
/// file into place, after which `/proc/self/exe` names a deleted inode.
pub struct Launch {
    pub path: PathBuf,
    id: FileId,
}

static LAUNCH: OnceLock<Launch> = OnceLock::new();

/// Records the launch path, first thing at start.
pub fn record() -> io::Result<()> {
    let path = std::env::current_exe()?;
    let id = FileId::at(&path)?;
    let _ = LAUNCH.set(Launch { path, id });
    Ok(())
}

/// The launch, once recorded; `None` in a process that never started as a machine (tests).
pub fn launch() -> Option<&'static Launch> {
    LAUNCH.get()
}

impl Launch {
    /// Whether the file at the launch path is no longer the one this process runs.
    pub fn swapped(&self) -> bool {
        FileId::at(&self.path).is_ok_and(|id| id != self.id)
    }
}

/// The binary a rollback restores: the one `cozy update` replaced, beside the launch path.
pub fn previous(launch: &Path) -> PathBuf {
    launch.with_file_name(".cozy-previous")
}

/// The release a `cozy` binary says it is (`cozy 0.10.1`): it also proves the file runs here.
pub fn version(path: &Path) -> io::Result<String> {
    let out = Command::new(path).arg("--version").stdin(Stdio::null()).output()?;
    let text = String::from_utf8_lossy(&out.stdout);
    match (out.status.success(), text.trim().strip_prefix("cozy ")) {
        (true, Some(version)) if !version.is_empty() => Ok(version.to_owned()),
        _ => Err(io::Error::other(format!(
            "{} --version answered {:?} ({})",
            path.display(),
            text.trim(),
            out.status
        ))),
    }
}

/// Replaces this process with the binary at `path`, with this process's arguments; a rental's
/// provider credential is handed on as at its first start. Returns only on failure.
#[cfg(unix)]
pub fn exec(path: &Path, provider: Option<&super::provider::ProviderSelf>) -> io::Error {
    if provider.is_some_and(|own| !super::provider::hand_on(own)) {
        return io::Error::other("cannot place the provider credential");
    }
    let mut args = std::env::args_os();
    let mut command = Command::new(path);
    if let Some(name) = args.next() {
        command.arg0(name);
    }
    command.args(args).exec()
}
