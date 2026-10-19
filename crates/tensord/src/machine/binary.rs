//! The `cozy` binary this machine runs from (design §5). `cozy update` renames a new binary over
//! the launch path and keeps the one it replaces beside it as `.cozy-previous`; an UPDATE run then
//! activates it: the service drains and exits `EXEC_EXIT`, and the stable parent execs the launch
//! path with the same arguments. A binary that never proves readiness is renamed back. Windows
//! has no exec: its stable parent starts each service from the launch path, so the next service
//! is the activated (or restored) binary.
use platform::durable::FileId;
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

/// The release a dev binary (`cozy machine update --tensord`) first replaced, beside the launch path.
pub fn release(launch: &Path) -> PathBuf {
    launch.with_file_name(".cozy-release")
}

/// Keeps the binary at the launch path for a rollback, and as the release when a dev binary
/// first replaces it.
pub fn keep(launch: &Path, release: bool) -> io::Result<()> {
    let copies = [Some(previous(launch)), release.then(|| self::release(launch))];
    for copy in copies.into_iter().flatten() {
        match std::fs::remove_file(&copy) {
            Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
            _ => {}
        }
        std::fs::hard_link(launch, &copy).or_else(|_| std::fs::copy(launch, &copy).map(drop))?;
    }
    Ok(())
}

/// Puts `staged` at the launch path, which this process runs from.
pub fn swap(staged: &Path) -> io::Result<()> {
    self_replace::self_replace(staged)
}

/// The release `path` says it is, once this machine can run it: its header's OS and arch are
/// this machine's, its loader finds the glibc it needs, and it answers `--version`. Err names
/// the mismatch.
pub fn check(path: &Path) -> Result<String, String> {
    use std::io::Read as _;
    let mut head = vec![0u8; 512];
    let n = std::fs::File::open(path)
        .and_then(|mut file| file.read(&mut head))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let here = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
    match built_for(&head[..n]) {
        None => return Err("not an executable: no ELF, Mach-O or PE header".into()),
        Some(built) if built != here => return Err(format!("built for {built}, this machine is {here}")),
        Some(_) => {}
    }
    version(path).map_err(|e| match glibc_needed(path) {
        Some((need, have)) => format!("needs glibc {need}, this machine has {have}"),
        None => format!("does not answer --version: {e}"),
    })
}

/// `<os> <arch>` from an executable's header, as `std::env::consts` spells them.
fn built_for(head: &[u8]) -> Option<String> {
    let u16le = |at: usize| head.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    let u32le = |at: usize| head.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    let (os, arch) = if head.starts_with(b"\x7fELF") {
        ("linux", u32::from(u16le(18)?))
    } else if head.starts_with(&[0xcf, 0xfa, 0xed, 0xfe]) {
        ("macos", u32le(4)?)
    } else if head.starts_with(b"MZ") {
        ("windows", u32::from(u16le(u32le(0x3c)? as usize + 4)?))
    } else {
        return None;
    };
    let arch = match (os, arch) {
        ("linux", 0x3e) | ("macos", 0x0100_0007) | ("windows", 0x8664) => "x86_64".into(),
        ("linux", 0xb7) | ("macos", 0x0100_000c) | ("windows", 0xaa64) => "aarch64".into(),
        (_, other) => format!("machine type {other:#x}"),
    };
    Some(format!("{os} {arch}"))
}

/// The newest glibc `path` asks for and this machine's, when this machine's is older.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn glibc_needed(path: &Path) -> Option<(String, String)> {
    let parse = |text: &[u8]| -> Option<(u32, u32)> {
        let text = std::str::from_utf8(text).ok()?;
        let (major, minor) = text.split_once('.')?;
        Some((major.parse().ok()?, minor.parse().ok()?))
    };
    let bytes = std::fs::read(path).ok()?;
    let need = bytes
        .windows(6)
        .enumerate()
        .filter(|(_, window)| *window == b"GLIBC_")
        .filter_map(|(at, _)| {
            let tail = &bytes[at + 6..];
            let end = tail.iter().position(|b| !(b.is_ascii_digit() || *b == b'.')).unwrap_or(tail.len());
            parse(&tail[..end])
        })
        .max()?;
    // SAFETY: glibc returns a static NUL-terminated string.
    let have = unsafe { std::ffi::CStr::from_ptr(libc::gnu_get_libc_version()) }.to_string_lossy().into_owned();
    (parse(have.as_bytes())? < need).then(|| (format!("{}.{}", need.0, need.1), have))
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn glibc_needed(_: &Path) -> Option<(String, String)> {
    None
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
