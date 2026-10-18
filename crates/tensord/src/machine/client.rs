//! The machine's Python client, embedded at build time (build.rs packs python/ as a wheel).
//! Package environments and the installer helper take it from the binary, so an image or a
//! local install ships only the machine, uv and the executor SDK wheels.
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Mutex, PoisonError},
};

const WHEEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/client.whl"));
const WHEEL_NAME: &str = env!("COZY_CLIENT_WHEEL_NAME");

/// The uv every machine runs when its image brought none: one version, fetched on first need.
pub const UV: &str = "0.12.24";
/// That release's archive for each OS and CPU, and its sha256 (cozy_platform::tools' pins):
/// a tar.gz holding `<archive>/uv`, or on Windows a zip holding `uv.exe`.
const UV_ARCHIVES: [(&str, &str, &str, &str); 6] = [
    ("linux", "x86_64", "uv-x86_64-unknown-linux-gnu", "b4dfaef47d491a7296981f8374a4595f55dbf84e8937c8ecd2983574d8bb3da6"),
    ("linux", "aarch64", "uv-aarch64-unknown-linux-gnu", "5231be65f496304623895dacdbf1de8504fec90303684bdf05805aa34414dd21"),
    ("macos", "aarch64", "uv-aarch64-apple-darwin", "0c4346de7abdb49495b393b9ec809fe387aa43e586be20fecb972216c1e71732"),
    ("macos", "x86_64", "uv-x86_64-apple-darwin", "4fa82e37cb94767661f532b001e470b67a186c7260e305bd84ddb78fd545c0b6"),
    ("windows", "x86_64", "uv-x86_64-pc-windows-msvc", "7c38608c8a18ee137d748a1773053b07ec8f3a30fab49aebaa6f4e4efeceb019"),
    ("windows", "aarch64", "uv-aarch64-pc-windows-msvc", "4b783bda5cc44bbae0651a837223873a7acee31152381aef5fc86ef6bef25997"),
];

/// The root's own uv: an image's, or the pinned one [`ensure_uv`] fetches there.
pub fn uv(root: &Path) -> PathBuf {
    root.join("usr/local/bin").join(cozy_platform::paths::exe("uv"))
}

/// Fetches the pinned uv to `uv` when it is absent, verified by its sha256. A bare name is the
/// PATH's uv (the development serve mode) and is left alone.
pub fn ensure_uv(uv: &Path) -> io::Result<()> {
    static FETCHING: Mutex<()> = Mutex::new(());
    if !uv.is_absolute() || uv.exists() {
        return Ok(());
    }
    let _fetching = FETCHING.lock().unwrap_or_else(PoisonError::into_inner);
    if uv.exists() {
        return Ok(());
    }
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    let (.., name, sha256) = UV_ARCHIVES
        .iter()
        .find(|(o, a, ..)| (*o, *a) == (os, arch))
        .ok_or_else(|| io::Error::other(format!("uv {UV} has no build for {os}/{arch}")))?;
    let windows = os == "windows";
    let url = format!(
        "https://github.com/astral-sh/uv/releases/download/{UV}/{name}.{}",
        if windows { "zip" } else { "tar.gz" }
    );
    tracing::info!("fetching uv {UV} to {}", uv.display());
    // Its own thread and runtime: callers may be on a runtime's blocking threads.
    let fetched = std::thread::scope(|scope| {
        scope
            .spawn(|| -> io::Result<Vec<u8>> {
                let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
                runtime.block_on(async {
                    let response = reqwest::get(&url).await.and_then(|r| r.error_for_status());
                    let bytes = response.map_err(io::Error::other)?.bytes().await.map_err(io::Error::other)?;
                    Ok(bytes.to_vec())
                })
            })
            .join()
            .map_err(|_| io::Error::other("the uv fetch panicked"))?
    })?;
    if format!("{:x}", Sha256::digest(&fetched)) != *sha256 {
        return Err(io::Error::other(format!("{url} does not match its pinned sha256")));
    }
    let mut bytes = vec![];
    if windows {
        let mut archive = zip::ZipArchive::new(io::Cursor::new(&fetched[..])).map_err(io::Error::other)?;
        let mut member = archive.by_name("uv.exe").map_err(|_| io::Error::other(format!("{url} holds no uv")))?;
        member.read_to_end(&mut bytes)?;
    } else {
        let wanted = Path::new(name).join("uv");
        let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(&fetched[..]));
        let mut entry = archive
            .entries()?
            .find(|entry| entry.as_ref().is_ok_and(|e| e.path().is_ok_and(|p| p == wanted)))
            .ok_or_else(|| io::Error::other(format!("{url} holds no uv")))??;
        entry.read_to_end(&mut bytes)?;
    }
    fs::create_dir_all(uv.parent().expect("a file in a directory"))?;
    let fetching = uv.with_extension("fetching");
    super::identity::write_atomic(&fetching, &bytes, 0o755)?;
    settle(&fetching)?;
    fs::rename(&fetching, uv)
}

/// Waits until the executable just written at `path` runs. A child another thread forked while
/// its file was open for writing holds that descriptor until the child execs, and any exec of
/// the file meanwhile fails "Text file busy" (ETXTBSY). Once it runs, no writer is left, so `uv`
/// appears under its name only then and no later exec of it meets that.
fn settle(path: &Path) -> io::Result<()> {
    loop {
        match Command::new(path).arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status() {
            Err(error) if error.raw_os_error() == Some(libc::ETXTBSY) => {
                std::thread::sleep(std::time::Duration::from_millis(2))
            }
            Err(error) => return Err(error),
            Ok(_) => return Ok(()),
        }
    }
}

/// Publish the embedded wheel under its content identity. A new build with the same
/// distribution version must not overwrite an earlier installer's selected input.
pub fn wheel(engine: &Path) -> io::Result<PathBuf> {
    let digest = format!("{:x}", Sha256::digest(WHEEL));
    let path = engine.join("client").join(digest).join(WHEEL_NAME);
    if fs::read(&path).ok().as_deref() != Some(WHEEL) {
        fs::create_dir_all(path.parent().expect("embedded wheel directory"))?;
        super::identity::write_atomic(&path, WHEEL, 0o644)?;
    }
    Ok(path)
}

/// The installer helper's interpreter under `engine`.
pub fn helper_python(engine: &Path) -> PathBuf {
    cozy_platform::paths::venv_python(&engine.join("helper"))
}

/// The installer helper: an environment over Python 3.12 holding the client with its installer
/// extra, made with uv at `engine/helper` and made again when the embedded client changes.
pub fn helper(engine: &Path, uv: &Path, wheel: &Path) -> io::Result<PathBuf> {
    static MAKING: Mutex<()> = Mutex::new(());
    let _making = MAKING.lock().unwrap_or_else(PoisonError::into_inner);
    ensure_uv(uv)?;
    let dir = engine.join("helper");
    let (python, marker) = (helper_python(engine), dir.join(".client-sha256"));
    let digest = format!("{:x}", Sha256::digest(WHEEL));
    if python.exists() && fs::read_to_string(&marker).ok().as_deref() == Some(digest.as_str()) {
        return Ok(python);
    }
    if dir.exists() {
        fs::remove_dir_all(&dir)?;
    }
    let run = |command: &mut Command| -> io::Result<()> {
        let output = command.output()?;
        if output.status.success() {
            return Ok(());
        }
        Err(io::Error::other(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ))
    };
    run(Command::new(uv)
        .args(["venv", "--no-project", "--no-config", "--python", "3.12"])
        .arg(&dir))?;
    run(Command::new(uv)
        .args(["pip", "install", "--no-config", "--python"])
        .arg(&python)
        .arg(format!("{}[installer]", wheel.display())))?;
    fs::write(&marker, digest)?;
    Ok(python)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cozy_platform::durable::FileId;

    /// A root with no uv gets the pinned one, verified, once; a bare name is the PATH's.
    #[test]
    fn a_root_without_uv_fetches_the_pinned_one_once() {
        let root = tempfile::tempdir().unwrap();
        let path = uv(root.path());
        ensure_uv(Path::new("uv")).unwrap();
        ensure_uv(&path).unwrap();
        let version = Command::new(&path).arg("--version").output().unwrap();
        assert!(String::from_utf8_lossy(&version.stdout).starts_with(&format!("uv {UV}")), "{version:?}");
        let fetched = FileId::at(&path).unwrap();
        ensure_uv(&path).unwrap();
        assert_eq!(FileId::at(&path).unwrap(), fetched, "fetched again");
    }

    #[test]
    fn embedded_client_selection_preserves_older_same_version_bytes() {
        let engine = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/client-selection")
            .join(uuid::Uuid::new_v4().to_string());
        let old = engine.join("client").join(WHEEL_NAME);
        fs::create_dir_all(old.parent().unwrap()).unwrap();
        fs::write(&old, b"previous embedded client build").unwrap();
        let selected = wheel(&engine).unwrap();
        assert!(
            fs::read(&old).unwrap() == b"previous embedded client build",
            "selecting a new build overwrote a prior selected wheel"
        );
        assert_ne!(selected, old);
        assert!(fs::read(&selected).unwrap() == WHEEL);
        assert_eq!(
            selected
                .parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap(),
            format!("{:x}", Sha256::digest(WHEEL))
        );
        let stamp = |path: &Path| (FileId::at(path).unwrap(), fs::metadata(path).unwrap().modified().unwrap());
        let before = stamp(&selected);
        assert_eq!(wheel(&engine).unwrap(), selected);
        assert_eq!(before, stamp(&selected));
        fs::remove_dir_all(engine).unwrap();
    }
}
