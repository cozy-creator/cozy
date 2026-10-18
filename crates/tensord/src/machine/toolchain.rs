//! The machine's build toolchain on demand (xs-043 TC): CUDA's compiler, its headers and runtime
//! pieces, and ninja, as pip wheels in `<engine>/toolchain`, installed only when asked (`cozy
//! machine toolchain install`, an update run). Executors then find them through `CUDA_HOME` and
//! `PATH`, locations only. A host C/C++ compiler is the OS's: its absence is said with the command
//! that installs it. Images that bake a toolchain read as ready with nothing installed.
use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Mutex,
};

use cozy_platform::tools;
use cozy_proto::machine::{KernelVerdict, Toolchain};
use crate::stop::Stop;

use crate::process::Supervised;

/// CUDA's compiler, headers and runtime pieces for the CUDA the machine's torch runs (13.0), and
/// ninja.
const PACKAGES: [&str; 4] = [
    "nvidia-cuda-nvcc==13.0.*",
    "nvidia-cuda-cccl==13.0.*",
    "nvidia-cuda-crt==13.0.*",
    "ninja",
];

/// Where the toolchain goes: beside the machine's other software.
pub fn dir(engine: &Path) -> PathBuf {
    engine.join("toolchain")
}

/// CUDA's home inside the toolchain: the wheels' `nvidia/cu13`.
fn cuda(dir: &Path) -> PathBuf {
    dir.join("nvidia/cu13")
}

/// The executors' environment over `base`: CUDA_HOME and the toolchain's binaries first on PATH,
/// when the toolchain holds nvcc. `path` is the machine's own PATH, which `base` may not name.
pub fn environment(base: &BTreeMap<String, String>, dir: &Path, path: &str) -> BTreeMap<String, String> {
    let cuda = cuda(dir);
    if !cuda.join("bin/nvcc").is_file() {
        return base.clone();
    }
    let mut environment = base.clone();
    let rest = base.get("PATH").map(String::as_str).unwrap_or(path);
    let first = [cuda.join("bin"), dir.join("bin")].map(|p| p.display().to_string()).join(":");
    environment.insert("PATH".into(), [first.as_str(), rest].into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join(":"));
    environment.insert("CUDA_HOME".into(), cuda.display().to_string());
    environment
}

/// The toolchain as this machine finds it: its own install, else `CUDA_HOME`'s, `/usr/local/cuda`'s
/// or PATH's nvcc; PATH's host compiler and ninja. Runs each compiler once: call it at start and
/// after an install, not per Status.
pub fn probe(dir: &Path, cuda_home: Option<&Path>, path: &[PathBuf]) -> Toolchain {
    let mut nvcc_dirs = vec![cuda(dir).join("bin")];
    nvcc_dirs.extend(cuda_home.map(|home| home.join("bin")));
    nvcc_dirs.push(PathBuf::from("/usr/local/cuda/bin"));
    nvcc_dirs.extend(path.iter().cloned());
    let nvcc = tools::on_path("nvcc", &nvcc_dirs)
        .and_then(|found| last_line(&found.path, &["--version"]))
        .unwrap_or_default();
    let host = ["c++", "g++", "cc", "gcc"]
        .iter()
        .find_map(|name| tools::on_path(name, path))
        .and_then(|found| first_line(&found.path, &["--version"]))
        .unwrap_or_default();
    let mut ninja_dirs = vec![dir.join("bin")];
    ninja_dirs.extend(path.iter().cloned());
    let ninja = tools::on_path("ninja", &ninja_dirs).is_some();
    let missing: Vec<String> = [("nvcc", !nvcc.is_empty()), ("cc", !host.is_empty()), ("ninja", ninja)]
        .into_iter()
        .filter(|(_, found)| !found)
        .map(|(name, _)| name.into())
        .collect();
    Toolchain {
        state: match nvcc.is_empty() || host.is_empty() {
            true => "absent".into(),
            false => "ready".into(),
        },
        install: match host.is_empty() {
            true => tools::system_install("g++", path).unwrap_or_default(),
            false => String::new(),
        },
        nvcc,
        host_compiler: host,
        missing,
        detail: String::new(),
    }
}

/// Installs the toolchain's wheels with the machine's pinned `uv` into a staging directory, then
/// puts it in place whole. A stop ends it (Interrupted).
pub fn install(dir: &Path, uv: &Path, stop: &Stop) -> io::Result<()> {
    super::client::ensure_uv(uv)?;
    let staging = dir.with_extension("partial");
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging)?;
    let mut child = std::process::Command::new(uv)
        .args(["pip", "install", "--quiet", "--no-config", "--python", "3.12", "--target"])
        .arg(&staging)
        .args(PACKAGES)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stderr = child.stderr.take().expect("piped stderr");
    let (ended, reason) = std::thread::scope(|scope| {
        let drained = scope.spawn(move || {
            let mut text = String::new();
            let _ = io::Read::read_to_string(&mut stderr, &mut text);
            text
        });
        (crate::process::supervise(&mut child, Some(stop), "uv", &|_| ()), drained.join().unwrap_or_default())
    });
    match ended? {
        Supervised::Exited(status) if status.success() => {}
        Supervised::Canceled => return Err(io::Error::new(io::ErrorKind::Interrupted, "the toolchain install was canceled")),
        Supervised::Wedged(verdict) => return Err(io::Error::other(format!("uv pip install: {verdict}"))),
        Supervised::Exited(_) => return Err(io::Error::other(format!("uv pip install: {}", reason.trim()))),
    }
    if !cuda(&staging).join("bin/nvcc").is_file() {
        return Err(io::Error::other("the toolchain's wheels hold no nvcc"));
    }
    let _ = fs::remove_dir_all(dir);
    fs::rename(&staging, dir)
}

/// Each `verified` directory and when it last changed.
type Seen = Vec<(PathBuf, std::time::SystemTime)>;

/// Each kernel's verdict the Runtime kept under the machine's kernel store
/// (`<kernels>/<namespace>/verified/*.json`), read again only when a `verified` directory changed.
pub fn verdicts(kernels: &Path) -> Vec<KernelVerdict> {
    static CACHE: Mutex<Option<(Seen, Vec<KernelVerdict>)>> = Mutex::new(None);
    let dirs: Seen = fs::read_dir(kernels)
        .into_iter()
        .flatten()
        .flatten()
        .map(|namespace| namespace.path().join("verified"))
        .filter_map(|dir| Some((dir.clone(), fs::metadata(&dir).and_then(|m| m.modified()).ok()?)))
        .collect();
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, verdicts)) = cache.as_ref().filter(|(seen, _)| *seen == dirs) {
        return verdicts.clone();
    }
    let mut verdicts: Vec<KernelVerdict> = dirs
        .iter()
        .flat_map(|(dir, _)| fs::read_dir(dir).into_iter().flatten().flatten())
        .map(|entry| entry.path())
        // A dotfile is a verdict being written.
        .filter(|path| path.extension().is_some_and(|e| e == "json") && !path.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')))
        .filter_map(|path| serde_json::from_slice::<KernelVerdict>(&fs::read(path).ok()?).ok())
        .collect();
    verdicts.sort_by(|a, b| (&a.kernel, a.sm, &a.driver).cmp(&(&b.kernel, b.sm, &b.driver)));
    *cache = Some((dirs, verdicts.clone()));
    verdicts
}

fn first_line(program: &Path, args: &[&str]) -> Option<String> {
    lines(program, args)?.into_iter().next()
}

/// nvcc says its release last: "Cuda compilation tools, release 13.0, V13.0.88" is the line before.
fn last_line(program: &Path, args: &[&str]) -> Option<String> {
    let lines = lines(program, args)?;
    lines.iter().rev().find(|l| l.contains("release")).or(lines.last()).cloned()
}

fn lines(program: &Path, args: &[&str]) -> Option<Vec<String>> {
    let out = std::process::Command::new(program).args(args).stdin(Stdio::null()).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<String> = text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_owned).collect();
    (!lines.is_empty()).then_some(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executors_find_an_installed_toolchain_first_and_nothing_else_changes() {
        let dir = tempfile::tempdir().unwrap();
        let base = BTreeMap::from([("LANG".to_string(), "C.UTF-8".to_string())]);
        assert_eq!(environment(&base, dir.path(), "/usr/bin"), base, "no toolchain: unchanged");
        let bin = cuda(dir.path()).join("bin");
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("nvcc"), b"").unwrap();
        let env = environment(&base, dir.path(), "/usr/bin:/bin");
        assert_eq!(env["CUDA_HOME"], cuda(dir.path()).display().to_string());
        assert_eq!(env["PATH"], format!("{}:{}:/usr/bin:/bin", bin.display(), dir.path().join("bin").display()));
        assert_eq!(env["LANG"], "C.UTF-8");
    }

    #[test]
    fn verdicts_are_read_from_each_namespace_and_a_write_in_flight_is_not() {
        let kernels = tempfile::tempdir().unwrap();
        let verified = kernels.path().join("u1000/verified");
        fs::create_dir_all(&verified).unwrap();
        let row = serde_json::json!({"kernel": "prebuilt-int8-sm89", "artifact": "a", "sm": 89, "driver": "580.65.06", "ok": true, "detail": "rel L2 1.2e-3"});
        fs::write(verified.join("prebuilt-int8-sm89-0f.json"), row.to_string()).unwrap();
        fs::write(verified.join(".prebuilt-fp8-sm89-1a.json.42"), "{").unwrap();
        let read = verdicts(kernels.path());
        assert_eq!(read.len(), 1);
        assert!(read[0].ok && read[0].sm == 89 && read[0].kernel == "prebuilt-int8-sm89");
    }
}
