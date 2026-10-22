//! A local machine with no selected/baked SDK resolves one pair on its first install. This is
//! a shared cached resolution, never an SDK activation or a change to sdk/current.
use super::{compile_sdk, identified, verify_wheel, Coded};
use crate::{origin::Machine, stop::Stop};
use sha2::{Digest, Sha256};
use std::{fs, io, path::{Path, PathBuf}, sync::{Arc, Mutex, OnceLock}};

pub struct DefaultSdk {
    root: PathBuf,
    uv: PathBuf,
    machine: Arc<Machine>,
    resolving: Mutex<()>,
    chosen: OnceLock<Resolved>,
}

pub struct Resolved {
    pub wheels: Vec<PathBuf>,
    pub requirements: Vec<String>,
}

impl std::fmt::Debug for DefaultSdk {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("DefaultSdk").field("root", &self.root).field("resolved", &self.chosen.get().is_some()).finish()
    }
}

impl DefaultSdk {
    pub fn new(root: PathBuf, uv: PathBuf, machine: Arc<Machine>) -> Arc<Self> {
        Arc::new(Self { root, uv, machine, resolving: Mutex::new(()), chosen: OnceLock::new() })
    }

    pub fn resolved(&self) -> Option<&Resolved> { self.chosen.get() }

    pub fn resolve(&self, stop: Option<&Stop>) -> io::Result<&Resolved> {
        let stop = stop.cloned().unwrap_or_default();
        self.resolve_with(|directory| {
            let solution = compile_sdk(&self.uv, "cozy-runtime>=0.31.0\ntensorfs>=0.14.0\n", &stop, true)?
                .map_err(|why| io::Error::other(Coded { code: "package_sdk_incompatible", message: why }))?;
            artifacts(&solution)?.into_iter().map(|wheel| {
                // uv selected the host tags. The existing downloader checks its exact digest;
                // no Linux-specific wheel heuristic or platform fallback enters this path.
                super::super::pypi::wheel_at(&wheel.url, &wheel.hashes.sha256, &wheel.filename()?, directory, &self.machine, &stop)
            }).collect()
        })
    }

    fn resolve_with(&self, resolve: impl FnOnce(&Path) -> io::Result<Vec<PathBuf>>) -> io::Result<&Resolved> {
        if let Some(chosen) = self.chosen.get() { return Ok(chosen); }
        let _one = self.resolving.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.chosen.get().is_none() {
            fs::create_dir_all(&self.root)?;
            let pending = tempfile::Builder::new().prefix("resolve-").tempdir_in(&self.root)?;
            let wheels = resolve(pending.path())?;
            if wheels.len() != 2 { return Err(io::Error::other("SDK resolution returned an incomplete pair")); }
            let mut digest = Sha256::new();
            let mut names = vec![];
            for (wheel, distribution) in wheels.iter().zip(["cozy_runtime", "tensorfs"]) {
                verify_wheel(wheel, distribution)?;
                let name = wheel.file_name().ok_or_else(|| io::Error::other("SDK wheel has no name"))?;
                digest.update(name.as_encoded_bytes());
                digest.update(file_digest(wheel)?);
                names.push(name.to_os_string());
            }
            identified(pending.path())?.require_memory_protocol()?;
            let digest = super::hex(&digest.finalize());
            let immutable = self.root.join(&digest);
            if !immutable.exists() {
                platform::durable::sync_dir(pending.path())?;
                match fs::rename(pending.path(), &immutable) {
                    Ok(()) => platform::durable::sync_dir(&self.root)?,
                    Err(_) if immutable.is_dir() => {}, // Another resolver published these exact bytes.
                    Err(error) => return Err(error),
                }
            }
            let immutable = platform::durable::canonical(&immutable)?;
            let wheels: Vec<_> = names.into_iter().map(|name| immutable.join(name)).collect();
            let mut held = Sha256::new();
            for wheel in &wheels {
                held.update(wheel.file_name().ok_or_else(|| io::Error::other("SDK wheel name missing"))?.as_encoded_bytes());
                held.update(file_digest(wheel)?);
            }
            if super::hex(&held.finalize()) != digest {
                return Err(io::Error::other("cached immutable SDK bytes changed"));
            }
            // The installer receives the same immutable filenames included in its identity.
            let requirements = wheels.iter().map(|path| path.to_string_lossy().into_owned()).collect();
            let _ = self.chosen.set(Resolved { wheels, requirements });
        }
        self.chosen.get().ok_or_else(|| io::Error::other("SDK resolution was not published"))
    }
}


fn file_digest(path: &Path) -> io::Result<Vec<u8>> {
    let mut digest = Sha256::new();
    io::copy(&mut fs::File::open(path)?, &mut digest)?;
    Ok(digest.finalize().to_vec())
}

#[derive(serde::Deserialize)]
struct Artifact {
    url: String,
    hashes: Hashes,
}
#[derive(serde::Deserialize)]
struct Hashes { sha256: String }

impl Artifact {
    fn filename(&self) -> io::Result<String> {
        let file = self.url.rsplit('/').next().unwrap_or_default();
        if !self.url.starts_with("https://") || !file.ends_with(".whl")
            || !file.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-+".contains(&c))
            || self.hashes.sha256.len() != 64 || !self.hashes.sha256.bytes().all(|c| c.is_ascii_hexdigit())
        { return Err(io::Error::other("SDK resolution returned no verified wheel URL")); }
        Ok(file.into())
    }
}

/// A non-universal uv pylock resolution has one selected host wheel per SDK distribution.
fn artifacts(solution: &str) -> io::Result<Vec<Artifact>> {
    #[derive(serde::Deserialize)]
    struct Package { name: String, version: String, #[serde(default)] wheels: Vec<Artifact> }
    #[derive(serde::Deserialize)]
    struct Plan { packages: Vec<Package> }
    let mut plan: Plan = toml::from_str(solution).map_err(io::Error::other)?;
    let mut chosen = vec![];
    let mut pair = super::Pair::default();
    for name in ["cozy-runtime", "tensorfs"] {
        let index = plan.packages.iter().position(|p| p.name.replace('_', "-") == name)
            .ok_or_else(|| io::Error::other(format!("SDK resolution returned no {name}")))?;
        let package = plan.packages.remove(index);
        if package.wheels.len() != 1 || plan.packages.iter().any(|p| p.name.replace('_', "-") == name) {
            return Err(io::Error::other(format!("SDK resolution returned ambiguous host wheels for {name}")));
        }
        if name == "cozy-runtime" { pair.runtime = package.version; } else { pair.tensorfs = package.version; }
        let wheel = package.wheels.into_iter().next().ok_or_else(|| io::Error::other("SDK wheel absent"))?;
        wheel.filename()?;
        chosen.push(wheel);
    }
    pair.require_memory_protocol()?;
    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fixture(directory: &Path, runtime: &str, tensorfs: &str) -> io::Result<Vec<PathBuf>> {
        [("cozy_runtime", runtime), ("tensorfs", tensorfs)].into_iter().map(|(name, version)| {
            let path = directory.join(format!("{name}-{version}-py3-none-any.whl"));
            let mut wheel = zip::ZipWriter::new(fs::File::create(&path)?);
            wheel.start_file(format!("{name}-{version}.dist-info/METADATA"), zip::write::SimpleFileOptions::default())?;
            write!(wheel, "Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")?;
            wheel.finish()?;
            Ok(path)
        }).collect()
    }

    fn resolver(root: &Path) -> Arc<DefaultSdk> {
        let store = crate::storage::open(&root.join("store")).unwrap();
        DefaultSdk::new(root.join("sdk/default"), "uv".into(), Machine::new(store, None).unwrap())
    }

    #[test]
    fn first_install_resolves_once_for_concurrent_consumers_into_stable_real_paths() {
        let root = tempfile::tempdir().unwrap();
        let default = resolver(root.path());
        assert!(!default.root.exists(), "constructing an idle machine must not download or select anything");
        let calls = AtomicUsize::new(0);
        let paths = std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..4).map(|_| scope.spawn(|| {
                default.resolve_with(|dir| { calls.fetch_add(1, Ordering::SeqCst); fixture(dir, "0.31.2", "0.14.3") }).unwrap().wheels.clone()
            })).collect();
            jobs.into_iter().map(|job| job.join().unwrap()).collect::<Vec<_>>()
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(paths.iter().all(|p| *p == paths[0] && p.iter().all(|w| w.is_file())));
        let sdk = crate::published::PackageSdk { default_sdk: Some(default.clone()), ..Default::default() };
        assert_eq!(sdk.requirements(), default.resolved().unwrap().requirements);
        assert!(!root.path().join("sdk/current").exists());
        let again = resolver(root.path());
        assert_eq!(again.resolve_with(|dir| fixture(dir, "0.31.2", "0.14.3")).unwrap().wheels, paths[0]);
    }

    #[test]
    fn failed_or_incompatible_resolutions_never_become_a_cached_selection() {
        let root = tempfile::tempdir().unwrap();
        let default = resolver(root.path());
        assert!(default.resolve_with(|_| Err(io::Error::other("download failed"))).is_err());
        assert!(default.resolved().is_none());
        assert!(default.resolve_with(|dir| fixture(dir, "0.30.3", "0.13.4")).is_err());
        assert!(default.resolved().is_none());
        assert_eq!(fs::read_dir(&default.root).unwrap().count(), 0, "failed unique candidates were removed");
        assert_eq!(default.resolve_with(|dir| fixture(dir, "0.31.0", "0.14.0")).unwrap().wheels.len(), 2);
    }

    #[test]
    fn changed_bytes_under_a_content_identity_refuse_without_rewriting_them() {
        let root = tempfile::tempdir().unwrap();
        let default = resolver(root.path());
        let path = default.resolve_with(|dir| fixture(dir, "0.31.0", "0.14.0")).unwrap().wheels[0].clone();
        fs::write(&path, b"changed").unwrap();
        let restarted = resolver(root.path());
        assert!(restarted.resolve_with(|dir| fixture(dir, "0.31.0", "0.14.0")).is_err());
        assert_eq!(fs::read(path).unwrap(), b"changed");
        assert!(restarted.resolved().is_none());
    }

    #[test]
    fn empty_and_broken_explicit_or_baked_selections_are_never_fresh() {
        let root = tempfile::tempdir().unwrap();
        let engine = root.path().join("var/lib/cozy/tensord");
        let paths = super::super::Paths::new(&engine, root.path());
        assert!(paths.unselected_sdk());
        super::super::check_startup_pair(&paths).unwrap();
        let selected = engine.join("sdk/current");
        fs::create_dir_all(&selected).unwrap();
        assert!(!paths.unselected_sdk());
        assert!(super::super::check_startup_pair(&paths).is_err());
        fs::remove_dir(&selected).unwrap();
        #[cfg(unix)] {
            std::os::unix::fs::symlink("missing", &selected).unwrap();
            assert!(!paths.unselected_sdk());
            assert!(super::super::check_startup_pair(&paths).is_err());
            fs::remove_file(&selected).unwrap();
        }
        let baked = root.path().join("opt/cozy/machine/wheels");
        fs::create_dir_all(&baked).unwrap();
        let paths = super::super::Paths::new(&engine, root.path());
        assert!(!paths.unselected_sdk());
        assert!(super::super::check_startup_pair(&paths).is_err());
        fs::remove_dir(&baked).unwrap();
        #[cfg(unix)] {
            std::os::unix::fs::symlink("missing", &baked).unwrap();
            let paths = super::super::Paths::new(&engine, root.path());
            assert!(!paths.unselected_sdk());
            assert!(super::super::check_startup_pair(&paths).is_err());
        }
    }

    #[test]
    fn a_host_plan_requires_both_verified_wheels_and_preserves_the_solvers_platform() {
        let plan = |platform: &str| format!(r#"
lock-version = "1.0"
[[packages]]
name = "cozy-runtime"
version = "0.31.0"
wheels = [{{url="https://example.test/cozy_runtime-0.31.0-py3-none-any.whl", hashes={{sha256="{hash}"}}}}]
[[packages]]
name = "tensorfs"
version = "0.14.0"
wheels = [{{url="https://example.test/tensorfs-0.14.0-cp312-abi3-{platform}.whl", hashes={{sha256="{hash}"}}}}]
"#, hash="a".repeat(64));
        for platform in ["manylinux_2_28_x86_64", "macosx_11_0_arm64", "win_amd64"] {
            let wheels = artifacts(&plan(platform)).unwrap();
            assert_eq!(wheels.len(), 2);
            assert!(wheels[1].filename().unwrap().ends_with(&format!("{platform}.whl")));
        }
        for bad in [plan("win_amd64").replace("0.14.0", "0.13.4"),
            plan("win_amd64").replace("https://example.test/", "file:///"),
            plan("win_amd64").replace(&"a".repeat(64), "missing")]
        { assert!(artifacts(&bad).is_err()); }
    }

}
