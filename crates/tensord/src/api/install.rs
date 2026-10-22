//! Materialize verified package carriers and call the trusted Python/uv installer.
//! The service commits actor/installation aliases in its one execution journal.
use super::workspaces::{RootSet, UploadedPackage};
use platform::durable::Mode;
use fs2::FileExt;
use serde::Deserialize;
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
};
use crate::stop::Stop;
use protocol::error::Error;
use crate::triage::masked;

#[derive(Clone)]
pub struct InstallerConfig {
    pub helper_python: PathBuf,
    /// The engine whose helper `helper_python` is, made at the first install that needs it.
    pub helper_engine: Option<PathBuf>,
    pub python: String,
    pub generations: PathBuf,
    pub client_wheel: PathBuf,
    pub staging_root: PathBuf,
    /// The machine's own Runtime/TensorFS wheels: a local package runs this pair too.
    pub sdk: Vec<PathBuf>,
    pub default_sdk: Option<Arc<crate::machine::update::DefaultSdk>>,
    /// The development Runtime an update named (`--runtime-wheel`, `PackageSdk::runtime_build`):
    /// it runs a local package whatever the package's own bounds on the pair, as a release.
    pub runtime_build: String,
    /// The machine's uv (`machine::client::uv`): the helper runs it, whatever PATH the machine had.
    pub uv: PathBuf,
    /// An image's baked uv cache: local environments symlink into it as published ones do, so
    /// overlayfs never copies its payload (Torch) into each environment of each edit.
    pub seed_cache: Option<PathBuf>,
}
impl InstallerConfig {
    pub fn resolve_sdk(&self, stop: Option<&Stop>) -> Result<(), Error> {
        if self.sdk.is_empty() {
            if let Some(default) = &self.default_sdk {
                default.resolve(stop).map_err(|e| Error::unavailable("package_sdk_unavailable", e.to_string()))?;
            }
        }
        Ok(())
    }

    pub fn sdk(&self) -> &[PathBuf] {
        self.default_sdk.as_ref().filter(|_| self.sdk.is_empty())
            .and_then(|default| default.resolved()).map_or(&self.sdk, |chosen| &chosen.wheels)
    }
}

#[derive(Clone, Debug, Deserialize, serde::Serialize)]
pub struct Dependency {
    pub name: String,
    pub version: String,
}
#[derive(Debug, Deserialize, serde::Serialize)]
pub struct GenerationRecord {
    pub identity: String,
    pub package: String,
    pub version: String,
    pub application: String,
    pub python: PathBuf,
    #[serde(default)]
    pub dependencies: Vec<Dependency>,
    pub interface: Box<serde_json::value::RawValue>,
}
pub struct PreparedGeneration {
    pub record: GenerationRecord,
    pub interface_bytes: Vec<u8>,
    pub hold: Arc<File>,
}
pub struct MaterializedPackage {
    pub root_set: RootSet,
    pub project: Option<PathBuf>,
    pub wheels: Vec<PathBuf>,
    pub requirements: Option<PathBuf>,
    root: PathBuf,
    _hold: File,
}
impl Drop for MaterializedPackage {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

pub fn materialize_uploaded(
    uploaded: &UploadedPackage,
    staging: &Path,
) -> Result<MaterializedPackage, Error> {
    fs::create_dir_all(staging).map_err(storage)?;
    let root = staging.join(uuid::Uuid::new_v4().to_string());
    fs::create_dir(&root).map_err(storage)?;
    platform::durable::set_mode(&root, 0o700).map_err(std::io::Error::from).map_err(storage)?;
    let hold = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(root.join("active.lock"))
        .map_err(storage)?;
    hold.lock_exclusive().map_err(storage)?;
    let mut materialized = MaterializedPackage {
        root_set: uploaded.root.clone(),
        project: None,
        wheels: vec![],
        requirements: None,
        root,
        _hold: hold,
    };
    for carrier in &uploaded.files {
        let file = carrier.open()?;
        if carrier.filename == uploaded.root.source_archive {
            let project = materialized.root.join("source");
            fs::create_dir(&project).map_err(storage)?;
            let mut archive = tar::Archive::new(file);
            for entry in archive.entries().map_err(storage)? {
                let mut entry = entry.map_err(storage)?;
                let path = entry.path().map_err(storage)?.into_owned();
                if path.is_absolute()
                    || path
                        .components()
                        .any(|c| !matches!(c, std::path::Component::Normal(_)))
                    || !(entry.header().entry_type().is_file()
                        || entry.header().entry_type().is_dir())
                {
                    return Err(Error::conflict("precondition_failed",
                        "unsafe source archive member during materialization",
                    ));
                }
                let target = project.join(&path);
                if entry.header().entry_type().is_dir() {
                    fs::create_dir_all(target).map_err(storage)?;
                    continue;
                }
                fs::create_dir_all(target.parent().expect("source member parent"))
                    .map_err(storage)?;
                let mut output = platform::durable::open_nofollow(&target, OpenOptions::new().create_new(true).write(true).mode(0o600)).map_err(std::io::Error::from)
                    .map_err(storage)?;
                std::io::copy(&mut entry, &mut output).map_err(storage)?;
                output.sync_all().map_err(storage)?;
                platform::durable::set_mode(&target, entry.header().mode().map_err(storage)? & 0o777).map_err(std::io::Error::from)
                .map_err(storage)?;
            }
            materialized.project = Some(project);
        } else {
            let path = materialized.root.join(&carrier.filename);
            let mut output = platform::durable::open_nofollow(&path, OpenOptions::new().create_new(true).write(true).mode(0o600)).map_err(std::io::Error::from)
                .map_err(storage)?;
            std::io::copy(&mut { file }, &mut output).map_err(storage)?;
            output.sync_all().map_err(storage)?;
            materialized.wheels.push(path);
        }
    }
    if !uploaded.root.dependency_requirements.is_empty() {
        let path = materialized.root.join("requirements.txt");
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&path)
            .map_err(storage)?;
        file.write_all(&uploaded.root.dependency_requirements)
            .map_err(storage)?;
        file.sync_all().map_err(storage)?;
        materialized.requirements = Some(path);
    }
    Ok(materialized)
}

pub fn prepare_uploaded(
    config: &InstallerConfig,
    uploaded: &UploadedPackage,
    stop: Option<&Stop>,
) -> Result<PreparedGeneration, Error> {
    config.resolve_sdk(stop)?;
    let materialized = materialize_uploaded(uploaded, &config.staging_root)?;
    let mut command = installer(config, &materialized.root_set)?;
    if let Some(project) = &materialized.project {
        command.arg("--project").arg(project);
    }
    for wheel in &materialized.wheels {
        command.arg("--wheel").arg(wheel);
    }
    if let Some(requirements) = &materialized.requirements {
        command.arg("--requirements").arg(requirements);
    }
    publish(config, command, stop, None)
}

/// A live project installed editable in `environment`, which is built when it is not there
/// yet: the generation it publishes describes the project now and runs in that environment.
#[allow(clippy::too_many_arguments)]
pub fn prepare_editable(
    config: &InstallerConfig,
    root: &RootSet,
    project: &Path,
    environment: &Path,
    wheels: &[PathBuf],
    indexes: &std::collections::BTreeMap<String, String>,
    source_digest: &str,
    stop: Option<&Stop>,
) -> Result<PreparedGeneration, Error> {
    // A machine killed mid-install leaves its helper building here (its own process group
    // outlives it): that helper ends before another builds the same environment.
    config.resolve_sdk(stop)?;
    let claim = environment.with_extension("installer");
    end_stale(&claim).map_err(storage)?;
    let mut command = installer(config, root)?;
    command.arg("--project").arg(project).arg("--editable-environment").arg(environment);
    command.arg("--source-digest").arg(source_digest);
    for (name, url) in indexes {
        command.arg("--index").arg(format!("{name}={url}"));
    }
    for wheel in wheels {
        command.arg("--wheel").arg(wheel);
    }
    publish(config, command, stop, Some(&claim))
}

/// Who builds at a claim: the helper, and the machine that started it.
type Claim = (platform::process::Birth, platform::process::Birth);

/// Ends the helper an earlier machine left building at `claim`, once its exit is seen.
fn end_stale(claim: &Path) -> std::io::Result<()> {
    let recorded = fs::read(claim).ok().and_then(|bytes| serde_json::from_slice::<Claim>(&bytes).ok());
    if let Some((helper, _)) = recorded.filter(|(_, machine)| platform::process::Birth::own().ok().as_ref() != Some(machine)) {
        if let Some(exact) = platform::process::Exact::open(&helper)? {
            exact.kill()?;
            exact.wait()?;
        }
    }
    match fs::remove_file(claim) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
        _ => Ok(()),
    }
}

/// The trusted helper's command for one package: this machine's interpreter, SDK pair,
/// description client and generations, and what the package declares of itself.
fn installer(config: &InstallerConfig, root: &RootSet) -> Result<Command, Error> {
    let captured_python = &root.python_version;
    let python = if !captured_python.is_empty() {
        let parts: Vec<_> = captured_python.split('.').collect();
        if !(2..=3).contains(&parts.len())
            || parts
                .iter()
                .any(|part| part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()))
        {
            return Err(Error::validation("invalid_request",
                "captured Python selection must be a numeric major.minor[.patch]",
            ));
        }
        captured_python
    } else {
        &config.python
    };
    if let Some(engine) = &config.helper_engine {
        crate::machine::client::helper(engine, &config.uv, &config.client_wheel)
            .map_err(|e| Error::unavailable("unavailable", format!("the installer helper: {e}")))?;
    }
    let mut command = Command::new(&config.helper_python);
    platform::process::inherit_nothing(&mut command);
    // The uploaded build backend runs here: it gets the machine's environment without
    // credentials or Runtime-owned names, never the whole service environment.
    command
        .env_clear()
        .envs(crate::launch_identity::inherited())
        .env("PATH", helper_path(&config.uv))
        .envs(config.seed_cache.iter().flat_map(|seed| {
            [("UV_CACHE_DIR", seed.as_os_str()), ("UV_LINK_MODE", std::ffi::OsStr::new("symlink"))]
        }))
        .arg("-m")
        .arg("tensord_client.installer")
        .arg("install-captured")
        .arg("--generations")
        .arg(&config.generations)
        .arg("--client-wheel")
        .arg(&config.client_wheel)
        .args(
            config
                .sdk()
                .iter()
                .flat_map(|wheel| [std::ffi::OsStr::new("--sdk-wheel"), wheel.as_os_str()]),
        )
        .arg(format!("--runtime-build={}", config.runtime_build))
        .arg("--python")
        .arg(python)
        .arg("--distribution")
        .arg(root.package.rsplit('/').next().expect("declared package"))
        .arg("--release")
        .arg(&root.release)
        .arg("--callees")
        .arg(serde_json::to_string(&root.callees).map_err(|error| Error::internal("internal", error.to_string()))?)
        .arg(format!("--python-requires={}", root.python_requires))
        .arg(format!("--python-version={}", root.python_version));
    Ok(command)
}

/// Run the helper to its exit and hold the generation it published. Its process group ends
/// with its run's cancel (`stop`), or when it stops moving past its measured patience
/// (`process::supervise`); a long install that moves is never killed.
fn publish(config: &InstallerConfig, mut command: Command, stop: Option<&Stop>, claim: Option<&Path>) -> Result<PreparedGeneration, Error> {
    let mut child = platform::process::own_group(&mut command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(storage)?;
    if let Some(claim) = claim {
        let births = platform::process::Birth::of(child.id()).and_then(|helper| Ok((helper, platform::process::Birth::own()?)));
        let claimed: Claim = births.map_err(|error| storage(error.into()))?;
        fs::write(claim, serde_json::to_vec(&claimed).map_err(std::io::Error::other).map_err(storage)?).map_err(storage)?;
    }
    let stderr = child.stderr.take().expect("piped installer stderr");
    let tail = std::thread::spawn(move || forward_tail(stderr));
    let mut stdout = child.stdout.take().expect("piped installer stdout");
    // Its typed record, bounded; the rest is drained so the helper never blocks on a pipe.
    let output = std::thread::spawn(move || {
        let mut output = Vec::new();
        stdout.by_ref().take((8 << 20) + 1).read_to_end(&mut output)?;
        std::io::copy(&mut stdout, &mut std::io::sink())?;
        Ok::<_, std::io::Error>(output)
    });
    let ended = crate::process::supervise(&mut child, stop, "the package installer", &|_| ()).map_err(storage)?;
    if let Some(claim) = claim {
        let _ = fs::remove_file(claim);
    }
    let output = output.join().map_err(|_| Error::internal("internal", "installer output reader panicked"))?.map_err(storage)?;
    let exited = match ended {
        crate::process::Supervised::Exited(status) => status,
        crate::process::Supervised::Canceled => {
            return Err(Error::conflict(crate::published::CANCELED, "the run was canceled while its package installed"))
        }
        crate::process::Supervised::Wedged(verdict) => return Err(Error::conflict("no_progress", verdict)),
    };
    let tail = tail.join().unwrap_or_default();
    if !exited.success() {
        #[derive(Deserialize)]
        #[serde(tag = "kind", rename = "install_failed")]
        struct InstallationFailure {
            code: String,
            detail: String,
        }
        // A typed reason stands alone; an operation that failed or a helper that crashed is
        // explained by the helper's own output, under it (uv's words, or a traceback).
        let failure = serde_json::from_slice::<InstallationFailure>(&output).ok();
        let message = match &failure {
            Some(f) if f.code != "package_dependency_operation_failed" => format!("{}: {}", f.code, f.detail),
            Some(f) => format!("{}: {}:\n{tail}", f.code, f.detail),
            None => format!("package_installation_failed: the trusted Python/uv installer stopped ({exited}):\n{tail}"),
        };
        return Err(match failure {
            Some(f) if f.code == "package_sdk_incompatible" => Error::conflict("package_sdk_incompatible", message),
            Some(f) if f.code.ends_with("_unsupported") => Error::not_found("capability_unavailable", message),
            _ => Error::conflict("precondition_failed", message),
        });
    }
    if output.len() > 8 << 20 {
        return Err(Error::capacity("exhausted",
            "prepared package interface exceeds its control-record bound",
        ));
    }
    let returned: GenerationRecord = serde_json::from_slice(&output)
        .map_err(|_| Error::internal("custody_corrupt", "installer returned an invalid typed generation record"))?;
    let generations = platform::durable::canonical(&config.generations).map_err(std::io::Error::from).map_err(storage)?;
    if returned.identity.len() != 32
        || !returned
            .identity
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(Error::internal("custody_corrupt",
            "installer returned an invalid native generation identity",
        ));
    }
    let directory = generations.join(&returned.identity);
    if directory
        .symlink_metadata()
        .map_err(storage)?
        .file_type()
        .is_symlink()
        || platform::durable::canonical(&directory).map_err(std::io::Error::from).map_err(storage)? != directory
    {
        return Err(Error::internal("custody_corrupt",
            "published generation path is not an owned native directory",
        ));
    }
    let hold = platform::durable::open_nofollow(&directory.join(".hold"), OpenOptions::new().read(true)).map_err(std::io::Error::from)
        .map_err(storage)?;
    hold.lock_shared().map_err(storage)?;
    let file = platform::durable::open_nofollow(&directory.join("generation.json"), OpenOptions::new().read(true)).map_err(std::io::Error::from)
        .map_err(storage)?;
    let record: GenerationRecord = serde_json::from_reader(file.take(8 << 20))
        .map_err(|_| Error::internal("custody_corrupt", "published generation manifest is invalid"))?;
    if record.identity != returned.identity
        || record.python != platform::paths::venv_python(&directory.join("env"))
        || !record.python.is_file()
    {
        return Err(Error::internal("custody_corrupt",
            "published generation does not name its native interpreter",
        ));
    }
    let interface_bytes = record.interface.get().as_bytes().to_vec();
    Ok(PreparedGeneration {
        record,
        interface_bytes,
        hold: Arc::new(hold),
    })
}
fn storage(error: std::io::Error) -> Error {
    Error::unavailable("unavailable", format!(
        "package installation storage/process unavailable: {error}"
    ))
}

/// The helper's stderr, passed on to the machine log as it comes. Returns what a refusal
/// quotes of it ([`stderr_tail`]).
fn forward_tail(mut stderr: impl Read) -> String {
    // One byte past the bound tells `stderr_tail` the stream was cut.
    let keep = crate::triage::MAX_STDERR as usize + 1;
    let (mut chunk, mut kept) = ([0u8; 8192], Vec::new());
    loop {
        match stderr.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                let _ = std::io::stderr().write_all(&chunk[..n]);
                kept.extend_from_slice(&chunk[..n]);
                kept.drain(..kept.len().saturating_sub(keep));
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    stderr_tail(&kept)
}

/// An installer's stderr as a refusal quotes it: its last [`crate::triage::MAX_STDERR`] bytes
/// from a line start, URL credentials masked. uv explains a failed resolution over many lines.
pub(crate) fn stderr_tail(stderr: &[u8]) -> String {
    let start = stderr.len().saturating_sub(crate::triage::MAX_STDERR as usize);
    let mut tail = &stderr[start..];
    if start > 0 {
        if let Some(line) = tail.iter().position(|&b| b == b'\n') {
            tail = &tail[line + 1..];
        }
    }
    masked(String::from_utf8_lossy(tail).trim())
}


/// The helper's PATH: the machine's uv's directory first, then the machine's own PATH. The helper
/// calls `uv` by name, and a machine started with no PATH still installs local packages.
fn helper_path(uv: &Path) -> std::ffi::OsString {
    let mut dirs: Vec<PathBuf> = uv
        .parent()
        .filter(|d| d.is_absolute())
        .map(Path::to_path_buf)
        .into_iter()
        .collect();
    dirs.extend(
        protocol::config::environment()
            .get("PATH")
            .iter()
            .flat_map(std::env::split_paths),
    );
    std::env::join_paths(dirs).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_quoted_installer_line_never_carries_url_credentials() {
        assert_eq!(
            super::masked("GET https://user:tok@hub.example/simple/ and file:///x/y.whl"),
            "GET https://***@hub.example/simple/ and file:///x/y.whl"
        );
    }

    /// A failed install quotes the installer's stderr whole up to a 64 KiB tail from a line
    /// start, never just its last 12 lines: uv's explanation starts well before its last line.
    #[test]
    fn a_failed_install_quotes_the_installers_stderr_tail() {
        let config = super::InstallerConfig {
            helper_python: "python3".into(),
            helper_engine: None,
            python: "3.12".into(),
            generations: std::env::temp_dir(),
            client_wheel: Default::default(),
            staging_root: std::env::temp_dir(),
            sdk: vec![],
            default_sdk: None,
            uv: "uv".into(),
            seed_cache: None,
            runtime_build: String::new(),
        };
        let mut helper = std::process::Command::new("sh");
        helper.args(["-c", r#"echo '{"kind":"install_failed","code":"package_dependency_operation_failed","detail":"uv pip install"}'
            i=0; while [ $i -lt 40 ]; do echo "  uv line $i https://user:tok@hub.example/simple" >&2; i=$((i+1)); done; exit 1"#]);
        let refused = super::publish(&config, helper, None, None).err().expect("the helper failed");
        let message = &refused.message;
        assert!(message.starts_with("package_dependency_operation_failed: uv pip install:\nuv line 0 "), "{message}");
        assert!(message.ends_with("uv line 39 https://***@hub.example/simple") && message.lines().count() == 41, "{message}");

        // Past the bound: whole lines only, so a cut never starts inside a URL's credentials.
        let line = "x".repeat(99) + "\n";
        let tail = super::forward_tail(std::io::Cursor::new(format!("head https://u:secret@h/{}", line.repeat(1000))));
        assert!(tail.len() <= crate::triage::MAX_STDERR as usize && tail.lines().all(|l| l.len() == 99), "{}", &tail[..120]);
        assert_eq!(tail.lines().count(), crate::triage::MAX_STDERR as usize / 100);
    }

    #[test]
    fn the_helper_finds_the_machines_uv_first() {
        let uv = crate::machine::client::uv(&std::env::temp_dir().join("machine"));
        let path = super::helper_path(&uv);
        let first = std::env::split_paths(&path).next();
        assert_eq!(first.as_deref(), uv.parent());
        // A bare `uv` (no root copy) leaves the machine's own PATH as it is.
        assert_eq!(
            super::helper_path(std::path::Path::new("uv")),
            std::env::var_os("PATH").unwrap_or_default()
        );
    }
}
