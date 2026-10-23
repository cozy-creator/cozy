//! A run's local source: unpublished code, named by the digest of its manifest object.
//!
//! A remote editable project and its libraries are kept under the exact manifest's source
//! home. Each environment points at that home, so a later install cannot rewrite code or data
//! an accepted run still reads. This computer's explicit live path is read in place, never
//! copied. Both use the shared dependency cache.
use crate::{
    api::{
        install::{prepare_editable, prepare_uploaded, InstallerConfig},
        workspaces::{RootSet, UploadedFile, UploadedPackage},
    },
    journal::Installation,
    objects::{Objects, Refused},
    service::Service,
};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
};
use crate::{cas::Cas, stop::Stop};
use tensorfs::{format::ObjectRef, sha256};

/// The manifest object (JSON). Members are objects the same signer wrote.
#[derive(Deserialize)]
pub struct Manifest {
    pub package: String,
    #[serde(default)]
    pub release: String,
    #[serde(default)]
    pub python_requires: String,
    #[serde(default)]
    pub python_version: String,
    /// The project's source tree as a tar archive.
    #[serde(default)]
    pub source: Option<Member>,
    /// Wheels the project vendors, by file name.
    #[serde(default)]
    pub wheels: Vec<Member>,
    /// The locked dependency requirements (`uv export` text).
    #[serde(default)]
    pub requirements: Option<Member>,
    /// Installed dependency distribution -> the package whose invocables it owns.
    #[serde(default)]
    pub callees: std::collections::BTreeMap<String, String>,
    /// A live project and the local path dependencies its lock names, file by file under
    /// their common directory; `project` is the project's own directory there.
    #[serde(default)]
    pub files: Vec<Member>,
    #[serde(default)]
    pub project: String,
    /// This computer's machine: that common directory itself, read in place and never copied.
    #[serde(default)]
    pub path: String,
    /// uv indexes the project's sources name without declaring, by name: an authored tree
    /// names its account's index (`tensorhub`), whose URL depends on who runs it.
    #[serde(default)]
    pub indexes: BTreeMap<String, String>,
    /// The local path dependencies its lock installs, as directories or files under the tree:
    /// their code is the environment's, not the project's own.
    #[serde(default)]
    pub locals: Vec<String>,
}
#[derive(Deserialize)]
pub struct Member {
    #[serde(default)]
    pub name: String,
    pub digest: String,
    pub length: u64,
    #[serde(default)]
    pub executable: bool,
}
impl Manifest {
    /// The project's directory under the tree, as a prefix of its files' names.
    fn own(&self) -> String {
        match self.project.is_empty() {
            true => String::new(),
            false => format!("{}/", self.project),
        }
    }
    /// Whether `name` is a local path dependency's file, or outside the project.
    fn beside(&self, name: &str) -> bool {
        !name.starts_with(&self.own()) || self.locals.iter().any(|dir| name == dir || name.starts_with(&format!("{dir}/")))
    }
    /// The project's own code by content, the same wherever it is installed: what keys its own
    /// memoized calls (an editable install's files say nothing of the code they point at).
    fn source_digest(&self) -> String {
        let own = self.own();
        let code: Vec<_> = self.files.iter().filter(|f| !self.beside(&f.name)).map(|f| (&f.name[own.len()..], &f.digest)).collect();
        format!("sha256:{}", sha256::hex_digest(serde_json::json!(code).to_string().as_bytes()))
    }
}
impl Member {
    /// What a tree holds for this file: its bytes and whether it is executable.
    fn held(&self) -> String {
        format!("{}{}", self.digest, if self.executable { "+x" } else { "" })
    }
}

/// The installer's holder, forgotten however its install ends.
struct Holder<'a>(&'a Mutex<Option<(String, String)>>);
impl Drop for Holder<'_> {
    fn drop(&mut self) {
        *self.0.lock().unwrap() = None;
    }
}

pub struct LocalSources {
    pub objects: Arc<Objects>,
    pub installer: InstallerConfig,
    pub cas: Cas,
    /// Each signer's packages: remote `source-<manifest>/{tree,env-*}`, direct paths `env-*`.
    editable: PathBuf,
    installing: Mutex<()>,
    /// The run whose install holds `installing`, and its package: what a second run waits for.
    holder: Mutex<Option<(String, String)>>,
}

/// `name` as a path inside a tree: relative, with no step out of it.
fn inside(name: &str) -> Result<&Path, Refused> {
    let path = Path::new(name);
    if path.components().all(|c| matches!(c, Component::Normal(_))) {
        return Ok(path);
    }
    Err(refused("local_source_invalid", format!("{name:?} is not a path inside the package's tree")))
}

/// The files the tree under `home` holds, by name; none while it is absent or half written.
fn held_tree(home: &Path) -> BTreeMap<String, String> {
    fs::read(home.join("tree.json")).ok().and_then(|raw| serde_json::from_slice(&raw).ok()).unwrap_or_default()
}

fn keyed(name: &str, prefix: &str, length: usize) -> bool {
    name.strip_prefix(prefix).is_some_and(|key| key.len() == length && key.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// Only this layout's environments leave. Unknown entries in a package home are preserved.
fn collect_environments(home: &Path, named: &HashSet<PathBuf>) -> usize {
    // In the spelling Python writes (Windows' canonicalize says \\?\C:\…), as `named` is.
    let root = platform::durable::canonical(home).unwrap_or_else(|_| home.to_path_buf());
    // A named but missing environment still retains its source for recovery.
    let kept = named.iter().filter(|env| env.parent() == Some(root.as_path())).count();
    for entry in fs::read_dir(home).into_iter().flatten().flatten() {
        let path = entry.path();
        if !keyed(&entry.file_name().to_string_lossy(), "env-", 16) || !path.is_dir() {
            continue;
        }
        if !named.contains(&platform::durable::canonical(&path).unwrap_or_else(|_| path.clone())) {
            let _ = fs::remove_dir_all(&path);
            let _ = fs::remove_file(path.with_extension("constraints.txt"));
            let _ = fs::remove_dir_all(path.with_extension("wheels"));
        }
    }
    kept
}

fn refused(code: &'static str, message: impl Into<String>) -> Refused {
    Refused {
        code,
        message: message.into(),
    }
}

/// One member as the object this signer wrote.
fn written_member(objects: &Objects, actor: &str, m: &Member) -> Result<ObjectRef, Refused> {
    match objects.path(actor, &m.digest)? {
        Some((_, length)) if length == m.length => Ok(ObjectRef {
            sha256: m.digest.trim_start_matches("sha256:").to_string(),
            length,
        }),
        _ => Err(refused(
            "local_source_incomplete",
            format!("{} {} was not written to this machine", m.name, m.digest),
        )),
    }
}

/// The manifest `digest` (`sha256:<hex>`) as this signer wrote it.
fn open_manifest(
    objects: &Objects,
    actor: &str,
    digest: &str,
) -> Result<(Manifest, ObjectRef), Refused> {
    let (path, length) = objects.path(actor, digest)?.ok_or_else(|| {
        refused(
            "local_source_incomplete",
            format!("the local package manifest {digest} was not written to this machine"),
        )
    })?;
    if length > 1 << 20 {
        return Err(refused(
            "local_source_invalid",
            "a local package manifest is at most 1 MiB",
        ));
    }
    let manifest = serde_json::from_slice(&fs::read(path)?).map_err(|e| {
        refused(
            "local_source_invalid",
            format!("the local package manifest is invalid: {e}"),
        )
    })?;
    let sha256 = digest.trim_start_matches("sha256:").to_string();
    Ok((manifest, ObjectRef { sha256, length }))
}

impl LocalSources {
    pub fn new(objects: Arc<Objects>, installer: InstallerConfig) -> Self {
        Self {
            editable: installer.generations.with_file_name("editable"),
            cas: objects.cas().clone(),
            objects,
            installer,
            installing: Mutex::new(()),
            holder: Mutex::new(None),
        }
    }

    /// Where `actor`'s live `package` is kept.
    fn home(&self, actor: &str, package: &str) -> Result<PathBuf, Refused> {
        let name = crate::catalog::normalized(package.strip_prefix("local/").unwrap_or_default());
        if name.is_empty() || name.contains('/') {
            return Err(refused("local_source_invalid", "a local package manifest names local/<name>"));
        }
        Ok(self.editable.join(&sha256::hex_digest(actor.as_bytes())[..16]).join(name))
    }

    /// A local source's objects, its manifest and every member this machine needs written:
    /// what a run takes custody of when it is accepted. Every captured member is retained:
    /// a cached generation may be collected while this run waits to install. The caller's
    /// object cache still avoids reuploading unchanged bytes. A directory read in place needs
    /// none of its files. One the signer did not write is `local_source_incomplete`, naming it.
    pub fn written(&self, actor: &str, digest: &str) -> Result<Vec<ObjectRef>, Refused> {
        let (manifest, own) = open_manifest(&self.objects, actor, digest)?;
        let mut written = vec![own];
        if !manifest.files.is_empty() || !manifest.path.is_empty() {
            self.home(actor, &manifest.package)?;
        }
        let files = manifest.files.iter().filter(|_| manifest.path.is_empty());
        let members = manifest.source.iter().chain(&manifest.wheels).chain(&manifest.requirements).chain(files);
        for m in members {
            written.push(written_member(&self.objects, actor, m)?);
        }
        Ok(written)
    }

    /// The environment a live manifest runs in: one per lock, project metadata, local path
    /// dependencies, carried wheels and this machine's interpreter, SDK pair and client.
    /// Remote environments live beside their immutable source; direct paths remain live.
    fn environment(&self, home: &Path, manifest: &Manifest) -> PathBuf {
        let own = manifest.own();
        let held = |name: &str| {
            let path = format!("{own}{name}");
            manifest.files.iter().find(|f| f.name == path).map(|f| f.digest.as_str()).unwrap_or_default()
        };
        let beside: Vec<_> = manifest.files.iter().filter(|f| manifest.beside(&f.name)).map(|f| (&f.name, f.held())).collect();
        let carried: Vec<_> = manifest.wheels.iter().map(|w| (&w.name, &w.digest)).collect();
        let key = serde_json::json!({
            "lock": held("uv.lock"), "project": held("pyproject.toml"), "beside": beside, "wheels": carried,
            "indexes": manifest.indexes,
            "sdk": self.installer.sdk, "client": self.installer.client_wheel,
            "python": self.installer.python, "python_version": manifest.python_version,
        });
        home.join(format!("env-{}", &sha256::hex_digest(key.to_string().as_bytes())[..16]))
    }

    /// Make the tree under `home` the manifest's files: changed ones are replaced from the
    /// objects the signer wrote, and ones it no longer names leave.
    fn sync(&self, actor: &str, home: &Path, files: &[Member]) -> Result<PathBuf, Refused> {
        let tree = home.join("tree");
        let held = held_tree(home);
        let next: BTreeMap<String, String> = files.iter().map(|f| (f.name.clone(), f.held())).collect();
        if next != held {
            // Half synced, the tree holds what neither manifest names: forget it until done.
            let _ = fs::remove_file(home.join("tree.json"));
        }
        for file in files {
            let target = tree.join(inside(&file.name)?);
            if held.get(&file.name) == Some(&file.held()) && target.is_file() {
                continue;
            }
            let object = written_member(&self.objects, actor, file)?;
            fs::create_dir_all(target.parent().expect("a file inside the tree"))?;
            let incoming = home.join(".incoming");
            fs::copy(self.cas.path(&object)?, &incoming)?;
            platform::durable::set_mode(&incoming, if file.executable { 0o755 } else { 0o644 }).map_err(std::io::Error::from)?;
            fs::rename(&incoming, &target)?;
        }
        for gone in held.keys().filter(|name| !next.contains_key(*name)) {
            let path = tree.join(gone);
            let _ = fs::remove_file(&path);
            // Directories it emptied leave with it.
            let mut parent = path.parent();
            while let Some(directory) = parent.filter(|d| *d != tree && fs::remove_dir(d).is_ok()) {
                parent = directory.parent();
            }
        }
        if next != held {
            fs::create_dir_all(&tree)?;
            fs::write(home.join(".tree.json"), serde_json::to_vec(&next).map_err(std::io::Error::other)?)?;
            fs::rename(home.join(".tree.json"), home.join("tree.json"))?;
        }
        Ok(tree)
    }

    /// Install editable against this manifest's remote tree, or the explicit direct path.
    #[allow(clippy::too_many_arguments)]
    fn install_live(
        &self,
        service: &Service,
        actor: &str,
        alias: String,
        manifest: Manifest,
        digest: &str,
        stop: Option<&Stop>,
        stage: &dyn Fn(&str, u64),
    ) -> Result<Installation, Refused> {
        let home = self.home(actor, &manifest.package)?;
        let home = if manifest.path.is_empty() { home.join(format!("source-{digest}")) } else { home };
        fs::create_dir_all(&home)?;
        let root = match manifest.path.is_empty() {
            true => self.sync(actor, &home, &manifest.files)?,
            false => PathBuf::from(&manifest.path),
        };
        let project = match manifest.project.is_empty() {
            true => root.clone(),
            false => root.join(inside(&manifest.project)?),
        };
        if !root.is_absolute() || !project.join("pyproject.toml").is_file() {
            return Err(refused(
                "local_source_absent",
                format!("{} is not a project directory on this machine", project.display()),
            ));
        }
        let environment = self.environment(&home, &manifest);
        let installing = match environment.join(".ready").is_file() {
            true => format!("syncing {}'s changed source", manifest.package),
            false => format!("installing {} on this machine", manifest.package),
        };
        stage(&installing, 0);
        let source_digest = manifest.source_digest();
        // Wheels the CLI carries for locked dependencies this machine cannot fetch itself (a
        // Hub's on the caller's computer, a private git source): beside the environment they
        // install into, and needed only while it is built.
        let mut wheels = vec![];
        if !environment.join(".ready").is_file() {
            let carried = environment.with_extension("wheels");
            let _ = fs::remove_dir_all(&carried);
            fs::create_dir_all(&carried)?;
            for wheel in &manifest.wheels {
                if !wheel.name.ends_with(".whl") || wheel.name.contains('/') {
                    return Err(refused("local_source_invalid", format!("{:?} is not a wheel file name", wheel.name)));
                }
                let object = written_member(&self.objects, actor, wheel)?;
                fs::copy(self.cas.path(&object)?, carried.join(&wheel.name))?;
                wheels.push(carried.join(&wheel.name));
            }
        }
        let root = RootSet {
            operation_id: String::new(),
            package: manifest.package.clone(),
            release: manifest.release.clone(),
            installation_id: alias.clone(),
            python_requires: manifest.python_requires,
            python_version: manifest.python_version,
            source_archive: String::new(),
            dependency_requirements: Vec::new(),
            callees: manifest.callees,
            files: Vec::new(),
        };
        for (name, url) in &manifest.indexes {
            let named = !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b));
            if !named || !(url.starts_with("https://") || url.starts_with("http://")) || url.contains(char::is_whitespace) {
                return Err(refused("local_source_invalid", format!("index {name:?} = {url:?} is not a named HTTP(S) index")));
            }
        }
        let read = |bytes| stage(&installing, bytes);
        let prepared = prepare_editable(&self.installer, &root, &project, &environment, &wheels, &manifest.indexes, &source_digest, stop, &read)
            .map_err(installer_refusal)?;
        let _ = fs::remove_dir_all(environment.with_extension("wheels"));
        let installed = service.engine.bind_installation(Installation {
            actor: actor.to_string(),
            alias,
            generation: prepared.record.identity,
            package: manifest.package,
            release: manifest.release,
            interface: prepared.interface_bytes,
            hub: String::new(),
        })?;
        service.changed_environment()?;
        Ok(installed)
    }

    /// Keep remote source homes while a generation names their environment. Only completed,
    /// recognized source homes leave; partial or unknown homes are preserved for recovery.
    pub fn collect(&self, actor: &str, package: &str) {
        let Ok(home) = self.home(actor, package) else {
            return;
        };
        let _installing = self.installing.lock().unwrap();
        // An unavailable or partial inventory says nothing about which sources are unused.
        let Ok(generations) = fs::read_dir(&self.installer.generations)
            .and_then(|entries| entries.collect::<std::io::Result<Vec<_>>>()) else { return; };
        let mut named = HashSet::new();
        for generation in generations {
            let env = generation.path().join("env");
            match fs::read_link(&env) {
                Ok(target) => { named.insert(platform::durable::canonical(&target).unwrap_or(target)); }
                // Published generations own an ordinary environment directory.
                Err(error) if error.kind() == std::io::ErrorKind::InvalidInput && env.is_dir() => {}
                Err(_) => return,
            }
        }
        let kept = collect_environments(&home, &named);
        for entry in fs::read_dir(&home).into_iter().flatten().flatten() {
            let path = entry.path();
            if keyed(&entry.file_name().to_string_lossy(), "source-", 64)
                && !held_tree(&path).is_empty() && collect_environments(&path, &named) == 0
            {
                let _ = fs::remove_dir_all(&path);
            }
        }
        if kept == 0 {
            // The pre-generation layout's tree, once no old environment points at it.
            if !held_tree(&home).is_empty() {
                let _ = fs::remove_dir_all(home.join("tree"));
                let _ = fs::remove_file(home.join("tree.json"));
            }
            let _ = fs::remove_dir(&home); // Only an empty home; never erase unknown entries.
        }
    }

    /// This signer's installation of the manifest `digest` (`sha256:<hex>`).
    /// `stage` hears what is built: nothing for an environment this machine holds. While
    /// another run's install holds the installer, run `run` is told it waits for that run.
    #[allow(clippy::too_many_arguments)]
    pub fn install(
        &self,
        service: &Service,
        actor: &str,
        digest: &str,
        stop: Option<&Stop>,
        run: &str,
        stage: &dyn Fn(&str, u64),
    ) -> Result<Installation, Refused> {
        // Like published installations, local aliases include the selected SDK and
        // client. The source digest alone cannot identify the environment after an update.
        let key = serde_json::json!({
            "manifest": digest,
            "sdk": self.installer.sdk,
            "client": self.installer.client_wheel,
            "python": self.installer.python,
        });
        let alias = format!("local-{}", &tensorfs::sha256::hex_digest(key.to_string().as_bytes())[..32]);
        let _installing = match self.installing.try_lock() {
            Ok(installing) => installing,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                let reason = match self.holder.lock().unwrap().clone() {
                    Some((holder, package)) => format!("waiting for run {holder} to install {package}"),
                    None => "waiting for another run's install".into(),
                };
                let tell = |reason| match run.is_empty() {
                    true => Ok(()),
                    false => service.engine.waiting(run, reason),
                };
                tell(Some(reason))?;
                let installing = self.installing.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                tell(None)?;
                installing
            }
        };
        let package = open_manifest(&self.objects, actor, digest).map(|(m, _)| m.package).unwrap_or_default();
        *self.holder.lock().unwrap() = Some((service.engine.caller_id(run), package));
        let _held = Holder(&self.holder);
        if let Some(held) = service.engine.installation(actor, &alias)? {
            if service.catalog.resolve(&held.generation).is_ok() {
                return Ok(held);
            }
        }
        let member = |m: &Member| written_member(&self.objects, actor, m);
        let (manifest, object) = open_manifest(&self.objects, actor, digest)?;
        if !manifest.files.is_empty() || !manifest.path.is_empty() {
            return self.install_live(service, actor, alias, manifest, &object.sha256, stop, stage);
        }
        let installing = format!("installing {} on this machine", manifest.package);
        stage(&installing, 0);
        if !manifest.package.starts_with("local/") || (manifest.source.is_none() && manifest.wheels.is_empty()) {
            return Err(refused(
                "local_source_invalid",
                "a local package manifest names local/<name> and its source archive or root wheel",
            ));
        }
        let requirements = match &manifest.requirements {
            Some(m) => {
                let object = member(m)?;
                fs::read(self.cas.path(&object)?)?
            }
            None => Vec::new(),
        };
        let (source_name, mut files) = match &manifest.source {
            Some(source) => ("source.tar".to_string(), vec![UploadedFile::held(
                "source.tar".into(), &member(source)?, self.cas.clone(),
            )]),
            None => (String::new(), vec![]),
        };
        for wheel in &manifest.wheels {
            if !wheel.name.ends_with(".whl") || wheel.name.contains('/') {
                return Err(refused(
                    "local_source_invalid",
                    format!("{:?} is not a wheel file name", wheel.name),
                ));
            }
            files.push(UploadedFile::held(
                wheel.name.clone(),
                &member(wheel)?,
                self.cas.clone(),
            ));
        }
        let uploaded = UploadedPackage {
            root: RootSet {
                operation_id: String::new(),
                package: manifest.package.clone(),
                release: manifest.release.clone(),
                installation_id: alias.clone(),
                python_requires: manifest.python_requires,
                python_version: manifest.python_version,
                source_archive: source_name,
                dependency_requirements: requirements,
                callees: manifest.callees,
                files: Vec::new(),
            },
            files,
        };
        let prepared = prepare_uploaded(&self.installer, &uploaded, stop, &|bytes| stage(&installing, bytes)).map_err(installer_refusal)?;
        let installed = service.engine.bind_installation(Installation {
            actor: actor.to_string(),
            alias,
            generation: prepared.record.identity,
            package: manifest.package,
            release: manifest.release,
            interface: prepared.interface_bytes,
            hub: String::new(),
        })?;
        service.changed_environment()?;
        Ok(installed)
    }
}


/// The trusted installer's end as a run's reason: canceled with its run, wedged (no CPU or
/// bytes moved past its measured patience), or failed.
fn installer_refusal(error: protocol::error::Error) -> Refused {
    let code = match error.code.as_str() {
        crate::published::CANCELED => crate::published::CANCELED,
        "no_progress" => "no_progress",
        _ => "package_installation_failed",
    };
    refused(code, error.message)
}
