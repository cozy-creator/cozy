//! A Runtime or TensorFS wheel the machine fetches itself: published on PyPI, checked against
//! the index's own SHA-256, or at any https URL, held to the digest its update names. It lands
//! through the machine's store with the one downloader: `stop` ends it, and what landed stays
//! for the next ask.
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use tensorfs::{
    err::{refuse, Code, Refusal, Result},
    fetch::{policy::is_loopback_origin, Location, Origin},
    format::ObjectRef,
};

use crate::{
    origin::{priority, Machine, Remote},
    stop::Stop,
};

const MAX_REDIRECTS: usize = 8;

pub fn wheel(distribution: &str, version: &str, uv: &Path, dir: &Path, machine: &Machine, stop: &Stop) -> io::Result<PathBuf> {
    let project = distribution.replace('_', "-");
    if version.is_empty() || !version.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-+!".contains(&c)) {
        return Err(io::Error::other("a published wheel requires a concrete version"));
    }
    let input = format!("{project}=={version}\n");
    let plan = super::update::compile_sdk(uv, &input, stop, true)?
        .map_err(|why| io::Error::other(format!("PyPI {project} {version}: {why}")))?;
    let wheel = selected(&plan, &project, version)?;
    wheel_at(&wheel.url, &wheel.hashes.sha256, &wheel.filename()?, dir, machine, stop)
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
        let name = self.url.rsplit('/').next().unwrap_or_default();
        if !self.url.starts_with("https://") || !name.ends_with(".whl")
            || !name.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-+".contains(&c))
            || self.hashes.sha256.len() != 64 || !self.hashes.sha256.bytes().all(|c| c.is_ascii_hexdigit())
        { return Err(io::Error::other("the resolver returned no verified wheel URL")); }
        Ok(name.into())
    }
}

/// uv resolves for the executor's Python and this machine's platform, returning one wheel.
/// Filename order and a matching architecture alone cannot establish Python ABI compatibility.
fn selected(plan: &str, distribution: &str, version: &str) -> io::Result<Artifact> {
    #[derive(serde::Deserialize)]
    struct Package { name: String, version: String, #[serde(default)] wheels: Vec<Artifact> }
    #[derive(serde::Deserialize)]
    struct Plan { packages: Vec<Package> }
    let parsed: Plan = toml::from_str(plan).map_err(io::Error::other)?;
    let mut packages = parsed.packages.into_iter().filter(|p| p.name.replace('_', "-") == distribution);
    let package = packages.next().ok_or_else(|| io::Error::other("the resolver omitted the requested wheel"))?;
    if packages.next().is_some() || package.version != version || package.wheels.len() != 1 {
        return Err(io::Error::other("the resolver returned ambiguous or different SDK wheels"));
    }
    let wheel = package.wheels.into_iter().next().ok_or_else(|| io::Error::other("no host wheel"))?;
    if !wheel.filename()?.starts_with(&format!("{}-{version}-", distribution.replace('-', "_"))) {
        return Err(io::Error::other("the selected filename differs from its package/version"));
    }
    Ok(wheel)
}

/// Wheel `file` at an https `url` on any host, redirects followed: its bytes are held to
/// `sha256`, so no host vouches for anything. Other bytes fail `wheel_digest_mismatch`.
pub fn wheel_at(url: &str, sha256: &str, file: &str, dir: &Path, machine: &Machine, stop: &Stop) -> io::Result<PathBuf> {
    // A private server's root as uv trusts it: the bundle SSL_CERT_FILE names. Only this fetch
    // reads it; the Hub's clients trust their own CA alone.
    let bundle = protocol::config::environment().get("SSL_CERT_FILE").and_then(|path| fs::read(path).ok()).filter(|pem| !pem.is_empty());
    let failed = |e: Refusal| match e.code {
        Code::OBJECT_ID_MISMATCH => io::Error::other(super::update::Coded {
            code: "wheel_digest_mismatch",
            message: format!("{file} from {url}: {}", e.detail),
        }),
        _ => io::Error::other(format!("{file} from {url}: {e}")),
    };
    let blob = object_at(machine, url, sha256, None, bundle, stop).map_err(failed)?;
    land(&blob, dir, file)
}

/// Object `sha256` at `url` (its length asked of the server when not given), through the store:
/// its path there. `trust` is a PEM bundle of roots beyond the public ones.
pub(crate) fn object_at(machine: &Machine, url: &str, sha256: &str, length: Option<u64>, trust: Option<Vec<u8>>, stop: &Stop) -> Result<PathBuf> {
    let length = match length {
        Some(length) => length,
        None => {
            let http = client(trust.as_deref()).map_err(|e| Refusal { code: Code::TRANSFER_FAILED, detail: e.to_string() })?;
            let probed = url.to_string();
            machine.block(stop, async move {
                // One byte asked: the answer states the whole length (`Content-Range`).
                let answer = http.get(&probed).header("range", "bytes=0-0").send().await.map_err(weather)?;
                let header = |name: &str| answer.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_string);
                let length = match answer.status().as_u16() {
                    206 => header("content-range").and_then(|r| r.rsplit('/').next()?.parse::<u64>().ok()),
                    200 => header("content-length").and_then(|l| l.parse::<u64>().ok()),
                    _ => None,
                };
                match length {
                    Some(length) => Ok(length),
                    None => refuse(Code::TRANSFER_FAILED, format!("a ranged GET answered HTTP {} with no length", answer.status())),
                }
            })?
        }
    };
    let object = ObjectRef { sha256: sha256.into(), length };
    fetch(machine, url, &object, trust.map(Arc::new), stop)
}

/// One object from one URL, through the store.
fn fetch(machine: &Machine, url: &str, object: &ObjectRef, trust: Option<Arc<Vec<u8>>>, stop: &Stop) -> Result<PathBuf> {
    let remote = Remote { origin: Arc::new(Url(url.into())), trust, local: is_loopback_origin(url) };
    machine.objects(&remote, std::slice::from_ref(object), priority::NOW, stop)?;
    machine.store().path(object).map_err(|e| e.refusal())
}

/// A file at a fixed URL: the one object it serves.
struct Url(String);

#[async_trait]
impl Origin for Url {
    async fn repo_of(&self, name: &str) -> Result<tensorfs::format::Repo> {
        refuse(Code::REF_NOT_FOUND, format!("{} names no model {name}", self.0))
    }

    async fn locate(&self, objects: &[ObjectRef]) -> Result<Vec<Location>> {
        Ok(objects.iter().map(|_| Location::Url { url: self.0.clone(), headers: vec![], expires: None }).collect())
    }

    fn public(&self) -> bool {
        false
    }
}

fn client(bundle: Option<&[u8]>) -> io::Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder().redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS));
    if let Some(pem) = bundle {
        for certificate in reqwest::Certificate::from_pem_bundle(pem).map_err(io::Error::other)? {
            builder = builder.add_root_certificate(certificate);
        }
    }
    builder.build().map_err(io::Error::other)
}

fn weather(error: reqwest::Error) -> Refusal {
    Refusal { code: Code::TRANSFER_FAILED, detail: error.to_string() }
}

fn land(blob: &Path, dir: &Path, file: &str) -> io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join(file);
    let _ = fs::remove_file(&path);
    fs::hard_link(blob, &path).or_else(|_| fs::copy(blob, &path).map(drop))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_plan_must_identify_one_exact_wheel() {
        let plan = format!(r#"
[[packages]]
name = "tensorfs"
version = "0.13.5"
wheels = [{{url = "https://files.pythonhosted.org/tensorfs-0.13.5-cp311-abi3-manylinux_2_17_x86_64.whl", hashes = {{sha256 = "{}"}}}}]
"#, "1".repeat(64));
        assert!(selected(&plan, "tensorfs", "0.13.5").unwrap().filename().unwrap().contains("cp311-abi3"));
        assert!(selected(&plan, "tensorfs", "0.13.4").is_err());
        assert!(selected(&format!("{plan}\n{plan}"), "tensorfs", "0.13.5").is_err());
        assert!(selected(&plan.replace("tensorfs-0.13.5", "tensorfs-0.13.4"), "tensorfs", "0.13.5").is_err());
        assert!(selected(&plan.replace(&"1".repeat(64), "invalid"), "tensorfs", "0.13.5").is_err());
    }

    #[test]
    #[ignore = "downloads a real published wheel and installs it for the executor's Python"]
    fn published_tensorfs_wheel_installs_for_executor_python() {
        let root = tempfile::tempdir().unwrap();
        let store = Arc::new(tensorfs::store::Store::open(root.path().join("store")).unwrap());
        let machine = Machine::new(store, None).unwrap();
        let uv = root.path().join(if cfg!(windows) { "uv.exe" } else { "uv" });
        let wheel = wheel("tensorfs", "0.13.5", &uv, &root.path().join("wheels"), &machine, &Stop::default()).unwrap();
        let output = std::process::Command::new(&uv)
            .args(["pip", "install", "--no-config", "--no-deps", "--python", "3.12", "--target"])
            .arg(root.path().join("installed")).arg(&wheel).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        assert!(root.path().join("installed/tensorfs-0.13.5.dist-info/METADATA").is_file());
    }
}
