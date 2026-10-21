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

const MAX_INDEX_BYTES: usize = 32 << 20;
const MAX_REDIRECTS: usize = 8;

pub fn wheel(distribution: &str, version: &str, dir: &Path, machine: &Machine, stop: &Stop) -> io::Result<PathBuf> {
    let project = distribution.replace('_', "-");
    let index = format!("https://pypi.org/pypi/{project}/{version}/json");
    let http = client(None)?;
    let raw = machine
        .block(stop, async move {
            let answer = http.get(&index).send().await.map_err(weather)?;
            if !answer.status().is_success() {
                return refuse(Code::REF_NOT_FOUND, format!("HTTP {}", answer.status()));
            }
            let raw = answer.bytes().await.map_err(weather)?;
            match raw.len() <= MAX_INDEX_BYTES {
                true => Ok(raw.to_vec()),
                false => refuse(Code::SIZE_CAP, "the index answer is too large"),
            }
        })
        .map_err(|e| io::Error::other(format!("PyPI {project} {version}: {e}")))?;
    #[derive(serde::Deserialize)]
    struct File {
        filename: String,
        url: String,
        size: u64,
        digests: Digests,
    }
    #[derive(serde::Deserialize)]
    struct Digests {
        sha256: String,
    }
    #[derive(serde::Deserialize)]
    struct Release {
        urls: Vec<File>,
    }
    let release: Release = serde_json::from_slice(&raw).map_err(io::Error::other)?;
    // pip's choice: the wheel built for this OS and CPU, else a pure one.
    let rank = |f: &File| {
        let name = f.filename.strip_suffix(".whl").filter(|name| !name.contains('/'))?;
        platform::wheel::rank(name.rsplit('-').next()?)
    };
    let (_, file) = release.urls.iter().filter_map(|f| Some((rank(f)?, f))).min_by_key(|(rank, _)| *rank).ok_or_else(|| {
        let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
        io::Error::other(format!("PyPI has no {os} {arch} wheel of {project} {version}"))
    })?;
    let object = ObjectRef { sha256: file.digests.sha256.to_lowercase(), length: file.size };
    let blob = fetch(machine, &file.url, &object, None, stop).map_err(|e| io::Error::other(format!("{}: {e}", file.filename)))?;
    land(&blob, dir, &file.filename)
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
    use std::env::consts::{ARCH, OS};

    /// `cozy machine update --runtime <version>` lands this OS and CPU's wheel from PyPI, through
    /// the machine's store, else the pure one: cozy-runtime 0.27.4 ships manylinux x86_64 and
    /// pure, tensorfs 0.11.0 four platforms and no pure one.
    #[test]
    fn pypi_lands_this_machines_wheel_else_a_pure_one() {
        let (tensorfs, runtime) = match (OS, ARCH) {
            ("linux", "x86_64") => ("cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64", "cp312-abi3-manylinux_2_28_x86_64"),
            ("linux", "aarch64") => ("cp312-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64", "py3-none-any"),
            ("windows", "x86_64") => ("cp312-abi3-win_amd64", "py3-none-any"),
            ("macos", "aarch64") => ("cp312-abi3-macosx_11_0_arm64", "py3-none-any"),
            other => panic!("no published wheels to expect on {other:?}"),
        };
        let dir = tempfile::tempdir().unwrap();
        let machine = crate::origin::Machine::new(crate::storage::open(&dir.path().join("store")).unwrap(), None).unwrap();
        let fetched = |distribution, version| {
            let wheel = super::wheel(distribution, version, &dir.path().join("wheels"), &machine, &Default::default()).unwrap();
            wheel.file_name().unwrap().to_string_lossy().into_owned()
        };
        assert_eq!(fetched("tensorfs", "0.11.0"), format!("tensorfs-0.11.0-{tensorfs}.whl"));
        assert_eq!(fetched("cozy-runtime", "0.27.4"), format!("cozy_runtime-0.27.4-{runtime}.whl"));
    }
}
