//! A `cozy` release the machine fetches itself (`cozy machine update --tensord <version>`): the
//! release's dist-manifest names this target's archive and its sha256, and the archive lands
//! through the machine's store held to that digest, as a published wheel does (`pypi`).
use std::{
    fs, io,
    path::{Path, PathBuf},
};

use crate::{origin::Machine, stop::Stop};

const RELEASES: &str = "https://github.com/cozy-creator/cozy-creator/releases/download";

/// The dist target whose archive this machine runs.
const TARGET: &str = if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
    "x86_64-unknown-linux-gnu"
} else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
    "aarch64-unknown-linux-gnu"
} else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
    "aarch64-apple-darwin"
} else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
    "x86_64-apple-darwin"
} else if cfg!(all(windows, target_arch = "x86_64")) {
    "x86_64-pc-windows-msvc"
} else {
    "aarch64-pc-windows-msvc"
};

#[derive(serde::Deserialize)]
struct Manifest {
    artifacts: std::collections::BTreeMap<String, Artifact>,
}

#[derive(serde::Deserialize)]
struct Artifact {
    kind: String,
    #[serde(default)]
    target_triples: Vec<String>,
    #[serde(default)]
    checksums: std::collections::BTreeMap<String, String>,
}

/// cozy `version`'s binary for this machine, unpacked into `dir`.
pub fn cozy(version: &str, dir: &Path, machine: &Machine, stop: &Stop) -> io::Result<PathBuf> {
    if version.is_empty() || !version.bytes().all(|c| c.is_ascii_alphanumeric() || b".-+".contains(&c)) {
        return Err(io::Error::other(format!("{version:?} is not a cozy release version")));
    }
    let base = format!("{RELEASES}/v{version}");
    let listed = format!("{base}/dist-manifest.json");
    let http = reqwest::Client::builder().user_agent(concat!("cozy/", env!("CARGO_PKG_VERSION"))).build().map_err(io::Error::other)?;
    let fetched = listed.clone();
    let manifest: Manifest = machine
        .block(stop, async move {
            let answer = http.get(&fetched).send().await.and_then(|a| a.error_for_status());
            let answer = answer.map_err(|e| tensorfs::err::Refusal { code: tensorfs::err::Code::TRANSFER_FAILED, detail: e.to_string() })?;
            answer.json().await.map_err(|e| tensorfs::err::Refusal { code: tensorfs::err::Code::TRANSFER_FAILED, detail: e.to_string() })
        })
        .map_err(|e| io::Error::other(format!("cozy {version}: {listed}: {e}")))?;
    let (name, sha256) = manifest
        .artifacts
        .iter()
        .find(|(_, a)| a.kind == "executable-zip" && a.target_triples.iter().any(|t| t == TARGET))
        .and_then(|(name, a)| Some((name.clone(), a.checksums.get("sha256")?.clone())))
        .ok_or_else(|| io::Error::other(format!("cozy {version} has no build for {TARGET}")))?;
    let url = format!("{base}/{name}");
    let archive = super::pypi::object_at(machine, &url, &sha256, None, None, stop).map_err(|e| io::Error::other(format!("{url}: {e}")))?;
    fs::create_dir_all(dir)?;
    let binary = dir.join(format!("cozy-{version}"));
    let mut out = fs::File::create(&binary)?;
    extract(&fs::read(archive)?, &mut out)?;
    Ok(binary)
}

/// dist's `.tar.gz`: one top directory holding `cozy`.
#[cfg(unix)]
fn extract(archive: &[u8], out: &mut impl io::Write) -> io::Result<u64> {
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(archive));
    for entry in tar.entries()? {
        let mut entry = entry?;
        if entry.path()?.file_name() == Some("cozy".as_ref()) {
            return io::copy(&mut entry, out);
        }
    }
    Err(io::Error::new(io::ErrorKind::NotFound, "the archive holds no cozy"))
}

/// dist's `.zip`: `cozy.exe` at its root.
#[cfg(windows)]
fn extract(archive: &[u8], out: &mut impl io::Write) -> io::Result<u64> {
    let mut zip = zip::ZipArchive::new(io::Cursor::new(archive))?;
    io::copy(&mut zip.by_name("cozy.exe")?, out)
}
