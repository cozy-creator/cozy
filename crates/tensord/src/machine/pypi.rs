//! A published Runtime or TensorFS wheel for this platform from PyPI, checked against the
//! index's own SHA-256. It lands through the machine's TensorFS store as a package file does:
//! a download that stops moving fails on its measured stillness, `stop` ends it, and what
//! landed stays for the next ask.
use std::{
    fs, io,
    path::{Path, PathBuf},
};
use tensorfs_core::{
    ids::ObjectRef,
    store::Store,
    transport::{self, Anonymous, Deadline, Ledger, PullCancellation, SourcePolicy},
};

const MAX_INDEX_BYTES: u64 = 32 << 20;

pub fn wheel(
    distribution: &str,
    version: &str,
    dir: &Path,
    store: &Store,
    stop: &PullCancellation,
) -> io::Result<PathBuf> {
    let project = distribution.replace('_', "-");
    let policy = SourcePolicy {
        allowed_hosts: vec!["pypi.org".into(), "files.pythonhosted.org".into()],
        ..Default::default()
    };
    let ledger = Ledger::new().with_cancellation(stop.clone());
    let (index, _) = transport::api_get(
        &format!("https://pypi.org/pypi/{project}/{version}/json"),
        &policy,
        &Anonymous,
        MAX_INDEX_BYTES,
        Deadline::none(),
        &ledger,
    )
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
    let release: Release = serde_json::from_slice(&index).map_err(io::Error::other)?;
    let arch = std::env::consts::ARCH;
    let file = release
        .urls
        .into_iter()
        .find(|f| {
            f.filename.ends_with(".whl")
                && f.filename.contains("manylinux")
                && f.filename.contains(arch)
                && !f.filename.contains('/')
        })
        .ok_or_else(|| {
            io::Error::other(format!("PyPI has no linux {arch} wheel of {project} {version}"))
        })?;
    let object = ObjectRef { sha256: file.digests.sha256.to_lowercase(), length: file.size };
    let blob = crate::published::fetch_file(store, &file.url, &object, &policy, &ledger)
        .map_err(|e| io::Error::other(format!("{}: {e}", file.filename)))?;
    fs::create_dir_all(dir)?;
    let path = dir.join(&file.filename);
    let _ = fs::remove_file(&path);
    fs::hard_link(&blob, &path).or_else(|_| fs::copy(&blob, &path).map(drop))?;
    Ok(path)
}
