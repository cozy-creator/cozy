//! A Runtime or TensorFS wheel the machine fetches itself: published on PyPI, checked against
//! the index's own SHA-256, or at any https URL, held to the digest its update names. It lands
//! through the machine's TensorFS store as a package file does: a download that stops moving
//! fails on its measured stillness, `stop` ends it, and what landed stays for the next ask.
use std::{
    fs, io,
    path::{Path, PathBuf},
};
use tensorfs_core::{
    err::Code,
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
    land(&blob, dir, &file.filename)
}

/// Wheel `file` at an https `url` on any host, redirects followed: its bytes are held to
/// `sha256`, so no host vouches for anything. Other bytes fail `wheel_digest_mismatch`.
pub fn wheel_at(
    url: &str,
    sha256: &str,
    file: &str,
    dir: &Path,
    store: &Store,
    stop: &PullCancellation,
) -> io::Result<PathBuf> {
    let policy = SourcePolicy {
        any_host: true,
        allow_local: transport::is_loopback_origin(url),
        max_redirects: 8,
        ..Default::default()
    };
    // A private server's root as uv trusts it: each certificate of the bundle SSL_CERT_FILE
    // names. Only this fetch reads it; the Hub's clients trust their own CA alone.
    if let Some(bundle) = std::env::var_os("SSL_CERT_FILE") {
        let pem = fs::read_to_string(&bundle).unwrap_or_default();
        let certificates = pem.split_inclusive("-----END CERTIFICATE-----");
        if certificates.filter(|c| transport::trust_roots(c.as_bytes()).is_ok()).count() == 0 {
            tracing::warn!("SSL_CERT_FILE {} holds no certificate", bundle.to_string_lossy());
        }
    }
    let ledger = Ledger::new().with_cancellation(stop.clone());
    let failed = |e: tensorfs_core::err::Refusal| match e.code {
        Code::OBJECT_ID_MISMATCH => io::Error::other(super::update::Coded {
            code: "wheel_digest_mismatch",
            message: format!("{file} from {url}: {}", e.detail),
        }),
        _ => io::Error::other(format!("{file} from {url}: {e}")),
    };
    let length = transport::probe_length(url, &policy, &Anonymous, Deadline::none(), &ledger).map_err(failed)?;
    let object = ObjectRef { sha256: sha256.into(), length };
    let blob = crate::published::fetch_file(store, url, &object, &policy, &ledger).map_err(failed)?;
    land(&blob, dir, file)
}

fn land(blob: &Path, dir: &Path, file: &str) -> io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join(file);
    let _ = fs::remove_file(&path);
    fs::hard_link(blob, &path).or_else(|_| fs::copy(blob, &path).map(drop))?;
    Ok(path)
}
