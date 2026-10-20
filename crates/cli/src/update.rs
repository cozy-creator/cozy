//! `cozy update [VERSION]` (design §5): replaces this binary with a release's, restarts Calcifer on it and
//! asks the local machine to update. The release's dist-manifest.json names this target's archive and its
//! sha256; the release also carries the archive's GitHub attestations (`<archive>.sigstore.jsonl`, one
//! Sigstore bundle per line), and one must verify as this repo's release workflow, pinned by GitHub
//! repository id so a rename keeps verifying and a repo that takes an old name never does.
use std::{
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
    process::ExitCode,
};

use protocol::error::{Error, Kind};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sigstore_verify::{
    VerificationPolicy, VerificationResult,
    trust_root::{SigstoreInstance, TrustedRoot},
    types::Bundle,
};

use crate::{Context, machine::Adopted};

const RELEASES: &str = "https://github.com/cozy-creator/cozy-creator/releases";
const ISSUER: &str = "https://token.actions.githubusercontent.com";
/// This repo's GitHub id, which a rename keeps.
const REPOSITORY_ID: &str = "1345607976";
/// Its release workflow, on any ref.
const WORKFLOW: &str = "/.github/workflows/release.yml@";

/// The dist target whose archive replaces this binary.
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
} else if cfg!(all(windows, target_arch = "aarch64")) {
    "aarch64-pc-windows-msvc"
} else {
    "unreleased"
};

#[derive(clap::Args, Debug)]
pub struct Args {
    /// The release to install, even when it is this one (default: the newest, when newer).
    version: Option<String>,
}

/// The fields of dist's manifest an update reads.
#[derive(Deserialize)]
struct Manifest {
    announcement_tag: String,
    releases: Vec<Release>,
    artifacts: BTreeMap<String, Artifact>,
}

#[derive(Deserialize)]
struct Release {
    app_name: String,
    app_version: String,
}

#[derive(Deserialize)]
struct Artifact {
    kind: String,
    #[serde(default)]
    target_triples: Vec<String>,
    #[serde(default)]
    checksums: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct Updated {
    from: &'static str,
    to: String,
    binary: PathBuf,
    archive: String,
    sha256: String,
    signer: String,
    calcifer: Step,
    machine: Adopted,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    Restarted,
    NotRunning,
    NotPorted,
}

pub async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    let current = env!("CARGO_PKG_VERSION");
    let binary = installed()?;
    let wanted = args.version.as_deref().map(|v| v.trim_start_matches('v'));
    let dir = match (&ctx.config.download_url, wanted) {
        (Some(url), _) => url.as_str().trim_end_matches('/').to_owned(),
        (None, Some(version)) => format!("{RELEASES}/download/v{version}"),
        (None, None) => format!("{RELEASES}/latest/download"),
    };
    let http = reqwest::Client::builder()
        .user_agent(concat!("cozy/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| Error::new(Kind::Internal, "http_client", e.to_string()))?;

    let manifest: Manifest = get(&http, &format!("{dir}/dist-manifest.json"))
        .await?
        .json()
        .await
        .map_err(|e| unavailable(&dir, e))?;
    let version = manifest
        .releases
        .iter()
        .find(|r| r.app_name == "cozy")
        .map(|r| r.app_version.clone())
        .ok_or_else(|| not_found(format!("{dir} holds no cozy release")))?;
    match wanted {
        Some(wanted) if wanted != version => {
            return Err(not_found(format!(
                "{dir} holds cozy {version}, not {wanted}"
            )));
        }
        None if !newer(&version, current) => {
            println!("cozy {current} is the newest release");
            return Ok(ExitCode::SUCCESS);
        }
        _ => {}
    }
    let (name, artifact) = manifest
        .artifacts
        .iter()
        .find(|(_, a)| a.kind == "executable-zip" && a.target_triples.iter().any(|t| t == TARGET))
        .ok_or_else(|| not_found(format!("cozy {version} has no build for {TARGET}")))?;
    let url = match &ctx.config.download_url {
        Some(_) => format!("{dir}/{name}"),
        None => format!("{RELEASES}/download/{}/{name}", manifest.announcement_tag),
    };
    let archive = get(&http, &url)
        .await?
        .bytes()
        .await
        .map_err(|e| unavailable(&url, e))?;
    let sha256 = hex::encode(Sha256::digest(&archive));
    if artifact.checksums.get("sha256") != Some(&sha256) {
        return Err(Error::new(
            Kind::Validation,
            "checksum_mismatch",
            format!("{url} has sha256 {sha256}, not the one its release names"),
        ));
    }
    let signer = attested(&http, &url, &archive).await?;
    let new = unpack(&archive, &binary).map_err(|e| local("unpack", e))?;
    let runs = std::process::Command::new(&new)
        .arg("--version")
        .output()
        .map_err(|e| local("run the new binary", e))?;
    if String::from_utf8_lossy(&runs.stdout).trim() != format!("cozy {version}") {
        return Err(Error::new(
            Kind::Validation,
            "new_binary_refused",
            format!(
                "cozy {version} for {TARGET} does not run here: {}",
                String::from_utf8_lossy(&runs.stderr).trim()
            ),
        ));
    }
    keep_previous(&binary).map_err(|e| local("keep the binary it replaces", e))?;
    self_replace::self_replace(&new).map_err(|e| local("replace this binary", e))?;

    let calcifer = match crate::daemon::restart(ctx, &binary).await {
        Ok(true) => Step::Restarted,
        Ok(false) => Step::NotRunning,
        Err(e) if e.code == "not_ported" => Step::NotPorted,
        Err(e) => {
            return Err(Error {
                remedy: Some(format!(
                    "{} is cozy {version} now; restart Calcifer with `cozy down` and `cozy up`",
                    binary.display()
                )),
                ..e
            });
        }
    };
    let machine = crate::machine::adopt(ctx, &version).await;
    let updated = Updated {
        from: current,
        to: version,
        binary,
        archive: name.clone(),
        sha256,
        signer,
        calcifer,
        machine,
    };
    let kept = matches!(updated.machine, Adopted::Kept { .. });
    if ctx.json {
        println!(
            "{}",
            serde_json::to_string(&updated).expect("an update serializes")
        );
    } else {
        println!(
            "cozy {} → {}: {}",
            updated.from,
            updated.to,
            updated.binary.display()
        );
        println!(
            "  {} sha256 {}, signed by {}",
            updated.archive, updated.sha256, updated.signer
        );
        println!("  calcifer: {}", step(&updated.calcifer));
        println!("  machine: {}", adopted(&updated.machine));
    }
    Ok(if kept {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}

/// Keeps the binary being replaced beside it as `.cozy-previous`, a machine's rollback when the
/// new one never serves (design §5).
fn keep_previous(binary: &Path) -> io::Result<()> {
    let previous = binary.with_file_name(".cozy-previous");
    match std::fs::remove_file(&previous) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
        _ => {}
    }
    std::fs::hard_link(binary, &previous).or_else(|_| std::fs::copy(binary, &previous).map(drop))
}

/// This binary's own file, through any link to it.
fn installed() -> Result<PathBuf, Error> {
    let exe = std::env::current_exe().map_err(|e| local("find this binary", e))?;
    // On Windows, canonicalize answers a `\\?\` path.
    if cfg!(windows) {
        return Ok(exe);
    }
    exe.canonicalize().map_err(|e| local("find this binary", e))
}

/// The release workflow run that signed `archive`, from the bundles the release carries beside it
/// (Sigstore public good).
async fn attested(http: &reqwest::Client, url: &str, archive: &[u8]) -> Result<String, Error> {
    let bundles = format!("{url}.sigstore.jsonl");
    let lines = get(http, &bundles)
        .await?
        .text()
        .await
        .map_err(|e| unavailable(&bundles, e))?;
    signer(&bundles, &lines, archive.into())
}

/// The signer of the first bundle in `lines` that attests `artifact` as this repo's release workflow.
fn signer(
    bundles: &str,
    lines: &str,
    artifact: sigstore_verify::types::Artifact<'_>,
) -> Result<String, Error> {
    let root = TrustedRoot::from_embedded(SigstoreInstance::PublicGood)
        .map_err(|e| Error::new(Kind::Internal, "sigstore_root", e.to_string()))?;
    let policy = VerificationPolicy::any_identity().require_issuer(ISSUER);
    let mut seen = Vec::new();
    for line in lines.lines().filter(|line| !line.trim().is_empty()) {
        let verified = Bundle::from_json(line)
            .map_err(|e| e.to_string())
            .and_then(|bundle| {
                sigstore_verify::verify(artifact.clone(), &bundle, &policy, &root)
                    .map_err(|e| e.to_string())
            })
            .and_then(|result| ours(&result));
        match verified {
            Ok(signer) => return Ok(signer),
            Err(e) => seen.push(e),
        }
    }
    Err(Error::new(
        Kind::Validation,
        "unattested",
        format!(
            "no bundle in {bundles} verifies as the release workflow of GitHub repository {REPOSITORY_ID}: [{}]",
            seen.join("; ")
        ),
    ))
}

/// The signer, when the certificate is this repository's (by id) and the signer is its own release workflow.
fn ours(result: &VerificationResult) -> Result<String, String> {
    let signer = result.identity().map_or("", |id| id.as_str());
    let claims = result.certificate().map(|cert| &cert.ci_claims);
    let repository = claims
        .and_then(|c| c.source_repository_uri.as_deref())
        .unwrap_or_default();
    let id = claims
        .and_then(|c| c.source_repository_identifier.as_deref())
        .unwrap_or_default();
    let release_workflow = !repository.is_empty()
        && signer
            .strip_prefix(repository)
            .is_some_and(|path| path.starts_with(WORKFLOW));
    if result.certificate_verified() && id == REPOSITORY_ID && release_workflow {
        Ok(signer.to_owned())
    } else {
        Err(format!("signed by {signer} (GitHub repository {id})"))
    }
}

/// Writes the archive's `cozy` beside `binary` (a file there can run, and the swap stays on one disk).
fn unpack(archive: &[u8], binary: &Path) -> io::Result<tempfile::TempPath> {
    let mut file = tempfile::Builder::new()
        .prefix(".cozy-update-")
        .suffix(std::env::consts::EXE_SUFFIX)
        .tempfile_in(binary.parent().unwrap_or(Path::new(".")))?;
    extract(archive, &mut file)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o755))?;
    }
    // Closed: neither OS runs a file another handle still writes.
    Ok(file.into_temp_path())
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
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "the archive holds no cozy",
    ))
}

/// dist's `.zip`: `cozy.exe` at its root.
#[cfg(windows)]
fn extract(archive: &[u8], out: &mut impl io::Write) -> io::Result<u64> {
    let mut zip = zip::ZipArchive::new(io::Cursor::new(archive))?;
    io::copy(&mut zip.by_name("cozy.exe")?, out)
}

async fn get(http: &reqwest::Client, url: &str) -> Result<reqwest::Response, Error> {
    let response = http
        .get(url)
        .send()
        .await
        .map_err(|e| unavailable(url, e))?;
    match response.status() {
        status if status.is_success() => Ok(response),
        reqwest::StatusCode::NOT_FOUND => Err(not_found(format!("{url}: not found"))),
        status => Err(Error::new(
            Kind::Unavailable,
            "release_unavailable",
            format!("{url}: {status}"),
        )),
    }
}

/// Whether release `version` is newer than `current`; one that does not parse never is.
/// Whether release `version` replaces `current`: a newer release, or any release over a dev build
/// (`+dev.g<sha>`), even the release it was built from.
fn newer(version: &str, current: &str) -> bool {
    match (
        semver::Version::parse(version),
        semver::Version::parse(current),
    ) {
        (Ok(_), Ok(current)) if current.build.as_str().starts_with("dev") => true,
        (Ok(version), Ok(current)) => version > current,
        _ => false,
    }
}

fn adopted(adopted: &Adopted) -> String {
    match adopted {
        Adopted::NotEnabled => "not enabled".into(),
        Adopted::NotRunning => "not running; it starts on the new binary".into(),
        Adopted::Restarted { version } => format!("restarted on {version}"),
        Adopted::Waits { runs, run } => {
            format!("waits for its {runs} runs; restarts when idle (`cozy run watch {run}`)")
        }
        Adopted::Kept { code, message, run } => match run {
            Some(run) => format!("{code}: {message} (`cozy run show {run}`)"),
            None => format!("{code}: {message}"),
        },
    }
}

fn step(step: &Step) -> &'static str {
    match step {
        Step::Restarted => "restarted on the new binary",
        Step::NotRunning => "not running",
        Step::NotPorted => "not in this build yet",
    }
}

fn not_found(message: String) -> Error {
    Error::new(Kind::NotFound, "release_not_found", message)
}

fn unavailable(url: &str, error: impl std::fmt::Display) -> Error {
    Error::new(
        Kind::Unavailable,
        "release_unavailable",
        format!("{url}: {error}"),
    )
}

fn local(op: &str, error: io::Error) -> Error {
    Error::new(Kind::Internal, "update_failed", format!("{op}: {error}"))
}

#[cfg(test)]
mod tests {
    use sigstore_verify::types::{Artifact, Sha256Hash};

    fn sha256(hex: &str) -> Artifact<'static> {
        Artifact::from_digest(Sha256Hash::from_hex(hex).unwrap())
    }

    /// v0.10.0's own bundle for its x86_64 Linux archive.
    #[test]
    fn a_release_of_this_repo_verifies() {
        let lines = include_str!("../tests/fixtures/sigstore/cozy-v0.10.0.jsonl");
        let archive = sha256("28ec835fdef1d0a1eaf231cfa05d76bdef207ee6387bcb5ae8bf11c3ebc1017c");
        let signer = super::signer("v0.10.0", lines, archive).unwrap();
        assert_eq!(
            signer,
            "https://github.com/cozy-creator/cozy-creator/.github/workflows/release.yml@refs/heads/release-v0.10.0"
        );
    }

    /// A dev build is older than every release, even the one it was built from: `cozy update`
    /// returns it to the newest.
    #[test]
    fn a_release_replaces_a_dev_build() {
        assert!(super::newer("0.10.2", "0.10.2+dev.g1234abcd"));
        assert!(super::newer("0.10.1", "0.10.2+dev.g1234abcd"));
        assert!(!super::newer("0.10.2", "0.10.2"));
        assert!(super::newer("0.10.3", "0.10.2"));
    }

    /// cli/cli's genuine GitHub Actions provenance of its own archive verifies, but is not this repo's.
    #[test]
    fn another_repos_release_is_refused_for_its_signer() {
        let lines = include_str!("../tests/fixtures/sigstore/cli-v2.102.0.jsonl");
        let archive = sha256("bb766f710eef8ede859c18578c72c327597cd4c8a85b06001b1f3843c6019386");
        let refusal = super::signer("gh v2.102.0", lines, archive).unwrap_err();
        assert_eq!(refusal.code, "unattested");
        let cli = "signed by https://github.com/cli/cli/.github/workflows/deployment.yml@refs/heads/trunk \
            (GitHub repository 212613049)";
        assert!(refusal.message.contains(cli), "{}", refusal.message);
    }
}
