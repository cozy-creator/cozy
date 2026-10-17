//! The wheels a release carries for its git dependencies. Machines install everything else straight
//! from the release's `uv.lock`, by the URLs and hashes locked there; a git source has no wheel to
//! lock, so publish builds it at its pinned commit and the machine installs it by its Hub digest.
use std::path::{Path, PathBuf};

use cozy_proto::error::{Error, Kind};
use serde::Deserialize;
use sha2::{Digest as _, Sha256};

use super::author::{Stage, refuse, run, uv};

/// A git source as the lock spells it: `https://host/repo?rev=<rev>#<commit>`.
struct Vcs {
    url: String,
    requested: Option<String>,
    commit: String,
}

impl Vcs {
    fn parse(locked: &str) -> Self {
        let (head, commit) = locked.split_once('#').unwrap_or((locked, ""));
        let (url, query) = head.split_once('?').unwrap_or((head, ""));
        let requested = query
            .split('&')
            .find_map(|pair| pair.strip_prefix("rev="))
            .map(str::to_owned);
        Self {
            url: url.to_owned(),
            requested,
            commit: commit.to_owned(),
        }
    }
}

/// The staged project's git-sourced packages, built to wheels.
pub fn git_wheels(stage: &Stage, python: &str) -> Result<Vec<PathBuf>, Error> {
    #[derive(Deserialize)]
    struct Lock {
        #[serde(default)]
        package: Vec<Locked>,
    }
    #[derive(Deserialize)]
    struct Locked {
        name: String,
        #[serde(default)]
        source: toml::Table,
    }
    let text =
        std::fs::read_to_string(stage.src.join("uv.lock")).map_err(|e| refuse("lock_failed", e))?;
    let lock: Lock = toml::from_str(&text).map_err(|e| refuse("lock_invalid", e))?;
    lock.package
        .iter()
        .filter_map(|p| Some((p, Vcs::parse(p.source.get("git")?.as_str()?))))
        .map(|(p, vcs)| git_wheel(stage, python, &p.name.to_ascii_lowercase(), &vcs))
        .collect()
}

/// A git dependency pinned to a full commit, built to a wheel whose build tag names its bytes:
/// `<name>-<version>-0<12 hex>-<tags>.whl`. A branch or tag could move, so it is refused.
fn git_wheel(stage: &Stage, python: &str, name: &str, vcs: &Vcs) -> Result<PathBuf, Error> {
    let commit = vcs.commit.as_str();
    let pinned = commit.len() == 40 && commit.bytes().all(|b| b.is_ascii_hexdigit());
    if !pinned || vcs.requested.as_deref() != Some(commit) {
        return Err(Error::new(
            Kind::Validation,
            "git_dependency_unpinned",
            format!(
                "{name} is locked to git {} at a revision that can move",
                vcs.url
            ),
        )
        .remedy(format!(
            "depend on `{name} @ git+{}@<40-hex commit>`",
            vcs.url
        )));
    }
    let checkout = stage.root.join("git").join(name);
    std::fs::create_dir_all(&checkout).map_err(|e| refuse("git_dependency_checkout", e))?;
    let git = |args: &[&str]| -> Result<String, Error> {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(&checkout)
            .args(args)
            .output()
            .map_err(|e| refuse("git_unavailable", e))?;
        match out.status.success() {
            true => Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned()),
            false => Err(refuse(
                "git_dependency_fetch",
                String::from_utf8_lossy(&out.stderr),
            )),
        }
    };
    git(&["init", "-q"])?;
    git(&["fetch", "-q", "--depth", "1", vcs.url.as_str(), commit])?;
    git(&["checkout", "-q", "FETCH_HEAD"])?;
    let epoch = git(&["log", "-1", "--format=%ct"])?;
    let out = stage.root.join("git-wheels").join(name);
    run(
        uv(&checkout)
            .args([
                "build",
                "--wheel",
                "--no-progress",
                "--python",
                python,
                "--out-dir",
            ])
            .arg(&out)
            .env("SOURCE_DATE_EPOCH", &epoch),
        "git_dependency_build",
    )?;
    let built = single(&out, ".whl")?;
    let bytes = std::fs::read(&built).map_err(|e| refuse("git_dependency_wheel", e))?;
    let tag = &hex::encode(Sha256::digest(&bytes))[..12];
    let file = built
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or_default();
    let mut parts: Vec<&str> = file.splitn(3, '-').collect();
    let renamed = match parts.len() {
        3 => {
            let build = format!("0{tag}");
            parts.insert(2, &build);
            parts.join("-")
        }
        _ => {
            return Err(refuse(
                "git_dependency_wheel",
                format!("{file} is not a wheel name"),
            ));
        }
    };
    let target = out.join(renamed);
    std::fs::rename(&built, &target).map_err(|e| refuse("git_dependency_wheel", e))?;
    Ok(target)
}

/// The one file with `suffix` in `dir`.
pub fn single(dir: &Path, suffix: &str) -> Result<PathBuf, Error> {
    let found: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| refuse("build_output_unreadable", e))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.to_string_lossy().ends_with(suffix))
        .collect();
    match <[PathBuf; 1]>::try_from(found) {
        Ok([path]) => Ok(path),
        Err(found) => Err(refuse(
            "build_output_unexpected",
            format!(
                "uv build left {} {suffix} files in {}",
                found.len(),
                dir.display()
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_locked_git_source_names_its_commit() {
        let vcs = Vcs::parse(
            "https://github.com/huggingface/diffusers?rev=7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941#7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941",
        );
        assert_eq!(vcs.url, "https://github.com/huggingface/diffusers");
        assert_eq!(vcs.requested.as_deref(), Some(vcs.commit.as_str()));
        assert!(
            Vcs::parse("https://x/y?branch=main#abc")
                .requested
                .is_none()
        );
    }
}
