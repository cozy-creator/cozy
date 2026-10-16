//! What a release's machines install besides its own wheel: one registry row per locked
//! dependency (the wheel for CPython on Linux x86_64 at the release's Python), and wheels built
//! from full-commit git dependencies, which the release carries.
//!
//! Today's Hub turns the rows into each machine's locked requirements. When machines read the
//! release's `uv.lock` themselves, this module goes.
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    str::FromStr as _,
};

use cozy_proto::error::{Error, Kind};
use hub_client::types::PackageRegistryPin;
use pep508_rs::{MarkerEnvironment, MarkerEnvironmentBuilder, MarkerTree};
use serde::Deserialize;
use sha2::{Digest as _, Sha256};

use super::author::{Stage, refuse, run, uv};

/// The oldest glibc on the machines' images.
const GLIBC: u32 = 36;

#[derive(Deserialize)]
struct PyLock {
    #[serde(default)]
    packages: Vec<Locked>,
}

#[derive(Deserialize)]
struct Locked {
    name: String,
    version: Option<String>,
    marker: Option<String>,
    #[serde(default)]
    wheels: Vec<Wheel>,
    vcs: Option<Vcs>,
}

#[derive(Deserialize)]
struct Wheel {
    url: String,
    #[serde(default)]
    size: i64,
    #[serde(default)]
    hashes: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct Vcs {
    url: String,
    #[serde(rename = "requested-revision")]
    requested: Option<String>,
    #[serde(rename = "commit-id")]
    commit: Option<String>,
}

/// The registry rows of the staged project's lock, and the wheels its git dependencies build to.
pub fn dependencies(
    stage: &Stage,
    python: &str,
) -> Result<(Vec<PackageRegistryPin>, Vec<PathBuf>), Error> {
    let exported = stage.root.join("pylock.toml");
    run(
        uv(&stage.src)
            .args([
                "export",
                "--frozen",
                "--no-dev",
                "--no-default-groups",
                "--no-emit-project",
            ])
            .args([
                "--no-emit-local",
                "--format",
                "pylock.toml",
                "--python",
                python,
                "--output-file",
            ])
            .arg(&exported),
        "registry_export_failed",
    )?;
    let text =
        std::fs::read_to_string(&exported).map_err(|e| refuse("registry_export_unreadable", e))?;
    let lock: PyLock = toml::from_str(&text).map_err(|e| refuse("registry_export_invalid", e))?;
    let env = environment(python)?;
    let (mut rows, mut wheels) = (vec![], vec![]);
    for package in lock.packages {
        if let Some(marker) = &package.marker {
            let tree =
                MarkerTree::from_str(marker).map_err(|e| refuse("registry_marker_invalid", e))?;
            if !tree.evaluate(&env, &[]) {
                continue;
            }
        }
        let name = package.name.to_ascii_lowercase();
        let version = package.version.unwrap_or_default();
        if let Some(vcs) = package.vcs {
            wheels.push(git_wheel(stage, python, &name, &vcs)?);
            continue;
        }
        let wheel = choose(&package.wheels, python).ok_or_else(|| {
            let detail = match package.wheels.is_empty() {
                true => "is available only as source",
                false => "has no wheel for CPython on Linux x86_64",
            };
            Error::new(
                Kind::Validation,
                "registry_dependency_platform_mismatch",
                format!("{name}=={version} {detail} at Python {python}"),
            )
            .remedy("depend on a release that ships a wheel for Linux x86_64")
        })?;
        let sha256 = wheel.hashes.get("sha256").cloned().unwrap_or_default();
        rows.push(PackageRegistryPin {
            name,
            version,
            url: wheel.url.clone(),
            sha256,
            size: wheel.size,
        });
    }
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((rows, wheels))
}

fn environment(python: &str) -> Result<MarkerEnvironment, Error> {
    let minor = python.splitn(3, '.').take(2).collect::<Vec<_>>().join(".");
    MarkerEnvironment::try_from(MarkerEnvironmentBuilder {
        implementation_name: "cpython",
        implementation_version: python,
        os_name: "posix",
        platform_machine: "x86_64",
        platform_python_implementation: "CPython",
        platform_release: "",
        platform_system: "Linux",
        platform_version: "",
        python_full_version: python,
        python_version: &minor,
        sys_platform: "linux",
    })
    .map_err(|e| refuse("python_version_invalid", e))
}

/// The wheel machines install: a pure one when there is one, else the most specific CPython tag
/// with the newest manylinux x86_64 at or below the images' glibc.
fn choose<'a>(wheels: &'a [Wheel], python: &str) -> Option<&'a Wheel> {
    let minor: u32 = python.split('.').nth(1).and_then(|m| m.parse().ok())?;
    wheels
        .iter()
        .filter_map(|wheel| {
            let file = wheel
                .url
                .rsplit('/')
                .next()?
                .strip_suffix(".whl")?
                .to_ascii_lowercase();
            let parts: Vec<&str> = file.split('-').collect();
            let [python_tags, abis, platforms] = parts.get(parts.len().checked_sub(3)?..)? else {
                return None;
            };
            let mut best = None;
            for py in python_tags.split('.') {
                for abi in abis.split('.') {
                    for platform in platforms.split('.') {
                        let score = match platform {
                            "any" if abi == "none" && pure(py, minor) => 1_000_000,
                            _ => match (rank(py, abi, minor), glibc(platform)) {
                                (Some(rank), Some(glibc)) => rank * 1000 + glibc,
                                _ => continue,
                            },
                        };
                        best = best.max(Some(score));
                    }
                }
            }
            best.map(|score| (score, wheel))
        })
        .max_by(|(a, x), (b, y)| a.cmp(b).then_with(|| x.url.cmp(&y.url)))
        .map(|(_, wheel)| wheel)
}

fn tag_minor(tag: &str, prefix: &str) -> Option<u32> {
    tag.strip_prefix(prefix)?.strip_prefix('3')?.parse().ok()
}

fn pure(py: &str, minor: u32) -> bool {
    py == "py3"
        || tag_minor(py, "py").is_some_and(|m| m <= minor)
        || tag_minor(py, "cp") == Some(minor)
}

fn rank(py: &str, abi: &str, minor: u32) -> Option<u32> {
    match abi {
        "abi3" => tag_minor(py, "cp").filter(|m| *m <= minor).map(|m| 300 + m),
        "none" if tag_minor(py, "cp") == Some(minor) => Some(200),
        "none" if py == "py3" => Some(112),
        "none" => tag_minor(py, "py").filter(|m| *m <= minor).map(|m| 100 + m),
        _ if tag_minor(abi, "cp") == Some(minor) && tag_minor(py, "cp") == Some(minor) => Some(400),
        _ => None,
    }
}

/// The glibc minor a manylinux x86_64 tag needs, when the images have it.
fn glibc(platform: &str) -> Option<u32> {
    let minor = match platform {
        "manylinux1_x86_64" => 5,
        "manylinux2010_x86_64" => 12,
        "manylinux2014_x86_64" => 17,
        _ => platform
            .strip_prefix("manylinux_2_")?
            .strip_suffix("_x86_64")?
            .parse()
            .ok()?,
    };
    (minor <= GLIBC).then_some(minor)
}

/// A git dependency pinned to a full commit, built to a wheel whose build tag names its bytes:
/// `<name>-<version>-0<12 hex>-<tags>.whl`. A branch or tag could move, so it is refused.
fn git_wheel(stage: &Stage, python: &str, name: &str, vcs: &Vcs) -> Result<PathBuf, Error> {
    let commit = vcs.commit.as_deref().unwrap_or_default();
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

    fn wheel(file: &str) -> Wheel {
        Wheel {
            url: format!("https://files.pythonhosted.org/x/{file}"),
            size: 1,
            hashes: BTreeMap::new(),
        }
    }

    #[test]
    fn the_pure_wheel_wins_then_the_most_specific_linux_x86_64_one() {
        let av = [
            wheel("av-19.0.1-cp312-abi3-manylinux_2_28_aarch64.whl"),
            wheel("av-19.0.1-cp312-abi3-manylinux_2_28_x86_64.whl"),
            wheel("av-19.0.1-cp312-abi3-musllinux_1_2_x86_64.whl"),
            wheel("av-19.0.1-cp314-cp314t-manylinux_2_28_x86_64.whl"),
        ];
        assert!(
            choose(&av, "3.12.12")
                .unwrap()
                .url
                .ends_with("cp312-abi3-manylinux_2_28_x86_64.whl")
        );
        let cffi = [
            wheel("cffi-2.1.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
            wheel("cffi-2.1.1-cp312-cp312-manylinux_2_39_x86_64.whl"),
            wheel("cffi-2.1.1-cp313-cp313-manylinux_2_17_x86_64.whl"),
        ];
        assert!(
            choose(&cffi, "3.12.12")
                .unwrap()
                .url
                .contains("manylinux2014")
        );
        let mixed = [
            wheel("x-1-py3-none-any.whl"),
            wheel("x-1-cp312-cp312-manylinux_2_28_x86_64.whl"),
        ];
        assert!(
            choose(&mixed, "3.12.12")
                .unwrap()
                .url
                .ends_with("py3-none-any.whl")
        );
        assert!(
            choose(
                &[wheel("y-1-cp313-cp313-manylinux_2_28_x86_64.whl")],
                "3.12.12"
            )
            .is_none()
        );
    }
}
