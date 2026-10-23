//! What `cozy run` runs: `[org/]pkg[/fn][@release]`, `./dir[/fn]` or `./script.py`. A package
//! with no org is first-party, the root's (xs-071).
use std::path::{Path, PathBuf};

use protocol::error::{Error, Kind};
use protocol::names::ROOT;

#[derive(Debug, PartialEq, Eq)]
pub enum Target {
    Published {
        /// `org/pkg`, lowercase (names are never case-sensitive); `_/pkg` for a root package.
        package: String,
        function: Option<String>,
        /// Empty: the machine runs the release it holds, else the newest.
        release: Option<String>,
        /// `x/y`'s other reading, account x's package y, when `package` is root package x with
        /// function y. Accounts and root names never collide, so the Hub holds at most one.
        owned: Option<String>,
    },
    Directory {
        path: PathBuf,
        function: Option<String>,
    },
    Script(PathBuf),
}

impl Target {
    /// Reads a target. `cwd` decides whether a spelling names a local file or directory.
    pub fn parse(raw: &str, cwd: &Path) -> Result<Self, Error> {
        let raw = raw.trim();
        let explicit = ["./", "../", "/", "~/"].iter().any(|p| raw.starts_with(p)) || raw == ".";
        if raw.ends_with(".py") && (explicit || cwd.join(raw).is_file()) {
            return Ok(Target::Script(PathBuf::from(raw)));
        }
        if explicit {
            let path = PathBuf::from(raw);
            if cwd.join(&path).is_dir() {
                return Ok(Target::Directory {
                    path,
                    function: None,
                });
            }
            let function = path.file_name().map(|n| n.to_string_lossy().into_owned());
            let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
            if !cwd.join(&parent).is_dir() {
                return Err(Error::new(
                    Kind::NotFound,
                    "target_not_found",
                    format!("{raw} is not a package directory or a script on this computer"),
                ));
            }
            return Ok(Target::Directory {
                path: parent,
                function,
            });
        }
        let (path, release) = match raw.rsplit_once('@') {
            Some((path, release)) => (path, Some(release)),
            None => (raw, None),
        };
        let parts: Vec<&str> = path.split('/').collect();
        let malformed = || {
            Error::new(
                Kind::Usage,
                "target_malformed",
                format!("{raw:?} is not [org/]package[/function][@release]"),
            )
            .remedy("omit /function to list the package's functions; ./dir or ./script.py runs local code")
            .next("cozy package search <words>")
        };
        if parts.len() == 4 {
            return Err(malformed().remedy(format!(
                "a release goes last: {}/{}/{}@{}",
                parts[0], parts[1], parts[3], parts[2]
            )));
        }
        if parts.len() > 3 || parts.iter().any(|p| p.is_empty()) || release.is_some_and(str::is_empty) {
            return Err(malformed());
        }
        let release = release.map(str::to_owned);
        let lower = |s: &str| s.to_lowercase();
        Ok(match parts[..] {
            [name] => Target::Published { package: format!("{ROOT}/{}", lower(name)), function: None, release, owned: None },
            [first, second] if first == ROOT => Target::Published {
                package: format!("{ROOT}/{}", lower(second)), function: None, release, owned: None,
            },
            [first, second] => Target::Published {
                package: format!("{ROOT}/{}", lower(first)),
                function: Some(second.to_owned()),
                release,
                owned: Some(format!("{}/{}", lower(first), lower(second))),
            },
            _ => Target::Published {
                package: format!("{}/{}", lower(parts[0]), lower(parts[1])),
                function: Some(parts[2].to_owned()),
                release,
                owned: None,
            },
        })
    }
}

/// The package and function a published target names, and its release: for `x/y`, root package
/// x's function y when the Hub holds root package x, else account x's package y (one read when it
/// is the root's; an older Hub answers the root's absent).
pub(crate) async fn resolve(
    ctx: &crate::Context,
    hub: &protocol::config::Hub,
    package: String,
    function: Option<String>,
    release: Option<&str>,
    owned: Option<String>,
) -> Result<(String, Option<String>, super::Resolved), Error> {
    let root = super::resolve(ctx, hub, &package, release).await;
    match (root, owned) {
        (Ok(resolved), _) => Ok((package, function, resolved)),
        (Err(e), Some(owned)) if matches!(e.kind, Kind::NotFound | Kind::Usage) => {
            match super::resolve(ctx, hub, &owned, release).await {
                Ok(resolved) => Ok((owned, None, resolved)),
                Err(e) if e.kind == Kind::NotFound => {
                    let first = package.trim_start_matches("_/");
                    Err(Error::new(Kind::NotFound, "package_not_found",
                        format!("the Hub has neither a package {first} nor {owned}"))
                        .next(format!("cozy package search {first}")))
                }
                Err(e) => Err(e),
            }
        }
        (Err(e), _) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn published(package: &str, function: Option<&str>, release: Option<&str>) -> Target {
        Target::Published {
            package: package.into(),
            function: function.map(Into::into),
            release: release.map(Into::into),
            owned: None,
        }
    }

    fn either(root: &str, function: &str, release: Option<&str>, owned: &str) -> Target {
        Target::Published {
            package: root.into(),
            function: Some(function.into()),
            release: release.map(Into::into),
            owned: Some(owned.into()),
        }
    }

    #[test]
    fn targets_read_as_written() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("pkg")).unwrap();
        std::fs::write(dir.path().join("try.py"), "").unwrap();
        std::fs::write(dir.path().join("fidika"), "").unwrap();
        let cwd = dir.path();
        for (raw, want) in [
            (
                "fidika/sdxl/generate",
                published("fidika/sdxl", Some("generate"), None),
            ),
            ("Fidika/SDXL", either("_/fidika", "SDXL", None, "fidika/sdxl")),
            ("SDXL", published("_/sdxl", None, None)),
            ("sdxl/generate", either("_/sdxl", "generate", None, "sdxl/generate")),
            ("_/sdxl", published("_/sdxl", None, None)),
            ("_/sdxl/generate@2.12.1", published("_/sdxl", Some("generate"), Some("2.12.1"))),
            (
                "fidika/sdxl/generate@2.10.0",
                published("fidika/sdxl", Some("generate"), Some("2.10.0")),
            ),
            (
                "fidika/sdxl@2.9.0",
                either("_/fidika", "sdxl", Some("2.9.0"), "fidika/sdxl"),
            ),
            (
                "./pkg",
                Target::Directory {
                    path: "./pkg".into(),
                    function: None,
                },
            ),
            (
                "./pkg/generate",
                Target::Directory {
                    path: "./pkg".into(),
                    function: Some("generate".into()),
                },
            ),
            ("try.py", Target::Script("try.py".into())),
            ("./missing.py", Target::Script("./missing.py".into())),
            ("org/name.py", either("_/org", "name.py", None, "org/name.py")),
        ] {
            assert_eq!(Target::parse(raw, cwd).unwrap(), want, "{raw}");
        }
        for (raw, kind, code) in [
            ("fidika//generate", Kind::Usage, "target_malformed"),
            ("fidika/sdxl/generate@", Kind::Usage, "target_malformed"),
            (
                "fidika/sdxl/2.10.0/generate",
                Kind::Usage,
                "target_malformed",
            ),
            ("./nowhere/generate", Kind::NotFound, "target_not_found"),
        ] {
            let error = Target::parse(raw, cwd).unwrap_err();
            assert_eq!((error.kind, error.code.as_str()), (kind, code), "{raw}");
        }
        let release_last = Target::parse("fidika/sdxl/2.10.0/generate", cwd).unwrap_err();
        assert!(
            release_last
                .remedy
                .unwrap()
                .contains("fidika/sdxl/generate@2.10.0")
        );
    }
}
