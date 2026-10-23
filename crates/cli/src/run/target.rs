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
