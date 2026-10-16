//! What `cozy run` runs: `org/pkg[/fn][@release]`, `./dir[/fn]` or `./script.py`.
use std::path::{Path, PathBuf};

use cozy_proto::error::{Error, Kind};

#[derive(Debug, PartialEq, Eq)]
pub enum Target {
    Published {
        /// `org/pkg`, lowercase: names are never case-sensitive.
        package: String,
        function: Option<String>,
        /// Empty: the machine runs the release it holds, else the newest.
        release: Option<String>,
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
                format!("{raw:?} is not org/package[/function][@release]"),
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
        if !(2..=3).contains(&parts.len())
            || parts.iter().any(|p| p.is_empty())
            || release.is_some_and(str::is_empty)
        {
            return Err(malformed());
        }
        Ok(Target::Published {
            package: format!("{}/{}", parts[0], parts[1]).to_lowercase(),
            function: parts.get(2).map(|f| (*f).to_owned()),
            release: release.map(str::to_owned),
        })
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
            ("Fidika/SDXL", published("fidika/sdxl", None, None)),
            (
                "fidika/sdxl/generate@2.10.0",
                published("fidika/sdxl", Some("generate"), Some("2.10.0")),
            ),
            (
                "fidika/sdxl@2.9.0",
                published("fidika/sdxl", None, Some("2.9.0")),
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
            ("org/name.py", published("org/name.py", None, None)),
        ] {
            assert_eq!(Target::parse(raw, cwd).unwrap(), want, "{raw}");
        }
        for (raw, kind, code) in [
            ("sdxl", Kind::Usage, "target_malformed"),
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
