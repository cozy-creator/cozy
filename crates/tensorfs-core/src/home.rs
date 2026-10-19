//! TensorFS-owned default Store location.
//!
//! Core Store operations remain explicit-root. This resolver exists for user-facing entrypoints
//! and never consults another product's home or configuration.
//!
//! A root the operator NAMED — a positional argument or `TENSORFS_HOME` — is a place they asked
//! for. `$HOME/.tensorfs` is a guess, so it is only ever OPENED: no command conjures a Store
//! there, and nothing creates a Store in whatever directory a process happens to be run from
//! (tfs-053). Creation stays with the verbs that say so, `tfs store init` and `tfs store ensure`,
//! over a root that was named.

use std::path::{Path, PathBuf};

use crate::err::{refuse, Code, Result};

pub const ENV: &str = "TENSORFS_HOME";

/// Where a resolved root came from. `Named` roots were asked for; `Fallback` is a guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origin {
    Named,
    Fallback,
}

/// Resolve an optional caller-supplied Store root.
///
/// Precedence is explicit argument, `TENSORFS_HOME`, then `$HOME/.tensorfs`. This answers WHERE
/// a Store would live; it neither requires nor creates one. `tfs home` prints exactly this.
pub fn resolve(explicit: Option<&Path>) -> Result<PathBuf> {
    Ok(resolve_env(explicit)?.0)
}

/// The Store root of a command that needs a Store.
///
/// A named root — argument or `TENSORFS_HOME` — is used as given, and `tfs store init`/`ensure`
/// may bring it into existence. The `$HOME/.tensorfs` fallback is accepted only for a Store that
/// ALREADY exists: with nothing named and nothing there, every command refuses instead of
/// materializing a database nobody asked for.
pub fn store_root(explicit: Option<&Path>) -> Result<PathBuf> {
    let (root, origin) = resolve_env(explicit)?;
    if origin == Origin::Named || root.is_dir() {
        return Ok(root);
    }
    refuse(
        Code::STORE_ROOT_ABSENT,
        format!(
            "no Store root was given and the default {} does not exist: name a root, set {ENV}, \
             or create one with `tfs store init <root>`",
            root.display()
        ),
    )
}

fn resolve_env(explicit: Option<&Path>) -> Result<(PathBuf, Origin)> {
    resolve_values(
        explicit,
        std::env::var_os(ENV).map(PathBuf::from).as_deref(),
        std::env::var_os("HOME").map(PathBuf::from).as_deref(),
    )
}

fn resolve_values(
    explicit: Option<&Path>,
    tensorfs_home: Option<&Path>,
    user_home: Option<&Path>,
) -> Result<(PathBuf, Origin)> {
    if let Some(root) = explicit {
        return Ok((nonempty("explicit TensorFS root", root)?, Origin::Named));
    }
    if let Some(root) = tensorfs_home {
        return Ok((nonempty(ENV, root)?, Origin::Named));
    }
    match user_home {
        Some(root) if !root.as_os_str().is_empty() => {
            Ok((root.join(".tensorfs"), Origin::Fallback))
        }
        _ => refuse(
            Code::IO_FAILED,
            "TensorFS root is absent: pass one explicitly, set TENSORFS_HOME, or set HOME",
        ),
    }
}

fn nonempty(source: &str, root: &Path) -> Result<PathBuf> {
    if root.as_os_str().is_empty() {
        return refuse(Code::IO_FAILED, format!("{source} is empty"));
    }
    Ok(root.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedence_is_explicit_then_tensorfs_home_then_user_home() {
        assert_eq!(
            resolve_values(
                Some(Path::new("/explicit")),
                Some(Path::new("/environment")),
                Some(Path::new("/user")),
            )
            .unwrap(),
            (PathBuf::from("/explicit"), Origin::Named),
        );
        assert_eq!(
            resolve_values(
                None,
                Some(Path::new("/environment")),
                Some(Path::new("/user"))
            )
            .unwrap(),
            (PathBuf::from("/environment"), Origin::Named),
        );
        assert_eq!(
            resolve_values(None, None, Some(Path::new("/user"))).unwrap(),
            (PathBuf::from("/user/.tensorfs"), Origin::Fallback),
        );
    }

    #[test]
    fn an_empty_selected_value_refuses_instead_of_falling_through() {
        assert_eq!(
            resolve_values(None, Some(Path::new("")), Some(Path::new("/user")))
                .unwrap_err()
                .code,
            Code::IO_FAILED,
        );
        assert_eq!(
            resolve_values(None, None, None).unwrap_err().code,
            Code::IO_FAILED,
        );
    }
}
