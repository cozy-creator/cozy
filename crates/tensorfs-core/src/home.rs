//! TensorFS-owned default Store location.
//!
//! Core Store operations remain explicit-root. This resolver exists for user-facing entrypoints
//! and never consults another product's home or configuration.

use std::path::{Path, PathBuf};

use crate::err::{refuse, Code, Result};

pub const ENV: &str = "TENSORFS_HOME";

/// Resolve an optional caller-supplied Store root.
///
/// Precedence is explicit argument, `TENSORFS_HOME`, then `$HOME/.tensorfs`.
pub fn resolve(explicit: Option<&Path>) -> Result<PathBuf> {
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
) -> Result<PathBuf> {
    if let Some(root) = explicit {
        return nonempty("explicit TensorFS root", root);
    }
    if let Some(root) = tensorfs_home {
        return nonempty(ENV, root);
    }
    match user_home {
        Some(root) if !root.as_os_str().is_empty() => Ok(root.join(".tensorfs")),
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
            Path::new("/explicit"),
        );
        assert_eq!(
            resolve_values(
                None,
                Some(Path::new("/environment")),
                Some(Path::new("/user"))
            )
            .unwrap(),
            Path::new("/environment"),
        );
        assert_eq!(
            resolve_values(None, None, Some(Path::new("/user"))).unwrap(),
            Path::new("/user/.tensorfs"),
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
