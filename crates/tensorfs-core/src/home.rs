//! TensorFS-owned default Store location.
//!
//! Core Store operations remain explicit-root. This resolver exists for user-facing entrypoints
//! and never consults another product's home or configuration.

use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use crate::catalog::WriterGuard;
use crate::err::{refuse, Code, Result};
use crate::store::{self, Store};

pub const ENV: &str = "TENSORFS_HOME";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveReport {
    pub source: PathBuf,
    pub destination: PathBuf,
}

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

/// Move one stopped Store into the resolved TensorFS home without copying or rehashing it.
///
/// The destination must not exist and both parents must be on the same filesystem. The Store's
/// recovery lock excludes readers and writers during the atomic rename. No redirect is left at
/// the old path.
pub fn move_store(source: &Path, explicit_destination: Option<&Path>) -> Result<MoveReport> {
    let source_metadata = fs::symlink_metadata(source).map_err(|error| io("stat source", error))?;
    if source_metadata.file_type().is_symlink() || !source_metadata.is_dir() {
        return refuse(
            Code::STORE_ERA,
            format!("{} is not a TensorFS Store directory", source.display()),
        );
    }
    let source = fs::canonicalize(source).map_err(|error| io("resolve source", error))?;
    Store::open(&source)?;

    let requested_destination = resolve(explicit_destination)?;
    match fs::symlink_metadata(&requested_destination) {
        Ok(_) => {
            return refuse(
                Code::IO_FAILED,
                format!(
                    "destination {} already exists; remove the conflict or choose another root",
                    requested_destination.display()
                ),
            )
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io("stat destination", error)),
    }
    let destination_name =
        requested_destination
            .file_name()
            .ok_or_else(|| crate::err::Refusal {
                code: Code::IO_FAILED,
                detail: format!(
                    "destination {} has no final path component",
                    requested_destination.display()
                ),
            })?;
    let destination_parent = requested_destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let destination_parent = fs::canonicalize(destination_parent)
        .map_err(|error| io("resolve destination parent", error))?;
    let destination = destination_parent.join(destination_name);
    if source == destination {
        return refuse(Code::IO_FAILED, "source and destination are the same Store");
    }
    let source_parent = source.parent().ok_or_else(|| crate::err::Refusal {
        code: Code::IO_FAILED,
        detail: format!("source {} has no parent", source.display()),
    })?;
    let source_device = fs::metadata(&source)
        .map_err(|error| io("stat source", error))?
        .dev();
    let destination_device = fs::metadata(&destination_parent)
        .map_err(|error| io("stat destination parent", error))?
        .dev();
    if source_device != destination_device {
        return refuse(
            Code::IO_FAILED,
            format!(
                "source {} and destination {} are on different filesystems; a hardcut requires an atomic rename",
                source.display(),
                destination.display()
            ),
        );
    }

    let _exclusive = WriterGuard::lock_rebuild(&source)?;
    fs::rename(&source, &destination).map_err(|error| io("rename Store", error))?;
    store::fsync_dir(&destination_parent)?;
    if source_parent != destination_parent {
        store::fsync_dir(source_parent)?;
    }
    Store::open(&destination)?;
    Ok(MoveReport {
        source,
        destination,
    })
}

fn io(what: &str, error: std::io::Error) -> crate::err::Refusal {
    crate::err::Refusal {
        code: Code::IO_FAILED,
        detail: format!("{what}: {error}"),
    }
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
