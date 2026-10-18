//! A local directory given to a Tree input: its files, and the TensorFS manifest naming them that
//! the machine materializes the tree from. Both are sent with the run like any input file.
use std::path::{Path, PathBuf};

use cozy_proto::{
    calcifer::StagedInput,
    error::{Error, Kind},
    ids::Digest,
};
use tensorfs_core::{
    ids::{Doc, ObjectRef},
    manifest::Manifest,
};

/// How a run's spec marks an input that is a tree's manifest.
pub const MEDIA: &str = "application/vnd.cozy.tree-manifest";

/// A directory staged: its manifest (written under `manifests`) and the files it names.
pub struct Staged {
    pub manifest: StagedInput,
    pub files: Vec<StagedInput>,
}

/// Stages `root`'s files, each digest remembered in `memo` as an input file's is.
pub fn stage(root: &Path, memo: Option<&Path>, manifests: &Path) -> Result<Staged, Error> {
    let mut found = vec![];
    walk(root, root, &mut found)?;
    found.sort();
    let mut files = vec![];
    let mut entries = vec![];
    for (name, path) in found {
        let length = std::fs::metadata(&path)
            .map_err(|e| unreadable(&path, e))?
            .len();
        let digest = super::digests::of(memo, &path)?;
        let sha256 = hex::encode(digest.bytes());
        entries.push((name, ObjectRef { sha256, length }));
        files.push(StagedInput {
            path,
            digest,
            length,
        });
    }
    let manifest = Manifest::from_files(entries).map_err(|e| {
        Error::new(
            Kind::Validation,
            "tree_invalid",
            format!("{}: {e}", root.display()),
        )
    })?;
    let bytes = manifest.canonical_bytes();
    let digest = Digest::of(&bytes);
    let path = manifests.join(hex::encode(digest.bytes()));
    if !path.is_file() {
        let write = std::fs::create_dir_all(manifests).and_then(|()| std::fs::write(&path, &bytes));
        write.map_err(|e| unreadable(&path, e))?;
    }
    let length = bytes.len() as u64;
    let manifest = StagedInput {
        path,
        digest,
        length,
    };
    Ok(Staged { manifest, files })
}

/// Every file under `dir`, named by its path below `root` with `/` between its parts.
fn walk(root: &Path, dir: &Path, found: &mut Vec<(String, PathBuf)>) -> Result<(), Error> {
    let entries = std::fs::read_dir(dir).map_err(|e| unreadable(dir, e))?;
    for entry in entries {
        let path = entry.map_err(|e| unreadable(dir, e))?.path();
        let kind = std::fs::metadata(&path).map_err(|e| unreadable(&path, e))?;
        if kind.is_dir() {
            walk(root, &path, found)?;
            continue;
        }
        let relative = path.strip_prefix(root).expect("under its root");
        let parts: Vec<String> = relative
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        found.push((parts.join("/"), path));
    }
    Ok(())
}

fn unreadable(path: &Path, e: std::io::Error) -> Error {
    Error::new(
        Kind::Usage,
        "tree_unreadable",
        format!("{}: {e}", path.display()),
    )
}
