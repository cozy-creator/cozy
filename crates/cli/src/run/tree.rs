//! A local directory given to a Tree input: its files, and the TensorFS manifest naming them that
//! the machine materializes the tree from. Both are sent with the run like any input file. The
//! manifests are a cache that keeps itself: each is touched when staged, one unused for `UNUSED`
//! goes when another is staged, and none is kept while the disk is low.
use std::{
    fs::{self, File},
    io,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

use protocol::{
    calcifer::StagedInput,
    error::{Error, Kind},
    ids::Digest,
};
use tensorfs::{
    ids::{Doc, ObjectRef},
    manifest::Manifest,
};

/// How a run's spec marks an input that is a tree's manifest.
pub const MEDIA: &str = "application/vnd.cozy.tree-manifest";

/// A manifest unused this long leaves the cache the next time one is staged.
const UNUSED: Duration = Duration::from_secs(7 * 24 * 60 * 60);

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
    let path = keep(manifests, &hex::encode(digest.bytes()), &bytes)
        .map_err(|e| unreadable(manifests, e))?;
    let length = bytes.len() as u64;
    let manifest = StagedInput {
        path,
        digest,
        length,
    };
    Ok(Staged { manifest, files })
}

/// The manifest's file: kept under `manifests` (a reused one touched, the unused swept), or, on
/// a low disk, a scratch file the cache never holds.
fn keep(manifests: &Path, name: &str, bytes: &[u8]) -> io::Result<PathBuf> {
    if low(manifests) {
        let path = std::env::temp_dir().join(format!("cozy-tree-{name}"));
        fs::write(&path, bytes)?;
        return Ok(path);
    }
    fs::create_dir_all(manifests)?;
    let (path, now) = (manifests.join(name), SystemTime::now());
    match File::options().write(true).open(&path) {
        Ok(file) => file.set_modified(now)?,
        Err(_) => {
            let part = tempfile::NamedTempFile::new_in(manifests)?;
            fs::write(part.path(), bytes)?;
            part.persist(&path).map_err(|e| e.error)?;
        }
    }
    for old in fs::read_dir(manifests)?.flatten() {
        let modified = old.metadata().and_then(|m| m.modified());
        if modified.is_ok_and(|at| at + UNUSED < now) {
            let _ = fs::remove_file(old.path());
        }
    }
    Ok(path)
}

/// Whether the disk holding `dir` has no more free than TensorFS keeps in reserve.
fn low(dir: &Path) -> bool {
    let probe = dir.ancestors().find(|d| d.exists()).unwrap_or(dir);
    let reserve = |total: u64| (total / 50).clamp(1 << 30, 10 << 30);
    platform::durable::space(probe).is_ok_and(|s| s.free <= reserve(s.total))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Staging touches the tree's manifest and sweeps one unused for longer than `UNUSED`; a
    /// manifest used within it stays.
    #[test]
    fn staging_keeps_its_manifest_and_sweeps_the_unused() {
        let dir = tempfile::tempdir().unwrap();
        let (tree, manifests) = (dir.path().join("tree"), dir.path().join("manifests"));
        fs::create_dir_all(tree.join("sub")).unwrap();
        fs::write(tree.join("a.pt"), "a").unwrap();
        fs::write(tree.join("sub/b.pt"), "b").unwrap();
        fs::create_dir_all(&manifests).unwrap();
        let aged = |name: &str, age: Duration| {
            let path = manifests.join(name);
            fs::write(&path, name).unwrap();
            let file = File::options().write(true).open(&path).unwrap();
            file.set_modified(SystemTime::now() - age).unwrap();
            path
        };
        let stale = aged("stale", UNUSED + Duration::from_secs(60));
        let recent = aged("recent", UNUSED - Duration::from_secs(60));

        let staged = stage(&tree, None, &manifests).unwrap();
        assert!(staged.manifest.path.starts_with(&manifests) || low(&manifests));
        assert!(
            !stale.exists() || low(&manifests),
            "the unused manifest stays"
        );
        assert!(recent.exists());
        let lengths: Vec<_> = staged.files.iter().map(|f| f.length).collect();
        assert_eq!(lengths, [1, 1]);

        // Staged again later, it is touched, not rewritten.
        let path = staged.manifest.path;
        let file = File::options().write(true).open(&path).unwrap();
        file.set_modified(SystemTime::now() - UNUSED).unwrap();
        stage(&tree, None, &manifests).unwrap();
        let touched = fs::metadata(&path).unwrap().modified().unwrap();
        assert!(touched > SystemTime::now() - Duration::from_secs(60));
    }
}
