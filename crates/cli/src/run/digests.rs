//! Input files' sha256, remembered by what names one version of a file (its path, length,
//! modification time and, on Unix, its inode), so a big video is read once, not on every run.
//! Each entry is a file under the memo directory, touched when used; one unused for `UNUSED` goes
//! the next time an entry is written.
use std::{
    fs::{self, File},
    io,
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use protocol::{
    error::{Error, Kind},
    ids::Digest,
};
use sha2::{Digest as _, Sha256};

const UNUSED: Duration = Duration::from_secs(30 * 24 * 60 * 60);

/// The file's sha256: remembered in `memo` (when given) for as long as the file is unchanged.
pub fn of(memo: Option<&Path>, path: &Path) -> Result<Digest, Error> {
    let Some(memo) = memo else { return hash(path) };
    let Some(version) = key(path) else {
        return hash(path);
    };
    let entry = memo.join(&version);
    if let Some(digest) = fs::read_to_string(&entry)
        .ok()
        .and_then(|t| t.trim().parse().ok())
    {
        let _ = File::options()
            .write(true)
            .open(&entry)
            .and_then(|f| f.set_modified(SystemTime::now()));
        return Ok(digest);
    }
    let digest = hash(path)?;
    // Only a version that held still while it was read is remembered.
    if key(path).as_deref() == Some(version.as_str()) {
        let _ = remember(memo, &entry, &digest);
    }
    Ok(digest)
}

fn remember(memo: &Path, entry: &Path, digest: &Digest) -> io::Result<()> {
    fs::create_dir_all(memo)?;
    let before = SystemTime::now() - UNUSED;
    for old in fs::read_dir(memo)?.flatten() {
        if old
            .metadata()
            .and_then(|m| m.modified())
            .is_ok_and(|at| at < before)
        {
            let _ = fs::remove_file(old.path());
        }
    }
    let part = tempfile::NamedTempFile::new_in(memo)?;
    fs::write(part.path(), digest.to_string())?;
    part.persist(entry).map_err(|e| e.error)?;
    Ok(())
}

/// What names this version of the file; None when it cannot be read.
fn key(path: &Path) -> Option<String> {
    let path = fs::canonicalize(path).ok()?;
    let meta = fs::metadata(&path).ok()?;
    let modified = meta
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    #[cfg(unix)]
    let inode = std::os::unix::fs::MetadataExt::ino(&meta);
    #[cfg(not(unix))]
    let inode = 0u64;
    let named = format!("{}\0{}\0{modified}\0{inode}", path.display(), meta.len());
    Some(hex::encode(Sha256::digest(named.as_bytes())))
}

fn hash(path: &Path) -> Result<Digest, Error> {
    let unreadable = |e: io::Error| {
        Error::new(
            Kind::NotFound,
            "input_file_unreadable",
            format!("input file {}: {e}", path.display()),
        )
    };
    let mut file = File::open(path).map_err(unreadable)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher).map_err(unreadable)?;
    Ok(Digest::from_bytes(hasher.finalize().into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A remembered file is not read again: bytes changed under the same length and time still
    /// answer the remembered digest. A new version (another length or time) is read.
    #[test]
    fn an_unchanged_file_is_read_once() {
        let dir = tempfile::tempdir().unwrap();
        let (memo, file) = (dir.path().join("memo"), dir.path().join("clip.mp4"));
        fs::write(&file, b"first").unwrap();
        let first = of(Some(&memo), &file).unwrap();
        assert_eq!(first, Digest::of(b"first"));
        let at = fs::metadata(&file).unwrap().modified().unwrap();
        fs::write(&file, b"FIRST").unwrap();
        File::options()
            .write(true)
            .open(&file)
            .unwrap()
            .set_modified(at)
            .unwrap();
        assert_eq!(
            of(Some(&memo), &file).unwrap(),
            first,
            "remembered, not read"
        );
        fs::write(&file, b"second!").unwrap();
        assert_eq!(of(Some(&memo), &file).unwrap(), Digest::of(b"second!"));
        assert_eq!(of(None, &file).unwrap(), Digest::of(b"second!"));
    }
}
