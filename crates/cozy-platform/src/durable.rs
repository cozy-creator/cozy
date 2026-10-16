//! Files that survive a crash whole. `File::sync_all` is already the strongest flush each OS has:
//! `fsync`, `F_FULLFSYNC` on macOS, `FlushFileBuffers` on Windows.
use crate::{Error, Result};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

static TEMPS: AtomicU64 = AtomicU64::new(0);

/// Replaces `path` with `bytes`: a reader sees the old file or the new one, never a part, and after
/// this returns a crash keeps the new one. `mode` is the Unix permission; Windows keeps the
/// profile's ACL, renames with POSIX semantics (a reader holding the old file does not block it),
/// and has no directory to sync: the rename is atomic in NTFS's journal.
pub fn replace(path: &Path, bytes: &[u8], mode: u32) -> Result<()> {
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    let Some(name) = path.file_name() else {
        return Err(Error::io("replace", path)(
            io::ErrorKind::InvalidInput.into(),
        ));
    };
    let (temp, mut file) = loop {
        let n = TEMPS.fetch_add(1, Ordering::Relaxed);
        let temp = dir.join(format!(
            ".{}.{}.{n}.tmp",
            name.to_string_lossy(),
            std::process::id()
        ));
        match create_new(&temp) {
            Ok(file) => break (temp, file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(Error::io("create", temp)(error)),
        }
    };
    let written = file
        .write_all(bytes)
        .and_then(|()| set_mode(&file, mode))
        .and_then(|()| file.sync_all())
        .map_err(Error::io("write", &temp));
    drop(file);
    let renamed = written.and_then(|()| fs::rename(&temp, path).map_err(Error::io("rename", path)));
    if renamed.is_err() {
        let _ = fs::remove_file(&temp);
    }
    renamed?;
    sync_dir(dir)
}

/// Makes the entries of `dir` (a create, rename or unlink in it) survive a crash.
pub fn sync_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(dir)
        .and_then(|dir| dir.sync_all())
        .map_err(Error::io("sync directory", dir))?;
    #[cfg(windows)]
    let _ = dir;
    Ok(())
}

/// Creates `path` (and its parents) and makes it private to this user: 0700 on Unix; on Windows
/// the profile's ACL already is.
pub fn private_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path).map_err(Error::io("create directory", path))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(path)
            .map_err(Error::io("stat", path))?
            .permissions()
            .mode();
        // Read before writing: a mode rewritten on every open is a journal write on a busy disk.
        if mode & 0o777 != 0o700 {
            fs::set_permissions(path, fs::Permissions::from_mode(0o700))
                .map_err(Error::io("protect", path))?;
        }
    }
    Ok(())
}

fn create_new(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    options.open(path)
}

fn set_mode(file: &File, mode: u32) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(mode))
    }
    #[cfg(windows)]
    {
        let _ = (file, mode);
        Ok(())
    }
}
