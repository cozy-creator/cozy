//! Windows: only frozen bytes; sealed shared memory is off here.
use crate::{Error, Result};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    sync::atomic::{AtomicU64, Ordering},
};

/// `bytes` as a file no process can write: written to this user's temporary directory, opened
/// again read-only, then deleted, so the read-only handle is the only way to it.
pub fn frozen(name: &str, bytes: &[u8]) -> Result<File> {
    static NAMES: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir();
    let (path, mut file) = loop {
        let n = NAMES.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!(".{name}.{}.{n}", std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => break (path, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(Error::io("create", path)(error)),
        }
    };
    let frozen = file.write_all(bytes).and_then(|()| File::open(&path));
    drop(file);
    let _ = fs::remove_file(&path);
    frozen.map_err(Error::io("freeze", path))
}
