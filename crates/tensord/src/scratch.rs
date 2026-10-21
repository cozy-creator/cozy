//! Test scratch: roots and the files a fixture generation names.
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Removes a test's scratch root. A test's background threads (a pool's, a dispatched run's)
/// may still hold its journal (Windows deletes no file a handle holds open) or write a file into
/// it as it goes (a directory then is not empty): such a root is left to the temp directory.
pub(crate) fn remove(root: &Path) {
    const SHARING_VIOLATION: i32 = 32;
    match fs::remove_dir_all(root) {
        Err(error) if cfg!(windows) && error.raw_os_error() == Some(SHARING_VIOLATION) => {}
        Err(error) if error.kind() == std::io::ErrorKind::DirectoryNotEmpty => {}
        removed => removed.unwrap(),
    }
}

/// The interpreter a fixture generation's record names, at its venv's path on this OS; empty,
/// for a test that never runs it.
pub(crate) fn interpreter(env: &Path) -> PathBuf {
    let python = platform::paths::venv_python(env);
    fs::create_dir_all(python.parent().unwrap()).unwrap();
    fs::write(&python, "").unwrap();
    python
}

/// A program that lives until its stdin closes: `cat`, or Windows' own `sort`.
pub(crate) fn until_stdin_closes() -> std::process::Command {
    std::process::Command::new(if cfg!(windows) { "sort" } else { "/bin/cat" })
}
