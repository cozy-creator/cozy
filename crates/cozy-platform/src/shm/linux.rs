use super::FULL_SEALS;
use crate::Result;
use std::{
    ffi::CString,
    io,
    os::{
        fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd},
        unix::fs::MetadataExt,
    },
};

/// A new sealable memfd, close-on-exec. `name` shows in `/proc/<pid>/fd`.
pub fn memfd_named(name: &str) -> Result<OwnedFd> {
    let name = CString::new(name).map_err(io::Error::other)?;
    // SAFETY: valid NUL-terminated name, no borrowed output pointers.
    let fd =
        unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
    if fd < 0 {
        return Err(io::Error::last_os_error().into());
    }
    // SAFETY: memfd_create returned a new descriptor that is now ours.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

pub fn seals(fd: BorrowedFd<'_>) -> Result<i32> {
    // SAFETY: live descriptor; fcntl has no pointer argument for this operation.
    let result = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GET_SEALS) };
    if result < 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(result)
}

/// Adds [`FULL_SEALS`].
pub fn seal(fd: BorrowedFd<'_>) -> Result<()> {
    seal_with(fd, FULL_SEALS)
}

pub fn seal_with(fd: BorrowedFd<'_>, seals: i32) -> Result<()> {
    // SAFETY: live descriptor, scalar seal bitmask.
    if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals) } < 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(())
}

/// Whether a process still holds the file `(dev, ino)`: an open descriptor of it or a mapping.
/// Only processes this one may inspect are asked; another user's process cannot have been
/// handed this process's files.
pub fn held(dev: u64, ino: u64) -> Result<bool> {
    let device = format!("{:02x}:{:02x}", libc::major(dev), libc::minor(dev));
    let ino_text = ino.to_string();
    let processes = std::fs::read_dir("/proc").into_iter().flatten().flatten();
    for process in processes.filter(|p| {
        p.file_name()
            .to_string_lossy()
            .bytes()
            .all(|b| b.is_ascii_digit())
    }) {
        let descriptors = std::fs::read_dir(process.path().join("fd"))
            .into_iter()
            .flatten()
            .flatten();
        if descriptors
            .filter_map(|fd| std::fs::metadata(fd.path()).ok())
            .any(|m| m.dev() == dev && m.ino() == ino)
        {
            return Ok(true);
        }
        let maps = std::fs::read_to_string(process.path().join("maps")).unwrap_or_default();
        if maps.lines().any(|line| {
            let mut fields = line.split_whitespace().skip(3);
            fields.next() == Some(device.as_str()) && fields.next() == Some(ino_text.as_str())
        }) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file is held while any process has it open, here or in a child, and not after.
    #[test]
    fn a_file_is_held_while_any_process_has_it_open() {
        use std::os::unix::process::CommandExt;
        let file = std::fs::File::from(memfd_named("held-test").unwrap());
        let identity = file.metadata().unwrap();
        let (dev, ino) = (identity.dev(), identity.ino());
        assert!(held(dev, ino).unwrap());
        let fd = file.as_raw_fd();
        let mut command = std::process::Command::new("sleep");
        command.arg("1000");
        // SAFETY: only async-signal-safe fcntl on a descriptor this test keeps open.
        unsafe {
            command.pre_exec(move || match libc::fcntl(fd, libc::F_SETFD, 0) {
                0 => Ok(()),
                _ => Err(io::Error::last_os_error()),
            });
        }
        let mut child = command.spawn().unwrap();
        drop(file);
        assert!(held(dev, ino).unwrap(), "the child's descriptor holds it");
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(!held(dev, ino).unwrap());
    }
}
