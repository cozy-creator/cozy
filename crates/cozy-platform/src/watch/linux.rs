//! Linux: inotify on the directory.
use crate::Result;
use std::{path::Path, time::Duration};

pub(super) struct Watch {
    fd: std::os::fd::OwnedFd,
}

impl Watch {
    pub(super) fn new(dir: &Path) -> Result<Self> {
        use std::os::{
            fd::{AsRawFd, FromRawFd, OwnedFd},
            unix::ffi::OsStrExt,
        };
        // SAFETY: plain flags; the descriptor is owned at once.
        let raw = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
        if raw < 0 {
            return Err(crate::Error::os("inotify_init1")(
                std::io::Error::last_os_error(),
            ));
        }
        // SAFETY: a descriptor inotify_init1 just returned and nothing else owns.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).map_err(|e| {
            crate::Error::io("watch", dir)(std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
        })?;
        let mask = libc::IN_MODIFY | libc::IN_CREATE | libc::IN_MOVED_TO | libc::IN_DELETE;
        // SAFETY: a valid descriptor and a NUL-terminated path.
        if unsafe { libc::inotify_add_watch(fd.as_raw_fd(), path.as_ptr(), mask) } < 0 {
            return Err(crate::Error::io("watch", dir)(
                std::io::Error::last_os_error(),
            ));
        }
        Ok(Self { fd })
    }

    pub(super) fn wait(&mut self, most: Duration) -> Result<bool> {
        use std::os::fd::AsRawFd;
        let fd = self.fd.as_raw_fd();
        let mut poll = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let ms = most.as_millis().min(i32::MAX as u128) as i32;
        // SAFETY: one valid pollfd.
        match unsafe { libc::poll(&mut poll, 1, ms) } {
            0 => return Ok(false),
            n if n < 0 => {
                let e = std::io::Error::last_os_error();
                return match e.kind() {
                    std::io::ErrorKind::Interrupted => Ok(false),
                    _ => Err(crate::Error::os("poll")(e)),
                };
            }
            _ => {}
        }
        // What changed does not matter, only that something did: drain the queue.
        let mut events = [0u8; 4096];
        // SAFETY: a buffer of the length given; the descriptor is non-blocking.
        while unsafe { libc::read(fd, events.as_mut_ptr().cast(), events.len()) } > 0 {}
        Ok(true)
    }
}
