//! Unix: Unix sockets.
use crate::{Error, Result, process::Exact};
use std::{
    io,
    os::{
        fd::{AsFd, AsRawFd, FromRawFd},
        unix::net::{UnixListener, UnixStream},
    },
    path::Path,
};

pub type LocalStream = UnixStream;

/// A listener at a path an executor dials.
pub struct LocalListener(UnixListener);

impl LocalListener {
    pub fn bind(path: &Path) -> Result<Self> {
        UnixListener::bind(path)
            .map(Self)
            .map_err(Error::io("listen", path))
    }

    /// Every dialer, one at a time, as `UnixListener::incoming` gives them.
    pub fn incoming(&self) -> impl Iterator<Item = Result<LocalStream>> + '_ {
        std::iter::repeat_with(|| self.accept())
    }

    pub fn accept(&self) -> Result<LocalStream> {
        Ok(self.0.accept().map_err(Error::os("accept"))?.0)
    }

    /// A dialer, or `None` once `process` has ended first: woken by the kernel, never a timer.
    pub fn accept_unless(&self, process: &Exact) -> Result<Option<LocalStream>> {
        loop {
            let mut poll = [
                libc::pollfd {
                    fd: self.0.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                },
                libc::pollfd {
                    fd: process.as_fd().as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                },
            ];
            // SAFETY: two live descriptors and their count.
            if unsafe { libc::poll(poll.as_mut_ptr(), 2, -1) } < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(Error::os("wait for a dialer")(error));
            }
            if poll[0].revents & libc::POLLIN != 0 {
                return self.accept().map(Some);
            }
            if poll[1].revents != 0 {
                return Ok(None);
            }
        }
    }
}

impl AsFd for LocalListener {
    fn as_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        self.0.as_fd()
    }
}

pub fn connect(path: &Path) -> Result<LocalStream> {
    UnixStream::connect(path).map_err(Error::io("connect", path))
}

/// Two connected ends.
pub fn pair() -> Result<(LocalStream, LocalStream)> {
    Ok(LocalStream::pair()?)
}

/// The stream whose descriptor the caller hands over.
///
/// # Safety
/// `raw` must be an open, connected stream the caller owns and gives up.
pub unsafe fn adopt(raw: i64) -> LocalStream {
    // SAFETY: the caller's promise above.
    unsafe { LocalStream::from_raw_fd(raw as i32) }
}
