//! A connected local byte stream between an owner and its executor. Unix: a Unix socket. Windows,
//! until its named-pipe arm: a loopback TCP connection made like CPython's `socketpair`, whose
//! accepted end must be the connector itself.
use super::Result;

#[cfg(unix)]
pub type LocalStream = std::os::unix::net::UnixStream;
#[cfg(windows)]
pub type LocalStream = std::net::TcpStream;

/// Two connected ends.
pub fn pair() -> Result<(LocalStream, LocalStream)> {
    #[cfg(unix)]
    return Ok(LocalStream::pair()?);
    #[cfg(windows)]
    {
        use std::net::{Ipv4Addr, TcpListener};
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        let client = LocalStream::connect(listener.local_addr()?)?;
        loop {
            let (server, peer) = listener.accept()?;
            if peer == client.local_addr()? {
                return Ok((client, server));
            }
        }
    }
}

/// The stream whose descriptor (Unix) or socket (Windows) the caller hands over.
///
/// # Safety
/// `raw` must be an open, connected stream the caller owns and gives up.
pub unsafe fn adopt(raw: i64) -> LocalStream {
    #[cfg(unix)]
    return unsafe { std::os::fd::FromRawFd::from_raw_fd(raw as i32) };
    #[cfg(windows)]
    return unsafe { std::os::windows::io::FromRawSocket::from_raw_socket(raw as u64) };
}
