//! Socket options a connection needs.
use crate::Result;

/// Keeps at most `bytes` of a TCP socket's data unsent in the kernel, so what a writer has handed
/// the socket is close to what has crossed the network. Linux only (`TCP_NOTSENT_LOWAT`).
#[cfg(target_os = "linux")]
pub fn bound_unsent(socket: &impl std::os::fd::AsRawFd, bytes: u32) -> Result<()> {
    let value = libc::c_int::try_from(bytes).unwrap_or(libc::c_int::MAX);
    // SAFETY: a valid descriptor and an int option of the size given.
    let set = unsafe {
        libc::setsockopt(
            socket.as_raw_fd(),
            libc::IPPROTO_TCP,
            libc::TCP_NOTSENT_LOWAT,
            (&raw const value).cast(),
            size_of::<libc::c_int>() as libc::socklen_t,
        )
    };
    match set {
        0 => Ok(()),
        _ => Err(crate::Error::os("bound unsent bytes")(
            std::io::Error::last_os_error(),
        )),
    }
}

#[cfg(not(target_os = "linux"))]
pub fn bound_unsent<S>(_: &S, _: u32) -> Result<()> {
    Err(crate::Error::unsupported(crate::Seam::Network))
}
