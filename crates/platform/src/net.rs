//! Listening sockets.
use std::{
    io,
    net::{SocketAddr, TcpListener},
};

/// A TCP listener on `address`. Windows takes its port exclusively (`SO_EXCLUSIVEADDRUSE`), so no
/// other process binding it with `SO_REUSEADDR` can take its connections.
pub fn listen(address: SocketAddr) -> io::Result<TcpListener> {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawSocket;
        use windows_sys::Win32::Networking::WinSock::{SO_EXCLUSIVEADDRUSE, SOL_SOCKET, setsockopt};
        let socket = socket2::Socket::new(socket2::Domain::for_address(address), socket2::Type::STREAM, None)?;
        let on = 1i32.to_ne_bytes();
        // SAFETY: a live socket and the option's four bytes.
        let set = unsafe {
            setsockopt(socket.as_raw_socket() as usize, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, on.as_ptr(), 4)
        };
        if set != 0 {
            return Err(io::Error::last_os_error());
        }
        socket.bind(&address.into())?;
        socket.listen(1024)?;
        Ok(socket.into())
    }
    #[cfg(not(windows))]
    TcpListener::bind(address)
}

#[cfg(all(test, windows))]
mod tests {
    #[test]
    fn a_reusing_socket_cannot_take_a_listened_port() {
        let ours = super::listen("127.0.0.1:0".parse().unwrap()).unwrap();
        let thief = socket2::Socket::new(socket2::Domain::IPV4, socket2::Type::STREAM, None).unwrap();
        thief.set_reuse_address(true).unwrap();
        assert!(thief.bind(&ours.local_addr().unwrap().into()).is_err());
    }
}
