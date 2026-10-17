use super::Peer;
use crate::{
    Result,
    process::{Birth, Exact},
};
use std::{
    fs::File,
    io,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
        unix::net::UnixStream,
    },
};

pub fn peer_credentials(stream: &UnixStream) -> Result<Peer> {
    let mut cred: libc::ucred = unsafe { std::mem::zeroed() };
    let mut size = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: correctly sized writable credential structure and size pointer.
    if unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            &mut cred as *mut _ as *mut libc::c_void,
            &mut size,
        )
    } < 0
    {
        return Err(io::Error::last_os_error().into());
    }
    // SAFETY: geteuid has no memory arguments.
    if cred.uid != unsafe { libc::geteuid() } {
        return Err(io::Error::new(io::ErrorKind::PermissionDenied, "peer UID differs").into());
    }
    Ok(Peer {
        pid: cred.pid as u32,
        uid: cred.uid,
    })
}

/// Whether the connecting process descends from this machine (an executor or anything
/// package code started). Same-UID package code is not a sandbox, but it cannot
/// administer the machine through its own process tree.
pub fn peer_descends_from_machine(stream: &UnixStream) -> Result<bool> {
    let mut pid = peer_credentials(stream)?.pid as i32;
    let machine = std::process::id() as i32;
    while pid > 1 {
        if pid == machine {
            return Ok(true);
        }
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"))?;
        pid = stat
            .rsplit_once(") ")
            .and_then(|(_, rest)| rest.split_whitespace().nth(1))
            .and_then(|ppid| ppid.parse().ok())
            .ok_or_else(|| io::Error::other("invalid process stat"))?;
    }
    Ok(false)
}

pub fn peer_pidfd(stream: &UnixStream) -> Result<File> {
    let cred = peer_credentials(stream)?;
    // Linux SO_PEERPIDFD (UAPI value 77, kernel 6.5) pins the connecting process atomically.
    let mut fd: i32 = -1;
    let mut fd_size = std::mem::size_of::<i32>() as libc::socklen_t;
    // SAFETY: correctly sized scalar output and size pointer.
    if unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            77,
            &mut fd as *mut _ as *mut libc::c_void,
            &mut fd_size,
        )
    } < 0
    {
        let error = io::Error::last_os_error();
        return match error.raw_os_error() {
            Some(libc::ENOPROTOOPT | libc::EINVAL) => Ok(pidfd_of_live(cred.pid as i32)?),
            _ => Err(error.into()),
        };
    }
    if fd < 0 {
        return Err(io::Error::last_os_error().into());
    }
    // SAFETY: getsockopt handed us a new descriptor that is now ours.
    let file = unsafe { File::from_raw_fd(fd) };
    // SAFETY: valid descriptor; scalar close-on-exec flag.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(file)
}

/// Older kernels: pin the credential's PID with pidfd_open, accepted only when the same
/// birth is read on both sides of the open. A peer that exited and whose PID was reused in
/// that instant is the remaining (negligible) window.
fn pidfd_of_live(pid: i32) -> io::Result<File> {
    let birth = Birth::of(u32::try_from(pid).map_err(io::Error::other)?)?;
    Exact::open(&birth)?
        .map(Exact::into_file)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "peer exited before it was pinned"))
}

/// One descriptor and a one-byte marker; the receiver refuses anything else.
pub fn send_fd(stream: &UnixStream, fd: &impl AsRawFd) -> Result<()> {
    Ok(send_rights(stream, &[fd.as_raw_fd()])?)
}

fn send_rights(stream: &UnixStream, fds: &[RawFd]) -> io::Result<()> {
    let marker = [0u8];
    let mut iov = libc::iovec {
        iov_base: marker.as_ptr() as *mut libc::c_void,
        iov_len: 1,
    };
    let bytes = std::mem::size_of_val(fds) as u32;
    // SAFETY: CMSG_SPACE computes a size from a byte count.
    let space = unsafe { libc::CMSG_SPACE(bytes) } as usize;
    // u64 storage satisfies cmsghdr alignment.
    let mut control = vec![0u64; space.div_ceil(8)];
    // SAFETY: zero is a valid initial msghdr.
    let mut message: libc::msghdr = unsafe { std::mem::zeroed() };
    message.msg_iov = &mut iov;
    message.msg_iovlen = 1;
    message.msg_control = control.as_mut_ptr().cast();
    message.msg_controllen = space as _;
    // SAFETY: the control buffer holds one header and `bytes` of data (CMSG_SPACE above).
    unsafe {
        let header = libc::CMSG_FIRSTHDR(&message);
        (*header).cmsg_level = libc::SOL_SOCKET;
        (*header).cmsg_type = libc::SCM_RIGHTS;
        (*header).cmsg_len = libc::CMSG_LEN(bytes) as _;
        std::ptr::copy_nonoverlapping(
            fds.as_ptr(),
            libc::CMSG_DATA(header).cast::<RawFd>(),
            fds.len(),
        );
    }
    // SAFETY: a live socket and a message whose buffers outlive the call.
    let sent = unsafe { libc::sendmsg(stream.as_raw_fd(), &message, libc::MSG_NOSIGNAL) };
    if sent < 0 {
        return Err(io::Error::last_os_error());
    }
    if sent != 1 {
        return Err(io::Error::other("descriptor marker truncated"));
    }
    Ok(())
}

pub fn recv_fd(stream: &UnixStream) -> Result<OwnedFd> {
    let mut marker = [255u8];
    // usize storage satisfies cmsghdr alignment; every iteration is bounded by
    // the kernel-returned control length. Parse delivered fds even on CTRUNC,
    // solely to close that prefix before rejecting the message.
    let mut control = [0usize; 8];
    let mut iov = libc::iovec {
        iov_base: marker.as_mut_ptr().cast(),
        iov_len: 1,
    };
    // SAFETY: zero is a valid initial msghdr; all writable buffers live through recvmsg.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr().cast();
    msg.msg_controllen = std::mem::size_of_val(&control);
    // SAFETY: correctly sized aligned control/data buffers and live socket.
    let count = unsafe { libc::recvmsg(stream.as_raw_fd(), &mut msg, libc::MSG_CMSG_CLOEXEC) };
    if count < 0 {
        return Err(io::Error::last_os_error().into());
    }
    let mut valid =
        count == 1 && marker == [0] && msg.msg_flags & (libc::MSG_CTRUNC | libc::MSG_TRUNC) == 0;
    let mut received = Vec::new();
    // SAFETY: msghdr contains buffers initialized by recvmsg; bounds checked below.
    unsafe {
        let mut header = libc::CMSG_FIRSTHDR(&msg);
        while !header.is_null() {
            let start = header as usize - msg.msg_control as usize;
            let length = (*header).cmsg_len;
            let base = libc::CMSG_LEN(0) as usize;
            if length < base || start.saturating_add(length) > msg.msg_controllen {
                valid = false;
                break;
            }
            if (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
                let bytes = length - base;
                valid &= bytes.is_multiple_of(std::mem::size_of::<i32>());
                for i in 0..bytes / std::mem::size_of::<i32>() {
                    let fd = std::ptr::read_unaligned(libc::CMSG_DATA(header).cast::<i32>().add(i));
                    received.push(OwnedFd::from_raw_fd(fd));
                }
            } else {
                valid = false;
            }
            header = libc::CMSG_NXTHDR(&msg, header);
        }
    }
    if !valid || received.len() != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected exactly one descriptor",
        )
        .into());
    }
    Ok(received.pop().unwrap())
}

/// The peer end of a lease is closed: the reader released, or its process is gone.
pub fn hung_up(lease: &UnixStream) -> bool {
    let mut poll = libc::pollfd {
        fd: lease.as_raw_fd(),
        events: libc::POLLIN | libc::POLLRDHUP,
        revents: 0,
    };
    // SAFETY: one live descriptor, instantaneous observation.
    let ready = unsafe { libc::poll(&mut poll, 1, 0) } > 0;
    ready && poll.revents & (libc::POLLHUP | libc::POLLRDHUP | libc::POLLERR) != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    #[test]
    fn older_kernel_fallback_pins_a_live_peer_and_refuses_a_gone_one() {
        let pinned = pidfd_of_live(std::process::id() as i32).unwrap();
        assert!(!crate::process::ended(&pinned));
        let mut child = std::process::Command::new("/bin/true").spawn().unwrap();
        let pid = child.id() as i32;
        child.wait().unwrap();
        assert!(pidfd_of_live(pid).is_err());
    }

    #[test]
    fn admin_peer_check_sees_the_machines_own_descendants() {
        let (same, _keep) = UnixStream::pair().unwrap();
        assert!(peer_descends_from_machine(&same).unwrap());
        let path = std::env::temp_dir().join(format!("admin-peer-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let mut child = std::process::Command::new("/bin/sh")
            .args(["-c", "exec python3 -c \"import socket,sys;s=socket.socket(socket.AF_UNIX);s.connect(sys.argv[1]);s.recv(1)\" \"$0\"", path.to_str().unwrap()])
            .spawn()
            .unwrap();
        let (grandchild, _) = listener.accept().unwrap();
        assert!(peer_descends_from_machine(&grandchild).unwrap());
        drop(grandchild);
        child.wait().unwrap();
        std::fs::remove_file(path).unwrap();
    }

    /// This process's descriptors open on `file`. Tests share the process and its
    /// descriptor table, so only descriptors of this test's own memfd are counted.
    fn descriptors_of(file: &std::fs::File) -> usize {
        let own = file.metadata().unwrap();
        std::fs::read_dir("/proc/self/fd")
            .unwrap()
            .filter_map(|entry| std::fs::metadata(entry.ok()?.path()).ok())
            .filter(|open| (open.dev(), open.ino()) == (own.dev(), own.ino()))
            .count()
    }

    #[test]
    fn truncated_descriptor_batch_closes_delivered_prefix() {
        let (send, receive) = UnixStream::pair().unwrap();
        let file = std::fs::File::from(crate::shm::memfd_named("rights-test").unwrap());
        send_rights(&send, &[file.as_raw_fd(); 32]).unwrap();
        assert!(recv_fd(&receive).is_err());
        assert_eq!(descriptors_of(&file), 1);
    }
}
