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

/// The peer, always this user (another is refused).
pub fn peer_credentials(stream: &UnixStream) -> Result<Peer> {
    let peer = peer(stream)?;
    // SAFETY: geteuid has no memory arguments.
    if peer.uid != unsafe { libc::geteuid() } {
        return Err(io::Error::new(io::ErrorKind::PermissionDenied, "peer UID differs").into());
    }
    Ok(peer)
}

/// The peer as the kernel names it, whoever it runs as.
#[cfg(target_os = "linux")]
pub fn peer(stream: &UnixStream) -> io::Result<Peer> {
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
        return Err(io::Error::last_os_error());
    }
    Ok(Peer {
        pid: cred.pid as u32,
        uid: cred.uid,
        gid: cred.gid,
    })
}

/// The peer as the kernel names it, whoever it runs as.
#[cfg(target_os = "macos")]
pub fn peer(stream: &UnixStream) -> io::Result<Peer> {
    let (mut uid, mut gid) = (0, 0);
    // SAFETY: a live socket and two writable ids.
    if unsafe { libc::getpeereid(stream.as_raw_fd(), &mut uid, &mut gid) } < 0 {
        return Err(io::Error::last_os_error());
    }
    let mut pid: libc::pid_t = 0;
    let mut size = std::mem::size_of::<libc::pid_t>() as libc::socklen_t;
    // SAFETY: a correctly sized writable pid and size pointer.
    if unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_LOCAL,
            libc::LOCAL_PEERPID,
            &mut pid as *mut _ as *mut libc::c_void,
            &mut size,
        )
    } < 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(Peer {
        pid: pid as u32,
        uid,
        gid,
    })
}

/// Whether the connecting process descends from this machine (an executor or anything
/// package code started). Same-UID package code is not a sandbox, but it cannot
/// administer the machine through its own process tree.
pub fn peer_descends_from_machine(stream: &UnixStream) -> Result<bool> {
    let mut pid = peer_credentials(stream)?.pid;
    let machine = std::process::id();
    while pid > 1 {
        if pid == machine {
            return Ok(true);
        }
        pid = crate::process::parent_of(pid)?;
    }
    Ok(false)
}

/// The peer pinned as a process handle (a pidfd on Linux, a kqueue on its exit on macOS).
pub fn peer_pidfd(stream: &UnixStream) -> Result<File> {
    let cred = peer_credentials(stream)?;
    #[cfg(target_os = "linux")]
    if let Some(file) = peer_pidfd_atomic(stream)? {
        return Ok(file);
    }
    Ok(pidfd_of_live(cred.pid as i32)?)
}

/// Linux SO_PEERPIDFD (UAPI value 77, kernel 6.5) pins the connecting process atomically;
/// `None` on an older kernel.
#[cfg(target_os = "linux")]
fn peer_pidfd_atomic(stream: &UnixStream) -> io::Result<Option<File>> {
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
            Some(libc::ENOPROTOOPT | libc::EINVAL) => Ok(None),
            _ => Err(error),
        };
    }
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: getsockopt handed us a new descriptor that is now ours.
    let file = unsafe { File::from_raw_fd(fd) };
    // SAFETY: valid descriptor; scalar close-on-exec flag.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(Some(file))
}

/// Older kernels and macOS: pin the credential's PID by its birth, accepted only when the same
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
    // macOS has no MSG_NOSIGNAL; Rust programs ignore SIGPIPE, so a closed peer is EPIPE there too.
    #[cfg(target_os = "linux")]
    let flags = libc::MSG_NOSIGNAL;
    #[cfg(target_os = "macos")]
    let flags = 0;
    let sent = unsafe { libc::sendmsg(stream.as_raw_fd(), &message, flags) };
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
    // solely to close that prefix before rejecting the message. Room for a thousand
    // descriptors: macOS installs a whole batch even when it reports it truncated, so a
    // batch must fit to be closed.
    let mut control = [0usize; 512];
    let mut iov = libc::iovec {
        iov_base: marker.as_mut_ptr().cast(),
        iov_len: 1,
    };
    // SAFETY: zero is a valid initial msghdr; all writable buffers live through recvmsg.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr().cast();
    msg.msg_controllen = std::mem::size_of_val(&control) as _;
    // SAFETY: correctly sized aligned control/data buffers and live socket.
    // macOS has no MSG_CMSG_CLOEXEC: each descriptor is marked close-on-exec as it is taken.
    #[cfg(target_os = "linux")]
    let flags = libc::MSG_CMSG_CLOEXEC;
    #[cfg(target_os = "macos")]
    let flags = 0;
    let count = unsafe { libc::recvmsg(stream.as_raw_fd(), &mut msg, flags) };
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
            let base = libc::CMSG_LEN(0) as usize;
            // macOS reports a truncated header's whole length: only what fits is read.
            let fits = (msg.msg_controllen as usize).saturating_sub(start);
            let length = ((*header).cmsg_len as usize).min(fits);
            if length < base {
                valid = false;
                break;
            }
            valid &= (*header).cmsg_len as usize == length;
            if (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
                let bytes = length - base;
                valid &= bytes.is_multiple_of(std::mem::size_of::<i32>());
                for i in 0..bytes / std::mem::size_of::<i32>() {
                    let fd = std::ptr::read_unaligned(libc::CMSG_DATA(header).cast::<i32>().add(i));
                    #[cfg(target_os = "macos")]
                    libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
                    received.push(OwnedFd::from_raw_fd(fd));
                }
            } else {
                valid = false;
            }
            if length < (*header).cmsg_len as usize {
                break;
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
#[cfg(target_os = "linux")]
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

/// The peer end of a lease is closed: the reader released, or its process is gone. macOS has no
/// POLLRDHUP: a readable lease whose peek reads nothing is at its end.
#[cfg(target_os = "macos")]
pub fn hung_up(lease: &UnixStream) -> bool {
    let mut poll = libc::pollfd {
        fd: lease.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one live descriptor, instantaneous observation.
    if unsafe { libc::poll(&mut poll, 1, 0) } <= 0 {
        return false;
    }
    if poll.revents & (libc::POLLHUP | libc::POLLERR) != 0 {
        return true;
    }
    let mut byte = 0u8;
    // SAFETY: one writable byte, peeked without waiting.
    let read = unsafe {
        libc::recv(
            lease.as_raw_fd(),
            (&mut byte as *mut u8).cast(),
            1,
            libc::MSG_PEEK | libc::MSG_DONTWAIT,
        )
    };
    read == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    #[test]
    fn older_kernel_fallback_pins_a_live_peer_and_refuses_a_gone_one() {
        let pinned = pidfd_of_live(std::process::id() as i32).unwrap();
        assert!(!crate::process::ended(&pinned));
        let mut child = std::process::Command::new("true").spawn().unwrap();
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
    /// descriptor table, so only descriptors of this test's own file are counted.
    fn descriptors_of(file: &std::fs::File) -> usize {
        let own = file.metadata().unwrap();
        (0..4096)
            .filter(|fd| {
                // SAFETY: plain-old-data the call fills; a closed number fails harmlessly.
                let mut stat: libc::stat = unsafe { std::mem::zeroed() };
                let open = unsafe { libc::fstat(*fd, &mut stat) } == 0;
                open && (stat.st_dev as u64, stat.st_ino as u64) == (own.dev(), own.ino())
            })
            .count()
    }

    #[test]
    fn a_descriptor_crosses_with_its_file_and_a_closed_peer_reads_as_hung_up() {
        use std::io::{Read, Seek, Write};
        let (send, receive) = UnixStream::pair().unwrap();
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"crossed").unwrap();
        send_fd(&send, &file).unwrap();
        let mut got = File::from(recv_fd(&receive).unwrap());
        // SAFETY: a scalar query of a live descriptor.
        let flags = unsafe { libc::fcntl(got.as_raw_fd(), libc::F_GETFD) };
        assert_ne!(flags & libc::FD_CLOEXEC, 0, "received close-on-exec");
        got.rewind().unwrap();
        let mut text = String::new();
        got.read_to_string(&mut text).unwrap();
        assert_eq!(text, "crossed");
        let me = peer_credentials(&receive).unwrap();
        assert_eq!(me.pid, std::process::id());
        assert!(!hung_up(&receive));
        drop(send);
        // Readable once the peer's end is closed everywhere (a child spawned meanwhile by another
        // test may hold it until it exits): woken by the kernel, not a timer.
        let mut poll = libc::pollfd {
            fd: receive.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one live descriptor.
        assert_eq!(unsafe { libc::poll(&mut poll, 1, -1) }, 1);
        assert!(hung_up(&receive), "revents {:#x}", poll.revents);
    }

    #[test]
    fn a_batch_of_descriptors_is_refused_and_every_one_closed() {
        let (send, receive) = UnixStream::pair().unwrap();
        let file = tempfile::tempfile().unwrap();
        send_rights(&send, &[file.as_raw_fd(); 32]).unwrap();
        assert!(recv_fd(&receive).is_err());
        assert_eq!(descriptors_of(&file), 1);
    }
}
