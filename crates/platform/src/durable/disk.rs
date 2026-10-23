//! What a crash can take from a Store, read off the filesystem the Store actually lives on, and
//! what the page cache holds of a file.
//!
//! An fsync protects bytes against the machine going down before its page cache is written.
//! Whether that protection is worth paying for depends on where the bytes live, and only the
//! mount can say that — never a flag, never an environment variable.
//!
//! The rule, from `statfs(2)` of the Store root (Linux):
//!
//! - **Ephemeral** — tmpfs, ramfs, overlayfs. RAM, or a container's writable layer, which the
//!   platform discards with the machine or container (a RunPod container disk is erased when
//!   the pod stops). Nothing a crash could leave behind is ever read again, and a process
//!   crash leaves the page cache intact, so no fsync buys anything.
//! - **Network** — NFS, SMB/CIFS, FUSE, Ceph, 9p, AFS, Coda, GFS2, OCFS2, Lustre, GPFS.
//!   Another machine may read it, and close-to-open is the only ordering that machine sees:
//!   bytes are flushed before their name is published, as before.
//! - **Persistent** — everything else (ext4, xfs, btrfs, zfs, ...), a `statfs` that fails,
//!   and every non-Linux platform. The safe side: it survives a crash of this machine.
#![allow(unsafe_code)]
use std::{fs::File, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiskClass {
    Ephemeral,
    Persistent,
    Network,
}

impl DiskClass {
    /// The class of the filesystem holding `path`.
    pub fn of(path: &Path) -> DiskClass {
        filesystem(path).class
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DiskClass::Ephemeral => "ephemeral",
            DiskClass::Persistent => "persistent",
            DiskClass::Network => "network",
        }
    }
}

/// The filesystem holding a path: its class and the name it was recognised by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Filesystem {
    pub class: DiskClass,
    /// `ext4`, `overlayfs`, `nfs`, ...; `other` for an unrecognised magic, `unknown` when
    /// `statfs` failed or the platform has no magic.
    pub kind: &'static str,
}

#[cfg(target_os = "linux")]
pub fn filesystem(path: &Path) -> Filesystem {
    use std::os::unix::ffi::OsStrExt;
    let magic = std::ffi::CString::new(path.as_os_str().as_bytes())
        .ok()
        .and_then(|c| {
            // SAFETY: plain-old-data the call fills, and a NUL-terminated path.
            let mut stat: libc::statfs = unsafe { std::mem::zeroed() };
            #[allow(clippy::unnecessary_cast)] // f_type's width differs by target
            // SAFETY: as above.
            (unsafe { libc::statfs(c.as_ptr(), &mut stat) } == 0).then_some(stat.f_type as u64)
        });
    match magic {
        Some(magic) => classify(magic & 0xffff_ffff),
        None => Filesystem {
            class: DiskClass::Persistent,
            kind: "unknown",
        },
    }
}

#[cfg(not(target_os = "linux"))]
pub fn filesystem(_path: &Path) -> Filesystem {
    Filesystem {
        class: DiskClass::Persistent,
        kind: "unknown",
    }
}

/// Superblock magics from `linux/magic.h` and the filesystems' own headers.
#[cfg(target_os = "linux")]
fn classify(magic: u64) -> Filesystem {
    use DiskClass::*;
    let (class, kind) = match magic {
        0x0102_1994 => (Ephemeral, "tmpfs"),
        0x8584_58f6 => (Ephemeral, "ramfs"),
        0x794c_7630 => (Ephemeral, "overlayfs"),
        0x6969 => (Network, "nfs"),
        0x517b => (Network, "smb"),
        0xff53_4d42 => (Network, "cifs"),
        0xfe53_4d42 => (Network, "smb2"),
        0x6573_5546 => (Network, "fuse"),
        0x00c3_6400 => (Network, "ceph"),
        0x0102_1997 => (Network, "9p"),
        0x5346_414f | 0x6b41_4653 => (Network, "afs"),
        0x7375_7245 => (Network, "coda"),
        0x0116_1970 => (Network, "gfs2"),
        0x7461_636f => (Network, "ocfs2"),
        0x0bd0_0bd0 => (Network, "lustre"),
        0x4750_4653 => (Network, "gpfs"),
        0xef53 => (Persistent, "ext4"),
        0x5846_5342 => (Persistent, "xfs"),
        0x9123_683e => (Persistent, "btrfs"),
        0x2fc1_2fc1 => (Persistent, "zfs"),
        0xf2f5_2010 => (Persistent, "f2fs"),
        _ => (Persistent, "other"),
    };
    Filesystem { class, kind }
}

/// A second description of `file`'s inode that bypasses the page cache, or `None` where the
/// filesystem refuses it. Linux reopens through `/proc/self/fd` with `O_DIRECT`; elsewhere no
/// direct read is offered and callers read buffered.
pub fn open_direct(file: &File) -> crate::Result<Option<File>> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::{AsRawFd, FromRawFd};
        let path = std::ffi::CString::new(format!("/proc/self/fd/{}", file.as_raw_fd())).unwrap();
        // SAFETY: plain open of a magic link to our own descriptor.
        let fd = unsafe {
            libc::open(
                path.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECT | libc::O_CLOEXEC,
            )
        };
        // SAFETY: a fresh descriptor from the kernel.
        Ok((fd >= 0).then(|| unsafe { File::from_raw_fd(fd) }))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = file;
        Ok(None)
    }
}

/// Whether the page cache holds every page of `[off, off+len)`; `None` where the OS cannot say,
/// which callers treat as cold.
pub fn cached(file: &File, off: u64, len: u64) -> Option<bool> {
    #[cfg(target_os = "linux")]
    {
        linux::cached(file, off, len)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, off, len);
        None
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use std::fs::File;
    use std::os::fd::AsRawFd;

    /// From `cachestat(2)`: a pure query (unlike an `RWF_NOWAIT` read, which starts readahead on
    /// a miss). Where that cannot answer, the pages are counted through a mapping: a kernel
    /// without it (< 6.5), a container whose seccomp filter refuses it, and overlayfs (a
    /// container's root), whose files report no pages of their own.
    pub(super) fn cached(file: &File, off: u64, len: u64) -> Option<bool> {
        #[repr(C)]
        struct Range {
            off: u64,
            len: u64,
        }
        #[repr(C)]
        #[derive(Default)]
        struct Stat {
            nr_cache: u64,
            nr_dirty: u64,
            nr_writeback: u64,
            nr_evicted: u64,
            nr_recently_evicted: u64,
        }
        const SYS_CACHESTAT: libc::c_long = 451;
        let page = page_size();
        let range = Range { off, len };
        let mut stat = Stat::default();
        // SAFETY: the documented cachestat ABI; both structs outlive the call.
        let rc = unsafe { libc::syscall(SYS_CACHESTAT, file.as_raw_fd(), &range, &mut stat, 0) };
        if rc == 0 && stat.nr_cache >= (off + len).div_ceil(page) - off / page {
            return Some(true);
        }
        Some((rc != 0 || overlay(file)) && resident(file, off, len, page))
    }

    fn page_size() -> u64 {
        // SAFETY: sysconf reads a constant.
        (unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64).max(4096)
    }

    fn overlay(file: &File) -> bool {
        const OVERLAYFS_SUPER_MAGIC: i64 = 0x794c7630;
        // SAFETY: out-param of the right type on our own descriptor.
        let mut fs: libc::statfs = unsafe { std::mem::zeroed() };
        #[allow(clippy::unnecessary_cast)] // f_type is not an i64 on every target
        // SAFETY: as above.
        let magic =
            unsafe { libc::fstatfs(file.as_raw_fd(), &mut fs) == 0 }.then_some(fs.f_type as i64);
        magic == Some(OVERLAYFS_SUPER_MAGIC)
    }

    /// Every page of `[off, off+len)` is in core, by `mincore(2)` over a mapping (never faults).
    fn resident(file: &File, off: u64, len: u64, page: u64) -> bool {
        let start = off - off % page;
        let n = (off + len - start) as usize;
        if n == 0 {
            return true;
        }
        // SAFETY: a read-only shared mapping of our descriptor, unmapped below.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                n,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                start as libc::off_t,
            )
        };
        if p == libc::MAP_FAILED {
            return false;
        }
        let mut pages = vec![0u8; n.div_ceil(page as usize)];
        // SAFETY: the vector holds one byte per page of the mapping.
        let ok = unsafe { libc::mincore(p, n, pages.as_mut_ptr()) } == 0
            && pages.iter().all(|b| b & 1 == 1);
        // SAFETY: unmaps the mapping made above, which nothing else references.
        unsafe { libc::munmap(p, n) };
        ok
    }
}
