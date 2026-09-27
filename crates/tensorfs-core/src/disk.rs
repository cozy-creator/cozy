//! What a crash can take from a Store, read off the filesystem the Store actually lives on.
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

use std::path::Path;

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
    match rustix::fs::statfs(path) {
        #[allow(clippy::unnecessary_cast)]
        Ok(stat) => classify((stat.f_type as u64) & 0xffff_ffff),
        Err(_) => Filesystem {
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

/// This kernel's boot, or `None` where there is no such identity. Page cache belongs to a
/// boot: bytes written under this boot and not yet on disk are still readable until it ends.
/// A container sees its host's boot, which is the one that owns the page cache.
pub(crate) fn boot_id() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let id = std::fs::read_to_string("/proc/sys/kernel/random/boot_id").ok()?;
        let id = id.trim();
        (id.len() == 36 && id.bytes().all(|b| b.is_ascii_hexdigit() || b == b'-'))
            .then(|| id.to_string())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn magics_name_their_class() {
        assert_eq!(classify(0x794c_7630).class, DiskClass::Ephemeral);
        assert_eq!(classify(0x0102_1994).class, DiskClass::Ephemeral);
        assert_eq!(classify(0x6969).class, DiskClass::Network);
        assert_eq!(classify(0x6573_5546).class, DiskClass::Network);
        assert_eq!(classify(0xef53).class, DiskClass::Persistent);
        assert_eq!(classify(0x1234_5678).class, DiskClass::Persistent);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn real_mounts_classify() {
        if Path::new("/dev/shm").is_dir() {
            assert_eq!(filesystem(Path::new("/dev/shm")).kind, "tmpfs");
            assert_eq!(DiskClass::of(Path::new("/dev/shm")), DiskClass::Ephemeral);
        }
        assert_eq!(
            DiskClass::of(Path::new("/proc/self")),
            DiskClass::Persistent,
            "an unrecognised filesystem is the safe side"
        );
        assert_eq!(
            DiskClass::of(Path::new("/no/such/path/anywhere")),
            DiskClass::Persistent
        );
        assert!(boot_id().is_some());
    }
}
