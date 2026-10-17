//! Files: a name's own inode, positional I/O, modes and owners, directory durability, space.
//! Windows has no POSIX owner or mode: a Store there keeps the profile's ACL, so modes and owners
//! are accepted and kept by the filesystem's defaults, and a directory needs no sync (NTFS
//! journals its entries).
use super::Result;
use std::{
    fs::{File, Metadata, OpenOptions},
    io,
    path::Path,
};

/// The absolute path with every symlink resolved, in the spelling a user writes: Windows'
/// `canonicalize` answers a verbatim `\\?\` path, which is shortened where it can be.
pub fn canonical(path: &Path) -> Result<std::path::PathBuf> {
    let canonical = std::fs::canonicalize(path)?;
    #[cfg(windows)]
    if let Some(rest) = canonical.to_str().and_then(|text| text.strip_prefix(r"\\?\")) {
        let plain = match rest.strip_prefix(r"UNC\") {
            Some(share) => format!(r"\\{share}"),
            None if rest.as_bytes().get(1) == Some(&b':') => rest.to_string(),
            None => return Ok(canonical),
        };
        if plain.len() < 260 {
            return Ok(plain.into());
        }
    }
    Ok(canonical)
}

/// Opens `path` itself: a symlink at the name is refused, and a FIFO there never blocks the open.
pub fn open_nofollow(path: &Path, opts: &OpenOptions) -> Result<File> {
    let mut opts = opts.clone();
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        Ok(opts.open(path)?)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
        };
        // The reparse point itself opens, so a link is seen and refused rather than followed.
        opts.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS);
        let file = opts.open(path)?;
        if file.metadata()?.file_type().is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{} is a symbolic link", path.display()),
            )
            .into());
        }
        Ok(file)
    }
}

/// `OpenOptions::mode`, the permission a file is created with, on every OS.
pub trait Mode {
    fn mode(&mut self, mode: u32) -> &mut Self;
}

impl Mode for OpenOptions {
    fn mode(&mut self, mode: u32) -> &mut Self {
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(self, mode);
        #[cfg(windows)]
        let _ = mode;
        self
    }
}

pub fn set_mode(path: &Path, mode: u32) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
    }
    #[cfg(windows)]
    let _ = (path, mode);
    Ok(())
}

pub fn set_file_mode(file: &File, mode: u32) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(mode))?;
    }
    #[cfg(windows)]
    let _ = (file, mode);
    Ok(())
}

/// A file's permission bits; `None` on Windows, which keeps an ACL instead.
pub fn mode(metadata: &Metadata) -> Option<u32> {
    #[cfg(unix)]
    return Some(std::os::unix::fs::PermissionsExt::mode(&metadata.permissions()));
    #[cfg(windows)]
    {
        let _ = metadata;
        None
    }
}

/// Who a file belongs to: uid and gid on Unix; on Windows every file is the profile's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Owner {
    #[cfg(unix)]
    uid: u32,
    #[cfg(unix)]
    gid: u32,
}

impl Owner {
    pub fn of(metadata: &Metadata) -> Owner {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Owner {
                uid: metadata.uid(),
                gid: metadata.gid(),
            }
        }
        #[cfg(windows)]
        {
            let _ = metadata;
            Owner {}
        }
    }

    pub fn assign(self, file: &File) -> Result<()> {
        #[cfg(unix)]
        std::os::unix::fs::fchown(file, Some(self.uid), Some(self.gid))?;
        #[cfg(windows)]
        let _ = file;
        Ok(())
    }
}

/// One file's identity: device and inode, or volume serial and file index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FileId {
    pub dev: u64,
    pub ino: u64,
}

impl FileId {
    pub fn of(file: &File) -> Result<FileId> {
        #[cfg(unix)]
        {
            Ok(Self::unix(&file.metadata()?))
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::Storage::FileSystem::{
                GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
            };
            // SAFETY: plain-old-data the call fills, on a handle this File owns.
            let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
            if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) } == 0 {
                return Err(io::Error::last_os_error().into());
            }
            Ok(FileId {
                dev: info.dwVolumeSerialNumber as u64,
                ino: (info.nFileIndexHigh as u64) << 32 | info.nFileIndexLow as u64,
            })
        }
    }

    /// The file or directory a path names, following symlinks.
    pub fn at(path: &Path) -> Result<FileId> {
        #[cfg(unix)]
        {
            Ok(Self::unix(&std::fs::metadata(path)?))
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_BACKUP_SEMANTICS;
            let file = OpenOptions::new()
                .access_mode(0)
                .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
                .open(path)?;
            Self::of(&file)
        }
    }

    #[cfg(unix)]
    fn unix(metadata: &Metadata) -> FileId {
        use std::os::unix::fs::MetadataExt;
        FileId {
            dev: metadata.dev(),
            ino: metadata.ino(),
        }
    }
}

/// The device holding `path`, from cached attributes where the OS has them (Linux's
/// `STATX_DONT_SYNC`), so a hung network mount cannot hold the caller; elsewhere a plain stat,
/// which can. Windows names a volume by its serial.
pub fn device(path: &Path) -> Result<u64> {
    #[cfg(target_os = "linux")]
    {
        let c = std::ffi::CString::new(std::os::unix::ffi::OsStrExt::as_bytes(path.as_os_str()))
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        // SAFETY: plain-old-data the call fills, and a NUL-terminated path. The raw syscall:
        // glibc's wrapper is newer than the manylinux floor the binary links against.
        let mut stat: libc::statx = unsafe { std::mem::zeroed() };
        let flags = libc::AT_STATX_DONT_SYNC;
        if unsafe { libc::syscall(libc::SYS_statx, libc::AT_FDCWD, c.as_ptr(), flags, libc::STATX_BASIC_STATS, &mut stat) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        Ok(u64::from(stat.stx_dev_major) << 32 | u64::from(stat.stx_dev_minor))
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        Ok(std::os::unix::fs::MetadataExt::dev(&std::fs::metadata(path)?))
    }
    #[cfg(windows)]
    {
        Ok(FileId::at(path)?.dev)
    }
}

/// Sizes `file` to `len` as a hole: writes fill it in any order and only what they write takes
/// disk. Unix files are sparse already; NTFS allocates a grown file whole unless marked sparse.
pub fn set_sparse_len(file: &File, len: u64) -> Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::{Ioctl::FSCTL_SET_SPARSE, IO::DeviceIoControl};
        let mut returned = 0u32;
        // SAFETY: no buffers in or out, on a handle this File owns.
        let ok = unsafe {
            DeviceIoControl(file.as_raw_handle(), FSCTL_SET_SPARSE, std::ptr::null(), 0, std::ptr::null_mut(), 0, &mut returned, std::ptr::null_mut())
        };
        if ok == 0 {
            return Err(io::Error::last_os_error().into());
        }
    }
    file.set_len(len)?;
    Ok(())
}

/// Bytes a file holds on disk: a sparse file only what was written (Unix). An NTFS file is
/// not sparse unless asked, so on Windows it holds its length.
pub fn allocated(metadata: &Metadata) -> u64 {
    #[cfg(unix)]
    return metadata.len().min(std::os::unix::fs::MetadataExt::blocks(metadata).saturating_mul(512));
    #[cfg(windows)]
    return metadata.len();
}

/// Positional I/O on every OS, by Unix `FileExt`'s names. Windows' positional calls move the
/// handle's cursor; it is put back, so a sequential reader of the same handle is unaffected.
pub trait FileAt {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize>;
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize>;

    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
        while !buf.is_empty() {
            match self.read_at(buf, offset) {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset += n as u64;
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    fn write_all_at(&self, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
        while !buf.is_empty() {
            match self.write_at(buf, offset) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(n) => {
                    buf = &buf[n..];
                    offset += n as u64;
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
}

impl FileAt for File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        #[cfg(unix)]
        return std::os::unix::fs::FileExt::read_at(self, buf, offset);
        #[cfg(windows)]
        return keep_cursor(self, |file| std::os::windows::fs::FileExt::seek_read(file, buf, offset));
    }

    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        #[cfg(unix)]
        return std::os::unix::fs::FileExt::write_at(self, buf, offset);
        #[cfg(windows)]
        return keep_cursor(self, |file| std::os::windows::fs::FileExt::seek_write(file, buf, offset));
    }
}

#[cfg(windows)]
fn keep_cursor(file: &File, positional: impl FnOnce(&File) -> io::Result<usize>) -> io::Result<usize> {
    use std::io::Seek;
    let mut handle = file;
    let cursor = handle.stream_position()?;
    let done = positional(file);
    handle.seek(io::SeekFrom::Start(cursor))?;
    done
}

/// Makes the entries of `dir` (a create, rename or unlink in it) survive a crash.
pub fn sync_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(dir)?.sync_all()?;
    #[cfg(windows)]
    let _ = dir;
    Ok(())
}

/// Makes a file's bytes survive a crash, by name. Unix syncs through any descriptor; Windows'
/// `FlushFileBuffers` needs a writable handle.
pub fn sync_file(path: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(path)?.sync_data()?;
    // A directory has nothing to flush on Windows: NTFS journals its entries.
    #[cfg(windows)]
    if !path.is_dir() {
        OpenOptions::new().write(true).open(path)?.sync_data()?;
    }
    Ok(())
}

/// Opens a small record another process may be replacing or deleting right now. Windows
/// answers an open that races a rename or a delete with access denied: the open is asked
/// again while the name keeps changing, and a refusal of an unchanged name stands; a name that
/// no longer identifies is gone.
pub fn open_racing(path: &Path) -> io::Result<File> {
    let mut refused: Option<FileId> = None;
    loop {
        match File::open(path) {
            Err(error) if cfg!(windows) && error.kind() == io::ErrorKind::PermissionDenied => {
                // A name that cannot even be identified is a delete in flight.
                let Ok(now) = FileId::at(path) else {
                    return Err(io::ErrorKind::NotFound.into());
                };
                if refused == Some(now) {
                    return Err(error);
                }
                refused = Some(now);
            }
            opened => return opened,
        }
    }
}

/// The filesystem is full, or this user's quota is.
pub fn quota_or_space(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded
    )
}

/// The filesystem holding a path, in bytes and inodes. Inode counts are zero where the
/// filesystem has no inode limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Space {
    pub total: u64,
    pub free: u64,
    pub inodes: u64,
    pub free_inodes: u64,
}

pub fn space(path: &Path) -> Result<Space> {
    #[cfg(unix)]
    {
        let c = std::ffi::CString::new(std::os::unix::ffi::OsStrExt::as_bytes(path.as_os_str()))
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        // SAFETY: plain-old-data the call fills, and a NUL-terminated path.
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statvfs(c.as_ptr(), &mut stat) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        let unit = (stat.f_frsize as u64).max(1);
        #[allow(clippy::unnecessary_cast)] // the widths differ by OS
        Ok(Space {
            total: (stat.f_blocks as u64).saturating_mul(unit),
            free: (stat.f_bavail as u64).saturating_mul(unit),
            inodes: stat.f_files as u64,
            free_inodes: stat.f_favail as u64,
        })
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        let (mut free, mut total) = (0u64, 0u64);
        // SAFETY: a NUL-terminated wide path and out-params of the right type.
        if unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut free, &mut total, std::ptr::null_mut()) }
            == 0
        {
            return Err(io::Error::last_os_error().into());
        }
        Ok(Space {
            total,
            free,
            inodes: 0,
            free_inodes: 0,
        })
    }
}

/// A symlink. Windows types it by what `target` is now, and needs Developer Mode or the
/// privilege: callers copy instead when it is refused.
pub fn symlink(target: &Path, link: &Path) -> Result<()> {
    #[cfg(unix)]
    std::os::unix::fs::symlink(target, link)?;
    #[cfg(windows)]
    if link.parent().unwrap_or(Path::new(".")).join(target).is_dir() {
        std::os::windows::fs::symlink_dir(target, link)?;
    } else {
        std::os::windows::fs::symlink_file(target, link)?;
    }
    Ok(())
}

/// A second description of `file`'s inode that bypasses the page cache, or `None` where the
/// filesystem refuses it. Linux reopens through `/proc/self/fd` with `O_DIRECT`; elsewhere no
/// direct read is offered and callers read buffered.
pub fn open_direct(file: &File) -> Result<Option<File>> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::{AsRawFd, FromRawFd};
        let path = std::ffi::CString::new(format!("/proc/self/fd/{}", file.as_raw_fd())).unwrap();
        // SAFETY: plain open of a magic link to our own descriptor.
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_DIRECT | libc::O_CLOEXEC) };
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
        let magic = unsafe { libc::fstatfs(file.as_raw_fd(), &mut fs) == 0 }.then_some(fs.f_type as i64);
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
            libc::mmap(std::ptr::null_mut(), n, libc::PROT_READ, libc::MAP_SHARED, file.as_raw_fd(), start as libc::off_t)
        };
        if p == libc::MAP_FAILED {
            return false;
        }
        let mut pages = vec![0u8; n.div_ceil(page as usize)];
        // SAFETY: the vector holds one byte per page of the mapping.
        let ok = unsafe { libc::mincore(p, n, pages.as_mut_ptr()) } == 0 && pages.iter().all(|b| b & 1 == 1);
        unsafe { libc::munmap(p, n) };
        ok
    }
}
