//! Files: atomic replacement and crash durability, a name's own inode, positional I/O, modes and
//! owners, space, and what the disk under a path is. `File::sync_all` is already the strongest
//! flush each OS has: `fsync`, `F_FULLFSYNC` on macOS, `FlushFileBuffers` on Windows. Windows has
//! no POSIX owner or mode: a store there keeps the profile's ACL, so modes and owners are accepted
//! and left to the filesystem's defaults, and a directory needs no sync (NTFS journals its entries).
use crate::{Error, Result};
use std::{
    fs::{self, File, Metadata, OpenOptions},
    io::{self, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

mod disk;
pub use disk::{DiskClass, Filesystem, cached, filesystem, open_direct};

static TEMPS: AtomicU64 = AtomicU64::new(0);

/// Replaces `path` with `bytes`: a reader sees the old file or the new one, never a part, and after
/// this returns a crash keeps the new one. `mode` is the Unix permission; Windows keeps the
/// profile's ACL, renames with POSIX semantics (a reader holding the old file does not block it),
/// and has no directory to sync: the rename is atomic in NTFS's journal.
pub fn replace(path: &Path, bytes: &[u8], mode: u32) -> Result<()> {
    renamed(path, bytes, mode, true)?;
    sync_dir(
        path.parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )
}

/// As `replace`, but nothing is made to survive a crash: for a record of a live process, which a
/// crash makes stale anyway. It never waits on the disk's write-back.
pub fn swap(path: &Path, bytes: &[u8], mode: u32) -> Result<()> {
    renamed(path, bytes, mode, false)
}

/// Writes `bytes` beside `path` and renames them over it, flushed first when `durable`.
fn renamed(path: &Path, bytes: &[u8], mode: u32, durable: bool) -> Result<()> {
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    let Some(name) = path.file_name() else {
        return Err(Error::io("replace", path)(
            io::ErrorKind::InvalidInput.into(),
        ));
    };
    let (temp, mut file) = loop {
        let n = TEMPS.fetch_add(1, Ordering::Relaxed);
        let temp = dir.join(format!(
            ".{}.{}.{n}.tmp",
            name.to_string_lossy(),
            std::process::id()
        ));
        match create_new(&temp) {
            Ok(file) => break (temp, file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(Error::io("create", temp)(error)),
        }
    };
    let written = file
        .write_all(bytes)
        .and_then(|()| set_file_mode(&file, mode).map_err(io::Error::from))
        .and_then(|()| if durable { file.sync_all() } else { Ok(()) })
        .map_err(Error::io("write", &temp));
    drop(file);
    let renamed = written.and_then(|()| fs::rename(&temp, path).map_err(Error::io("rename", path)));
    if renamed.is_err() {
        let _ = fs::remove_file(&temp);
    }
    renamed
}

/// Makes the entries of `dir` (a create, rename or unlink in it) survive a crash.
pub fn sync_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(dir)
        .and_then(|dir| dir.sync_all())
        .map_err(Error::io("sync directory", dir))?;
    #[cfg(windows)]
    let _ = dir;
    Ok(())
}

/// A handle on the directory `path`, held to keep naming it (Linux reaches a directory through its
/// descriptor). Windows opens a directory only with backup semantics.
pub fn open_dir(path: &Path) -> Result<File> {
    #[cfg(unix)]
    let opened = File::open(path);
    #[cfg(windows)]
    let opened = {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_BACKUP_SEMANTICS;
        OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(path)
    };
    opened.map_err(Error::io("open directory", path))
}

/// Creates `path` (and its parents) and makes it private to this user: 0700 on Unix; on Windows
/// the profile's ACL already is.
pub fn private_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path).map_err(Error::io("create directory", path))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(path)
            .map_err(Error::io("stat", path))?
            .permissions()
            .mode();
        // Read before writing: a mode rewritten on every open is a journal write on a busy disk.
        if mode & 0o777 != 0o700 {
            fs::set_permissions(path, fs::Permissions::from_mode(0o700))
                .map_err(Error::io("protect", path))?;
        }
    }
    Ok(())
}

/// The absolute path with every symlink resolved, in the spelling a user writes: Windows'
/// `canonicalize` answers a verbatim `\\?\` path, which is shortened where it can be.
pub fn canonical(path: &Path) -> Result<std::path::PathBuf> {
    let canonical = std::fs::canonicalize(path)?;
    #[cfg(windows)]
    if let Some(rest) = canonical
        .to_str()
        .and_then(|text| text.strip_prefix(r"\\?\"))
    {
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
    return Some(std::os::unix::fs::PermissionsExt::mode(
        &metadata.permissions(),
    ));
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
                BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
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
        if unsafe {
            libc::syscall(
                libc::SYS_statx,
                libc::AT_FDCWD,
                c.as_ptr(),
                flags,
                libc::STATX_BASIC_STATS,
                &mut stat,
            )
        } != 0
        {
            return Err(io::Error::last_os_error().into());
        }
        Ok(u64::from(stat.stx_dev_major) << 32 | u64::from(stat.stx_dev_minor))
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        Ok(std::os::unix::fs::MetadataExt::dev(&std::fs::metadata(
            path,
        )?))
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
        use windows_sys::Win32::System::{IO::DeviceIoControl, Ioctl::FSCTL_SET_SPARSE};
        let mut returned = 0u32;
        // SAFETY: no buffers in or out, on a handle this File owns.
        let ok = unsafe {
            DeviceIoControl(
                file.as_raw_handle(),
                FSCTL_SET_SPARSE,
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                0,
                &mut returned,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error().into());
        }
    }
    file.set_len(len)?;
    Ok(())
}

/// Hard links to the file `path` names itself, a symlink not followed: unlinking one of several
/// frees nothing.
pub fn links(path: &Path) -> Result<u64> {
    #[cfg(unix)]
    return Ok(std::os::unix::fs::MetadataExt::nlink(&std::fs::symlink_metadata(path)?));
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{
            BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
        };
        let file = itself(path)?;
        // SAFETY: plain-old-data the call fills, on a handle this File owns.
        let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) } == 0 {
            return Err(io::Error::last_os_error().into());
        }
        Ok(info.nNumberOfLinks as u64)
    }
}

/// Sets a file's or a directory's times through a handle opened only for that: Windows sets
/// them only through attribute-write access, which Unix never needs (a 0444 file's owner can).
pub fn set_times(path: &Path, times: fs::FileTimes) -> Result<()> {
    #[cfg(unix)]
    let file = File::open(path)?;
    #[cfg(windows)]
    let file = {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_FLAG_BACKUP_SEMANTICS, FILE_WRITE_ATTRIBUTES,
        };
        OpenOptions::new()
            .access_mode(FILE_WRITE_ATTRIBUTES)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(path)?
    };
    Ok(file.set_times(times)?)
}

/// When `path` itself last changed status: a link, an unlink, its mode or its bytes. Unix
/// ctime; NTFS ChangeTime.
pub fn changed(path: &Path) -> Result<std::time::SystemTime> {
    use std::time::{Duration, UNIX_EPOCH};
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let metadata = std::fs::symlink_metadata(path)?;
        Ok(UNIX_EPOCH + Duration::new(metadata.ctime().max(0) as u64, metadata.ctime_nsec() as u32))
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_BASIC_INFO, FileBasicInfo, GetFileInformationByHandleEx,
        };
        let file = itself(path)?;
        // SAFETY: plain-old-data the call fills, of the size it is told, on a handle this File owns.
        let mut info: FILE_BASIC_INFO = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<FILE_BASIC_INFO>() as u32;
        if unsafe {
            GetFileInformationByHandleEx(file.as_raw_handle(), FileBasicInfo, (&mut info as *mut FILE_BASIC_INFO).cast(), size)
        } == 0
        {
            return Err(io::Error::last_os_error().into());
        }
        // 100 ns intervals since 1601, 11,644,473,600 s before the Unix epoch.
        let since = (info.ChangeTime as u64).saturating_sub(11_644_473_600 * 10_000_000);
        Ok(UNIX_EPOCH + Duration::from_nanos(since.saturating_mul(100)))
    }
}

/// The file or directory `path` names itself, a reparse point not followed, open only to read
/// its attributes.
#[cfg(windows)]
fn itself(path: &Path) -> Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_READ_ATTRIBUTES,
    };
    Ok(OpenOptions::new()
        .access_mode(FILE_READ_ATTRIBUTES)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?)
}

/// Bytes a file holds on disk: a sparse file only what was written (Unix). An NTFS file is
/// not sparse unless asked, so on Windows it holds its length.
pub fn allocated(metadata: &Metadata) -> u64 {
    #[cfg(unix)]
    return metadata
        .len()
        .min(std::os::unix::fs::MetadataExt::blocks(metadata).saturating_mul(512));
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
        return keep_cursor(self, |file| {
            std::os::windows::fs::FileExt::seek_read(file, buf, offset)
        });
    }

    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        #[cfg(unix)]
        return std::os::unix::fs::FileExt::write_at(self, buf, offset);
        #[cfg(windows)]
        return keep_cursor(self, |file| {
            std::os::windows::fs::FileExt::seek_write(file, buf, offset)
        });
    }
}

#[cfg(windows)]
fn keep_cursor(
    file: &File,
    positional: impl FnOnce(&File) -> io::Result<usize>,
) -> io::Result<usize> {
    use std::io::Seek;
    let mut handle = file;
    let cursor = handle.stream_position()?;
    let done = positional(file);
    handle.seek(io::SeekFrom::Start(cursor))?;
    done
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
        if unsafe {
            GetDiskFreeSpaceExW(wide.as_ptr(), &mut free, &mut total, std::ptr::null_mut())
        } == 0
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
    if link
        .parent()
        .unwrap_or(Path::new("."))
        .join(target)
        .is_dir()
    {
        std::os::windows::fs::symlink_dir(target, link)?;
    } else {
        std::os::windows::fs::symlink_file(target, link)?;
    }
    Ok(())
}

fn create_new(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
}
