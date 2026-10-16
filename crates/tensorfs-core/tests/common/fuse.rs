#![allow(dead_code)]
//! A FUSE filesystem over a backing directory, as a volume: a stated capacity (`statfs` says
//! how full it is) and reads that block on demand, through the real kernel path.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::{self, File, Metadata, OpenOptions};
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fuser::{
    FileAttr, FileType, Filesystem, MountOption, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory,
    ReplyEmpty, ReplyEntry, ReplyOpen, ReplyStatfs, ReplyWrite, Request, TimeOrNow,
};

/// What a test changes while the filesystem is mounted.
#[derive(Default)]
pub struct Control {
    pub capacity: AtomicU64,
    blocked: Mutex<bool>,
    unblocked: Condvar,
    /// Reads that reached the filesystem.
    pub reads: AtomicU64,
}

impl Control {
    pub fn block(&self, blocked: bool) {
        *self.blocked.lock().unwrap() = blocked;
        self.unblocked.notify_all();
    }
}

/// A mounted volume; dropping it unmounts.
pub struct Mounted {
    pub at: PathBuf,
    pub backing: PathBuf,
    pub control: Arc<Control>,
    _session: fuser::BackgroundSession,
}

/// Mount `backing` at `at` with `capacity` bytes.
pub fn mount(backing: &Path, at: &Path, capacity: u64) -> Mounted {
    fs::create_dir_all(backing).unwrap();
    fs::create_dir_all(at).unwrap();
    let control = Arc::new(Control::default());
    control.capacity.store(capacity, Ordering::Release);
    let fs = Volume {
        root: backing.to_path_buf(),
        paths: HashMap::from([(fuser::FUSE_ROOT_ID, PathBuf::new())]),
        files: HashMap::new(),
        next: fuser::FUSE_ROOT_ID + 1,
        control: Arc::clone(&control),
    };
    let session = fuser::spawn_mount2(fs, at, &[MountOption::FSName("tfs-volume".into())])
        .expect("mount the FUSE volume (needs /dev/fuse and fusermount3)");
    Mounted {
        at: at.to_path_buf(),
        backing: backing.to_path_buf(),
        control,
        _session: session,
    }
}

struct Volume {
    root: PathBuf,
    paths: HashMap<u64, PathBuf>,
    files: HashMap<u64, File>,
    next: u64,
    control: Arc<Control>,
}

const NOW: Duration = Duration::ZERO;

fn errno(error: std::io::Error) -> i32 {
    error.raw_os_error().unwrap_or(libc::EIO)
}

fn attr(ino: u64, m: &Metadata) -> FileAttr {
    FileAttr {
        ino,
        size: m.len(),
        blocks: m.blocks(),
        atime: m.accessed().unwrap_or(UNIX_EPOCH),
        mtime: m.modified().unwrap_or(UNIX_EPOCH),
        ctime: UNIX_EPOCH + Duration::from_secs(m.ctime().max(0) as u64),
        crtime: UNIX_EPOCH,
        kind: if m.is_dir() {
            FileType::Directory
        } else {
            FileType::RegularFile
        },
        perm: (m.mode() & 0o7777) as u16,
        nlink: m.nlink() as u32,
        uid: m.uid(),
        gid: m.gid(),
        rdev: 0,
        blksize: 4096,
        flags: 0,
    }
}

/// Bytes of every file under `dir`.
fn used(dir: &Path) -> u64 {
    fs::read_dir(dir).map_or(0, |entries| {
        entries
            .flatten()
            .map(|e| match e.metadata() {
                Ok(m) if m.is_dir() => used(&e.path()),
                Ok(m) => m.len(),
                Err(_) => 0,
            })
            .sum()
    })
}

impl Volume {
    fn path(&self, ino: u64) -> PathBuf {
        self.root.join(&self.paths[&ino])
    }

    /// The inode of `relative`, numbered on first sight.
    fn ino(&mut self, relative: PathBuf) -> u64 {
        if let Some((ino, _)) = self.paths.iter().find(|(_, p)| **p == relative) {
            return *ino;
        }
        self.next += 1;
        self.paths.insert(self.next, relative);
        self.next
    }

    fn child(&mut self, parent: u64, name: &OsStr) -> (u64, PathBuf) {
        let relative = self.paths[&parent].join(name);
        (self.ino(relative.clone()), self.root.join(relative))
    }
}

impl Filesystem for Volume {
    fn lookup(&mut self, _: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let (ino, path) = self.child(parent, name);
        match fs::symlink_metadata(path) {
            Ok(m) => reply.entry(&NOW, &attr(ino, &m), 0),
            Err(e) => reply.error(errno(e)),
        }
    }

    fn getattr(&mut self, _: &Request<'_>, ino: u64, _: Option<u64>, reply: ReplyAttr) {
        match fs::symlink_metadata(self.path(ino)) {
            Ok(m) => reply.attr(&NOW, &attr(ino, &m)),
            Err(e) => reply.error(errno(e)),
        }
    }

    fn setattr(
        &mut self,
        _: &Request<'_>,
        ino: u64,
        _: Option<u32>,
        _: Option<u32>,
        _: Option<u32>,
        size: Option<u64>,
        atime: Option<TimeOrNow>,
        mtime: Option<TimeOrNow>,
        _: Option<SystemTime>,
        _: Option<u64>,
        _: Option<SystemTime>,
        _: Option<SystemTime>,
        _: Option<SystemTime>,
        _: Option<u32>,
        reply: ReplyAttr,
    ) {
        let path = self.path(ino);
        let time = |t: TimeOrNow| match t {
            TimeOrNow::SpecificTime(t) => t,
            TimeOrNow::Now => SystemTime::now(),
        };
        let set = (|| {
            let file = OpenOptions::new()
                .write(size.is_some())
                .read(true)
                .open(&path)?;
            if let Some(size) = size {
                file.set_len(size)?;
            }
            let mut times = fs::FileTimes::new();
            if let Some(t) = atime {
                times = times.set_accessed(time(t));
            }
            if let Some(t) = mtime {
                times = times.set_modified(time(t));
            }
            file.set_times(times)?;
            fs::symlink_metadata(&path)
        })();
        match set {
            Ok(m) => reply.attr(&NOW, &attr(ino, &m)),
            Err(e) => reply.error(errno(e)),
        }
    }

    fn mkdir(
        &mut self,
        _: &Request<'_>,
        parent: u64,
        name: &OsStr,
        _: u32,
        _: u32,
        reply: ReplyEntry,
    ) {
        let (ino, path) = self.child(parent, name);
        match fs::create_dir(&path).and_then(|()| fs::symlink_metadata(&path)) {
            Ok(m) => reply.entry(&NOW, &attr(ino, &m), 0),
            Err(e) => reply.error(errno(e)),
        }
    }

    fn unlink(&mut self, _: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let (_, path) = self.child(parent, name);
        match fs::remove_file(path) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(errno(e)),
        }
    }

    fn rename(
        &mut self,
        _: &Request<'_>,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
        _: u32,
        reply: ReplyEmpty,
    ) {
        let (from_ino, from) = self.child(parent, name);
        let to_relative = self.paths[&newparent].join(newname);
        match fs::rename(&from, self.root.join(&to_relative)) {
            Ok(()) => {
                self.paths.insert(from_ino, to_relative);
                reply.ok()
            }
            Err(e) => reply.error(errno(e)),
        }
    }

    fn open(&mut self, _: &Request<'_>, ino: u64, flags: i32, reply: ReplyOpen) {
        let write = flags & libc::O_ACCMODE != libc::O_RDONLY;
        match OpenOptions::new()
            .read(true)
            .write(write)
            .open(self.path(ino))
        {
            Ok(file) => {
                self.next += 1;
                self.files.insert(self.next, file);
                reply.opened(self.next, fuser::consts::FOPEN_DIRECT_IO)
            }
            Err(e) => reply.error(errno(e)),
        }
    }

    fn create(
        &mut self,
        _: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _: u32,
        flags: i32,
        reply: ReplyCreate,
    ) {
        let (ino, path) = self.child(parent, name);
        let mut options = OpenOptions::new();
        options.read(true).write(true).mode(mode);
        match flags & libc::O_EXCL != 0 {
            true => options.create_new(true),
            false => options.create(true),
        };
        match options
            .open(&path)
            .and_then(|f| f.metadata().map(|m| (f, m)))
        {
            Ok((file, m)) => {
                self.next += 1;
                self.files.insert(self.next, file);
                reply.created(
                    &NOW,
                    &attr(ino, &m),
                    0,
                    self.next,
                    fuser::consts::FOPEN_DIRECT_IO,
                )
            }
            Err(e) => reply.error(errno(e)),
        }
    }

    /// Blocks the whole filesystem while reads are blocked, as a hard mount whose server stopped.
    fn read(
        &mut self,
        _: &Request<'_>,
        _: u64,
        fh: u64,
        offset: i64,
        size: u32,
        _: i32,
        _: Option<u64>,
        reply: ReplyData,
    ) {
        self.control.reads.fetch_add(1, Ordering::AcqRel);
        let control = &self.control;
        drop(
            control
                .unblocked
                .wait_while(control.blocked.lock().unwrap(), |b| *b)
                .unwrap(),
        );
        let mut data = vec![0; size as usize];
        match self.files[&fh].read_at(&mut data, offset as u64) {
            Ok(n) => reply.data(&data[..n]),
            Err(e) => reply.error(errno(e)),
        }
    }

    fn write(
        &mut self,
        _: &Request<'_>,
        _: u64,
        fh: u64,
        offset: i64,
        data: &[u8],
        _: u32,
        _: i32,
        _: Option<u64>,
        reply: ReplyWrite,
    ) {
        match self.files[&fh].write_all_at(data, offset as u64) {
            Ok(()) => reply.written(data.len() as u32),
            Err(e) => reply.error(errno(e)),
        }
    }

    fn flush(&mut self, _: &Request<'_>, _: u64, _: u64, _: u64, reply: ReplyEmpty) {
        reply.ok()
    }

    fn fsync(&mut self, _: &Request<'_>, _: u64, fh: u64, _: bool, reply: ReplyEmpty) {
        match self.files[&fh].sync_all() {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(errno(e)),
        }
    }

    fn release(
        &mut self,
        _: &Request<'_>,
        _: u64,
        fh: u64,
        _: i32,
        _: Option<u64>,
        _: bool,
        reply: ReplyEmpty,
    ) {
        self.files.remove(&fh);
        reply.ok()
    }

    fn readdir(
        &mut self,
        _: &Request<'_>,
        ino: u64,
        _: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        let parent = self.paths[&ino].clone();
        let mut names: Vec<_> = match fs::read_dir(self.path(ino)) {
            Ok(entries) => entries.flatten().map(|e| e.file_name()).collect(),
            Err(e) => return reply.error(errno(e)),
        };
        names.sort();
        for (at, name) in names.into_iter().enumerate().skip(offset as usize) {
            let path = self.root.join(&parent).join(&name);
            let kind = match path.is_dir() {
                true => FileType::Directory,
                false => FileType::RegularFile,
            };
            let ino = self.ino(parent.join(&name));
            if reply.add(ino, at as i64 + 1, kind, &name) {
                break;
            }
        }
        reply.ok()
    }

    fn statfs(&mut self, _: &Request<'_>, _: u64, reply: ReplyStatfs) {
        let capacity = self.control.capacity.load(Ordering::Acquire) / 4096;
        let free = capacity.saturating_sub(used(&self.root).div_ceil(4096));
        reply.statfs(capacity, free, free, 0, 0, 4096, 255, 4096)
    }
}
