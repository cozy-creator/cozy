//! macOS: kqueue vnode events on the directory (an entry added, removed or renamed) and on each
//! regular file in it (written, extended, deleted or renamed). A directory event rescans it, so
//! a file rotated in is watched from then on.
use crate::{Error, Result};
use std::{
    collections::HashMap,
    fs, io,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::{ffi::OsStrExt, fs::MetadataExt},
    },
    path::{Path, PathBuf},
    ptr,
    time::Duration,
};

const DIRECTORY: u32 = libc::NOTE_WRITE | libc::NOTE_DELETE | libc::NOTE_RENAME;
const FILE: u32 = libc::NOTE_WRITE
    | libc::NOTE_EXTEND
    | libc::NOTE_DELETE
    | libc::NOTE_RENAME
    | libc::NOTE_ATTRIB;

pub(super) struct Watch {
    queue: OwnedFd,
    dir: PathBuf,
    directory: OwnedFd,
    /// Each regular file's descriptor, by (device, inode): a renamed file keeps its watch.
    files: HashMap<(u64, u64), OwnedFd>,
}

impl Watch {
    pub(super) fn new(dir: &Path) -> Result<Self> {
        // SAFETY: kqueue takes nothing and returns a new descriptor or -1.
        let raw = unsafe { libc::kqueue() };
        if raw < 0 {
            return Err(Error::os("kqueue")(io::Error::last_os_error()));
        }
        // SAFETY: a descriptor kqueue just returned and nothing else owns.
        let queue = unsafe { OwnedFd::from_raw_fd(raw) };
        let directory = open_events(dir).map_err(Error::io("watch", dir))?;
        let mut watch = Self {
            queue,
            dir: dir.to_path_buf(),
            directory,
            files: HashMap::new(),
        };
        watch
            .add(watch.directory.as_raw_fd(), DIRECTORY)
            .map_err(Error::io("watch", dir))?;
        watch.rescan()?;
        Ok(watch)
    }

    pub(super) fn wait(&mut self, most: Duration) -> Result<bool> {
        if !self.next(Some(most))? {
            return Ok(false);
        }
        // Drain what is already queued: one wake per wait.
        while self.next(Some(Duration::ZERO))? {}
        Ok(true)
    }

    /// One batch of events within `period`; a directory event or a file leaving rescans.
    fn next(&mut self, period: Option<Duration>) -> Result<bool> {
        let timeout = period.map(|period| libc::timespec {
            tv_sec: period.as_secs() as libc::time_t,
            tv_nsec: period.subsec_nanos() as libc::c_long,
        });
        // SAFETY: plain-old-data the call fills.
        let mut events: [libc::kevent; 16] = unsafe { std::mem::zeroed() };
        // SAFETY: a live queue, no changes, a writable event buffer of the length given.
        let n = unsafe {
            libc::kevent(
                self.queue.as_raw_fd(),
                ptr::null(),
                0,
                events.as_mut_ptr(),
                events.len() as libc::c_int,
                timeout.as_ref().map_or(ptr::null(), |t| t as *const _),
            )
        };
        if n < 0 {
            let error = io::Error::last_os_error();
            return match error.kind() {
                io::ErrorKind::Interrupted => Ok(false),
                _ => Err(Error::os("kevent")(error)),
            };
        }
        let events = &events[..n as usize];
        let directory = self.directory.as_raw_fd() as usize;
        if events.iter().any(|event| {
            event.ident == directory || event.fflags & (libc::NOTE_DELETE | libc::NOTE_RENAME) != 0
        }) {
            self.rescan()?;
        }
        Ok(n > 0)
    }

    /// Watches every regular file now in the directory and lets go of the ones that left it.
    fn rescan(&mut self) -> Result<()> {
        let mut present = HashMap::new();
        let entries = fs::read_dir(&self.dir).map_err(Error::io("list", &self.dir))?;
        for entry in entries.flatten() {
            let Ok(metadata) = fs::symlink_metadata(entry.path()) else {
                continue;
            };
            if metadata.is_file() {
                present.insert((metadata.dev(), metadata.ino()), entry.path());
            }
        }
        // Closing a descriptor removes its kevent.
        self.files.retain(|key, _| present.contains_key(key));
        for (key, path) in present {
            if self.files.contains_key(&key) {
                continue;
            }
            // A file gone between the listing and the open is the next rescan's business.
            let Ok(file) = open_events(&path) else {
                continue;
            };
            self.add(file.as_raw_fd(), FILE)
                .map_err(Error::io("watch", &path))?;
            self.files.insert(key, file);
        }
        Ok(())
    }

    fn add(&self, fd: libc::c_int, notes: u32) -> io::Result<()> {
        let change = libc::kevent {
            ident: fd as libc::uintptr_t,
            filter: libc::EVFILT_VNODE,
            flags: libc::EV_ADD | libc::EV_CLEAR,
            fflags: notes,
            data: 0,
            udata: ptr::null_mut(),
        };
        // SAFETY: a live queue, one change to register and no events asked back.
        let added = unsafe {
            libc::kevent(
                self.queue.as_raw_fd(),
                &change,
                1,
                ptr::null_mut(),
                0,
                ptr::null(),
            )
        };
        match added {
            0 => Ok(()),
            _ => Err(io::Error::last_os_error()),
        }
    }
}

/// A descriptor only for events: it keeps no volume busy and reads nothing.
fn open_events(path: &Path) -> io::Result<OwnedFd> {
    let path = std::ffi::CString::new(path.as_os_str().as_bytes())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    // SAFETY: a NUL-terminated path; the descriptor is owned at once.
    let raw = unsafe { libc::open(path.as_ptr(), libc::O_EVTONLY | libc::O_CLOEXEC) };
    if raw < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a descriptor open just returned and nothing else owns.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}
