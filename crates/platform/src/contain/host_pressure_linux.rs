//! The Linux trigger: PSI on the system or this cgroup, inotify on each limited cgroup v2's
//! `memory.events`, and an eventfd TensorD's own events write.
use super::{HostMode, WINDOW};
use std::{
    ffi::CString,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::{ffi::OsStrExt, fs::OpenOptionsExt},
    },
    path::{Path, PathBuf},
    sync::Arc,
};

/// What wakes the watcher: a PSI trigger, if this process may arm one, inotify on each limited
/// cgroup v2's `memory.events` it may watch, and its [`Nudge`].
pub struct Pressure {
    psi: Option<File>,
    events: Option<OwnedFd>,
    nudge: Nudge,
    how: String,
}

/// TensorD's own events wake the watcher for a look: the only wakeups where the kernel allows
/// none, and ones between the kernel's (cgroup v1 is read on every look, as v2 is).
#[derive(Clone)]
pub struct Nudge(Arc<OwnedFd>);

impl Nudge {
    pub fn send(&self) {
        // SAFETY: an eventfd write of one u64 from our own stack; a full counter only coalesces.
        unsafe { libc::write(self.0.as_raw_fd(), (&1u64 as *const u64).cast(), 8) };
    }
}

impl Pressure {
    /// Armed on whatever this process may watch, each limit on its own; `how` says what and why
    /// not. An error only when not even an eventfd exists.
    pub fn arm(mode: HostMode) -> crate::Result<Self> {
        // SAFETY: plain flags; the descriptor is owned below.
        let raw = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        if raw < 0 {
            return Err(io::Error::last_os_error().into());
        }
        // SAFETY: a descriptor eventfd just returned and nothing else owns.
        let nudge = Nudge(Arc::new(unsafe { OwnedFd::from_raw_fd(raw) }));
        let cgroups = super::super::host_memory::cgroups();
        let (path, kind) = match (mode, own_cgroup().filter(|c| c.join("memory.pressure").exists())) {
            (HostMode::Dedicated, Some(cgroup)) => (cgroup.join("memory.pressure"), "full"),
            // System `full` can stay quiet while one container stalls on a busy host.
            _ => (PathBuf::from("/proc/pressure/memory"), "some"),
        };
        let limited: Vec<PathBuf> = match cgroups {
            Some((groups, limits, "memory.current")) => groups
                .into_iter()
                .filter(|g| limits.iter().any(|l| super::super::host_memory::number(&g.join(l)).is_some()))
                .map(|g| g.join("memory.events"))
                .collect(),
            _ => vec![],
        };
        let mut how = vec![];
        let psi = trigger(&path, kind)
            .inspect(|_| how.push(format!("psi {} {kind}", path.display())))
            .inspect_err(|error| how.push(format!("no psi trigger on {} ({error})", path.display())))
            .ok();
        let events = watch(&limited, &mut how);
        how.push("tensord events".into());
        Ok(Self { psi, events, nudge, how: how.join("; ") })
    }

    /// What it is armed on, for the record.
    pub fn how(&self) -> &str {
        &self.how
    }

    pub fn nudge(&self) -> Nudge {
        self.nudge.clone()
    }

    /// Blocks until the kernel reports a stall, a watched `memory.events` changes, or a nudge
    /// comes: no timeout, no work in between.
    pub fn wait(&mut self) -> crate::Result<()> {
        let fd = |file: Option<i32>, events| libc::pollfd { fd: file.unwrap_or(-1), events, revents: 0 };
        let mut fds = [
            fd(self.psi.as_ref().map(AsRawFd::as_raw_fd), libc::POLLPRI),
            fd(self.events.as_ref().map(AsRawFd::as_raw_fd), libc::POLLIN),
            fd(Some(self.nudge.0.as_raw_fd()), libc::POLLIN),
        ];
        loop {
            // SAFETY: live descriptors (-1 is ignored by poll).
            if unsafe { libc::poll(fds.as_mut_ptr(), 3, -1) } > 0 {
                if fds[0].revents & libc::POLLERR != 0 {
                    return Err(io::Error::other("the PSI trigger is gone").into());
                }
                let mut buf = [0u8; 4096];
                for drained in [self.events.as_ref(), Some(&*self.nudge.0)].into_iter().flatten() {
                    // SAFETY: reads into our own buffer from a nonblocking descriptor, until empty.
                    while unsafe { libc::read(drained.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len()) } > 0 {}
                }
                return Ok(());
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error.into());
            }
        }
    }
}

/// A PSI trigger on one pressure file (`memory.pressure` of a cgroup, or the system's).
fn trigger(path: &Path, kind: &str) -> io::Result<File> {
    let mut trigger = OpenOptions::new().read(true).write(true).custom_flags(libc::O_NONBLOCK).open(path)?;
    trigger.write_all(format!("{kind} 1 {}\0", WINDOW.as_micros()).as_bytes())?;
    Ok(trigger)
}

/// One inotify descriptor watching each file it can for IN_MODIFY, each noted in `how`; None
/// when it watches none.
fn watch(files: &[PathBuf], how: &mut Vec<String>) -> Option<OwnedFd> {
    if files.is_empty() {
        how.push("no limited cgroup v2".into());
        return None;
    }
    // SAFETY: plain flags; the descriptor is owned below.
    let raw = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
    if raw < 0 {
        how.push(format!("no inotify ({})", io::Error::last_os_error()));
        return None;
    }
    // SAFETY: a descriptor inotify_init1 just returned and nothing else owns.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    let mut watched = false;
    for file in files {
        let added = CString::new(file.as_os_str().as_bytes()).map_err(io::Error::other).and_then(|path| {
            // SAFETY: a live inotify descriptor and a NUL-terminated path.
            match unsafe { libc::inotify_add_watch(fd.as_raw_fd(), path.as_ptr(), libc::IN_MODIFY) } {
                ..0 => Err(io::Error::last_os_error()),
                _ => Ok(()),
            }
        });
        match added {
            Ok(()) => {
                how.push(format!("inotify {}", file.display()));
                watched = true;
            }
            Err(error) => how.push(format!("no inotify on {} ({error})", file.display())),
        }
    }
    watched.then_some(fd)
}

/// This process's cgroup v2 directory.
fn own_cgroup() -> Option<PathBuf> {
    let text = fs::read_to_string("/proc/self/cgroup").ok()?;
    let relative = text.lines().find_map(|line| line.strip_prefix("0::"))?;
    Some(PathBuf::from("/sys/fs/cgroup").join(relative.trim_start_matches('/')))
}

#[cfg(test)]
mod tests {
    use super::super::{Growth, Reserve};
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn growth_is_the_most_unforeseen_anonymous_memory_grew_between_two_looks_and_settles_per_call() {
        let mut growth = Growth::default();
        let looks = |rows: &[(&str, u64)]| rows.iter().map(|(call, gib)| (call.to_string(), gib << 30)).collect::<BTreeMap<_, _>>();
        assert_eq!(growth.observe(looks(&[("a", 2)])), 0, "one look measures nothing");
        // A call's unforeseen anonymous memory grows 6 GiB between two looks; another starts.
        assert_eq!(growth.observe(looks(&[("a", 8), ("b", 5)])), 6 << 30);
        assert_eq!(growth.observe(looks(&[("a", 7), ("b", 6)])), 6 << 30, "a high-water");
        // A call ends measured: what was seen stays, the rest halves with each later one.
        growth.settle();
        assert_eq!(growth.most(), 6 << 30);
        assert_eq!(growth.observe(looks(&[("b", 7)])), 6 << 30);
        growth.settle();
        assert_eq!(growth.most(), (1 << 30) + (5 << 30) / 2);
        growth.settle();
        assert_eq!(growth.most(), (7 << 30) / 4);
    }

    #[test]
    fn the_reserve_keeps_a_recent_peak_and_forgets_it_on_quiet_passes() {
        let mut reserve = Reserve::default();
        assert_eq!(reserve.observe(4 << 30), 0);
        // A 6 GiB burst came and went: 6 GiB stays theirs for now.
        assert_eq!(reserve.observe(10 << 30), 0);
        assert_eq!(reserve.observe(4 << 30), 6 << 30);
        reserve.quiet(4 << 30);
        assert_eq!(reserve.observe(4 << 30), 3 << 30);
        reserve.quiet(4 << 30);
        reserve.quiet(4 << 30);
        assert_eq!(reserve.observe(4 << 30), (3 << 30) / 4);
    }

    /// Whatever the kernel allows, the watcher arms and a nudge wakes it.
    #[test]
    fn the_watcher_arms_and_a_nudge_wakes_it() {
        let mut pressure = Pressure::arm(HostMode::Shared).unwrap();
        assert!(pressure.how().ends_with("tensord events"), "{}", pressure.how());
        if Path::new("/proc/pressure/memory").exists() {
            assert!(pressure.how().starts_with("psi /proc/pressure/memory some"), "{}", pressure.how());
        }
        let nudge = pressure.nudge();
        let woke = std::thread::spawn(move || pressure.wait().is_ok());
        nudge.send();
        assert!(woke.join().unwrap());
    }

    /// A limit whose `memory.events` cannot be watched is named, and the others are still watched.
    #[test]
    fn each_memory_events_is_watched_on_its_own() {
        let dir = tempfile::tempdir().unwrap();
        let (watchable, missing) = (dir.path().join("memory.events"), dir.path().join("gone/memory.events"));
        fs::write(&watchable, "max 0\n").unwrap();
        let mut how = vec![];
        let fd = watch(&[missing.clone(), watchable.clone()], &mut how).expect("one is watched");
        assert!(how[0].starts_with(&format!("no inotify on {}", missing.display())), "{how:?}");
        assert_eq!(how[1], format!("inotify {}", watchable.display()));
        fs::write(&watchable, "max 1\n").unwrap();
        let mut poll = libc::pollfd { fd: fd.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        // SAFETY: one live descriptor; the write above has already queued its event.
        assert_eq!(unsafe { libc::poll(&mut poll, 1, 0) }, 1);
        assert!(watch(&[missing], &mut how).is_none());
    }
}
