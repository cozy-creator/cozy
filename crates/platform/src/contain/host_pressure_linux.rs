//! The Linux trigger: PSI on the system or this cgroup, inotify on the limited cgroups'
//! `memory.events`, and cgroup v1's limit-hit counters.
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
};

/// What wakes the watcher: a PSI trigger, if this process may arm one, and inotify on the
/// `memory.events` of every limited cgroup v2 on its path; v1 qualifies wakeups by its finite
/// limits' `memory.failcnt`.
pub struct Pressure {
    psi: Option<File>,
    events: Option<OwnedFd>,
    limit_hits: Vec<PathBuf>,
    hits: u64,
    how: String,
}

impl Pressure {
    /// Armed on whatever this process may watch; an error names why nothing could be.
    pub fn arm(mode: HostMode) -> crate::Result<Self> {
        let cgroups = super::super::host_memory::cgroups();
        if let Some((groups, _, "memory.usage_in_bytes")) = &cgroups
            && mode == HostMode::Dedicated
        {
            let limited = v1_limits(groups.clone());
            if !limited.is_empty() {
                return Self::arm_v1(limited, "/proc/pressure/memory".into());
            }
        }
        let (path, kind) = match (mode, own_cgroup().filter(|c| c.join("memory.pressure").exists())) {
            (HostMode::Dedicated, Some(cgroup)) => (cgroup.join("memory.pressure"), "full"),
            (HostMode::Dedicated, None) => (PathBuf::from("/proc/pressure/memory"), "full"),
            (HostMode::Shared, _) => (PathBuf::from("/proc/pressure/memory"), "some"),
        };
        let limited: Vec<PathBuf> = match cgroups {
            Some((groups, limits, "memory.current")) => groups
                .into_iter()
                .filter(|g| limits.iter().any(|l| super::super::host_memory::number(&g.join(l)).is_some()))
                .map(|g| g.join("memory.events"))
                .collect(),
            _ => vec![],
        };
        let psi = trigger(&path, kind);
        let events = watch(&limited);
        let mut how: Vec<String> = vec![];
        match &psi {
            Ok(_) => how.push(format!("psi {} {kind}", path.display())),
            Err(error) => how.push(format!("no psi trigger on {} ({error})", path.display())),
        }
        match &events {
            Ok(Some(_)) => how.extend(limited.iter().map(|e| format!("inotify {}", e.display()))),
            Ok(None) => how.push("no limited cgroup".into()),
            Err(error) => how.push(format!("no inotify on memory.events ({error})")),
        }
        let how = how.join("; ");
        match (psi, events) {
            (Err(_), Ok(None) | Err(_)) => Err(io::Error::other(how).into()),
            (psi, events) => Ok(Self { psi: psi.ok(), events: events.ok().flatten(), limit_hits: vec![], hits: 0, how }),
        }
    }

    fn arm_v1(groups: Vec<PathBuf>, system_pressure: PathBuf) -> crate::Result<Self> {
        // System `full` can stay quiet while one container stalls on a busy host. `some` is only
        // the wakeup: an unrelated host stall is no look without a local limit hit. These counters
        // also work on read-only cgroup mounts; no event_control registration is needed.
        let limit_hits: Vec<PathBuf> = groups.into_iter().map(|group| group.join("memory.failcnt")).collect();
        if limit_hits.is_empty() {
            return Err(io::Error::other("cgroup v1 memory limit counters are unavailable").into());
        }
        let psi = trigger(&system_pressure, "some")?;
        let how = format!("psi {} some, cgroup v1 limit hits", system_pressure.display());
        let mut pressure = Self { psi: Some(psi), events: None, limit_hits, hits: 0, how };
        pressure.hits = pressure.limit_hits()?;
        Ok(pressure)
    }

    /// What it is armed on, for the record.
    pub fn how(&self) -> &str {
        &self.how
    }

    /// Blocks until the kernel reports a stall or a limited cgroup's `memory.events` changes (v1:
    /// a stall with a local limit hit): no timeout, no work in between.
    pub fn wait(&mut self) -> crate::Result<()> {
        loop {
            self.poll()?;
            if self.local()? {
                return Ok(());
            }
        }
    }

    /// Whether a wakeup is this machine's: always on v2, on v1 when a limit counter moved.
    fn local(&mut self) -> crate::Result<bool> {
        if self.limit_hits.is_empty() {
            return Ok(true);
        }
        let hits = self.limit_hits()?;
        Ok(std::mem::replace(&mut self.hits, hits) != hits)
    }

    fn poll(&self) -> crate::Result<()> {
        let fd = |file: Option<i32>, events| libc::pollfd { fd: file.unwrap_or(-1), events, revents: 0 };
        let mut fds = [
            fd(self.psi.as_ref().map(AsRawFd::as_raw_fd), libc::POLLPRI),
            fd(self.events.as_ref().map(AsRawFd::as_raw_fd), libc::POLLIN),
        ];
        loop {
            // SAFETY: live descriptors (-1 is ignored by poll).
            if unsafe { libc::poll(fds.as_mut_ptr(), 2, -1) } > 0 {
                if fds[0].revents & libc::POLLERR != 0 {
                    return Err(io::Error::other("the PSI trigger is gone").into());
                }
                if let Some(events) = &self.events {
                    let mut buf = [0u8; 4096];
                    // SAFETY: reads into our own buffer from a nonblocking descriptor, until empty.
                    while unsafe { libc::read(events.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len()) } > 0 {}
                }
                return Ok(());
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error.into());
            }
        }
    }

    fn limit_hits(&self) -> crate::Result<u64> {
        let mut total = 0u64;
        for path in &self.limit_hits {
            let hits = fs::read_to_string(path)?
                .trim()
                .parse::<u64>()
                .map_err(|_| io::Error::other("invalid cgroup v1 memory limit counter"))?;
            total = total.saturating_add(hits);
        }
        Ok(total)
    }
}

/// A PSI trigger on one pressure file (`memory.pressure` of a cgroup, or the system's).
fn trigger(path: &Path, kind: &str) -> io::Result<File> {
    let mut trigger = OpenOptions::new().read(true).write(true).custom_flags(libc::O_NONBLOCK).open(path)?;
    trigger.write_all(format!("{kind} 1 {}\0", WINDOW.as_micros()).as_bytes())?;
    Ok(trigger)
}

/// One inotify descriptor watching each file for IN_MODIFY; None when there is none to watch.
fn watch(files: &[PathBuf]) -> io::Result<Option<OwnedFd>> {
    if files.is_empty() {
        return Ok(None);
    }
    // SAFETY: plain flags; the descriptor is owned below.
    let raw = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
    if raw < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a descriptor inotify_init1 just returned and nothing else owns.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    for file in files {
        let path = CString::new(file.as_os_str().as_bytes()).map_err(io::Error::other)?;
        // SAFETY: a live inotify descriptor and a NUL-terminated path.
        if unsafe { libc::inotify_add_watch(fd.as_raw_fd(), path.as_ptr(), libc::IN_MODIFY) } < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(Some(fd))
}

fn v1_limits(groups: Vec<PathBuf>) -> Vec<PathBuf> {
    groups
        .into_iter()
        .filter(|group| {
            super::super::host_memory::number(&group.join("memory.limit_in_bytes")).is_some()
        })
        .collect()
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
    use super::super::super::host_memory::HostMemory;
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct V1Files(PathBuf);
    impl V1Files {
        fn new() -> Self {
            static FIXTURES: AtomicU32 = AtomicU32::new(0);
            let n = FIXTURES.fetch_add(1, Ordering::Relaxed);
            let root =
                std::env::temp_dir().join(format!("cozy-pressure-{}-{n}", std::process::id()));
            fs::create_dir_all(root.join("container")).unwrap();
            fs::write(root.join("memory.failcnt"), "30\n").unwrap();
            fs::write(root.join("container/memory.failcnt"), "10\n").unwrap();
            fs::write(root.join("memory.limit_in_bytes"), "134217728\n").unwrap();
            fs::write(root.join("container/memory.limit_in_bytes"), "67108864\n").unwrap();
            fs::write(root.join("system-pressure"), "full avg10=0.00 total=0\n").unwrap();
            Self(root)
        }
        fn arm(&self) -> crate::Result<Pressure> {
            Pressure::arm_v1(vec![self.0.join("container"), self.0.clone()], self.0.join("system-pressure"))
        }
    }
    impl Drop for V1Files {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn dedicated_v1_arms_some_and_reads_container_and_parent_limits_without_control_writes() {
        let files = V1Files::new();
        for path in [files.0.join("memory.failcnt"), files.0.join("container/memory.failcnt")] {
            fs::set_permissions(path, fs::Permissions::from_mode(0o444)).unwrap();
        }
        // No cgroup.event_control or memory.pressure_level exists in this fixture.
        let pressure = files.arm().unwrap();
        let trigger = fs::read(files.0.join("system-pressure")).unwrap();
        assert!(trigger.starts_with(b"some 1 2000000\0"));
        assert_eq!(pressure.hits, 40);
    }

    #[test]
    fn only_a_local_limit_hit_makes_a_v1_wakeup_a_look() {
        let files = V1Files::new();
        let mut pressure = files.arm().unwrap();
        // A host stall with no local limit hit: another tenant's, not a look.
        assert!(!pressure.local().unwrap());
        fs::write(files.0.join("container/memory.failcnt"), "14\n").unwrap();
        assert!(pressure.local().unwrap());
        // A parent's limit counts too, and an administrator's reset is a change like any other.
        fs::write(files.0.join("memory.failcnt"), "31\n").unwrap();
        assert!(pressure.local().unwrap());
        assert!(!pressure.local().unwrap());
        fs::write(files.0.join("memory.failcnt"), "0\n").unwrap();
        assert!(pressure.local().unwrap());
    }

    #[test]
    fn unreadable_v1_counter_is_not_replaced_by_unrelated_host_psi() {
        let files = V1Files::new();
        let pressure = files.arm().unwrap();
        fs::write(files.0.join("container/memory.failcnt"), "unreadable\n").unwrap();
        assert!(pressure.limit_hits().is_err());
    }

    #[test]
    fn a_missing_controlling_ancestor_counter_cannot_be_silently_omitted() {
        let files = V1Files::new();
        fs::remove_file(files.0.join("memory.failcnt")).unwrap();
        assert!(files.arm().is_err());
    }

    #[test]
    fn an_unlimited_v1_host_keeps_system_pressure_instead_of_an_inert_limit_counter() {
        let files = V1Files::new();
        fs::write(files.0.join("memory.limit_in_bytes"), "9223372036854771712\n").unwrap();
        let groups = vec![files.0.join("container"), files.0.clone()];
        assert_eq!(v1_limits(groups.clone()), vec![files.0.join("container")]);
        fs::write(files.0.join("container/memory.limit_in_bytes"), "9223372036854771712\n").unwrap();
        assert!(v1_limits(groups).is_empty());
    }

    fn host(available: i64, shmem: i64, bound: i64) -> HostMemory {
        HostMemory { available, shmem, mem_available: available, bound }
    }

    #[test]
    fn growth_is_the_most_non_shared_memory_grew_between_two_looks() {
        let mut growth = Growth::default();
        assert_eq!(growth.observe(host(20 << 30, 0, 31 << 30)), 0, "one look measures nothing");
        // An executor's anonymous memory grows 6 GiB between two looks.
        assert_eq!(growth.observe(host(14 << 30, 0, 31 << 30)), 6 << 30);
        // The tier fills 8 GiB of sealed layouts and the call grows 1 GiB: 1 GiB is growth.
        assert_eq!(growth.observe(host(5 << 30, 8 << 30, 31 << 30)), 6 << 30);
        assert_eq!(growth.observe(host(4 << 30, 8 << 30, 31 << 30)), 6 << 30, "a high-water");
        // The tightest limit is another one now: no growth across the change.
        assert_eq!(growth.observe(host(1 << 30, 8 << 30, 128 << 30)), 6 << 30);
        assert_eq!(growth.observe(host(-1, 8 << 30, 128 << 30)), 6 << 30, "unreadable: no look");
        assert_eq!(growth.observe(host(-6 << 30, 8 << 30, 128 << 30)), 6 << 30);
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

    #[test]
    fn a_trigger_arms_on_the_systems_memory_pressure() {
        // Kernels without PSI (or a sandbox that hides it) have nothing to arm.
        if !Path::new("/proc/pressure/memory").exists() {
            return;
        }
        let pressure = Pressure::arm(HostMode::Shared).expect("an unprivileged 2 s trigger");
        assert!(pressure.how().starts_with("psi /proc/pressure/memory some"), "{}", pressure.how());
    }
}
