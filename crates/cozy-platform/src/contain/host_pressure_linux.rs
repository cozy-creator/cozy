//! The Linux trigger: PSI on the system or this cgroup, `memory.events`, and cgroup v1's
//! limit-hit counters.
use super::{HostMode, Sample, WINDOW};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{FileExt, OpenOptionsExt},
    },
    path::PathBuf,
};

/// One armed PSI trigger. Dedicated v2 also watches memory.events; dedicated v1
/// qualifies system wakeups with its finite cgroup limits' memory.failcnt counters.
pub struct Pressure {
    trigger: File,
    path: PathBuf,
    kind: &'static str,
    events: Option<File>,
    limit_hits: Vec<PathBuf>,
}

impl Pressure {
    pub fn arm(mode: HostMode) -> crate::Result<Self> {
        if mode == HostMode::Dedicated
            && let Some((groups, _, "memory.usage_in_bytes")) = super::super::host_memory::cgroups()
        {
            let limited = v1_limits(groups);
            if !limited.is_empty() {
                return Self::arm_v1(limited, "/proc/pressure/memory".into());
            }
        }
        let cgroup = own_cgroup().filter(|cgroup| cgroup.join("memory.pressure").exists());
        match (mode, cgroup) {
            (HostMode::Dedicated, Some(cgroup)) => {
                let mut pressure = Self::arm_at(cgroup.join("memory.pressure"), "full")?;
                pressure.events = File::open(cgroup.join("memory.events")).ok();
                Ok(pressure)
            }
            (HostMode::Dedicated, None) => Self::arm_at("/proc/pressure/memory".into(), "full"),
            (HostMode::Shared, _) => Self::arm_at("/proc/pressure/memory".into(), "some"),
        }
    }

    fn arm_v1(groups: Vec<PathBuf>, system_pressure: PathBuf) -> crate::Result<Self> {
        // System `full` can stay quiet while one container stalls on a busy host.
        // `some` is only the wakeup: unrelated host stalls cannot release this pod's
        // holdings without a local limit hit. These counters also work on read-only
        // cgroup mounts; no event_control registration or polling timer is needed.
        let mut pressure = Self::arm_at(system_pressure, "some")?;
        pressure.limit_hits = groups
            .into_iter()
            .map(|group| group.join("memory.failcnt"))
            .collect();
        if pressure.limit_hits.is_empty() {
            return Err(io::Error::other("cgroup v1 memory limit counters are unavailable").into());
        }
        pressure.sample()?;
        Ok(pressure)
    }

    /// A trigger on one pressure file (`memory.pressure` of a cgroup, or the system's).
    pub fn arm_at(path: PathBuf, kind: &'static str) -> crate::Result<Self> {
        let mut trigger = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&path)?;
        trigger.write_all(format!("{kind} 1 {}\0", WINDOW.as_micros()).as_bytes())?;
        Ok(Self {
            trigger,
            path,
            kind,
            events: None,
            limit_hits: Vec::new(),
        })
    }

    /// Blocks until the kernel reports a stall (or, dedicated, a `memory.events` change): no
    /// timeout, no work in between.
    pub fn wait(&self) -> crate::Result<()> {
        let mut fds = [self.trigger.as_raw_fd(), -1].map(|fd| libc::pollfd {
            fd,
            events: libc::POLLPRI,
            revents: 0,
        });
        if let Some(events) = &self.events {
            fds[1].fd = events.as_raw_fd();
            // kernfs re-arms a change notice when this descriptor reads the file
            let _ = events.read_at(&mut [0u8; 512], 0);
        }
        loop {
            // SAFETY: live descriptors (-1 is ignored by poll).
            if unsafe { libc::poll(fds.as_mut_ptr(), 2, -1) } > 0 {
                return match fds[0].revents & libc::POLLERR {
                    0 => Ok(()),
                    _ => Err(io::Error::other("the PSI trigger is gone").into()),
                };
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error.into());
            }
        }
    }

    /// Microseconds stalled so far, of this trigger's kind.
    pub fn stalled_us(&self) -> crate::Result<u64> {
        let text = fs::read_to_string(&self.path)?;
        Ok(text
            .lines()
            .find(|line| line.starts_with(self.kind))
            .and_then(|line| {
                line.split_whitespace()
                    .find_map(|f| f.strip_prefix("total="))
            })
            .and_then(|total| total.parse().ok())
            .ok_or_else(|| io::Error::other("no PSI total"))?)
    }

    pub fn sample(&self) -> crate::Result<Sample> {
        if self.limit_hits.is_empty() {
            return self.stalled_us().map(Sample::StalledUs);
        }
        let mut total = 0u64;
        for path in &self.limit_hits {
            let hits = fs::read_to_string(path)?
                .trim()
                .parse::<u64>()
                .map_err(|_| io::Error::other("invalid cgroup v1 memory limit counter"))?;
            total = total.saturating_add(hits);
        }
        Ok(Sample::LimitHits(total))
    }
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
    use super::super::{Feedback, Reserve};
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Instant;

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
        fn arm(&self) -> Pressure {
            Pressure::arm_v1(
                vec![self.0.join("container"), self.0.clone()],
                self.0.join("system-pressure"),
            )
            .unwrap()
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
        for path in [
            files.0.join("memory.failcnt"),
            files.0.join("container/memory.failcnt"),
        ] {
            fs::set_permissions(path, fs::Permissions::from_mode(0o444)).unwrap();
        }
        // No cgroup.event_control or memory.pressure_level exists in this fixture.
        let pressure = files.arm();
        let trigger = fs::read(files.0.join("system-pressure")).unwrap();
        assert!(trigger.starts_with(b"some 1 2000000\0"));
        assert_eq!(pressure.sample().unwrap(), Sample::LimitHits(40));
        assert_eq!(pressure.sample().unwrap().stalled_us(), None);
    }

    #[test]
    fn unrelated_host_stalls_cannot_reclaim_v1_holdings_but_local_limit_hits_can() {
        let files = V1Files::new();
        let pressure = files.arm();
        let start = Instant::now();
        let mut feedback = Feedback::new(pressure.sample().unwrap().total(), start);
        // Host `some` grows while `full` is quiet: another tenant alone cannot
        // evict our caches or prevent their quiet bring-back.
        fs::write(
            files.0.join("system-pressure"),
            "some total=9000000\nfull total=0\n",
        )
        .unwrap();
        assert_eq!(
            feedback.observe(pressure.sample().unwrap(), start + WINDOW),
            None
        );
        // This container now reaches its limit. Parent-only pressure counts too.
        fs::write(files.0.join("container/memory.failcnt"), "14\n").unwrap();
        assert_eq!(
            feedback.observe(pressure.sample().unwrap(), start + WINDOW * 2),
            Some(true)
        );
        fs::write(files.0.join("memory.failcnt"), "31\n").unwrap();
        assert_eq!(
            feedback.observe(pressure.sample().unwrap(), start + WINDOW * 3),
            Some(true)
        );
        assert_eq!(
            feedback.observe(pressure.sample().unwrap(), start + WINDOW * 4),
            None
        );
    }

    #[test]
    fn resetting_v1_counters_does_not_invent_pressure_or_disable_the_next_episode() {
        let start = Instant::now();
        let mut feedback = Feedback::new(40, start);
        assert_eq!(
            feedback.observe(Sample::LimitHits(44), start + WINDOW),
            Some(true)
        );
        assert_eq!(
            feedback.observe(Sample::LimitHits(0), start + WINDOW * 2),
            None
        );
        assert_eq!(
            feedback.observe(Sample::LimitHits(1), start + WINDOW * 3),
            Some(true)
        );
    }

    #[test]
    fn unreadable_v1_counter_is_not_replaced_by_unrelated_host_psi() {
        let files = V1Files::new();
        let pressure = files.arm();
        fs::write(files.0.join("container/memory.failcnt"), "unreadable\n").unwrap();
        assert!(pressure.sample().is_err());
    }

    #[test]
    fn a_missing_controlling_ancestor_counter_cannot_be_silently_omitted() {
        let files = V1Files::new();
        fs::remove_file(files.0.join("memory.failcnt")).unwrap();
        assert!(
            Pressure::arm_v1(
                vec![files.0.join("container"), files.0.clone()],
                files.0.join("system-pressure"),
            )
            .is_err()
        );
    }

    #[test]
    fn an_unlimited_v1_host_keeps_system_pressure_instead_of_an_inert_limit_counter() {
        let files = V1Files::new();
        fs::write(
            files.0.join("memory.limit_in_bytes"),
            "9223372036854771712\n",
        )
        .unwrap();
        let groups = vec![files.0.join("container"), files.0.clone()];
        assert_eq!(v1_limits(groups.clone()), vec![files.0.join("container")]);
        fs::write(
            files.0.join("container/memory.limit_in_bytes"),
            "9223372036854771712\n",
        )
        .unwrap();
        assert!(v1_limits(groups).is_empty());
    }

    /// Events every window with these stall shares; true where a rung went.
    fn run(shares: &[f64]) -> Vec<bool> {
        let start = Instant::now();
        let mut total = 0;
        let mut feedback = Feedback::new(0, start);
        shares
            .iter()
            .enumerate()
            .map(|(i, share)| {
                total += (share * WINDOW.as_micros() as f64) as u64;
                feedback.give(total, start + WINDOW * (i as u32 + 1))
            })
            .collect()
    }

    #[test]
    fn rungs_go_while_each_lowers_the_stall() {
        // A memory hog: each rung given back lowers the stall, until it is gone.
        assert_eq!(run(&[0.6, 0.4, 0.2, 0.05]), [true; 4]);
    }

    #[test]
    fn a_stall_giving_back_cannot_touch_sheds_one_rung_however_long_it_lasts() {
        // Refaults on a slow disk: the share wanders around 0.3 whatever is given.
        let refaults = [0.3, 0.31, 0.29, 0.3, 0.32, 0.3, 0.28, 0.31, 0.3, 0.29];
        let given = run(&refaults);
        assert_eq!(given.iter().filter(|g| **g).count(), 1, "{given:?}");
    }

    #[test]
    fn a_stall_rising_beyond_the_noise_after_a_stop_is_answered() {
        let given = run(&[0.3, 0.31, 0.29, 0.3, 0.6, 0.4]);
        assert_eq!(given, [true, false, false, false, true, true]);
    }

    #[test]
    fn a_quiet_window_starts_a_new_episode_that_remembers_where_giving_stopped() {
        let start = Instant::now();
        let mut feedback = Feedback::new(0, start);
        assert!(feedback.give(600_000, start + WINDOW), "0.3: a rung");
        assert!(
            !feedback.give(1_200_000, start + WINDOW * 2),
            "0.3 again: stopped"
        );
        assert!(
            !feedback.give(1_800_000, start + WINDOW * 3),
            "0.3: the noise is 0"
        );
        // No stall for three windows, then the same background burst: nothing.
        assert!(!feedback.give(2_400_000, start + WINDOW * 6));
        // A hog's stall well above it is answered, and falls rung by rung.
        assert!(feedback.give(3_600_000, start + WINDOW * 7), "0.6");
        assert!(feedback.give(4_400_000, start + WINDOW * 8), "0.4");
    }

    #[test]
    fn nothing_left_to_give_waits_for_more_pressure_than_the_noise() {
        let start = Instant::now();
        let mut feedback = Feedback::new(0, start);
        assert!(feedback.give(400_000, start + WINDOW));
        feedback.exhausted();
        assert!(
            !feedback.give(900_000, start + WINDOW * 2),
            "noise is measured first"
        );
        assert!(
            !feedback.give(1_380_000, start + WINDOW * 3),
            "within the noise"
        );
        assert!(
            feedback.give(2_580_000, start + WINDOW * 4),
            "well above it"
        );
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
        if !std::path::Path::new("/proc/pressure/memory").exists() {
            return;
        }
        let pressure = Pressure::arm(HostMode::Shared).expect("an unprivileged 2 s trigger");
        assert!(pressure.stalled_us().is_ok());
    }
}
