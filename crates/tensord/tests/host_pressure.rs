//! The host tier under a real memory limit (MEM/HOST-PRESSURE.md), on the pod's terms: a
//! memory-limited scope whose PSI trigger write is denied (EACCES, as RunPod denies it). Sealed
//! layouts TensorD keeps, a page cache filling the scope, and a call's process growing its
//! anonymous memory past what is left. With nothing watching (the shipped behaviour when no PSI
//! trigger arms) the kernel kills the call; with the watcher on the scope's `memory.events` the
//! layouts go back by measured headroom and the call completes.
#![cfg(target_os = "linux")]
use platform::contain::host_pressure::{HostMode, Pressure};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    os::unix::{fs::PermissionsExt, process::ExitStatusExt},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
use tensord::{host_tier::HostTier, memory::MemoryLog};

const MIB: u64 = 1 << 20;
/// The scope: what it holds is sized against this.
const LIMIT: u64 = 768 * MIB;
/// Sealed layouts kept, a page cache read, and the call's growth: together past the limit.
const LAYOUT: u64 = 128 * MIB;
const LAYOUTS: usize = 3;
const CACHE: u64 = 256 * MIB;
const CALL: u64 = 448 * MIB;

fn scope(name: &str) -> Option<std::process::ExitStatus> {
    let limit = format!("MemoryMax={LIMIT}");
    // The kernel's OOM kill ends one process; systemd would otherwise stop the whole scope.
    let args = ["--user", "--scope", "--quiet", "--collect", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue", "-p", &limit];
    if !Command::new("systemd-run").args(args).arg("true").status().is_ok_and(|s| s.success()) {
        eprintln!("this session cannot create a memory-limited scope");
        return None;
    }
    let exe = std::env::current_exe().unwrap();
    let test = [name, "--exact", "--ignored", "--nocapture", "--test-threads=1"];
    Some(Command::new("systemd-run").args(args).arg(exe).args(test).status().unwrap())
}

fn own_cgroup() -> PathBuf {
    let text = fs::read_to_string("/proc/self/cgroup").unwrap();
    let relative = text.lines().find_map(|l| l.strip_prefix("0::")).unwrap();
    PathBuf::from("/sys/fs/cgroup").join(relative.trim_start_matches('/'))
}

/// This scope's PSI trigger write is denied, as a container's is: EACCES.
fn deny_psi() -> PathBuf {
    let cgroup = own_cgroup();
    fs::set_permissions(cgroup.join("memory.pressure"), fs::Permissions::from_mode(0o444)).unwrap();
    let denied = OpenOptions::new().write(true).open(cgroup.join("memory.pressure"));
    assert_eq!(denied.unwrap_err().raw_os_error(), Some(libc::EACCES));
    cgroup
}

fn event(cgroup: &Path, name: &str) -> u64 {
    let text = fs::read_to_string(cgroup.join("memory.events")).unwrap();
    text.lines().find_map(|l| l.strip_prefix(name)?.trim().parse().ok()).unwrap()
}

/// A sealed layout of `bytes`, filled here (charged to this scope), deposited by an executor
/// that has ended.
fn deposit(tier: &HostTier, n: usize, bytes: u64) {
    let layout = File::from(platform::shm::memfd_named("layout").unwrap());
    let chunk = vec![1u8; MIB as usize];
    for _ in 0..bytes / MIB {
        (&layout).write_all(&chunk).unwrap();
    }
    platform::shm::seal(std::os::fd::AsFd::as_fd(&layout)).unwrap();
    tier.deposits.deposit("sha256:m", &format!("layout-{n}"), layout, "run-gone", &[]).unwrap();
}

/// A clean page cache of `bytes`, as a downloaded checkpoint leaves: a file written, synced and
/// read back. (Dirty pages are no room: the kernel cannot give them back in time.)
fn cache(dir: &Path, bytes: u64) {
    let path = dir.join("cache");
    let written = Command::new("dd").args(["if=/dev/zero", &format!("of={}", path.display()), "bs=1M", "conv=fsync"])
        .arg(format!("count={}", bytes / MIB)).stderr(Stdio::null()).status().unwrap();
    assert!(written.success());
    assert!(Command::new("cat").arg(&path).stdout(Stdio::null()).status().unwrap().success());
}

/// A call's process: its anonymous memory grows to `bytes`, 16 MiB each 10 ms, as an executor's
/// load and denoise do, then it exits.
fn call(bytes: u64) -> std::process::ExitStatus {
    let script = format!(
        "import time\nheld = []\nfor _ in range({}):\n    b = bytearray({}); b[::4096] = b'\\1' * len(b[::4096]); held.append(b); time.sleep(0.01)\nprint('done', flush=True)",
        bytes / (16 * MIB),
        16 * MIB
    );
    let mut child = Command::new("python3").args(["-c", &script]).stdout(Stdio::piped()).spawn().unwrap();
    let mut line = String::new();
    let _ = BufReader::new(child.stdout.take().unwrap()).read_line(&mut line);
    child.wait().unwrap()
}

fn scenario(watch: bool) {
    let cgroup = deny_psi();
    let dir = tempfile::tempdir().unwrap();
    let tier = Arc::new(HostTier::new(dir.path(), HostMode::Dedicated, MemoryLog::at(dir.path().into())));
    let given = Arc::new(AtomicUsize::new(0));
    if watch {
        let (weak, count) = (Arc::downgrade(&tier), given.clone());
        tier.watch(move || {
            let gave = weak.upgrade()?.deposits.release_one();
            count.fetch_add(gave as usize, Ordering::AcqRel);
            Some(gave)
        });
    }
    for n in 0..LAYOUTS {
        deposit(&tier, n, LAYOUT);
    }
    cache(dir.path(), CACHE);
    let killed = event(&cgroup, "oom_kill");
    let status = call(CALL);
    let log = fs::read_to_string(dir.path().join("memory.log")).unwrap_or_default();
    let (oom_kill, max) = (event(&cgroup, "oom_kill"), event(&cgroup, "max"));
    eprintln!("watch={watch}: call {status:?}, oom_kill {killed} -> {oom_kill}, max {max}, layouts given {}\n{log}", given.load(Ordering::Acquire));
    if watch {
        assert!(log.contains("\"no psi trigger") && log.contains("inotify"), "{log}");
        assert!(status.success() && oom_kill == killed, "the call completes with no OOM kill");
        assert!(given.load(Ordering::Acquire) >= 1 && given.load(Ordering::Acquire) <= LAYOUTS);
    } else {
        assert_eq!(status.signal(), Some(libc::SIGKILL), "nothing gives back: the kernel kills the call");
        assert_eq!(oom_kill, killed + 1);
    }
}

#[test]
fn layouts_yield_to_a_growing_call_where_psi_is_denied() {
    for name in ["inside_with_nothing_watching", "inside_with_the_watcher"] {
        if let Some(status) = scope(name) {
            assert!(status.success(), "{name}");
        }
    }
}

#[test]
#[ignore = "run inside a memory-limited scope by the test above"]
fn inside_with_nothing_watching() {
    scenario(false);
}

#[test]
#[ignore = "run inside a memory-limited scope by the test above"]
fn inside_with_the_watcher() {
    scenario(true);
}

/// Where PSI is denied the watcher arms on `memory.events`, and the kernel's file-modified event
/// wakes it as the scope reaches its limit.
#[test]
fn memory_events_wake_the_watcher_where_psi_is_denied() {
    if let Some(status) = scope("inside_memory_events_wake") {
        assert!(status.success());
    }
}

#[test]
#[ignore = "run inside a memory-limited scope by the test above"]
fn inside_memory_events_wake() {
    let cgroup = deny_psi();
    let mut pressure = Pressure::arm(HostMode::Dedicated).unwrap();
    let events = cgroup.join("memory.events");
    assert!(pressure.how().contains(&format!("inotify {}", events.display())), "{}", pressure.how());
    let (woke, max) = (Arc::new(AtomicUsize::new(0)), event(&cgroup, "max"));
    let counted = woke.clone();
    std::thread::spawn(move || {
        while pressure.wait().is_ok() {
            counted.fetch_add(1, Ordering::AcqRel);
        }
    });
    let dir = tempfile::tempdir().unwrap();
    // Each round fills the scope past its limit; a second round the kernel counts with no wake
    // is a watcher that does not hear it.
    for round in 0.. {
        let before = event(&cgroup, "max");
        cache(dir.path(), 2 * LIMIT);
        assert!(event(&cgroup, "max") > before, "the scope reached its limit");
        if woke.load(Ordering::Acquire) > 0 {
            break;
        }
        assert!(round < 1, "{} limit events and no wake", event(&cgroup, "max") - max);
    }
}
