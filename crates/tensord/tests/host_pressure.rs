//! The host tier under a real memory limit, on the pod's terms: a memory-limited scope whose PSI
//! trigger write is denied (EACCES, as RunPod denies it). Sealed layouts TensorD keeps, a page
//! cache filling the scope, and a call's process growing its anonymous memory past what is left.
//! With nothing watching the kernel kills the call; with the watcher on the scope's
//! `memory.events` idle layouts go back by measured headroom and the call completes. A layout a
//! warm executor maps is counted once, in the budget, never as its call's need.
#![cfg(target_os = "linux")]
use platform::contain::host_pressure::{HostMode, Pressure};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    os::{
        fd::AsRawFd,
        unix::{fs::PermissionsExt, process::ExitStatusExt},
    },
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
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

/// Runs test `name` inside a memory-limited scope with no swap, where this session can make one: a
/// user systemd manager the memory controller is delegated to. Elsewhere None, saying why; the
/// test proves nothing without the scope.
fn scope(name: &str) -> Option<std::process::ExitStatus> {
    let limit = format!("MemoryMax={LIMIT}");
    // The kernel's OOM kill ends one process; systemd would otherwise stop the whole scope.
    let args = ["--user", "--scope", "--quiet", "--collect", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue", "-p", &limit];
    let probe = Command::new("systemd-run").args(args).arg("true").output();
    if !probe.as_ref().is_ok_and(|p| p.status.success()) {
        let why = probe.map_or_else(|e| e.to_string(), |p| String::from_utf8_lossy(&p.stderr).trim().to_owned());
        eprintln!("skipped {name}: no memory-limited scope here (`systemd-run {}`: {why})", args.join(" "));
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
    assert_eq!(fs::read_to_string(cgroup.join("memory.max")).unwrap().trim(), LIMIT.to_string(), "inside the scope");
    fs::set_permissions(cgroup.join("memory.pressure"), fs::Permissions::from_mode(0o444)).unwrap();
    let denied = OpenOptions::new().write(true).open(cgroup.join("memory.pressure"));
    assert_eq!(denied.unwrap_err().raw_os_error(), Some(libc::EACCES));
    cgroup
}

fn event(cgroup: &Path, name: &str) -> u64 {
    let text = fs::read_to_string(cgroup.join("memory.events")).unwrap();
    text.lines().find_map(|l| l.strip_prefix(name)?.trim().parse().ok()).unwrap()
}

/// A sealed layout of `bytes`, filled here (charged to this scope).
fn layout(bytes: u64) -> File {
    let layout = File::from(platform::shm::memfd_named("layout").unwrap());
    let chunk = vec![1u8; MIB as usize];
    for _ in 0..bytes / MIB {
        (&layout).write_all(&chunk).unwrap();
    }
    platform::shm::seal(std::os::fd::AsFd::as_fd(&layout)).unwrap();
    layout
}

/// A clean page cache of `bytes`, as a downloaded checkpoint leaves: a file written past the
/// cache and read back. (Dirty pages are no room: the kernel cannot give them back in time.)
fn cache(dir: &Path, bytes: u64) {
    let path = dir.join("cache");
    let written = Command::new("dd").args(["if=/dev/zero", &format!("of={}", path.display()), "bs=1M", "oflag=direct"])
        .arg(format!("count={}", bytes / MIB)).stderr(Stdio::null()).status().unwrap();
    assert!(written.success());
    assert!(Command::new("cat").arg(&path).stdout(Stdio::null()).status().unwrap().success());
}

/// An executor: it maps `mapped` (every page resident), then grows its anonymous memory by the
/// bytes each line names, 16 MiB each 10 ms, as a load and a denoise do, with a progress frame
/// for each.
struct Executor(Child, ChildStdin, BufReader<std::process::ChildStdout>);

impl Executor {
    fn spawn(mapped: Option<&File>) -> Self {
        let path = mapped.map_or(String::new(), |f| format!("/proc/{}/fd/{}", std::process::id(), f.as_raw_fd()));
        let script = format!(
            "import mmap, sys, time\n\
             if '{path}':\n    f = open('{path}', 'rb'); m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ); s = sum(m[i] for i in range(0, len(m), 4096))\n\
             held = []\n\
             print('ready', flush=True)\n\
             for line in sys.stdin:\n    for _ in range(int(line) // {chunk}):\n        b = bytearray({chunk}); b[::4096] = b'\\1' * len(b[::4096]); held.append(b); print('step', flush=True); time.sleep(0.01)\n    print('grown', flush=True)\n",
            chunk = 16 * MIB
        );
        let mut child = Command::new("python3").args(["-c", &script]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let (stdin, mut stdout) = (child.stdin.take().unwrap(), BufReader::new(child.stdout.take().unwrap()));
        let mut line = String::new();
        stdout.read_line(&mut line).unwrap();
        assert_eq!(line.trim(), "ready");
        Self(child, stdin, stdout)
    }

    /// Whether it grew by `bytes` (false: it was killed). Each progress frame nudges the tier,
    /// as TensorD's do (`Callbacks::progress`).
    fn grow(&mut self, bytes: u64, tier: &HostTier) -> bool {
        if writeln!(self.1, "{bytes}").is_err() {
            return false;
        }
        let mut line = String::new();
        while { line.clear(); self.2.read_line(&mut line).is_ok_and(|n| n > 0) } {
            match line.trim() {
                "step" => tier.nudge(),
                done => return done == "grown",
            }
        }
        false
    }

    fn birth(&self) -> platform::process::Birth {
        platform::process::Birth::of(self.0.id()).unwrap()
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// The tier with its watcher, if `watch`, giving back one idle layout per rung; the count given.
fn tier(dir: &Path, watch: bool) -> (Arc<HostTier>, Arc<AtomicUsize>) {
    let tier = Arc::new(HostTier::new(dir, HostMode::Dedicated, MemoryLog::at(dir.into())));
    let given = Arc::new(AtomicUsize::new(0));
    if watch {
        let (weak, count) = (Arc::downgrade(&tier), given.clone());
        tier.watch(move || {
            let gave = weak.upgrade()?.deposits.release_one();
            count.fetch_add(gave as usize, Ordering::AcqRel);
            Some(gave)
        });
    }
    (tier, given)
}

fn scenario(watch: bool) {
    let cgroup = deny_psi();
    let dir = tempfile::tempdir().unwrap();
    let (tier, given) = tier(dir.path(), watch);
    for n in 0..LAYOUTS {
        tier.deposits.deposit("sha256:m", &format!("layout-{n}"), layout(LAYOUT), &[]).unwrap();
    }
    cache(dir.path(), CACHE);
    let killed = event(&cgroup, "oom_kill");
    let mut executor = Executor::spawn(None);
    let call = tier.begin("run", "plan", vec![executor.0.id()]);
    let grew = executor.grow(CALL, &tier);
    drop(call);
    let status = (!grew).then(|| executor.0.wait().unwrap());
    let log = fs::read_to_string(dir.path().join("memory.log")).unwrap_or_default();
    let (oom_kill, max) = (event(&cgroup, "oom_kill"), event(&cgroup, "max"));
    eprintln!("watch={watch}: call grew {grew} ({status:?}), oom_kill {killed} -> {oom_kill}, max {max}, layouts given {}\n{log}", given.load(Ordering::Acquire));
    if watch {
        assert!(log.contains("\"no psi trigger") && log.contains("inotify"), "{log}");
        assert!(grew && oom_kill == killed, "the call completes with no OOM kill");
        assert!(given.load(Ordering::Acquire) >= 1 && given.load(Ordering::Acquire) <= LAYOUTS);
    } else {
        assert_eq!(status.and_then(|s| s.signal()), Some(libc::SIGKILL), "nothing gives back: the kernel kills the call");
        assert_eq!(oom_kill, killed + 1);
    }
}

#[test]
fn layouts_yield_to_a_growing_call_where_psi_is_denied() {
    for name in ["inside_with_nothing_watching", "inside_with_the_watcher", "inside_a_mapped_layout_is_counted_once"] {
        assert!(scope(name).is_none_or(|status| status.success()), "{name}");
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

/// A warm executor maps a 256 MiB layout. Its measured call grew 128 MiB, so the same call again
/// needs only what it has still to grow: a 128 MiB layout seals beside it (the 0.11.2 need, its
/// resident high-water less its anonymous bytes, held the mapped layout as need and refused).
/// When the call then grows past its measure, the idle layout goes back, the mapped one stays
/// and the executor lives.
#[test]
#[ignore = "run inside a memory-limited scope by the test above"]
fn inside_a_mapped_layout_is_counted_once() {
    let cgroup = deny_psi();
    let dir = tempfile::tempdir().unwrap();
    let (tier, given) = tier(dir.path(), true);
    let mapped = layout(2 * LAYOUT);
    let mut executor = Executor::spawn(Some(&mapped));
    tier.deposits.deposit("m", "mapped", mapped, &[executor.birth()]).unwrap();
    let killed = event(&cgroup, "oom_kill");
    {
        let call = tier.begin("run-1", "plan", vec![executor.0.id()]);
        call.shape("s");
        assert!(executor.grow(LAYOUT, &tier));
        call.measured();
    }
    let call = tier.begin("run-2", "plan", vec![executor.0.id()]);
    call.shape("s");
    let admission = tier.deposits.seal("m", "idle", "run-2#0", LAYOUT, false, |reserved| {
        let admission = tier.admit(LAYOUT, reserved);
        eprintln!("{admission:?}");
        assert!(admission.need < 32 * MIB, "only what the call has still to grow: {admission:?}");
        admission.seal
    });
    assert_eq!(admission, Ok(true), "a layout seals beside a mapped one");
    tier.deposits.deposit("m", "idle", layout(LAYOUT), &[]).unwrap();
    assert!(executor.grow(2 * LAYOUT, &tier), "the call grows past its measure and lives");
    let log = fs::read_to_string(dir.path().join("memory.log")).unwrap_or_default();
    eprintln!("oom_kill {killed} -> {}, max {}, layouts given {}\n{log}", event(&cgroup, "oom_kill"), event(&cgroup, "max"), given.load(Ordering::Acquire));
    assert_eq!(event(&cgroup, "oom_kill"), killed);
    assert_eq!(given.load(Ordering::Acquire), 1, "the idle layout went back");
    assert!(tier.deposits.holds("mapped") && !tier.deposits.holds("idle"));
    drop(call);
}

/// Where PSI is denied the watcher arms on `memory.events`, and the kernel's file-modified event
/// wakes it as the scope reaches its limit.
#[test]
fn memory_events_wake_the_watcher_where_psi_is_denied() {
    assert!(scope("inside_memory_events_wake").is_none_or(|status| status.success()));
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
