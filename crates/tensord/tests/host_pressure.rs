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
    io::{BufRead, BufReader},
    os::unix::{fs::PermissionsExt, process::ExitStatusExt},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
use tensorfs::{format::ObjectRef, io::{source::Source as FileSource, ItemSource, ObjectRange}};
use varena::{Fill, Fingerprint, GpuId, HostBackend, Level, Memory, SetSpec, Tier};
use varena::layout::{Item, Layout, Part, Region};

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

/// A real Varena RAM-tier layout filled through the normal Memory.load path. The source
/// is a sparse zero file (reclaimable cache); the tier's shared pages are charged to this scope.
fn deposit(memory: &Memory, dir: &Path, gpu: &str, n: usize, bytes: u64) {
    let path = dir.join(format!("source-{n}"));
    File::create(&path).unwrap().set_len(bytes).unwrap();
    let object = ObjectRef { sha256: format!("{n:064x}"), length: bytes };
    let source = FileSource::files(vec![(object.clone(), File::open(&path).unwrap())], false, 0);
    let layout = Layout {
        parts: vec![Part { what: format!("model/value{n}#data"), region: 0, offset: 0, nbytes: bytes }],
        regions: vec![Region { offset: 0, nbytes: bytes, span: bytes, items: vec![Item {
            offset: 0, len: bytes, source: ItemSource::Object(ObjectRange { obj: object, off: 0, len: bytes }), part_end: true,
        }] }],
        nbytes: bytes,
        digest: format!("host-pressure-{n}"),
        names: vec![vec![format!("value{n}")]],
    };
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "host-pressure", "0/1", gpu);
    memory.load(SetSpec { name: format!("layout-{n}"), layout: Arc::new(layout), source: Arc::new(source), fingerprint }, Level::Ram).unwrap();
    memory.settle();
}

/// A clean page cache of `bytes`, as a downloaded checkpoint leaves: a file written, synced and
/// read back. (Dirty pages are no room: the kernel cannot give them back in time.)
fn cache(dir: &Path, bytes: u64) {
    let path = dir.join("cache");
    // Bound dirty pages while crossing the limit. A final-only fsync can leave more than
    // the entire cgroup dirty and kill the watcher fixture before reclaimable cache exists.
    assert_eq!(bytes % (16 * MIB), 0);
    let written = Command::new("dd").args(["if=/dev/zero", &format!("of={}", path.display()), "bs=16M", "oflag=dsync"])
        .arg(format!("count={}", bytes / (16 * MIB))).stderr(Stdio::null()).status().unwrap();
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
    assert!(varena::os::namespace(&format!("host-pressure-{}-", std::process::id())));
    // A previous measured peak is already consumed. Observed growth must still make the
    // tier yield at a limit; saturating its deficit to zero would suppress this case.
    let need: varena::Need = Arc::new(|| Some(0));
    let pressure = if watch {
        let pressure = Pressure::arm(HostMode::Dedicated).unwrap();
        assert!(pressure.how().contains("no psi trigger") && pressure.how().contains("inotify"), "{}", pressure.how());
        varena::Pressure::Host(pressure)
    } else { varena::Pressure::Headroom };
    let tier = Tier::serve(pressure, need).unwrap().unwrap();
    let gpu = GpuId { uuid: format!("host-pressure-{}", std::process::id()), model: "host".into(), memory: 4 * LIMIT };
    let memory = Memory::open(gpu.clone(), None, Box::new(HostBackend::new(gpu.memory))).unwrap();
    for n in 0..LAYOUTS {
        deposit(&memory, dir.path(), &gpu.uuid, n, LAYOUT);
    }
    let resident = || tier.resident().iter().map(|(_, bytes)| *bytes).sum::<u64>();
    let before = resident();
    assert!(before >= LAYOUT * LAYOUTS as u64, "the normal RAM-tier path actually filled the layouts: {before}");
    cache(dir.path(), CACHE);
    let killed = event(&cgroup, "oom_kill");
    let status = call(CALL);
    let (oom_kill, max) = (event(&cgroup, "oom_kill"), event(&cgroup, "max"));
    let given = before.saturating_sub(resident());
    eprintln!("watch={watch}: call {status:?}, oom_kill {killed} -> {oom_kill}, max {max}, tier bytes returned {given}");
    if watch {
        assert!(status.success() && oom_kill == killed, "the call completes with no OOM kill");
        assert!(given >= LAYOUT, "the actual Varena tier returned no layout");
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
