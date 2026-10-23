//! The host rule under a real memory limit, on the pod's terms: a memory-limited scope whose PSI
//! trigger write is denied (EACCES, as RunPod denies it), a page cache filling it, an idle
//! neighbour holding its memory, and an executor's call growing its anonymous memory,
//! unmeasured, past what is left. With nothing watching the kernel OOM-kills one of them; with
//! TensorD's watcher the host reads short, the neighbour ends (the pool's part, played here),
//! and the call completes with no OOM kill. Page cache is room, never need.
#![cfg(target_os = "linux")]
use platform::contain::host_pressure::{HostMode, Pressure};
use std::{
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    os::unix::{fs::PermissionsExt, process::ExitStatusExt},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::Arc,
};
use tensord::memory::{Call, Gpu, Host, Leases, MemoryLog, Peaks, Peers, Running, Tenants};
use varena::lease::Peak;
use varena::{GpuId, HostBackend};

const MIB: u64 = 1 << 20;
/// The scope: what it holds is sized against this.
const LIMIT: u64 = 768 * MIB;
/// The idle neighbour, a page cache read, and the call's growth: together past the limit.
const NEIGHBOUR: u64 = 384 * MIB;
const CACHE: u64 = 256 * MIB;
const CALL: u64 = 448 * MIB;
/// A call grows by a load's step, then by a denoise's.
const LOAD: u64 = 32 * MIB;
const STEP: u64 = 16 * MIB;

/// Runs test `name` inside a memory-limited scope with no swap. A session that cannot create
/// one fails the test: it proves nothing without one.
fn scope(name: &str) -> std::process::ExitStatus {
    let limit = format!("MemoryMax={LIMIT}");
    // The kernel's OOM kill ends one process; systemd would otherwise stop the whole scope.
    let args = ["--user", "--scope", "--quiet", "--collect", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue", "-p", &limit];
    let probe = Command::new("systemd-run").args(args).arg("true").status();
    assert!(probe.as_ref().is_ok_and(|s| s.success()), "this session cannot create a memory-limited scope ({probe:?}): `systemd-run {}` must work", args.join(" "));
    let exe = std::env::current_exe().unwrap();
    let test = [name, "--exact", "--ignored", "--nocapture", "--test-threads=1"];
    Command::new("systemd-run").args(args).arg(exe).args(test).status().unwrap()
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

/// A clean page cache of `bytes`, as a downloaded checkpoint leaves: a file written past the
/// cache and read back. (Dirty pages are no room: the kernel cannot give them back in time.)
fn cache(dir: &Path, bytes: u64) {
    let path = dir.join("cache");
    let written = Command::new("dd").args(["if=/dev/zero", &format!("of={}", path.display()), "bs=1M", "oflag=direct"])
        .arg(format!("count={}", bytes / MIB)).stderr(Stdio::null()).status().unwrap();
    assert!(written.success());
    assert!(Command::new("cat").arg(&path).stdout(Stdio::null()).status().unwrap().success());
}

/// This binary as a child playing `role`, one command per line.
struct Agent {
    child: Child,
    to: ChildStdin,
    from: BufReader<ChildStdout>,
}

impl Agent {
    fn spawn(role: &str) -> Agent {
        let exe = std::env::current_exe().unwrap();
        let mut child = Command::new(exe).args([role, "--exact", "--ignored", "--nocapture", "--test-threads=1"]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let (to, from) = (child.stdin.take().unwrap(), BufReader::new(child.stdout.take().unwrap()));
        Agent { child, to, from }
    }

    /// Its answer; `None`: it is gone (killed).
    fn ask(&mut self, command: &str) -> Option<String> {
        writeln!(self.to, "{command}").ok()?;
        self.to.flush().ok()?;
        let mut line = String::new();
        loop {
            line.clear();
            if self.from.read_line(&mut line).ok()? == 0 {
                return None;
            }
            // libtest's own `test <role> ... ` may lead the first answer's line.
            if let Some((_, answer)) = line.split_once("child> ") {
                return Some(answer.trim().to_string());
            }
        }
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A child: `hold <bytes>` touches that much anonymous memory and keeps it (an idle executor, or
/// a call growing).
#[test]
#[ignore = "a child process, spawned by the scenarios"]
fn child() {
    let mut held = Vec::<Vec<u8>>::new();
    for line in std::io::stdin().lines() {
        let line = line.unwrap();
        let words: Vec<&str> = line.split_whitespace().collect();
        let answer = match words.as_slice() {
            ["hold", bytes] | ["grow", bytes] => {
                held.push(vec![1u8; bytes.parse().unwrap()]);
                "held".to_string()
            }
            other => panic!("no command {other:?}"),
        };
        println!("child> {answer}");
    }
}

/// The test's GPU: a manager over host pages, its executors this test's children.
fn gpu(namespace: &str, tenants: &Tenants) -> (Arc<Gpu>, GpuId) {
    assert!(varena::os::namespace(namespace));
    let id = GpuId { uuid: format!("host-pressure-{}", std::process::id()), model: "host".into(), memory: 4 * LIMIT };
    let inventory: tensord::machine::receipt::Gpu = serde_json::from_value(serde_json::json!({"device_index": 0, "device_name": id.model,
        "device_uuid": id.uuid, "driver_version": "", "memory_bytes": id.memory, "pci_bus_id": ""})).unwrap();
    let hosting = Arc::new(Peers::new(tenants.clone(), vec![platform::memory::euid()], MemoryLog::default()));
    let card = id.memory;
    let leases = Leases { hosting, open: Some(Box::new(move || Ok(Arc::new(HostBackend::new(card)) as Arc<dyn varena::Backend>))), tenants: tenants.clone() };
    let peaks = Peaks::at(std::env::temp_dir().join(format!("host-pressure-peaks-{}", std::process::id())));
    let gpu = Gpu::create("0", &[inventory], None, HostMode::Dedicated, leases, &peaks, &MemoryLog::default()).expect("a manager");
    (Arc::new(gpu), id)
}

/// `call` runs request `run` of `pkg|s` on the GPU, as TensorD records an admitted call.
fn running(call: &Agent, running: &Running) {
    running.lock().unwrap().insert("run".into(), Call { pid: call.child.id(), package: "pkg".into(), shape: "s".into() });
}

/// An unmeasured call grows past what is left beside an idle neighbour.
fn scenario(watch: bool) {
    let cgroup = deny_psi();
    let (namespace, tenants, calls) = (format!("host-pressure-{}-", std::process::id()), Tenants::default(), Running::default());
    let (gpu, _) = gpu(&namespace, &tenants);
    let dir = tempfile::tempdir().unwrap();
    let host = watch.then(|| {
        let how = Pressure::arm(HostMode::Dedicated).unwrap().how().to_string();
        assert!(how.contains("no psi trigger") && how.contains("inotify"), "{how}");
        Host::watch(HostMode::Dedicated, vec![gpu.clone()], calls.clone(), &MemoryLog::default()).expect("watched")
    });
    let mut neighbour = Agent::spawn("child");
    assert_eq!(neighbour.ask(&format!("hold {NEIGHBOUR}")).as_deref(), Some("held"));
    cache(dir.path(), CACHE);
    let killed = event(&cgroup, "oom_kill");
    let mut call = Agent::spawn("child");
    running(&call, &calls);
    // The pool's part: on a short look, its one idle process ends.
    let mut ended = false;
    let mut look = |neighbour: &mut Agent| {
        if let Some(host) = &host {
            host.look();
            if host.short() && !ended {
                let _ = neighbour.child.kill();
                let _ = neighbour.child.wait();
                ended = true;
            }
        }
    };
    look(&mut neighbour);
    let mut grew = true;
    for step in std::iter::once(LOAD).chain(std::iter::repeat_n(STEP, ((CALL - LOAD) / STEP) as usize)) {
        if call.ask(&format!("grow {step}")).is_none() {
            grew = false;
            break;
        }
        // TensorD looks at each of an executor's frames.
        look(&mut neighbour);
    }
    let status = (!grew).then(|| call.child.wait().unwrap());
    let oom_kill = event(&cgroup, "oom_kill");
    eprintln!("watch={watch}: call grew {grew} ({status:?}), neighbour ended {ended}, oom_kill {killed} -> {oom_kill}");
    if watch {
        assert!(grew && oom_kill == killed, "the call completes with no OOM kill");
        assert!(ended, "the host read short and the idle neighbour ended");
    } else {
        // Nothing ends the neighbour: the kernel OOM-kills (the larger then, or both).
        let neighbour_killed = neighbour.child.try_wait().unwrap().and_then(|s| s.signal()) == Some(libc::SIGKILL);
        let call_killed = status.and_then(|s| s.signal()) == Some(libc::SIGKILL);
        assert!(oom_kill > killed && (neighbour_killed || call_killed), "OOM kills: call {call_killed}, neighbour {neighbour_killed}");
    }
    drop((call, neighbour, gpu));
}

/// A call measured to outgrow the host alone (its recorded peak is twice the limit), beside small
/// idle neighbours: ending them cannot help. Every look is short; the first ends one, the next
/// find the budget no higher than when that end was signalled, so none more ends, and the call
/// meets its own limit.
fn futile() {
    let cgroup = deny_psi();
    let (namespace, tenants, calls) = (format!("host-pressure-{}-", std::process::id()), Tenants::default(), Running::default());
    let (gpu, _) = gpu(&namespace, &tenants);
    let host = Host::watch(HostMode::Dedicated, vec![gpu.clone()], calls.clone(), &MemoryLog::default()).expect("watched");
    gpu.manager.restore(&[Peak::Host { key: "pkg|s".into(), bytes: 2 * LIMIT }]).unwrap();
    let mut neighbours: Vec<Agent> = (0..3).map(|_| Agent::spawn("child")).collect();
    for n in &mut neighbours {
        assert_eq!(n.ask(&format!("hold {MIB}")).as_deref(), Some("held"));
    }
    let killed = event(&cgroup, "oom_kill");
    let call = Agent::spawn("child");
    running(&call, &calls);
    let mut ended = 0;
    let mut grew = true;
    // Steps larger than a neighbour gives back: an end cannot raise the budget past the growth.
    for _ in 0..(2 * LIMIT / STEP) {
        host.look();
        if host.short() && ended < neighbours.len() {
            let n = &mut neighbours[ended];
            let _ = n.child.kill();
            let _ = n.child.wait();
            ended += 1;
        }
        if call.ask(&format!("grow {STEP}")).is_none() {
            grew = false;
            break;
        }
    }
    let alive = neighbours.iter_mut().skip(ended).filter_map(|n| n.child.try_wait().unwrap().is_none().then_some(())).count();
    let oom_kill = event(&cgroup, "oom_kill");
    eprintln!("futile: call grew {grew}, neighbours ended {ended}, alive {alive}, oom_kill {killed} -> {oom_kill}");
    assert!(!grew && oom_kill > killed, "the call met its own limit");
    assert!(ended <= 1 && alive == neighbours.len() - ended, "ends stop once they cannot help: {ended} ended, {alive} alive");
    drop((call, neighbours, gpu));
}

/// Needs a user systemd session (a hosted runner has none): run with `--ignored` where one is.
#[test]
#[ignore = "needs a user systemd session for its memory-limited scopes"]
fn an_unmeasured_calls_growth_ends_an_idle_neighbour_with_no_oom_kill() {
    for name in ["inside_with_nothing_watching", "inside_with_the_watcher", "inside_where_ending_cannot_help"] {
        assert!(scope(name).success(), "{name}");
    }
}

#[test]
#[ignore = "run inside a memory-limited scope by the test above"]
fn inside_where_ending_cannot_help() {
    futile();
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
