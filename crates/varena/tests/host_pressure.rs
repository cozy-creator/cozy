//! The RAM tier under a real memory limit, on the pod's terms: a memory-limited scope whose PSI
//! trigger write is denied (EACCES, as RunPod denies it). Layouts in the tier, a page cache
//! filling the scope, and an executor's call growing its anonymous memory past what is left.
//! With nothing watching the kernel kills the call; with the watcher (the platform's, the one
//! TensorD arms) layouts go back by measured headroom and the call completes. A layout a warm
//! executor maps is counted once, in the budget, never as its call's need.
#![cfg(target_os = "linux")]
use platform::contain::host_pressure::{HostMode, Pressure};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    os::unix::{fs::PermissionsExt, process::ExitStatusExt},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
use tensorfs::{format::ObjectRef, io::{source::Source as FileSource, ItemSource, ObjectRange}};
use varena::layout::{Item, Layout, Part, Region};
use varena::{Fill, Fingerprint, GpuId, HostBackend, Ledger, Level, Measured, Memory, Plan, SetSpec, Step, Tier, Torch};

const MIB: u64 = 1 << 20;
/// The scope: what it holds is sized against this.
const LIMIT: u64 = 768 * MIB;
/// Layouts kept, a page cache read, and the call's growth: together past the limit.
const LAYOUT: u64 = 128 * MIB;
const LAYOUTS: usize = 3;
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

fn digest(n: usize) -> String {
    format!("host-pressure-{n}")
}

/// Set `layout-<n>`: one region of `bytes`, stored in a sparse zero file (reclaimable cache).
fn set(dir: &Path, gpu: &str, n: usize, bytes: u64) -> SetSpec {
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
        digest: digest(n),
        names: vec![vec![format!("value{n}")]],
    };
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "host-pressure", "0/1", gpu);
    SetSpec { name: format!("layout-{n}"), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
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

/// This test's GPU table and tier (on the scope's watcher, if `watch`), in a namespace of its own.
struct Machine {
    gpu: GpuId,
    ledger: Ledger,
    tier: Tier,
    dir: tempfile::TempDir,
}

impl Machine {
    fn open(watch: bool) -> Machine {
        let namespace = format!("host-pressure-{}-", std::process::id());
        assert!(varena::os::namespace(&namespace));
        let gpu = GpuId { uuid: format!("host-pressure-{}", std::process::id()), model: "host".into(), memory: 4 * LIMIT };
        let ledger = Ledger::create(&gpu, None).unwrap();
        let pressure = watch.then(|| {
            let pressure = Pressure::arm(HostMode::Dedicated).unwrap();
            assert!(pressure.how().contains("no psi trigger") && pressure.how().contains("inotify"), "{}", pressure.how());
            pressure
        });
        let tier = Tier::serve(pressure, varena::tier::need(&[&ledger])).unwrap().expect("this namespace's tier");
        Machine { gpu, ledger, tier, dir: tempfile::tempdir().unwrap() }
    }

    fn memory(&self) -> Memory {
        Memory::open(self.gpu.clone(), None, Box::new(HostBackend::new(self.gpu.memory))).unwrap()
    }

    fn resident(&self, n: usize) -> u64 {
        self.tier.resident().into_iter().find(|(d, _)| *d == digest(n)).map_or(0, |(_, bytes)| bytes)
    }

    /// An executor of this machine's GPU.
    fn executor(&self) -> Executor {
        let exe = std::env::current_exe().unwrap();
        let test = ["executor", "--exact", "--ignored", "--nocapture", "--test-threads=1"];
        let mut child = Command::new(exe).args(test).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let (to, from) = (child.stdin.take().unwrap(), BufReader::new(child.stdout.take().unwrap()));
        let mut executor = Executor { child, to, from };
        let namespace = format!("host-pressure-{}-", std::process::id());
        let open = format!("open {namespace} {} {} {}", self.gpu.uuid, self.gpu.memory, self.dir.path().display());
        assert_eq!(executor.ask(&open).as_deref(), Some("ready"));
        executor
    }
}

impl Drop for Machine {
    fn drop(&mut self) {
        varena::os::remove_table(&self.gpu.uuid);
        varena::tier::remove_claims();
    }
}

/// An executor process (this binary's `executor`) on its own varena `Memory`, one command per
/// line.
struct Executor {
    child: Child,
    to: ChildStdin,
    from: BufReader<ChildStdout>,
}

impl Executor {
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
            // libtest's own `test executor ... ` may lead the first answer's line.
            if let Some((_, answer)) = line.split_once("executor> ") {
                return Some(answer.trim().to_string());
            }
        }
    }

    /// Whether its call grew by `bytes` (false: it was killed), a load's step and then denoise
    /// steps, the tier looking after each as TensorD's frames have it.
    fn grow(&mut self, bytes: u64, tier: &Tier) -> bool {
        let steps = std::iter::once(LOAD).chain(std::iter::repeat_n(STEP, ((bytes - LOAD) / STEP) as usize));
        for step in steps {
            if self.ask(&format!("grow {step}")).is_none() {
                return false;
            }
            tier.look();
        }
        true
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
#[ignore = "an executor process, spawned by the scenarios"]
fn executor() {
    let mut memory = None;
    let (mut request, mut pass, mut held, mut spec, mut sets) = (None, None, Vec::<Vec<u8>>::new(), None, std::collections::HashMap::new());
    for line in std::io::stdin().lines() {
        let line = line.unwrap();
        let words: Vec<&str> = line.split_whitespace().collect();
        let answer = match words.as_slice() {
            ["open", namespace, uuid, bytes, dir] => {
                assert!(varena::os::namespace(namespace));
                let gpu = GpuId { uuid: uuid.to_string(), model: "host".into(), memory: bytes.parse().unwrap() };
                memory = Some(Memory::open(gpu.clone(), None, Box::new(HostBackend::new(gpu.memory))).unwrap());
                spec = Some((PathBuf::from(dir), gpu.uuid));
                "ready".to_string()
            }
            ["load", n, bytes, level] => {
                let (dir, uuid) = spec.as_ref().unwrap();
                let level = if *level == "gpu" { Level::Gpu } else { Level::Ram };
                let memory = memory.as_ref().unwrap();
                let spec = set(dir, uuid, n.parse().unwrap(), bytes.parse().unwrap());
                sets.insert(spec.name.clone(), memory.load(spec, level).unwrap());
                memory.settle();
                "loaded".to_string()
            }
            // A call: on `using`, its pass runs, so the tier hears its next use.
            ["request", id, using @ ..] => {
                let steps = using.iter().map(|s| Step { sets: vec![s.to_string()], passes: 1 }).collect();
                let begun = memory.as_ref().unwrap().request(id, Plan { steps }).unwrap();
                pass = using.first().map(|s| begun.pass(sets[*s], "fwd", "1x1", Torch::default()).unwrap());
                request = Some(begun);
                "begun".to_string()
            }
            ["grow", bytes] => {
                held.push(vec![1u8; bytes.parse().unwrap()]);
                "grown".to_string()
            }
            ["end"] => {
                if let Some(pass) = pass.take() {
                    pass.end(Measured::default(), Torch::default());
                }
                format!("ended {}", request.take().unwrap().end().host.map_or("none".into(), |h| h.to_string()))
            }
            other => panic!("no command {other:?}"),
        };
        println!("executor> {answer}");
    }
}

/// An unmeasured call grows past what is left: with nothing watching the kernel kills it; with
/// the watcher, what it is seen to grow stays free and layouts go back.
fn scenario(watch: bool) {
    let cgroup = deny_psi();
    let machine = Machine::open(watch);
    let memory = machine.memory();
    for n in 0..LAYOUTS {
        memory.load(set(machine.dir.path(), &machine.gpu.uuid, n, LAYOUT), Level::Ram).unwrap();
        memory.settle();
    }
    let before: u64 = (0..LAYOUTS).map(|n| machine.resident(n)).sum();
    assert!(before >= LAYOUT * LAYOUTS as u64, "the normal RAM-tier path filled the layouts: {before}");
    cache(machine.dir.path(), CACHE);
    let killed = event(&cgroup, "oom_kill");
    let mut executor = machine.executor();
    assert_eq!(executor.ask("request run").as_deref(), Some("begun"));
    machine.tier.look();
    let grew = executor.grow(CALL, &machine.tier);
    let status = (!grew).then(|| executor.child.wait().unwrap());
    let (oom_kill, max) = (event(&cgroup, "oom_kill"), event(&cgroup, "max"));
    let given = before.saturating_sub((0..LAYOUTS).map(|n| machine.resident(n)).sum());
    eprintln!("watch={watch}: call grew {grew} ({status:?}), oom_kill {killed} -> {oom_kill}, max {max}, tier bytes returned {given}");
    if watch {
        assert!(grew && oom_kill == killed, "the call completes with no OOM kill");
        assert!(given >= LAYOUT, "the tier returned no layout");
        assert!(executor.ask("end").is_some());
    } else {
        assert_eq!(status.and_then(|s| s.signal()), Some(libc::SIGKILL), "nothing gives back: the kernel kills the call");
        assert_eq!(oom_kill, killed + 1);
    }
}

#[test]
fn layouts_yield_to_a_growing_call_where_psi_is_denied() {
    for name in ["inside_with_nothing_watching", "inside_with_the_watcher", "inside_a_mapped_layout_is_counted_once"] {
        assert!(scope(name).success(), "{name}");
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

/// A warm executor maps a 128 MiB layout from the tier and holds its 128 MiB copy on its device
/// (host pages here). Its measured call grew 128 MiB, so the same call again needs only what it
/// has still to grow: a 128 MiB idle layout fills beside it (the old need, its resident
/// high-water less its anonymous bytes, held the mapped bytes as need and refused). When the
/// call then grows past its measure, the idle layout goes back, the mapped one stays and the
/// executor lives.
#[test]
#[ignore = "run inside a memory-limited scope by the test above"]
fn inside_a_mapped_layout_is_counted_once() {
    let cgroup = deny_psi();
    let machine = Machine::open(true);
    let mut executor = machine.executor();
    for level in ["ram", "gpu"] {
        assert_eq!(executor.ask(&format!("load 0 {LAYOUT} {level}")).as_deref(), Some("loaded"));
    }
    let killed = event(&cgroup, "oom_kill");
    machine.ledger.dispatch("run-1", "pkg", "s").unwrap();
    assert_eq!(executor.ask("request run-1 layout-0").as_deref(), Some("begun"));
    machine.tier.look();
    assert!(executor.grow(LAYOUT, &machine.tier));
    let ended = executor.ask("end").unwrap();
    let peak: u64 = ended.strip_prefix("ended ").and_then(|h| h.parse().ok()).unwrap_or_else(|| panic!("{ended}"));
    assert!((LAYOUT..2 * LAYOUT).contains(&peak), "its anonymous peak, no mapped bytes: {} MiB", peak / MIB);
    machine.ledger.dispatch("run-2", "pkg", "s").unwrap();
    assert_eq!(executor.ask("request run-2 layout-0").as_deref(), Some("begun"));
    machine.tier.await_uses(|uses| uses.contains(&(digest(0), Some(0))));
    machine.tier.look();
    let memory = machine.memory();
    memory.load(set(machine.dir.path(), &machine.gpu.uuid, 1, LAYOUT), Level::Ram).unwrap();
    memory.settle();
    assert!(machine.resident(1) >= LAYOUT, "a layout fills beside a mapped one");
    assert!(executor.grow(2 * LAYOUT, &machine.tier), "the call grows past its measure and lives");
    eprintln!("oom_kill {killed} -> {}, max {}, held {:?}", event(&cgroup, "oom_kill"), event(&cgroup, "max"), machine.tier.held());
    assert_eq!(event(&cgroup, "oom_kill"), killed);
    assert_eq!(machine.tier.held(), [digest(0)], "the idle layout went back, the mapped one stayed");
    assert!(executor.ask("end").is_some());
}

/// Where PSI is denied the watcher arms on `memory.events`, and the kernel's file-modified event
/// wakes it as the scope reaches its limit.
#[test]
fn memory_events_wake_the_watcher_where_psi_is_denied() {
    assert!(scope("inside_memory_events_wake").success());
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
