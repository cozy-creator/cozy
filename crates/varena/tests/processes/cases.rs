//! The real Memory over host pages, across real processes. The parent plays TensorD (the lease
//! manager) where a case leases; each child is this binary as an agent
//! that runs one command per stdin line over its own `Memory` and answers on stdout.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Arc;

use tensorfs::format::ObjectRef;
use tensorfs::io::source::Source;
use tensorfs::io::{ItemSource, ObjectRange};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Fill, Fingerprint, GpuId, HostBackend, Level, Memory, Pass, Pin, Plan, Request, SetId, SetSpec, Step};

const MB: u64 = 1 << 20;


/// The tables the running case made: removed when it ends, passed or failed.
static TABLES: parking_lot::Mutex<Vec<String>> = parking_lot::Mutex::new(Vec::new());

/// The namespace these processes' tables and services live in, apart from any other on the
/// machine: the parent's pid names it, and its children are told it.
fn prefix(parent: u32) -> String {
    format!("proc{parent}-")
}

pub fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [role, prefix] = args.as_slice() {
        if role == "child" {
            assert!(varena::os::namespace(prefix));
            return agent();
        }
    }
    assert!(varena::os::namespace(&prefix(std::process::id())));
    let filter: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    type Case = (&'static str, fn(&Path));
    let mut cases: Vec<Case> = vec![
        ("a_failed_request_frees_everything", a_failed_request_frees_everything),
        ("exhausted_carries_stats", exhausted_carries_stats),
        ("a_joint_step_runs_a_pass_per_set", a_joint_step_runs_a_pass_per_set),
    ];
    // The kernel's list of sockets is Linux's.
    #[cfg(target_os = "linux")]
    cases.extend([
        ("weights_by_lease_return_on_kill_and_are_adopted", weights_by_lease_return_on_kill_and_are_adopted as fn(&Path)),
        ("a_refill_in_place_is_never_adopted_under_its_old_identity", a_refill_in_place_is_never_adopted_under_its_old_identity),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let mut failed = vec![];
    for (name, case) in cases {
        if !filter.is_empty() && !filter.iter().any(|f| name.contains(f.as_str())) {
            continue;
        }
        let path = dir.path().join(name);
        std::fs::create_dir(&path).unwrap();
        print!("test {name} ... ");
        std::io::stdout().flush().unwrap();
        let outcome = std::panic::catch_unwind(|| case(&path));
        TABLES.lock().drain(..).for_each(|t| varena::os::remove_table(&t));
        match outcome {
            Ok(()) => println!("ok"),
            Err(_) => {
                println!("FAILED");
                failed.push(name);
            }
        }
    }
    if !failed.is_empty() {
        eprintln!("failed: {failed:?}");
        std::process::exit(1);
    }
}

fn gpu(case: &str, memory: u64) -> GpuId {
    let uuid = format!("t{}-{case}", std::process::id());
    TABLES.lock().push(uuid.clone());
    GpuId { uuid, model: "host".into(), memory }
}

/// A byte of set `seed`'s file at `offset`.
fn pattern(seed: u64, offset: u64) -> u8 {
    ((offset / 4096 * 131 + seed * 17 + offset % 4096) % 251) as u8
}

fn seed(name: &str) -> u64 {
    name.bytes().map(u64::from).sum()
}

/// A stored file of `regions` regions of one tensor each.
fn write_set(dir: &Path, name: &str, regions: u64) -> PathBuf {
    let path = dir.join(format!("{name}.bin"));
    let s = seed(name);
    let bytes: Vec<u8> = (0..regions * REGION_ALIGN).map(|o| pattern(s, o)).collect();
    std::fs::write(&path, bytes).unwrap();
    path
}

fn spec(name: &str, path: &Path, fill: Fill, uuid: &str) -> SetSpec {
    let length = std::fs::metadata(path).unwrap().len();
    let regions = length / REGION_ALIGN;
    let obj = ObjectRef { sha256: format!("obj-{name}"), length };
    let layout = Layout {
        parts: (0..regions).map(|r| Part { what: format!("t{r}"), region: r as u32, offset: r * REGION_ALIGN, nbytes: REGION_ALIGN }).collect(),
        regions: (0..regions)
            .map(|r| Region {
                offset: r * REGION_ALIGN,
                nbytes: REGION_ALIGN,
                span: REGION_ALIGN,
                items: vec![Item { offset: r * REGION_ALIGN, len: REGION_ALIGN, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: r * REGION_ALIGN, len: REGION_ALIGN }), part_end: true }],
            })
            .collect(),
        nbytes: regions * REGION_ALIGN,
        digest: format!("layout-{name}-{uuid}"),
        names: (0..regions).map(|r| vec![format!("t{r}")]).collect(),
    };
    let source = Source::files(vec![(obj, File::open(path).unwrap())], false, 0);
    let fingerprint = Fingerprint::new(&layout, &[], fill, "release-1", "0/1", uuid);
    SetSpec { name: name.into(), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

// ---- the child

/// Runs commands over one Memory until stdin ends. Every command answers one line.
fn agent() {
    let mut mem: Option<Memory> = None;
    let mut uuid = String::new();
    let mut sets: HashMap<String, (SetId, String)> = HashMap::new();
    let mut requests: HashMap<String, Request> = HashMap::new();
    let mut passes: HashMap<String, Pass> = HashMap::new();
    let mut pins: Vec<Pin> = vec![];
    let mut yielded: HashMap<String, varena::Yielded> = HashMap::new();
    let mut out = std::io::stdout();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let w: Vec<&str> = line.split_whitespace().collect();
        let reply: String = match w.as_slice() {
            ["open", id, memory] => {
                uuid = id.to_string();
                let memory: u64 = memory.parse().unwrap();
                let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory };
                mem = Some(Memory::open(g, Box::new(HostBackend::new(memory))).unwrap());
                format!("ok {}", std::process::id())
            }
            ["unload", set] => {
                let m = mem.as_ref().unwrap();
                m.unload(sets[*set].0);
                m.settle();
                "ok".into()
            }
            ["refill", set, fill] => {
                let m = mem.as_ref().unwrap();
                m.fill(sets[*set].0, Fill::Declared(fill.to_string()));
                m.settle();
                "ok".into()
            }
            ["load", name, path, fill, level] => {
                let fill = match *fill {
                    "stored" => Fill::Stored,
                    "undeclared" => Fill::Undeclared,
                    other => Fill::Declared(other.into()),
                };
                let level = if *level == "gpu" { Level::Gpu } else { Level::Ram };
                let m = mem.as_ref().unwrap();
                match m.load(spec(name, Path::new(path), fill, &uuid), level) {
                    Ok(id) => {
                        m.settle();
                        sets.insert(name.to_string(), (id, path.to_string()));
                        "ok".into()
                    }
                    Err(e) => format!("err {}", e.code()),
                }
            }
            ["request", id, plan] => {
                let steps = plan.split(';').map(|s| Step { sets: s.split(',').map(String::from).collect(), passes: 1 }).collect();
                match mem.as_ref().unwrap().request(id, Plan { steps }) {
                    Ok(r) => {
                        requests.insert(id.to_string(), r);
                        "ok".into()
                    }
                    Err(e) => format!("err {}", e.code()),
                }
            }
            ["pass", req, set, shape] => match requests[*req].pass(sets[*set].0, shape) {
                Ok(p) => {
                    passes.insert(req.to_string(), p);
                    "ok".into()
                }
                Err(e) => format!("err {}", e.code()),
            },
            ["passas", req, set, shape, name] => match requests[*req].pass(sets[*set].0, shape) {
                Ok(p) => {
                    passes.insert(name.to_string(), p);
                    "ok".into()
                }
                Err(e) => format!("err {}", e.code()),
            },
            ["endpassas", name, peak] => {
                if let Some(p) = passes.remove(*name) {
                    p.end(peak.parse().unwrap());
                }
                "ok".into()
            }
            ["endpass", req] => {
                if let Some(p) = passes.remove(*req) {
                    p.end(0);
                }
                "ok".into()
            }
            ["endpass", req, peak] => {
                if let Some(p) = passes.remove(*req) {
                    p.end(peak.parse().unwrap());
                }
                "ok".into()
            }
            ["bind", set, block] => match mem.as_ref().unwrap().bind(sets[*set].0, block.parse().unwrap(), 0) {
                Ok(pin) => {
                    let fresh = pin.fresh;
                    pins.push(pin);
                    format!("ok {fresh}")
                }
                Err(varena::Error::Exhausted(e)) => format!("exhausted {} {}", e.stats.request, e.need),
                Err(e) => format!("err {}", e.code()),
            },
            ["filled"] => {
                pins.iter().filter(|p| p.fresh).for_each(Pin::filled);
                "ok".into()
            }
            ["exact", set, block] => {
                let s = seed(set);
                let b: u64 = block.parse().unwrap();
                let pin = mem.as_ref().unwrap().bind(sets[*set].0, b as u32, 0).unwrap();
                // SAFETY: a bound host page.
                let bytes = unsafe { std::slice::from_raw_parts(pin.ptr as *const u8, pin.len as usize) };
                let ok = bytes.iter().enumerate().all(|(i, v)| *v == pattern(s, b * REGION_ALIGN + i as u64));
                format!("ok {ok}")
            }
            ["unpin"] => {
                pins.clear();
                "ok".into()
            }
            ["range", set, blocks] => {
                let m = mem.as_ref().unwrap();
                let blocks: Vec<u32> = (0..blocks.parse().unwrap()).collect();
                match m.retain(sets[*set].0, &blocks).and_then(|r| r.prepare(0)) {
                    Ok(range) => {
                        pins.extend(range);
                        "ok".into()
                    }
                    Err(varena::Error::Exhausted(e)) => format!("exhausted {} {}", e.stats.request, e.need),
                    Err(e) => format!("err {}", e.code()),
                }
            }
            ["yield", req] => {
                yielded.insert(req.to_string(), requests[*req].yielding());
                "ok".into()
            }
            ["unyield", req] => match yielded.remove(*req).map(varena::Yielded::end) {
                Some(Err(e)) => format!("err {}", e.code()),
                _ => "ok".into(),
            },
            ["trace", req] => {
                let s = requests[*req].stats();
                let origins: Vec<String> = s.trace.iter().map(|r| format!("{:?}", r.origin)).collect();
                format!("ok {}", origins.join(","))
            }
            ["end", req] => {
                passes.remove(*req);
                let s = requests.remove(*req).unwrap().end();
                format!("ok leftover={} kept={} store={} evicted={:?} spec={} leased={}", s.leftover, s.moved.kept.bytes, s.moved.store.bytes, s.evicted, s.moved.speculative, s.moved.leased.bytes)
            }
            ["mapped"] => format!("ok {}", mem.as_ref().unwrap().classes().iter().sum::<u64>()),
            ["settle"] => {
                mem.as_ref().unwrap().settle();
                "ok".into()
            }
            #[cfg(target_os = "linux")]
            ["lease", socket] => {
                let stream = std::os::unix::net::UnixStream::connect(socket).unwrap();
                match mem.as_ref().unwrap().lease_over(stream).unwrap() {
                    None => "ok".into(),
                    Some((code, why)) => format!("refused {code:?} {why}"),
                }
            }
            ["exit"] => {
                // Its memory retires before the process ends (the row goes, TensorD wakes).
                drop((std::mem::take(&mut yielded), std::mem::take(&mut pins), std::mem::take(&mut passes), std::mem::take(&mut requests)));
                if let Some(m) = mem.take() {
                    m.retire().unwrap();
                }
                writeln!(out, "ok").unwrap();
                return;
            }
            _ => format!("err unknown {line}"),
        };
        writeln!(out, "{reply}").unwrap();
        out.flush().unwrap();
    }
}

// ---- the parent

struct Agent {
    child: Child,
    to: ChildStdin,
    from: BufReader<ChildStdout>,
}

impl Agent {
    fn spawn() -> Agent {
        Agent::start(Command::new(std::env::current_exe().unwrap()))
    }

    fn start(mut command: Command) -> Agent {
        let mut child = command.arg("child").arg(prefix(std::process::id())).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let to = child.stdin.take().unwrap();
        let from = BufReader::new(child.stdout.take().unwrap());
        Agent { child, to, from }
    }

    fn send(&mut self, line: &str) {
        writeln!(self.to, "{line}").unwrap();
        self.to.flush().unwrap();
    }

    fn reply(&mut self) -> String {
        let mut reply = String::new();
        self.from.read_line(&mut reply).unwrap();
        reply.trim().to_string()
    }

    fn raw(&mut self, line: &str) -> String {
        self.send(line);
        self.reply()
    }

    fn say(&mut self, line: &str) -> String {
        let reply = self.raw(line);
        assert!(!reply.starts_with("err"), "{line} -> {reply}");
        reply
    }

    fn ok(&mut self, line: &str) -> String {
        let reply = self.say(line);
        assert!(reply.starts_with("ok"), "{line} -> {reply}");
        reply.trim_start_matches("ok").trim().to_string()
    }

    fn exit(mut self) {
        self.ok("exit");
        assert!(self.child.wait().unwrap().success());
    }
}

impl Drop for Agent {
    /// A case that failed leaves no child holding its rows.
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[cfg(target_os = "linux")]
struct Leases {
    gpu: GpuId,
    socket: PathBuf,
    manager: Arc<varena::lease::Manager>,
}

#[cfg(target_os = "linux")]
impl Leases {
    /// This process as TensorD: the GPU's manager over host pages, on a socket in `dir`.
    fn host(dir: &Path, case: &str, capacity: u64) -> Leases {
        struct Ours;
        impl varena::lease::Hosting for Ours {
            fn identify(&self, peer: &platform::handles::Peer) -> Option<varena::lease::Identity> {
                Some(varena::lease::Identity { executor: peer.pid.to_string(), domain: "test".into() })
            }
        }
        let gpu = gpu(case, capacity);
        let socket = dir.join("lease.sock");
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        let open: varena::lease::Opener = Box::new(move || Ok(Arc::new(HostBackend::new(capacity)) as Arc<dyn varena::Backend>));
        let manager = varena::lease::Manager::create(&gpu, None, varena::lease::Mode::Dedicated, Some(open), listener, Arc::new(Ours)).unwrap();
        Leases { gpu, socket, manager }
    }

    /// A child whose Memory leases its pages, `set` loaded on the GPU and a request `r` open
    /// with a pass over it.
    fn agent(&self, set: &str, path: &Path) -> Agent {
        let mut x = Agent::spawn();
        x.ok(&format!("open {} {}", self.gpu.uuid, self.gpu.memory));
        x.ok(&format!("lease {}", self.socket.display()));
        x.ok(&format!("load {set} {} stored gpu", path.display()));
        x.ok(&format!("request r {set}"));
        x.ok(&format!("pass r {set} 1x1"));
        x
    }

    fn until(&self, done: &dyn Fn(&varena::lease::Holdings) -> bool) -> varena::lease::Holdings {
        let start = std::time::Instant::now();
        loop {
            let h = self.manager.holdings();
            if done(&h) {
                return h;
            }
            assert!(start.elapsed() < std::time::Duration::from_secs(20), "the manager never reached it: {h:?}");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}

/// Slice 1's bar on CPU (varena-leases §8): two executors' Memories get their weights by lease
/// from a manager this process hosts; kill -9 one and its leases return once its exit shows, its
/// landed weights retained; its successor adopts them with no copy and no new chunk.
#[cfg(target_os = "linux")]
fn weights_by_lease_return_on_kill_and_are_adopted(dir: &Path) {
    let leases = Leases::host(dir, "lease", 64 * MB);
    let (k, j) = (write_set(dir, "k", 4), write_set(dir, "j", 2));
    let mut a = leases.agent("k", &k);
    for block in 0..4 {
        assert_eq!(a.ok(&format!("exact k {block}")), "true", "a's block {block} bit-exact");
    }
    a.ok("unpin");
    assert!(a.ok("end r").contains(&format!("leased={}", 8 * MB)), "a's weights came by lease");
    let mut b = leases.agent("j", &j);
    for block in 0..2 {
        assert_eq!(b.ok(&format!("exact j {block}")), "true", "b's block {block} bit-exact");
    }
    b.ok("unpin");
    b.ok("end r");
    let a_pid = a.child.id();
    let h = leases.until(&|h| h.executors.len() == 2 && h.executors.iter().any(|e| e.pid == a_pid && e.landed == 8 * MB));
    assert_eq!((h.allocated, h.created), (12 * MB, 12 * MB), "{h:?}");
    a.child.kill().unwrap();
    a.child.wait().unwrap();
    let h = leases.until(&|h| h.executors.len() == 1);
    assert_eq!((h.page_cache, h.allocated, h.created), (8 * MB, 4 * MB, 12 * MB), "a's landed weights retained: {h:?}");
    let mut c = leases.agent("k", &k);
    for block in 0..4 {
        assert_eq!(c.ok(&format!("exact k {block}")), "true", "c's adopted block {block} bit-exact");
    }
    assert_eq!(c.ok("trace r"), "Kept,Kept,Kept,Kept,Hit,Hit,Hit,Hit", "adopted at load, not copied");
    c.ok("unpin");
    let end = c.ok("end r");
    assert!(end.contains("store=0"), "{end}");
    let h = leases.until(&|h| h.executors.len() == 2);
    assert_eq!((h.page_cache, h.allocated, h.created), (0, 12 * MB, 12 * MB), "adopted with no new chunk: {h:?}");
    b.exit();
    c.exit();
}

/// A grant's identity never changes (varena-leases review M-1): a set refilled in place under
/// another fill identity takes new grants for its pages, so an executor killed after it leaves
/// nothing an adopter of the first identity could take for its own.
#[cfg(target_os = "linux")]
fn a_refill_in_place_is_never_adopted_under_its_old_identity(dir: &Path) {
    let leases = Leases::host(dir, "refill", 64 * MB);
    let k = write_set(dir, "k", 4);
    let mut a = leases.agent("k", &k);
    for block in 0..4 {
        assert_eq!(a.ok(&format!("exact k {block}")), "true");
    }
    a.ok("unpin");
    a.ok("end r");
    // The same set under another fill identity: its pages refill, each under a new grant.
    a.ok(&format!("load k {} lora-x gpu", k.display()));
    a.ok("request s k");
    a.ok("pass s k 1x1");
    assert_eq!(a.ok("bind k 0"), "true", "fresh: the caller bakes it");
    a.ok("filled");
    a.ok("unpin");
    a.ok("end s");
    a.child.kill().unwrap();
    a.child.wait().unwrap();
    leases.until(&|h| h.executors.is_empty());
    let mut c = leases.agent("k", &k);
    for block in 0..4 {
        assert_eq!(c.ok(&format!("exact k {block}")), "true", "block {block} is the stored weights, never the baked ones");
    }
    let trace = c.ok("trace r");
    assert!(!trace.contains("Kept"), "adopted under its old identity: {trace}");
    c.ok("unpin");
    c.ok("end r");
    c.exit();
}

/// A request that fails mid-way (an exhausted allocation) frees every pin: leftover 0, and the
/// same process serves the next request.
fn a_failed_request_frees_everything(dir: &Path) {
    let g = gpu("fail", 8 * MB);
    let (path, more) = (write_set(dir, "f", 2), write_set(dir, "x", 3));
    let mut a = Agent::spawn();
    a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.ok(&format!("load f {} stored ram", path.display()));
    a.ok(&format!("load x {} stored ram", more.display()));
    a.ok("request r1 f");
    a.ok("pass r1 f 1x1");
    a.ok("bind f 0");
    a.ok("bind f 1");
    // With half the card pinned, a range of three pages is Exhausted at once.
    assert!(a.say("range x 3").starts_with("exhausted r1"));
    a.ok("unpin");
    assert!(a.ok("end r1").starts_with("leftover=0"));
    assert_eq!(a.ok("mapped").parse::<u64>().unwrap(), 4 * MB, "only f's pages remain mapped");
    a.ok("request r2 f");
    a.ok("pass r2 f 1x1");
    assert_eq!(a.ok("exact f 1"), "true");
    a.ok("unpin");
    assert!(a.ok("end r2").starts_with("leftover=0"));
    a.exit();
}

/// Exhaustion is the one memory error, and it carries the request's record.
fn exhausted_carries_stats(dir: &Path) {
    let g = gpu("exh", 4 * MB);
    let path = write_set(dir, "e", 4);
    let mut a = Agent::spawn();
    a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.ok(&format!("load e {} stored ram", path.display()));
    a.ok("request r e");
    a.ok("pass r e 1x1");
    a.ok("bind e 0");
    a.ok("bind e 1");
    assert_eq!(a.say("bind e 2"), format!("exhausted r {}", 2 * MB));
    a.ok("unpin");
    a.ok("end r");
    a.exit();
}

/// A joint step ({a, b}): each set runs its own pass; interleaved binds land in their own
/// passes.
fn a_joint_step_runs_a_pass_per_set(dir: &Path) {
    let g = gpu("joint", 32 * MB);
    let (ja, jb) = (write_set(dir, "ja", 2), write_set(dir, "jb", 2));
    let mut a = Agent::spawn();
    a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.ok(&format!("load ja {} stored ram", ja.display()));
    a.ok(&format!("load jb {} stored ram", jb.display()));
    a.ok("request r ja,jb");
    a.ok("passas r ja 1x1 ja");
    a.ok("passas r jb 1x1 jb");
    for (set, block) in [("ja", 0), ("jb", 0), ("ja", 1), ("jb", 1)] {
        assert_eq!(a.ok(&format!("exact {set} {block}")), "true");
    }
    let trace = a.ok("trace r");
    assert_eq!(trace.split(',').count(), 4, "every bind is the request's: {trace}");
    a.ok("unpin");
    a.ok(&format!("endpassas ja {}", 3 * MB));
    a.ok(&format!("endpassas jb {}", 5 * MB));
    a.ok("end r");
    a.exit();
}

