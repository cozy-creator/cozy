//! The real Memory, ledger, custody and RAM tier over host pages, across real processes. The
//! parent plays TensorD (the ledger, custody, the tier); each child is this binary as an agent
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
use varena::{Fill, Fingerprint, GpuId, HostBackend, Level, Measured, Memory, Next, Pass, Pin, Plan, Request, SetId, SetSpec, Step, Torch};

const MB: u64 = 1 << 20;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("child") {
        return agent();
    }
    let filter: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    type Case = (&'static str, fn(&Path));
    let cases: [Case; 6] = [
        ("an_idle_holder_answers_an_ask", an_idle_holder_answers_an_ask),
        ("a_killed_holder_leaves_orphans_a_successor_imports", a_killed_holder_leaves_orphans_a_successor_imports),
        ("a_failed_request_frees_everything", a_failed_request_frees_everything),
        ("a_panic_fails_the_request_not_the_process", a_panic_fails_the_request_not_the_process),
        ("exhausted_carries_stats", exhausted_carries_stats),
        ("the_tier_gives_back_in_order", the_tier_gives_back_in_order),
    ];
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
        match std::panic::catch_unwind(|| case(&path)) {
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
    GpuId { uuid: format!("t{}-{case}", std::process::id()), model: "host".into(), memory }
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
    let shrunk = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut out = std::io::stdout();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let w: Vec<&str> = line.split_whitespace().collect();
        let reply: String = match w.as_slice() {
            ["open", id, memory] => {
                uuid = id.to_string();
                let memory: u64 = memory.parse().unwrap();
                let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory };
                mem = Some(Memory::open(g, None, Box::new(HostBackend::new(memory))).unwrap());
                format!("ok {}", std::process::id())
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
            ["cache", owner, bytes] => {
                let m = mem.as_ref().unwrap();
                let s = shrunk.clone();
                let c = m.cache(owner, Box::new(move |want| {
                    s.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    want
                }));
                m.cached(c, bytes.parse().unwrap());
                "ok".into()
            }
            ["shrunk"] => format!("ok {}", shrunk.load(std::sync::atomic::Ordering::Relaxed)),
            ["pass", req, set, shape] => match requests[*req].pass(sets[*set].0, shape, Torch::default()) {
                Ok(p) => {
                    passes.insert(req.to_string(), p);
                    "ok".into()
                }
                Err(e) => format!("err {}", e.code()),
            },
            ["endpass", req] => {
                if let Some(p) = passes.remove(*req) {
                    p.end(Measured::default(), Torch::default());
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
            ["oom", req, need] => match requests[*req].oom(Some(need.parse().unwrap()), None, Torch::default()) {
                Next::Retry => "retry".into(),
                Next::Cut(l) => format!("cut {l}"),
                Next::Park(b) => format!("park {b}"),
                Next::Exhausted(e) => format!("exhausted {} {}", e.stats.request, e.need),
            },
            ["parked", req, bytes] => {
                requests[*req].parked(bytes.parse().unwrap());
                "ok".into()
            }
            ["trace", req] => {
                let s = requests[*req].stats();
                let origins: Vec<String> = s.trace.iter().map(|r| format!("{:?}", r.origin)).collect();
                format!("ok {}", origins.join(","))
            }
            ["end", req] => {
                passes.remove(*req);
                let s = requests.remove(*req).unwrap().end();
                format!("ok leftover={} kept={} tier={} store={}", s.leftover, s.moved.kept.bytes, s.moved.tier.bytes, s.moved.store.bytes)
            }
            ["limit"] => format!("ok {}", mem.as_ref().unwrap().torch_limit()),
            ["fault", point] => {
                varena::faults::arm(if *point == "decide" { varena::faults::Point::Decide } else { varena::faults::Point::Fill });
                "ok".into()
            }
            ["exit"] => {
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
        let mut child = Command::new(std::env::current_exe().unwrap()).arg("child").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let to = child.stdin.take().unwrap();
        let from = BufReader::new(child.stdout.take().unwrap());
        Agent { child, to, from }
    }

    fn raw(&mut self, line: &str) -> String {
        writeln!(self.to, "{line}").unwrap();
        self.to.flush().unwrap();
        let mut reply = String::new();
        self.from.read_line(&mut reply).unwrap();
        reply.trim().to_string()
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

/// A holder with every page of its set mapped and nothing running answers an ask: the
/// demander's pass and binds get its pages' room.
fn an_idle_holder_answers_an_ask(dir: &Path) {
    let g = gpu("ask", 8 * MB);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let (a_path, b_path) = (write_set(dir, "a", 4), write_set(dir, "b", 2));
    let mut a = Agent::spawn();
    a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.ok(&format!("load a {} stored gpu", a_path.display()));
    let held = |l: &varena::Ledger, pid: &str| l.status().rows.iter().find(|r| r.pid.to_string() == pid).map(|r| r.classes.iter().sum::<u64>()).unwrap();
    let mut b = Agent::spawn();
    let b_pid = b.ok(&format!("open {} {}", g.uuid, g.memory));
    let a_pid = a.child.id().to_string();
    assert_eq!(held(&ledger, &a_pid), 8 * MB, "a holds the card");
    b.ok(&format!("load b {} stored ram", b_path.display()));
    b.ok("request r b");
    b.ok("pass r b 1x1");
    b.ok("bind b 0");
    b.ok("bind b 1");
    assert_eq!(b.ok("exact b 1"), "true");
    assert_eq!(held(&ledger, &a_pid), 4 * MB, "a gave back what b needed");
    assert_eq!(held(&ledger, &b_pid), 4 * MB);
    b.ok("unpin");
    b.ok("end r");
    a.exit();
    b.exit();
}

/// `kill -9` of a holder leaves its kept pages as orphans; a new process imports them bit-exact
/// by fingerprint; another fill identity refills instead.
fn a_killed_holder_leaves_orphans_a_successor_imports(dir: &Path) {
    let g = gpu("kill", 16 * MB);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let path = write_set(dir, "k", 4);
    let mut a = Agent::spawn();
    a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.ok(&format!("load k {} stored gpu", path.display()));
    a.child.kill().unwrap();
    a.child.wait().unwrap();
    ledger.reap(a.child.id());
    let mut b = Agent::spawn();
    b.ok(&format!("open {} {}", g.uuid, g.memory));
    b.ok(&format!("load k {} stored ram", path.display()));
    b.ok("request r k");
    b.ok("pass r k 1x1");
    for block in 0..4 {
        assert_eq!(b.ok(&format!("exact k {block}")), "true", "block {block} bit-exact");
    }
    assert_eq!(b.ok("trace r"), "Kept,Kept,Kept,Kept", "imported, not copied");
    b.ok("unpin");
    let end = b.ok("end r");
    assert!(end.contains(&format!("kept={}", 8 * MB)) && end.contains("store=0"), "{end}");
    b.exit();
    // A fingerprint mismatch (another fill identity) refills from the store or tier.
    let mut c = Agent::spawn();
    c.ok(&format!("open {} {}", g.uuid, g.memory));
    c.ok(&format!("load k {} lora-x ram", path.display()));
    c.ok("request r k");
    c.ok("pass r k 1x1");
    assert_eq!(c.ok("bind k 0"), "true", "fresh: the caller bakes it");
    c.ok("filled");
    let trace = c.ok("trace r");
    assert!(!trace.contains("Kept"), "{trace}");
    c.ok("unpin");
    c.ok("end r");
    c.exit();
}

/// A request that fails mid-way (an exhausted op) frees every pin, room and cache: leftover 0,
/// and the same process serves the next request.
fn a_failed_request_frees_everything(dir: &Path) {
    let g = gpu("fail", 8 * MB);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let path = write_set(dir, "f", 2);
    let mut a = Agent::spawn();
    a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.ok(&format!("load f {} stored ram", path.display()));
    let limit = a.ok("limit");
    a.ok("request r1 f");
    a.ok("pass r1 f 1x1");
    a.ok("bind f 0");
    a.ok("bind f 1");
    // The free bytes are claimed once (the cap was stale); with every page pinned nothing else
    // frees: park (the host has headroom), and a park of nothing ends it.
    assert_eq!(a.say("oom r1 67108864"), "retry");
    assert!(a.say("oom r1 67108864").starts_with("park "));
    a.ok("parked r1 0");
    assert!(a.say("oom r1 67108864").starts_with("exhausted r1"));
    a.ok("cache r1 1048576");
    a.ok("unpin");
    assert!(a.ok("end r1").starts_with("leftover=0"));
    assert_eq!(a.ok("shrunk"), "1", "the request's cache was shrunk");
    assert_eq!(a.ok("limit").parse::<u64>().unwrap() + 4 * MB, limit.parse::<u64>().unwrap(), "only the mapped pages remain");
    a.ok("request r2 f");
    a.ok("pass r2 f 1x1");
    assert_eq!(a.ok("exact f 1"), "true");
    a.ok("unpin");
    assert!(a.ok("end r2").starts_with("leftover=0"));
    a.exit();
}

/// A panic in `decide` and one in a fill each fail their call `Internal`; the next request in
/// the same process succeeds.
fn a_panic_fails_the_request_not_the_process(dir: &Path) {
    let g = gpu("panic", 8 * MB);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let path = write_set(dir, "p", 2);
    let mut a = Agent::spawn();
    a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.ok(&format!("load p {} stored ram", path.display()));
    a.ok("request r1 p");
    a.ok("fault decide");
    assert_eq!(a.raw("pass r1 p 1x1"), "err INTERNAL");
    assert!(a.ok("end r1").starts_with("leftover=0"));
    a.ok("request r2 p");
    a.ok("pass r2 p 1x1");
    assert_eq!(a.ok("exact p 0"), "true");
    a.ok("unpin");
    a.ok("end r2");
    // A fill thread's panic fails the bind waiting on it.
    let q = write_set(dir, "q", 1);
    a.ok(&format!("load q {} stored ram", q.display()));
    a.ok("request r3 q");
    a.ok("pass r3 q 1x1");
    a.ok("fault fill");
    assert_eq!(a.raw("bind q 0"), "err INTERNAL");
    assert!(a.ok("end r3").starts_with("leftover=0"));
    a.ok("request r4 q");
    a.ok("pass r4 q 1x1");
    assert_eq!(a.ok("exact q 0"), "true");
    a.ok("unpin");
    a.ok("end r4");
    a.exit();
}

/// Exhaustion is the one memory error, and it carries the request's record.
fn exhausted_carries_stats(dir: &Path) {
    let g = gpu("exh", 4 * MB);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
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

/// On injected pressure the tier gives back orphaned layouts first, then unused, then the
/// furthest declared.
fn the_tier_gives_back_in_order(dir: &Path) {
    use std::os::fd::FromRawFd;
    let g = gpu("tier", 64 * MB);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    // SAFETY: eventfd(2) with no shared state; the descriptor is ours.
    let event = unsafe { platform::handles::Owned::from_raw_fd(libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK)) };
    let signal = event.try_clone().unwrap();
    let tier = varena::Tier::serve(varena::Pressure::Events(event)).unwrap().expect("another process serves this user's RAM tier");
    let paths: Vec<PathBuf> = ["l1", "l2", "l3", "l4"].iter().map(|n| write_set(dir, n, 1)).collect();
    let digest = |n: &str| format!("layout-{n}-{}", g.uuid);
    // l1: loaded by a process that ended (orphaned).
    let mut x = Agent::spawn();
    x.ok(&format!("open {} {}", g.uuid, g.memory));
    x.ok(&format!("load l1 {} stored ram", paths[0].display()));
    x.exit();
    // l2 unused, l3 declared three steps ahead, l4 in use now.
    let mut y = Agent::spawn();
    y.ok(&format!("open {} {}", g.uuid, g.memory));
    for (n, p) in ["l2", "l3", "l4"].iter().zip(&paths[1..]) {
        y.ok(&format!("load {n} {} stored ram", p.display()));
    }
    y.ok("request r l4;l4;l4;l3");
    y.ok("pass r l4 1x1");
    let mut left = vec![digest("l1"), digest("l2"), digest("l3"), digest("l4")];
    assert_eq!(tier.held().into_iter().filter(|d| left.contains(d)).collect::<Vec<_>>(), left);
    for n in 1..=4 {
        // SAFETY: an 8-byte write to our eventfd.
        unsafe { libc::write(std::os::fd::AsRawFd::as_raw_fd(&signal), 1u64.to_ne_bytes().as_ptr().cast(), 8) };
        tier.await_releases(n);
        left.remove(0);
        assert_eq!(tier.held().into_iter().filter(|d| d.ends_with(&g.uuid)).collect::<Vec<_>>(), left, "after {n} rungs");
    }
    y.ok("end r");
    y.exit();
}
