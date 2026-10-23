//! The real Memory, ledger and custody over host pages, across real processes. The parent
//! plays TensorD (the ledger, custody); each child is this binary as an agent
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
        ("an_idle_holder_answers_an_ask", an_idle_holder_answers_an_ask),
        ("a_killed_holder_leaves_orphans_a_successor_imports", a_killed_holder_leaves_orphans_a_successor_imports),
        ("a_failed_request_frees_everything", a_failed_request_frees_everything),
        ("a_panic_fails_the_request_not_the_process", a_panic_fails_the_request_not_the_process),
        ("exhausted_carries_stats", exhausted_carries_stats),
        ("kept_pages_survive_a_holder_that_served", kept_pages_survive_a_holder_that_served),
        ("binds_outlive_the_custody_holder", binds_outlive_the_custody_holder),
        ("a_table_with_live_rows_outlives_tensord", a_table_with_live_rows_outlives_tensord),
        ("a_transferred_grant_keeps_its_table_after_ledger_drop", a_transferred_grant_keeps_its_table_after_ledger_drop),
        ("a_stale_reader_never_frees_a_newer_row", a_stale_reader_never_frees_a_newer_row),
        ("a_joint_step_runs_a_pass_per_set", a_joint_step_runs_a_pass_per_set),
        ("a_child_runs_while_its_parent_yields", a_child_runs_while_its_parent_yields),
        ("a_reaper_with_a_stale_view_frees_nothing", a_reaper_with_a_stale_view_frees_nothing),
        ("a_request_asks_peers_at_its_open_and_the_hook_takes_its_own_idle_page_first", a_request_asks_peers_at_its_open_and_the_hook_takes_its_own_idle_page_first),
        ("a_standalone_process_wakes_tensord", a_standalone_process_wakes_tensord),
        ("a_yielded_parent_is_its_familys_alone_and_waits_for_its_room", a_yielded_parent_is_its_familys_alone_and_waits_for_its_room),
    ];
    // The kernel's list of sockets and stopped threads are Linux's.
    #[cfg(target_os = "linux")]
    cases.extend([
        ("weights_by_lease_return_on_kill_and_are_adopted", weights_by_lease_return_on_kill_and_are_adopted as fn(&Path)),
        ("an_ask_to_a_stopped_victim_ends_when_it_dies", an_ask_to_a_stopped_victim_ends_when_it_dies as fn(&Path)),
        ("an_unmeasured_request_asks_peers_for_pages_never_caches", an_unmeasured_request_asks_peers_for_pages_never_caches),
        ("a_joiners_limit_leaves_its_peers_room", a_joiners_limit_leaves_its_peers_room),
        ("a_silent_asker_holds_up_no_answer", a_silent_asker_holds_up_no_answer),
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
    let shrunk = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut out = std::io::stdout();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let w: Vec<&str> = line.split_whitespace().collect();
        let reply: String = match w.as_slice() {
            ["grant", id, memory] => {
                use std::os::fd::FromRawFd;
                uuid = id.to_string();
                let memory: u64 = memory.parse().unwrap();
                let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory };
                // SAFETY: descriptors 3 and 4 are the grant the parent handed this process.
                let grant = unsafe { varena::Grant { table: platform::handles::Owned::from_raw_fd(3), row: platform::handles::Owned::from_raw_fd(4), limit: None } };
                mem = Some(Memory::open(g, Some(grant), Box::new(HostBackend::new(memory))).unwrap());
                format!("ok {}", std::process::id())
            }
            ["open", id, memory] => {
                uuid = id.to_string();
                let memory: u64 = memory.parse().unwrap();
                let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory };
                mem = Some(Memory::open(g, None, Box::new(HostBackend::new(memory))).unwrap());
                format!("ok {}", std::process::id())
            }
            ["open", id, memory, limit] => {
                uuid = id.to_string();
                let memory: u64 = memory.parse().unwrap();
                let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory };
                let grant = varena::Grant::open(&g, Some(limit.parse().unwrap())).unwrap();
                mem = Some(Memory::open(g, Some(grant), Box::new(HostBackend::new(memory))).unwrap());
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
            ["cache", owner, bytes] => {
                let m = mem.as_ref().unwrap();
                let s = shrunk.clone();
                let c = m.cache(owner, &[], move |want| {
                    s.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    want
                });
                m.cached(c, bytes.parse().unwrap());
                "ok".into()
            }
            ["shrunk"] => format!("ok {}", shrunk.load(std::sync::atomic::Ordering::Relaxed)),
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
            ["room", need] => match mem.as_ref().unwrap().make_room(need.parse().unwrap()) {
                Ok(_) => "ok".into(),
                Err(varena::Error::Exhausted(e)) => format!("exhausted {} {}", e.stats.request, e.need),
                Err(e) => format!("err {}", e.code()),
            },
            ["yield", req, family] => {
                yielded.insert(req.to_string(), requests[*req].yielding(family));
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
            ["limit"] => {
                let m = mem.as_ref().unwrap();
                m.make_room(0).unwrap();
                format!("ok {}", m.free())
            }
            ["settle"] => {
                mem.as_ref().unwrap().settle();
                "ok".into()
            }
            #[cfg(target_os = "linux")]
            ["lease", socket] => {
                let stream = std::os::unix::net::UnixStream::connect(socket).unwrap();
                match mem.as_ref().unwrap().lease(stream).unwrap() {
                    None => "ok".into(),
                    Some((code, why)) => format!("refused {code:?} {why}"),
                }
            }
            ["fault", point] => {
                varena::faults::arm(if *point == "decide" { varena::faults::Point::Decide } else { varena::faults::Point::Fill });
                "ok".into()
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

    /// A child holding TensorD's grant as descriptors 3 and 4.
    fn granted(grant: varena::Grant) -> Agent {
        use std::os::fd::AsRawFd;
        use std::os::unix::process::CommandExt;
        let (table, row) = (grant.table.as_raw_fd(), grant.row.as_raw_fd());
        let mut command = Command::new(std::env::current_exe().unwrap());
        // SAFETY: only async-signal-safe dup2 calls between fork and exec.
        unsafe {
            command.pre_exec(move || {
                if libc::dup2(table, 3) < 0 || libc::dup2(row, 4) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let agent = Agent::start(command);
        drop(grant);
        agent
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

/// A holder with every page of its set mapped and nothing running answers an ask: an
/// unmeasured request opening beside it gets its idle pages' room.
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
    assert_eq!(held(&ledger, &a_pid), 0, "b's unmeasured request asked a for its idle pages");
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
    a.ok("settle");
    a.child.kill().unwrap();
    a.child.wait().unwrap();
    ledger.reap(a.child.id());
    let orphans: u64 = ledger.status().rows.iter().map(|r| r.classes[1]).sum();
    assert_eq!(orphans, 8 * MB, "S5: class O shows the orphans");
    let mut b = Agent::spawn();
    b.ok(&format!("open {} {}", g.uuid, g.memory));
    // One load path: its load imports what custody keeps, at any level.
    b.ok(&format!("load k {} stored ram", path.display()));
    b.ok("request r k");
    b.ok("pass r k 1x1");
    for block in 0..4 {
        assert_eq!(b.ok(&format!("exact k {block}")), "true", "block {block} bit-exact");
    }
    assert_eq!(b.ok("trace r"), "Kept,Kept,Kept,Kept,Hit,Hit,Hit,Hit", "imported at load (its fills in the record), not copied");
    b.ok("unpin");
    let end = b.ok("end r");
    assert!(end.contains("store=0"), "{end}");
    b.exit();
    // A fingerprint mismatch (another fill identity) refills from the store.
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

/// Slice 1's bar on CPU (varena-leases §8): two executors' Memories get their weights by lease
/// from a manager this process hosts; kill -9 one and its leases return once its exit shows, its
/// landed weights retained; its successor adopts them with no copy and no new chunk.
#[cfg(target_os = "linux")]
fn weights_by_lease_return_on_kill_and_are_adopted(dir: &Path) {
    use std::os::unix::net::UnixListener;
    use std::time::{Duration, Instant};
    use varena::lease::{Holdings, Hosting, Identity, Manager};
    struct Ours;
    impl Hosting for Ours {
        fn identify(&self, peer: &platform::handles::Peer) -> Option<Identity> {
            Some(Identity { executor: peer.pid.to_string(), domain: "test".into() })
        }
    }
    let until = |m: &Manager, done: &dyn Fn(&Holdings) -> bool| -> Holdings {
        let start = Instant::now();
        loop {
            let h = m.holdings();
            if done(&h) {
                return h;
            }
            assert!(start.elapsed() < Duration::from_secs(20), "the manager never reached it: {h:?}");
            std::thread::sleep(Duration::from_millis(5));
        }
    };
    let g = gpu("lease", 64 * MB);
    let socket = dir.join("lease.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let open: varena::lease::Opener = Box::new(|| Ok(Arc::new(HostBackend::new(64 * MB)) as Arc<dyn varena::Backend>));
    let manager = Manager::create(&g, None, varena::ledger::Mode::Dedicated, open, listener, Arc::new(Ours)).unwrap();
    let (k, j) = (write_set(dir, "k", 4), write_set(dir, "j", 2));
    let start = |set: &str, path: &Path| {
        let mut x = Agent::spawn();
        x.ok(&format!("open {} {}", g.uuid, g.memory));
        x.ok(&format!("lease {}", socket.display()));
        x.ok(&format!("load {set} {} stored gpu", path.display()));
        x.ok(&format!("request r {set}"));
        x.ok(&format!("pass r {set} 1x1"));
        x
    };
    let mut a = start("k", &k);
    for block in 0..4 {
        assert_eq!(a.ok(&format!("exact k {block}")), "true", "a's block {block} bit-exact");
    }
    a.ok("unpin");
    assert!(a.ok("end r").contains(&format!("leased={}", 8 * MB)), "a's weights came by lease");
    let mut b = start("j", &j);
    for block in 0..2 {
        assert_eq!(b.ok(&format!("exact j {block}")), "true", "b's block {block} bit-exact");
    }
    b.ok("unpin");
    b.ok("end r");
    let a_pid = a.child.id();
    let h = until(&manager, &|h| h.executors.len() == 2 && h.executors.iter().any(|e| e.pid == a_pid && e.landed == 8 * MB));
    assert_eq!((h.leased, h.created), (12 * MB, 12 * MB), "{h:?}");
    a.child.kill().unwrap();
    a.child.wait().unwrap();
    let h = until(&manager, &|h| h.executors.len() == 1);
    assert_eq!((h.retained, h.leased, h.created), (8 * MB, 4 * MB, 12 * MB), "a's landed weights retained: {h:?}");
    let mut c = start("k", &k);
    for block in 0..4 {
        assert_eq!(c.ok(&format!("exact k {block}")), "true", "c's adopted block {block} bit-exact");
    }
    assert_eq!(c.ok("trace r"), "Kept,Kept,Kept,Kept,Hit,Hit,Hit,Hit", "adopted at load, not copied");
    c.ok("unpin");
    let end = c.ok("end r");
    assert!(end.contains("store=0"), "{end}");
    let h = until(&manager, &|h| h.executors.len() == 2);
    assert_eq!((h.retained, h.leased, h.created), (0, 12 * MB, 12 * MB), "adopted with no new chunk: {h:?}");
    b.exit();
    c.exit();
}

/// A request that fails mid-way (an exhausted allocation) frees every pin and cache: leftover 0,
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
    // With every page pinned, the whole card is Exhausted at once.
    assert!(a.say("room 8388608").starts_with("exhausted r1"));
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
    a.ok("pass r1 p 1x1");
    a.ok("bind p 0");
    a.ok("bind p 1");
    a.ok("unpin");
    // make_room beyond the free bytes decides what to take.
    a.ok("fault decide");
    assert_eq!(a.raw(&format!("room {}", 8 * MB)), "err INTERNAL");
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

/// S5's finding: a holder that served a request (its pass measured, so a successor's pass
/// needs room) and was then killed still leaves its kept pages to its successor: they are
/// adopted before any demand could release them as orphans.
fn kept_pages_survive_a_holder_that_served(dir: &Path) {
    let g = gpu("served", 8 * MB);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let path = write_set(dir, "s", 3);
    let mut a = Agent::spawn();
    a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.ok(&format!("load s {} stored gpu", path.display()));
    a.ok("request r s");
    a.ok("pass r s 1x1");
    for block in 0..3 {
        a.ok(&format!("bind s {block}"));
    }
    a.ok("unpin");
    a.ok(&format!("endpass r {}", MB));
    a.ok("end r");
    a.ok("settle");
    a.child.kill().unwrap();
    a.child.wait().unwrap();
    ledger.reap(a.child.id());
    let mut b = Agent::spawn();
    b.ok(&format!("open {} {}", g.uuid, g.memory));
    b.ok(&format!("load s {} stored ram", path.display()));
    b.ok("request r s");
    b.ok("pass r s 1x1");
    for block in 0..3 {
        assert_eq!(b.ok(&format!("exact s {block}")), "true");
    }
    assert_eq!(b.ok("trace r"), "Kept,Kept,Kept,Hit,Hit,Hit", "every page kept, none copied");
    b.ok("unpin");
    let end = b.ok("end r");
    assert!(end.contains("store=0"), "{end}");
    b.exit();
}

#[cfg(target_os = "linux")]
/// Review 1, finding 2: an ask to a victim whose ask thread cannot run (stopped) waits on the
/// victim itself: when it dies, the asker goes on, with no timer. Review 2, finding 3: TensorD
/// sees it stalled only while an ask is pending to it, and the ask is in the asker's own row.
fn an_ask_to_a_stopped_victim_ends_when_it_dies(dir: &Path) {
    let g = gpu("stop", 10 * MB);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let (a_path, b_path) = (write_set(dir, "va", 4), write_set(dir, "vb", 2));
    let mut a = Agent::granted(ledger.grant(std::process::id()).unwrap());
    a.ok(&format!("grant {} {}", g.uuid, g.memory));
    let mut b = Agent::granted(ledger.grant(std::process::id()).unwrap());
    b.ok(&format!("grant {} {}", g.uuid, g.memory));
    b.ok(&format!("load vb {} stored ram", b_path.display()));
    b.ok(&format!("load vn {} stored gpu", write_set(dir, "vn", 1).display()));
    // b opens first: nobody holds anything to ask for yet.
    b.ok("request r vb");
    a.ok(&format!("load va {} stored gpu", a_path.display()));
    // Earlier events (rows claimed, pages mapped) are not the ask's.
    ledger.clear_events();
    let pid = a.child.id() as i32;
    assert!(ledger.stalled().is_empty(), "a running or sleeping ask thread is never stalled");
    let mut status = 0;
    // SAFETY: a signal to our own child, then waiting until it has stopped.
    unsafe {
        libc::kill(pid, libc::SIGSTOP);
        libc::waitpid(pid, &mut status, libc::WUNTRACED);
    }
    assert!(ledger.stalled().is_empty(), "stopped, but nobody asks it: not stalled");
    b.send("pass r vb 1x1");
    // The ask is a ledger event: wait for it, until the stopped victim's row is asked.
    while !ledger.status().rows.iter().any(|r| r.pid == pid as u32 && r.asked > 0) {
        ledger.await_event().unwrap();
    }
    assert_eq!(ledger.stalled(), vec![pid as u32], "asked and its ask thread stopped: measured, not counted");
    a.child.kill().unwrap();
    a.child.wait().unwrap();
    assert_eq!(b.reply(), "ok", "the asker went on when the victim died");
    b.ok("bind vb 0");
    b.ok("unpin");
    b.ok("end r");
    b.exit();
}

/// Review 1, finding 9: when custody's holder exits, binds go on: the next call reconnects,
/// serving custody from this process.
fn binds_outlive_the_custody_holder(dir: &Path) {
    let g = gpu("holder", 16 * MB);
    let (x_path, y_path) = (write_set(dir, "hx", 2), write_set(dir, "hy", 2));
    let mut a = Agent::spawn();
    a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.ok(&format!("load hx {} stored gpu", x_path.display()));
    let mut b = Agent::spawn();
    b.ok(&format!("open {} {}", g.uuid, g.memory));
    b.ok(&format!("load hy {} stored ram", y_path.display()));
    a.ok("settle");
    a.child.kill().unwrap();
    a.child.wait().unwrap();
    b.ok("request r hy");
    b.ok("pass r hy 1x1");
    assert_eq!(b.ok("exact hy 1"), "true");
    b.ok("unpin");
    b.ok("end r");
    b.exit();
    // The table goes with the last of its rows.
    drop(varena::Ledger::create(&g, None).unwrap());
}

/// The grant survives in a real child before its Memory is opened: dropping the parent's ledger
/// keeps that very generation (v8: TensorD's limit goes with it, and another opener's limit caps
/// only that opener's row).
fn a_transferred_grant_keeps_its_table_after_ledger_drop(_dir: &Path) {
    let g = gpu("transferred-cap", 64 * MB);
    let ledger = varena::Ledger::create(&g, Some(12 * MB)).unwrap();
    let mut child = Agent::granted(ledger.grant(std::process::id()).unwrap());
    drop(ledger);
    drop(varena::Grant::open(&g, Some(8 * MB)).unwrap());
    child.ok(&format!("grant {} {}", g.uuid, g.memory));
    assert_eq!(child.ok("limit"), (64 * MB).to_string(), "TensorD's limit went with it");
    let mut observer = Agent::spawn();
    observer.ok(&format!("open {} {}", g.uuid, g.memory));
    let path = write_set(_dir, "seen", 1);
    child.ok(&format!("load seen {} stored gpu", path.display()));
    assert_eq!(observer.ok("limit"), (64 * MB - REGION_ALIGN).to_string(), "one table: the child's page counts for the observer");
    child.exit();
    observer.exit();
}

/// Review 1, finding 10: TensorD's exit keeps a table an executor still uses, and a restarted
/// TensorD reopens it.
fn a_table_with_live_rows_outlives_tensord(dir: &Path) {
    let g = gpu("restart", 8 * MB);
    let first = varena::Ledger::create(&g, None).unwrap();
    let path = write_set(dir, "rt", 1);
    let mut a = Agent::spawn();
    let pid: u32 = a.ok(&format!("open {} {}", g.uuid, g.memory)).parse().unwrap();
    a.ok(&format!("load rt {} stored gpu", path.display()));
    drop(first);
    let second = varena::Ledger::create(&g, None).unwrap();
    let row = second.status().rows.into_iter().find(|r| r.pid == pid);
    assert!(row.as_ref().is_some_and(|r| r.live && r.classes[3] == 2 * MB), "the executor's row is still there: {row:?}");
    a.exit();
    drop(second);
}

/// Review 1, finding 1: a row reaped from a stale view is never freed again once its offset is
/// reused; the new holder's row stays.
fn a_stale_reader_never_frees_a_newer_row(dir: &Path) {
    let g = gpu("stale", 8 * MB);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let path = write_set(dir, "st", 1);
    let mut a = Agent::spawn();
    a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.child.kill().unwrap();
    a.child.wait().unwrap();
    // Two reapers saw the dead row; the first frees it and a new process reuses its offset.
    ledger.reap(a.child.id());
    let mut b = Agent::spawn();
    let pid: u32 = b.ok(&format!("open {} {}", g.uuid, g.memory)).parse().unwrap();
    b.ok(&format!("load st {} stored gpu", path.display()));
    ledger.reap(a.child.id());
    ledger.reap(pid);
    assert!(ledger.status().rows.iter().any(|r| r.pid == pid && r.live), "the new row survived stale reaps");
    b.exit();
}

/// v8: a live joiner opening with a small limit caps its own row; its peer's room and the
/// table's capacity stay the card's.
fn a_joiners_limit_leaves_its_peers_room(dir: &Path) {
    let _ = dir;
    let g = gpu("joiner", 64 * MB);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let mut peer = Agent::spawn();
    peer.ok(&format!("open {} {}", g.uuid, g.memory));
    let mut joiner = Agent::spawn();
    joiner.ok(&format!("open {} {} {}", g.uuid, g.memory, 8 * MB));
    assert_eq!(joiner.ok("limit"), (8 * MB).to_string(), "the joiner's own row is capped");
    assert_eq!(peer.ok("limit"), (64 * MB).to_string(), "its peer's room is untouched");
    assert_eq!(ledger.status().capacity, 64 * MB, "and the table's capacity");
    joiner.exit();
    peer.exit();
}

/// Co-residency (lead, v7: block code runs once, with no OOM to escalate from): an unmeasured
/// request asks its peers for every idle page at its open, pages only: an idle holder's graph
/// caches survive. Its allocations then fit.
fn an_unmeasured_request_asks_peers_for_pages_never_caches(dir: &Path) {
    let g = gpu("peers", 8 * MB);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let (a_path, b_path) = (write_set(dir, "pa", 2), write_set(dir, "pb", 1));
    let held = |l: &varena::Ledger, pid: &str| l.status().rows.iter().find(|r| r.pid.to_string() == pid).map(|r| r.classes.iter().sum::<u64>()).unwrap();
    let mut a = Agent::spawn();
    let a_pid = a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.ok(&format!("load pa {} stored gpu", a_path.display()));
    a.ok(&format!("cache graphs {MB}"));
    let mut b = Agent::spawn();
    b.ok(&format!("open {} {}", g.uuid, g.memory));
    b.ok(&format!("load pb {} stored gpu", b_path.display()));
    b.ok("request r pb");
    assert_eq!(held(&ledger, &a_pid), 0, "the request asked a for its idle pages");
    assert_eq!(a.ok("shrunk"), "0", "a's caches survive");
    b.ok("pass r pb 1x1");
    assert_eq!(b.say(&format!("room {}", 2 * REGION_ALIGN)), "ok", "its allocations fit");
    b.ok("end r");
    a.exit();
    b.exit();
}

/// A joint step ({a, b}): each set runs its own pass; interleaved binds land in their own
/// passes.
fn a_joint_step_runs_a_pass_per_set(dir: &Path) {
    let g = gpu("joint", 32 * MB);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
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

/// Round 2, finding 15 (S5): a request waiting on a child call yields its room and its step's
/// pages: the child (parent W + child W > capacity) runs beside it, and the parent completes.
fn a_child_runs_while_its_parent_yields(dir: &Path) {
    let g = gpu("yield", 8 * MB);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let (p_path, c_path) = (write_set(dir, "yp", 2), write_set(dir, "yc", 3));
    let mut parent = Agent::granted(ledger.grant(std::process::id()).unwrap());
    let parent_pid = parent.ok(&format!("grant {} {}", g.uuid, g.memory));
    parent.ok(&format!("load yp {} stored ram", p_path.display()));
    parent.ok("request r yp");
    parent.ok("pass r yp 1x1");
    parent.ok("bind yp 0");
    parent.ok("bind yp 1");
    parent.ok("unpin");
    let held = |pid: &str| ledger.status().rows.iter().find(|r| r.pid.to_string() == pid).map_or(0, |r| r.classes.iter().sum::<u64>());
    assert_eq!(held(&parent_pid), 4 * MB);
    // The parent calls its child and waits on it.
    parent.ok("yield r fam");
    let mut child = Agent::granted(ledger.grant(std::process::id()).unwrap());
    child.ok(&format!("grant {} {}", g.uuid, g.memory));
    child.ok(&format!("load yc {} stored ram", c_path.display()));
    child.ok("request c yc");
    child.ok("pass c yc 1x1");
    for block in 0..3 {
        assert_eq!(child.ok(&format!("exact yc {block}")), "true");
    }
    // Its pages were a later use while it waited: its unmeasured child asked for them. (Not
    // yielded they are its working set, and its child would have streamed beside them.)
    assert_eq!(held(&parent_pid), 0, "the yielded parent gave its child its working set");
    child.ok("unpin");
    child.ok("end c");
    child.exit();
    // The child returned: the parent takes its room back and finishes.
    parent.ok("unyield r");
    for block in 0..2 {
        assert_eq!(parent.ok(&format!("exact yp {block}")), "true");
    }
    parent.ok("unpin");
    parent.ok("end r");
    parent.exit();
}

#[cfg(target_os = "linux")]
/// Round 2, finding 9: a row's ask thread serves every asker at once: one connected and
/// silent (mid-message) holds up nobody else's answer.
fn a_silent_asker_holds_up_no_answer(dir: &Path) {
    use std::os::linux::net::SocketAddrExt;
    let g = gpu("silent", 8 * MB);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let (a_path, b_path) = (write_set(dir, "sa", 4), write_set(dir, "sb", 2));
    let mut a = Agent::spawn();
    a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.ok(&format!("load sa {} stored gpu", a_path.display()));
    // a's ask socket, as the kernel lists it.
    let sockets = std::fs::read_to_string("/proc/net/unix").unwrap();
    let name = sockets.split_whitespace().find_map(|w| w.strip_prefix('@').filter(|n| n.contains(&format!("-ask-{}-", g.uuid)))).unwrap().to_string();
    let mut silent = std::os::unix::net::UnixStream::connect_addr(&std::os::unix::net::SocketAddr::from_abstract_name(&name).unwrap()).unwrap();
    silent.write_all(&[9, 0]).unwrap();
    let mut b = Agent::spawn();
    b.ok(&format!("open {} {}", g.uuid, g.memory));
    b.ok(&format!("load sb {} stored ram", b_path.display()));
    b.ok("request r sb");
    b.ok("pass r sb 1x1");
    assert_eq!(b.ok("exact sb 1"), "true", "a answered b beside the silent asker");
    drop(silent);
    b.ok("unpin");
    b.ok("end r");
    a.exit();
    b.exit();
}

/// Round 2, finding 17: a reaper that read a row before it died and frees it after a new
/// process took its offset frees nothing: the free compares the whole word.
fn a_reaper_with_a_stale_view_frees_nothing(_dir: &Path) {
    let g = gpu("stalereap", 8 * MB);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let mut a = Agent::spawn();
    a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.child.kill().unwrap();
    a.child.wait().unwrap();
    let mut b = Agent::spawn();
    varena::faults::hold(varena::faults::Point::Reap);
    std::thread::scope(|s| {
        // TensorD reads the rows (a's dead) and stops there.
        let reaper = s.spawn(|| ledger.reap(a.child.id()));
        varena::faults::reached();
        // b opens: it reaps a's row itself and takes its offset.
        b.ok(&format!("open {} {}", g.uuid, g.memory));
        varena::faults::release();
        reaper.join().unwrap();
    });
    let pid = b.child.id();
    assert!(ledger.status().rows.iter().any(|r| r.pid == pid && r.live), "the new row survived the stale reap");
    b.exit();
}

/// Co-residency (owner, 10-22): an unmeasured request asks its peers for their idle pages when
/// it opens; the hook's `make_room` then takes its own idle page before a declared one.
fn a_request_asks_peers_at_its_open_and_the_hook_takes_its_own_idle_page_first(dir: &Path) {
    let g = gpu("ownask", 6 * REGION_ALIGN);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let path = |n: &str, r| write_set(dir, n, r);
    let mut a = Agent::spawn();
    a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.ok(&format!("load on {} stored gpu", path("on", 1).display()));
    let mut b = Agent::spawn();
    let b_pid = b.ok(&format!("open {} {}", g.uuid, g.memory));
    b.ok(&format!("load ow {} stored ram", path("ow", 2).display()));
    b.ok(&format!("load od {} stored gpu", path("od", 2).display()));
    b.ok(&format!("load ox {} stored gpu", path("ox", 1).display()));
    b.ok("request r ow;od");
    let row = |pid: &str| ledger.status().rows.into_iter().find(|r| r.pid.to_string() == pid).unwrap();
    assert_eq!(row(&a.child.id().to_string()).classes.iter().sum::<u64>(), 0, "a gave its idle page at b's open");
    b.ok("pass r ow 1x1");
    b.ok("bind ow 0");
    b.ok("bind ow 1");
    // The card: ow's two pages, od's two, ox's one, one free. The hook needs two.
    assert_eq!(b.say(&format!("room {}", 2 * REGION_ALIGN)), "ok");
    assert_eq!(row(&b_pid).classes[varena::table::Class::D.index()], 2 * REGION_ALIGN, "no D page went");
    b.ok("unpin");
    b.ok("end r");
    a.exit();
    b.exit();
}

/// S5 review, 6: a standalone process's frees reach TensorD: the wake is the GPU's, by name.
fn a_standalone_process_wakes_tensord(_dir: &Path) {
    let g = gpu("wake", 8 * MB);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let mut a = Agent::spawn();
    let pid = a.ok(&format!("open {} {}", g.uuid, g.memory));
    ledger.clear_events();
    a.exit();
    ledger.await_event().unwrap();
    assert!(ledger.status().rows.iter().all(|r| r.pid.to_string() != pid));
}

/// Review 4, NEW-3: a yielded parent is no stranger's: admission counts its W (its peak, else
/// its step's pages) for every request but its own family's child; its wait ends at once, and
/// it binds again once the child gave the card up.
fn a_yielded_parent_is_its_familys_alone_and_waits_for_its_room(dir: &Path) {
    let g = gpu("family", 8 * MB);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let (p_path, c_path) = (write_set(dir, "fp", 2), write_set(dir, "fc", 4));
    let mut parent = Agent::granted(ledger.grant(std::process::id()).unwrap());
    let parent_pid = parent.ok(&format!("grant {} {}", g.uuid, g.memory));
    parent.ok(&format!("load fp {} stored ram", p_path.display()));
    parent.ok("request r fp");
    parent.ok("pass r fp 1x1");
    parent.ok("bind fp 0");
    parent.ok("bind fp 1");
    parent.ok("unpin");
    parent.ok("yield r fam");
    ledger.restore(&[varena::Peak::Request { key: "child|1x1".into(), bytes: 6 * MB }]).unwrap();
    assert!(!ledger.fits("child", "1x1", None), "a stranger counts the parent's W");
    assert!(ledger.fits("child", "1x1", Some("fam")), "its family's child does not");
    // The child takes the whole card, pinned.
    let mut child = Agent::granted(ledger.grant(std::process::id()).unwrap());
    child.ok(&format!("grant {} {}", g.uuid, g.memory));
    child.ok(&format!("load fc {} stored ram", c_path.display()));
    child.ok("request c fc");
    child.ok("pass c fc 1x1");
    for block in 0..4 {
        child.ok(&format!("bind fc {block}"));
    }
    assert_eq!(ledger.status().free, 0, "the card is the child's");
    parent.ok("unyield r");
    let _ = parent_pid;
    child.ok("unpin");
    child.ok("end c");
    child.exit();
    for block in 0..2 {
        assert_eq!(parent.ok(&format!("exact fp {block}")), "true");
    }
    parent.ok("unpin");
    parent.ok("end r");
    parent.exit();
}
