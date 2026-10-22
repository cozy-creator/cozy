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
use varena::{Access, Backend, Event, Fill, Fingerprint, GpuId, HostBackend, Level, Measured, Memory, Next, Pass, Physical, Pin, Plan, Request, SetId, SetSpec, Step, Torch, Va};

const MB: u64 = 1 << 20;

/// A tier's need source with no call to keep growth free for.
fn nothing() -> varena::Need {
    Arc::new(|| Some(0))
}

/// A tier's need source the case sets: what running calls keep free (`SHORT`: more than the
/// host has, so a pressure event gives back).
fn need(keep: &Arc<std::sync::atomic::AtomicU64>) -> varena::Need {
    let keep = keep.clone();
    Arc::new(move || Some(keep.load(std::sync::atomic::Ordering::SeqCst)))
}

const SHORT: u64 = u64::MAX / 2;

/// The tables the running case made: removed when it ends, passed or failed.
static TABLES: parking_lot::Mutex<Vec<String>> = parking_lot::Mutex::new(Vec::new());

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("child") => return agent(),
        Some("isolated") => {}
        // The services' names are per user and network namespace: a namespace of its own keeps
        // these processes' custody, tier and ask sockets apart from any other on the machine.
        _ => {
            let exe = std::env::current_exe().unwrap();
            let status = Command::new("unshare").args(["--user", "--net"]).arg(exe).arg("isolated").args(&args).status().expect("unshare (util-linux) runs these tests in a namespace of their own");
            std::process::exit(status.code().unwrap_or(1));
        }
    }
    let filter: Vec<&String> = args.iter().skip(1).filter(|a| !a.starts_with('-')).collect();
    type Case = (&'static str, fn(&Path));
    let cases: [Case; 24] = [
        ("an_idle_holder_answers_an_ask", an_idle_holder_answers_an_ask),
        ("a_killed_holder_leaves_orphans_a_successor_imports", a_killed_holder_leaves_orphans_a_successor_imports),
        ("a_failed_request_frees_everything", a_failed_request_frees_everything),
        ("a_panic_fails_the_request_not_the_process", a_panic_fails_the_request_not_the_process),
        ("exhausted_carries_stats", exhausted_carries_stats),
        ("the_tier_gives_back_in_order", the_tier_gives_back_in_order),
        ("kept_pages_survive_a_holder_that_served", kept_pages_survive_a_holder_that_served),
        ("an_ask_to_a_stopped_victim_ends_when_it_dies", an_ask_to_a_stopped_victim_ends_when_it_dies),
        ("binds_outlive_the_custody_holder", binds_outlive_the_custody_holder),
        ("a_table_with_live_rows_outlives_tensord", a_table_with_live_rows_outlives_tensord),
        ("a_transferred_grant_keeps_its_cap_after_ledger_drop", a_transferred_grant_keeps_its_cap_after_ledger_drop),
        ("a_stale_reader_never_frees_a_newer_row", a_stale_reader_never_frees_a_newer_row),
        ("the_tier_takes_back_a_pinned_layout", the_tier_takes_back_a_pinned_layout),
        ("a_joint_step_runs_a_pass_per_set", a_joint_step_runs_a_pass_per_set),
        ("a_child_runs_while_its_parent_yields", a_child_runs_while_its_parent_yields),
        ("the_tier_waits_on_no_executor_and_says_when_the_host_is_short", the_tier_waits_on_no_executor_and_says_when_the_host_is_short),
        ("a_silent_asker_holds_up_no_answer", a_silent_asker_holds_up_no_answer),
        ("a_reaper_with_a_stale_view_frees_nothing", a_reaper_with_a_stale_view_frees_nothing),
        ("an_oom_counts_its_own_unmaps_after_an_ask", an_oom_counts_its_own_unmaps_after_an_ask),
        ("a_standalone_process_wakes_tensord", a_standalone_process_wakes_tensord),
        ("the_tier_comes_back_after_a_release", the_tier_comes_back_after_a_release),
        ("a_reload_fills_from_the_tier", a_reload_fills_from_the_tier),
        ("held_layouts_outlive_a_tier_restart", held_layouts_outlive_a_tier_restart),
        ("a_yielded_parent_is_its_familys_alone_and_waits_for_its_room", a_yielded_parent_is_its_familys_alone_and_waits_for_its_room),
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

/// The host backend, recording which host ranges are page-locked for the copy engine now.
struct Recording {
    host: HostBackend,
    registered: Arc<parking_lot::Mutex<std::collections::HashSet<u64>>>,
}

impl Backend for Recording {
    fn capacity(&self) -> u64 {
        self.host.capacity()
    }
    fn free(&self) -> varena::Result<u64> {
        self.host.free()
    }
    fn process_used(&self) -> varena::Result<u64> {
        self.host.process_used()
    }
    fn opened_context(&self) -> u64 {
        self.host.opened_context()
    }
    fn granularity(&self) -> u64 {
        self.host.granularity()
    }
    fn reserve(&self, len: u64) -> varena::Result<Va> {
        self.host.reserve(len)
    }
    fn release(&self, va: Va) {
        self.host.release(va)
    }
    fn create(&self, len: u64) -> varena::Result<Physical> {
        self.host.create(len)
    }
    fn destroy(&self, page: Physical) {
        self.host.destroy(page)
    }
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> varena::Result<()> {
        self.host.map(va, offset, page, access)
    }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> varena::Result<()> {
        self.host.protect(va, offset, len, access)
    }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> varena::Result<()> {
        self.host.unmap(va, offset, len)
    }
    fn export(&self, page: &Physical) -> varena::Result<platform::handles::Owned> {
        self.host.export(page)
    }
    fn import(&self, handle: platform::handles::Owned, len: u64) -> varena::Result<Physical> {
        self.host.import(handle, len)
    }
    fn target(&self, va: Va, offset: u64, len: u64) -> tensorfs::io::Dst {
        self.host.target(va, offset, len)
    }
    fn save(&self, va: Va, offset: u64, host: &mut [u8], after: Event) -> varena::Result<()> {
        self.host.save(va, offset, host, after)
    }
    fn fence(&self) -> varena::Result<Event> {
        self.host.fence()
    }
    fn record(&self, stream: usize) -> varena::Result<Event> {
        self.host.record(stream)
    }
    fn query(&self, event: Event) -> varena::Result<bool> {
        self.host.query(event)
    }
    fn elapsed(&self, from: Event, to: Event) -> varena::Result<Option<u64>> {
        self.host.elapsed(from, to)
    }
    fn wait(&self, event: Event, stream: usize) -> varena::Result<()> {
        self.host.wait(event, stream)
    }
    fn synchronize(&self, stream: usize) -> varena::Result<()> {
        self.host.synchronize(stream)
    }
    fn finish(&self, event: Event) -> varena::Result<()> {
        self.host.finish(event)
    }
    fn register_host(&self, ptr: u64, _: u64) -> varena::Result<()> {
        self.registered.lock().insert(ptr);
        Ok(())
    }
    fn unregister_host(&self, ptr: u64) -> varena::Result<()> {
        self.registered.lock().remove(&ptr);
        Ok(())
    }
    fn retire(&self, event: Event) {
        self.host.retire(event)
    }
    fn drain(&self) -> varena::Result<()> {
        self.host.drain()
    }
    fn drain_save(&self) -> varena::Result<()> {
        self.host.drain_save()
    }
}

/// Runs commands over one Memory until stdin ends. Every command answers one line.
fn agent() {
    let mut mem: Option<Memory> = None;
    let mut uuid = String::new();
    let mut sets: HashMap<String, (SetId, String)> = HashMap::new();
    let mut requests: HashMap<String, Request> = HashMap::new();
    let mut passes: HashMap<String, Pass> = HashMap::new();
    let mut pins: Vec<Pin> = vec![];
    let mut yielded: HashMap<String, varena::Yielded> = HashMap::new();
    let registered = Arc::new(parking_lot::Mutex::new(std::collections::HashSet::new()));
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
                let grant = unsafe { varena::Grant { table: platform::handles::Owned::from_raw_fd(3), row: platform::handles::Owned::from_raw_fd(4) } };
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
            ["openrec", id, memory] => {
                uuid = id.to_string();
                let memory: u64 = memory.parse().unwrap();
                let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory };
                mem = Some(Memory::open(g, None, Box::new(Recording { host: HostBackend::new(memory), registered: registered.clone() })).unwrap());
                format!("ok {}", std::process::id())
            }
            ["registered"] => format!("ok {}", registered.lock().len()),
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
                        // The tier's copies of the filled pages too: the cases count classes.
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
                let c = m.cache(owner, Box::new(move |want| {
                    s.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    want
                }));
                m.cached(c, bytes.parse().unwrap());
                "ok".into()
            }
            ["shrunk"] => format!("ok {}", shrunk.load(std::sync::atomic::Ordering::Relaxed)),
            ["pass", req, set, shape] => match requests[*req].pass(sets[*set].0, "fwd", shape, Torch::default()) {
                Ok(p) => {
                    passes.insert(req.to_string(), p);
                    "ok".into()
                }
                Err(e) => format!("err {}", e.code()),
            },
            ["passas", req, set, shape, name] => match requests[*req].pass(sets[*set].0, "fwd", shape, Torch::default()) {
                Ok(p) => {
                    passes.insert(name.to_string(), p);
                    "ok".into()
                }
                Err(e) => format!("err {}", e.code()),
            },
            ["endpassas", name, peak] => {
                if let Some(p) = passes.remove(*name) {
                    p.end(Measured { allocated_peak: peak.parse().unwrap(), ..Measured::default() }, Torch::default());
                }
                "ok".into()
            }
            ["endpass", req] => {
                if let Some(p) = passes.remove(*req) {
                    p.end(Measured::default(), Torch::default());
                }
                "ok".into()
            }
            ["endpass", req, peak] => {
                if let Some(p) = passes.remove(*req) {
                    p.end(Measured { allocated_peak: peak.parse().unwrap(), ..Measured::default() }, Torch::default());
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
            ["oom", req, set, need] => match requests[*req].oom(Some(sets[*set].0), Some(need.parse().unwrap()), None, Torch::default()) {
                Next::Retry => "retry".into(),
                Next::Cut(l) => format!("cut {l}"),
                Next::Park(b) => format!("park {b}"),
                Next::Exhausted(e) => format!("exhausted {} {}", e.stats.request, e.need),
            },
            ["yield", req, family] => {
                yielded.insert(req.to_string(), requests[*req].yielding(family));
                "ok".into()
            }
            ["unyield", req] => match yielded.remove(*req).map(varena::Yielded::end) {
                Some(Err(e)) => format!("err {}", e.code()),
                _ => "ok".into(),
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
                format!("ok leftover={} kept={} tier={} store={} evicted={:?} spec={}", s.leftover, s.moved.kept.bytes, s.moved.tier.bytes, s.moved.store.bytes, s.evicted, s.moved.speculative)
            }
            ["limit"] => format!("ok {}", mem.as_ref().unwrap().torch_limit()),
            ["settle"] => {
                mem.as_ref().unwrap().settle();
                "ok".into()
            }
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
        let mut child = command.arg("child").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
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
    assert!(end.contains("tier=0 store=0"), "{end}");
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
    // frees: park (the host has headroom), and a park of nothing ends it. (The whole card: a
    // need beyond it is Exhausted at once.)
    assert_eq!(a.say("oom r1 f 8388608"), "retry");
    assert!(a.say("oom r1 f 8388608").starts_with("park "));
    a.ok("parked r1 0");
    assert!(a.say("oom r1 f 8388608").starts_with("exhausted r1"));
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
    let keep = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let tier = varena::Tier::serve(varena::Pressure::Events(event), need(&keep)).unwrap().expect("another process serves this user's RAM tier");
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
    // The executor tells the tier its next uses on its own thread.
    tier.await_uses(|u| u.contains(&(digest("l4"), Some(0))) && u.contains(&(digest("l3"), Some(3))));
    let order = vec![digest("l1"), digest("l2"), digest("l3"), digest("l4")];
    assert_eq!(tier.held().into_iter().filter(|d| order.contains(d)).collect::<Vec<_>>(), order);
    // The running calls need more than the host has: each event gives back while that lasts,
    // a rung its users must answer ending its turn.
    keep.store(SHORT, std::sync::atomic::Ordering::SeqCst);
    for n in 1..=4 {
        // SAFETY: an 8-byte write to our eventfd.
        unsafe { libc::write(std::os::fd::AsRawFd::as_raw_fd(&signal), 1u64.to_ne_bytes().as_ptr().cast(), 8) };
        tier.await_releases(n);
        let held: Vec<String> = tier.held().into_iter().filter(|d| d.ends_with(&g.uuid)).collect();
        assert!(order.ends_with(&held) && held.len() as u64 <= 4 - n, "in order after {n} rungs: {held:?}");
    }
    y.ok("end r");
    y.exit();
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
    assert!(end.contains("tier=0 store=0"), "{end}");
    b.exit();
}

/// Review 1, finding 2: an ask to a victim whose ask thread cannot run (stopped) waits on the
/// victim itself: when it dies, the asker goes on, with no timer. Review 2, finding 3: TensorD
/// sees it stalled only while an ask is pending to it, and the ask is in the asker's own row.
/// Finding 11: while the ask is out, the asker claims what it asked and what it unmapped of its
/// own for the same demand.
fn an_ask_to_a_stopped_victim_ends_when_it_dies(dir: &Path) {
    let g = gpu("stop", 10 * MB);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    // TensorD serves the tier too: a stopped executor must not be the one serving it.
    let _tier = varena::Tier::serve(varena::Pressure::Headroom, nothing()).unwrap();
    let (a_path, b_path) = (write_set(dir, "va", 4), write_set(dir, "vb", 2));
    let mut a = Agent::granted(ledger.grant(std::process::id()).unwrap());
    a.ok(&format!("grant {} {}", g.uuid, g.memory));
    a.ok(&format!("load va {} stored gpu", a_path.display()));
    let mut b = Agent::granted(ledger.grant(std::process::id()).unwrap());
    let b_pid = b.ok(&format!("grant {} {}", g.uuid, g.memory));
    b.ok(&format!("load vb {} stored ram", b_path.display()));
    b.ok(&format!("load vn {} stored gpu", write_set(dir, "vn", 1).display()));
    // Earlier events (rows claimed, pages mapped) are not the ask's.
    ledger.clear_events();
    b.ok("request r vb");
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
    let torch = ledger.status().rows.iter().find(|r| r.pid.to_string() == b_pid).map(|r| r.torch);
    assert_eq!(torch, Some(2 * MB + 2 * MB), "b claims its own unmapped page and the asked bytes meanwhile");
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

/// Review 1, finding 10: TensorD's exit keeps a table an executor still uses, and a restarted
/// The grant survives in a real child before its Memory is opened. Dropping the parent's
/// ledger and lowering the named cap must update that very same generation.
fn a_transferred_grant_keeps_its_cap_after_ledger_drop(_dir: &Path) {
    let g = gpu("transferred-cap", 64 * MB);
    let ledger = varena::Ledger::create(&g, Some(12 * MB)).unwrap();
    let mut child = Agent::granted(ledger.grant(std::process::id()).unwrap());
    drop(ledger);
    drop(varena::Grant::open(&g, Some(8 * MB)).unwrap());
    child.ok(&format!("grant {} {}", g.uuid, g.memory));
    assert_eq!(child.ok("limit"), (8 * MB).to_string());
    let mut observer = Agent::spawn();
    observer.ok(&format!("open {} {}", g.uuid, g.memory));
    assert_eq!(observer.ok("limit"), (8 * MB).to_string());
    child.exit();
    assert_eq!(observer.ok("limit"), (8 * MB).to_string(), "the surviving opener retains the cap");
    observer.exit();
}

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

/// Under an injected pressure event, a layout an executor maps and has page-locked is asked
/// back from it first (it unpins and unmaps it), then punched out: its bytes really return
/// before the next layout is touched. The executor goes on from the store.
fn the_tier_takes_back_a_pinned_layout(dir: &Path) {
    use std::os::fd::FromRawFd;
    let g = gpu("pinned", 64 * MB);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    // SAFETY: eventfd(2) with no shared state; the descriptor is ours.
    let event = unsafe { platform::handles::Owned::from_raw_fd(libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK)) };
    let signal = event.try_clone().unwrap();
    let keep = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let tier = varena::Tier::serve(varena::Pressure::Events(event), need(&keep)).unwrap().expect("this namespace's tier");
    let (p1, p2) = (write_set(dir, "p1", 2), write_set(dir, "p2", 2));
    let digest = |n: &str| format!("layout-{n}-{}", g.uuid);
    let mut y = Agent::spawn();
    y.ok(&format!("openrec {} {}", g.uuid, g.memory));
    y.ok(&format!("load p1 {} stored ram", p1.display()));
    y.ok(&format!("load p2 {} stored ram", p2.display()));
    // A device page filled from the tier: the region is mapped and page-locked in y.
    assert_eq!(y.ok("exact p1 0"), "true");
    assert_eq!(y.ok("registered"), "1", "the region a fill read is page-locked");
    let resident = |d: &str| tier.resident().into_iter().find(|(k, _)| k == d).map_or(0, |(_, b)| b);
    // Each layout's file: its ready words' page and its two regions.
    assert!(resident(&digest("p1")) >= 4 * MB && resident(&digest("p2")) >= 4 * MB);
    keep.store(SHORT, std::sync::atomic::Ordering::SeqCst);
    // SAFETY: an 8-byte write to our eventfd.
    unsafe { libc::write(std::os::fd::AsRawFd::as_raw_fd(&signal), 1u64.to_ne_bytes().as_ptr().cast(), 8) };
    tier.await_releases(1);
    assert!(resident(&digest("p1")) < REGION_ALIGN, "the pinned layout's regions returned: {}", resident(&digest("p1")));
    assert!(resident(&digest("p2")) >= 4 * MB, "the next one is untouched");
    assert_eq!(tier.acked(), REGION_ALIGN, "y unpinned the one region it had page-locked, when asked");
    // Review 2, finding 13: y unlocked it before it answered, so before the tier punched it.
    assert_eq!(y.ok("registered"), "0", "unlocked before the punch");
    assert_eq!(y.ok("exact p1 1"), "true", "y goes on from the store");
    y.ok("unpin");
    y.exit();
}

/// A joint step ({a, b}): each set runs its own pass; interleaved binds land in their own
/// passes, and each pass's peak is recorded under its own set.
fn a_joint_step_runs_a_pass_per_set(dir: &Path) {
    let g = gpu("joint", 32 * MB);
    let ledger = varena::Ledger::create(&g, None).unwrap();
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
    let peaks = ledger.peaks();
    let peak = |set: &str| {
        peaks.iter().find_map(|p| match p {
            varena::Peak::Pass { key, room } if key.starts_with(&format!("{set}|")) => Some(room.delta),
            _ => None,
        })
    };
    assert_eq!((peak("ja"), peak("jb")), (Some(3 * MB), Some(5 * MB)), "{peaks:?}");
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
    // Its pages were a later use while it waited: it gave what its child asked for. (Not
    // yielded they are its working set, and its child would have streamed beside them.)
    assert_eq!(held(&parent_pid), 2 * MB, "the yielded parent gave its child what it needed");
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

/// Round 2, finding 2: the tier never waits on an executor. It asks a stopped executor for its
/// layout and goes on serving; the executor's death is its answer. Review 3, N2: the ask is
/// marked in the asked row, so TensorD sees the stopped user stalled, and under pressure with
/// nothing else to give the tier says the host is short meanwhile.
fn the_tier_waits_on_no_executor_and_says_when_the_host_is_short(dir: &Path) {
    use std::os::fd::FromRawFd;
    let g = gpu("short", 64 * MB);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    // SAFETY: eventfd(2) with no shared state; the descriptor is ours.
    let event = unsafe { platform::handles::Owned::from_raw_fd(libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK)) };
    let signal = event.try_clone().unwrap();
    let pressure = || {
        // SAFETY: an 8-byte write to our eventfd.
        unsafe { libc::write(std::os::fd::AsRawFd::as_raw_fd(&signal), 1u64.to_ne_bytes().as_ptr().cast(), 8) };
    };
    let keep = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let tier = varena::Tier::serve(varena::Pressure::Events(event), need(&keep)).unwrap().expect("this namespace's tier");
    let (s1, s2) = (write_set(dir, "s1", 2), write_set(dir, "s2", 2));
    let digest = |n: &str| format!("layout-{n}-{}", g.uuid);
    let mut y = Agent::spawn();
    y.ok(&format!("open {} {}", g.uuid, g.memory));
    y.ok(&format!("load s1 {} stored ram", s1.display()));
    assert_eq!(y.ok("exact s1 0"), "true");
    let pid = y.child.id() as i32;
    let mut status = 0;
    // SAFETY: a signal to our own child, then waiting until it has stopped.
    unsafe {
        libc::kill(pid, libc::SIGSTOP);
        libc::waitpid(pid, &mut status, libc::WUNTRACED);
    }
    keep.store(SHORT, std::sync::atomic::Ordering::SeqCst);
    pressure();
    while ledger.stalled() != vec![pid as u32] {
        ledger.await_event().unwrap();
    }
    assert!(tier.held().contains(&digest("s1")), "nothing punched before its user answered");
    // Nothing else to give while s1 waits on its stalled user: the host is short.
    pressure();
    tier.await_short().unwrap();
    // The tier serves another executor meanwhile (the host has room again for its fill).
    keep.store(0, std::sync::atomic::Ordering::SeqCst);
    let mut z = Agent::spawn();
    z.ok(&format!("open {} {}", g.uuid, g.memory));
    z.ok(&format!("load s2 {} stored ram", s2.display()));
    keep.store(SHORT, std::sync::atomic::Ordering::SeqCst);
    // TensorD ends the stalled user: its death is its answer.
    y.child.kill().unwrap();
    y.child.wait().unwrap();
    tier.await_releases(1);
    assert!(!tier.held().contains(&digest("s1")), "released once its user died");
    pressure();
    tier.await_releases(2);
    assert!(tier.held().iter().all(|d| !d.ends_with(&g.uuid)), "{:?}", tier.held());
    z.exit();
}

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

/// Round 2, finding 17 (review 1, finding 5, through Memory): an OOM whose own N unmap and
/// another row's answer together cover the need retries with both counted, evicting no D page.
fn an_oom_counts_its_own_unmaps_after_an_ask(dir: &Path) {
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
    b.ok("pass r ow 1x1");
    b.ok("bind ow 0");
    b.ok("bind ow 1");
    // The card is full: b's own N page and a's N page together are the need.
    assert_eq!(b.say(&format!("oom r ow {}", 2 * REGION_ALIGN)), "retry");
    let row = |pid: &str| ledger.status().rows.into_iter().find(|r| r.pid.to_string() == pid).unwrap();
    assert_eq!(row(&b_pid).classes[varena::table::Class::D.index()], 2 * REGION_ALIGN, "no D page went");
    assert_eq!(row(&a.child.id().to_string()).classes.iter().sum::<u64>(), 0, "a gave its page");
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

/// Round 2, finding 8: after the tier took a layout back, the set opens it again at its next
/// pass start, and its filled pages go to it again.
fn the_tier_comes_back_after_a_release(dir: &Path) {
    use std::os::fd::FromRawFd;
    let g = gpu("back", 64 * MB);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    // SAFETY: eventfd(2) with no shared state; the descriptor is ours.
    let event = unsafe { platform::handles::Owned::from_raw_fd(libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK)) };
    let signal = event.try_clone().unwrap();
    let keep = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let tier = varena::Tier::serve(varena::Pressure::Events(event), need(&keep)).unwrap().expect("this namespace's tier");
    let path = write_set(dir, "tb", 2);
    let digest = format!("layout-tb-{}", g.uuid);
    let mut y = Agent::spawn();
    y.ok(&format!("open {} {}", g.uuid, g.memory));
    y.ok(&format!("load tb {} stored ram", path.display()));
    assert_eq!(y.ok("exact tb 0"), "true");
    keep.store(SHORT, std::sync::atomic::Ordering::SeqCst);
    // SAFETY: an 8-byte write to our eventfd.
    unsafe { libc::write(std::os::fd::AsRawFd::as_raw_fd(&signal), 1u64.to_ne_bytes().as_ptr().cast(), 8) };
    tier.await_releases(1);
    assert!(!tier.held().contains(&digest));
    keep.store(0, std::sync::atomic::Ordering::SeqCst);
    y.ok("unpin");
    y.ok("request r tb");
    y.ok("pass r tb 1x1");
    assert_eq!(y.ok("exact tb 1"), "true");
    y.ok("settle");
    assert!(tier.held().contains(&digest), "the filled page went to the tier again");
    y.ok("unpin");
    y.ok("end r");
    y.exit();
}

/// The vertical slice's reload: an executor that ended (its kept pages gone with custody, here
/// none kept for it) leaves its layout in the tier; the next one's load fills from there.
fn a_reload_fills_from_the_tier(dir: &Path) {
    let g = gpu("reload", 64 * MB);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let tier = varena::Tier::serve(varena::Pressure::Headroom, nothing()).unwrap().expect("this namespace's tier");
    let path = write_set(dir, "rl", 2);
    let mut a = Agent::spawn();
    a.ok(&format!("open {} {}", g.uuid, g.memory));
    a.ok(&format!("load rl {} undeclared ram", path.display()));
    a.exit();
    let mut b = Agent::spawn();
    b.ok(&format!("open {} {}", g.uuid, g.memory));
    b.ok(&format!("load rl {} undeclared gpu", path.display()));
    b.ok("request r rl");
    assert_eq!(b.ok("trace r"), "Tier,Tier", "the load filled from the tier, not the store");
    b.exit();
    drop(tier);
}

/// The vertical slice's TensorD restart: the tier's server goes and comes back; the executors
/// that hold its layouts hand them (their files, ready regions and all) to the new one, so the
/// next load still fills from the tier.
fn held_layouts_outlive_a_tier_restart(dir: &Path) {
    let g = gpu("tierback", 64 * MB);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let first = varena::Tier::serve(varena::Pressure::Headroom, nothing()).unwrap().expect("this namespace's tier");
    let (p, q) = (write_set(dir, "hp", 2), write_set(dir, "hq", 1));
    let digest = format!("layout-hp-{}", g.uuid);
    let mut c = Agent::spawn();
    c.ok(&format!("open {} {}", g.uuid, g.memory));
    c.ok(&format!("load hp {} undeclared ram", p.display()));
    // The tier's server restarts.
    drop(first);
    let second = varena::Tier::serve(varena::Pressure::Headroom, nothing()).unwrap().expect("the tier again");
    // c's next tier call finds the new tier, and hands it what it holds.
    c.ok(&format!("load hq {} undeclared ram", q.display()));
    second.await_uses(|u| u.iter().any(|(d, _)| *d == digest));
    let mut d = Agent::spawn();
    d.ok(&format!("open {} {}", g.uuid, g.memory));
    d.ok(&format!("load hp {} undeclared gpu", p.display()));
    d.ok("request r hp");
    assert_eq!(d.ok("trace r"), "Tier,Tier", "filled from the layout c handed over");
    d.exit();
    c.exit();
}

/// Review 4, NEW-3: a yielded parent is no stranger's: admission counts its remaining W for
/// every request but its own family's child; and its wait's end waits for its room (the
/// table's changes wake it), never Exhausted while the card can hold it.
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
    // The wait ends while the card is the child's: the parent waits for its room.
    parent.send("unyield r");
    while !ledger.status().rows.iter().any(|r| r.pid.to_string() == parent_pid && r.waiting) {
        ledger.await_event().unwrap();
    }
    child.ok("unpin");
    child.ok("end c");
    child.exit();
    assert_eq!(parent.reply(), "ok", "its room came back once the child gave the card up");
    for block in 0..2 {
        assert_eq!(parent.ok(&format!("exact fp {block}")), "true");
    }
    parent.ok("unpin");
    parent.ok("end r");
    parent.exit();
}
