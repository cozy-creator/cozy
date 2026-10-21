//! TensorD on varena's ledger (design varena-slim §2–§4), with real processes as executors: two
//! stand-ins (this binary's `executor_stand_in`, run as children) open varena's `Memory` over the
//! host-page backend with the grant TensorD's ledger gave them. TensorD grants, reaps, admits
//! and persists peaks; the stand-ins load, run and are killed.
#![cfg(unix)]
use std::{
    io::{BufRead, BufReader, Write},
    os::{
        fd::{FromRawFd, OwnedFd},
        unix::net::UnixStream,
    },
    process::{Child, Command, Stdio},
    sync::{mpsc, Arc},
    time::Duration,
};
use platform::contain::host_pressure::HostMode;
use tensord::{
    execution::Engine,
    journal::Invocation,
    machine::receipt,
    memory::{self, Gpu, MemoryLog, Peaks},
};
use varena::{
    layout::Layout, Fill, Fingerprint, GpuId, Grant, HostBackend, Level, Measured, Memory, PartId, Peak, Plan, SetSpec, Step, Torch,
};

const MIB: u64 = 1 << 20;
/// Three regions of one part each: a set the host backend maps page by page.
const PARTS: [&str; 3] = ["c/w0", "c/w1", "c/w2"];
const PACKAGE: &str = "s5/package@1";
/// The room a pass at shape `8` measures: two such requests fit on the card.
const ROOM: u64 = 6 * MIB;
/// A shape whose pass measures more than half the card beside its weights: two never fit.
const BIG: &str = "64";
const BIG_ROOM: u64 = 40 * MIB;

/// A failure backstop only: every wait below ends on the event it waits for.
const BACKSTOP: Duration = Duration::from_secs(60);

fn gpu_id(tag: &str) -> GpuId {
    GpuId { uuid: format!("GPU-s5-{tag}-{}", uuid::Uuid::new_v4()), model: "S5 Host Pages".into(), memory: 64 * MIB }
}

fn inventory(id: &GpuId) -> receipt::Gpu {
    serde_json::from_value(serde_json::json!({"device_index": 0, "device_name": id.model, "device_uuid": id.uuid,
        "driver_version": "", "memory_bytes": id.memory, "pci_bus_id": ""}))
    .unwrap()
}

/// The stand-ins' one set: the same bytes, so the same fingerprint, in every process.
fn spec(gpu: &GpuId) -> SetSpec {
    let spans: Vec<tensorfs::io::Span> = PARTS
        .iter()
        .enumerate()
        .map(|(i, tensor)| tensorfs::io::Span {
            what: format!("{tensor}#data"),
            len: MIB,
            source: tensorfs::io::ItemSource::Inline(Arc::from(vec![i as u8 + 1; MIB as usize])),
        })
        .collect();
    let regions: Vec<Vec<String>> = PARTS.iter().map(|p| vec![p.to_string()]).collect();
    let layout = Layout::build(&spans, &regions).unwrap();
    let parts: Vec<PartId> = PARTS.iter().map(|p| PartId { name: format!("{p}#data"), encoding: "raw".into(), dtype: "u8".into() }).collect();
    let fingerprint = Fingerprint::new(&layout, &parts, Fill::Stored, PACKAGE, "0/1", &gpu.uuid);
    SetSpec {
        name: format!("{PACKAGE}/c"),
        layout: Arc::new(layout),
        source: Arc::new(tensorfs::io::source::Source::files(vec![], false, 0)),
        fingerprint,
    }
}

fn plan() -> Plan {
    Plan { steps: vec![Step { sets: vec![format!("{PACKAGE}/c")], passes: 2 }] }
}

/// An executor stand-in: run by `StandIn::spawn` with its control socket at stdin. It opens its
/// memory with the grant that arrives first and loads the set at its level (`gpu`:
/// warmed, `ram`: in the RAM tier only), then answers one line per command: `load` (a request
/// binding every block: `<origins> <kept bytes> <store bytes>`), `open <id>` (a request),
/// `pass <shape>` (its next pass starts), `done <room>` (that pass measured at `room`), `yield`
/// and `resume` (around a wait on a child call) and `close` (the request ends); `run <id> <shape>`
/// is open and pass, `end <room>` done and close.
#[test]
#[ignore = "memory_ledger's tests run it as an executor"]
fn executor_stand_in() {
    let Ok(gpu) = std::env::var("S5_STAND_IN") else { return };
    let [uuid, model, bytes, level, opened] = gpu.split('|').collect::<Vec<_>>()[..] else { panic!("{gpu}") };
    let gpu = GpuId { uuid: uuid.into(), model: model.into(), memory: bytes.parse().unwrap() };
    // SAFETY: the spawning test installs its private socket at the child's stdin.
    let control = unsafe { UnixStream::from_raw_fd(0) };
    // Granted by TensorD, or a standalone engine that opens the GPU's table by name.
    let grant = (opened == "granted").then(|| Grant { table: platform::handles::recv_fd(&control).unwrap() });
    let memory = Memory::open(gpu.clone(), grant, Box::new(HostBackend::new(gpu.memory))).unwrap();
    let set = memory.load(spec(&gpu), if level == "gpu" { Level::Gpu } else { Level::Ram }).unwrap();
    let (mut out, lines) = (control.try_clone().unwrap(), BufReader::new(control).lines());
    let (mut request, mut pass, mut _yielded) = (None, None, None);
    for line in lines.map_while(Result::ok) {
        let commands: Vec<(String, String)> = match line.split_once(' ').unwrap_or((&line, "")) {
            ("run", args) => {
                let (id, shape) = args.split_once(' ').unwrap();
                vec![("open".into(), id.into()), ("pass".into(), shape.into())]
            }
            ("end", room) => vec![("done".into(), room.into()), ("close".into(), String::new())],
            (command, args) => vec![(command.into(), args.into())],
        };
        let mut reply = String::new();
        for (command, args) in commands {
            reply = match (command.as_str(), args.as_str()) {
                ("load", _) => {
                    let request = memory.request(&format!("load-{}", std::process::id()), plan()).unwrap();
                    let pass = request.pass(set, "1", Torch::default()).unwrap();
                    let pins: Vec<_> = (0..PARTS.len() as u32).map(|block| memory.bind(set, block, 0).unwrap()).collect();
                    drop(pins);
                    pass.end(Measured::default(), Torch::default());
                    let stats = request.end();
                    let origins: Vec<String> = stats.trace.iter().map(|bind| format!("{:?}", bind.origin)).collect();
                    format!("{} {} {}", origins.join(","), stats.moved.kept.bytes, stats.moved.store.bytes)
                }
                ("open", id) => {
                    request = Some(memory.request(id, plan()).unwrap());
                    "open".into()
                }
                ("pass", shape) => {
                    pass = Some(request.as_ref().unwrap().pass(set, shape, Torch::default()).unwrap());
                    "passing".into()
                }
                ("done", room) => {
                    let room: u64 = room.parse().unwrap();
                    pass.take().unwrap().end(Measured { allocated_peak: room, slack: room / 8, ..Default::default() }, Torch::default());
                    "done".into()
                }
                ("yield", _) => {
                    _yielded = Some(request.as_ref().unwrap().yielding());
                    "yielding".into()
                }
                ("resume", _) => {
                    _yielded = None;
                    "resumed".into()
                }
                ("close", _) => {
                    request.take().unwrap().end();
                    "ended".into()
                }
                (other, _) => panic!("unknown command {other}"),
            };
        }
        writeln!(out, "{reply}").unwrap();
    }
}

/// A stand-in executor, granted `gpu`'s ledger at its spawn (or, standalone, opening it by name).
struct StandIn {
    child: Child,
    control: UnixStream,
    replies: BufReader<UnixStream>,
}

impl StandIn {
    fn spawn(gpu: &Gpu, id: &GpuId, level: &str) -> StandIn {
        let standin = StandIn::start(id, level, "granted");
        let grant = gpu.ledger.grant(standin.pid()).unwrap();
        platform::handles::send_fd(&standin.control, &grant.table).unwrap();
        standin
    }

    /// An engine with no TensorD: its memory opens the GPU's table by name.
    fn standalone(id: &GpuId, level: &str) -> StandIn {
        StandIn::start(id, level, "named")
    }

    fn start(id: &GpuId, level: &str, opened: &str) -> StandIn {
        let (control, theirs) = UnixStream::pair().unwrap();
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "executor_stand_in", "--ignored", "--nocapture"])
            .env("S5_STAND_IN", format!("{}|{}|{}|{level}|{opened}", id.uuid, id.model, id.memory))
            .stdin(Stdio::from(OwnedFd::from(theirs)))
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let replies = BufReader::new(control.try_clone().unwrap());
        StandIn { child, control, replies }
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn ask(&mut self, command: &str) -> String {
        writeln!(self.control, "{command}").unwrap();
        let mut reply = String::new();
        self.replies.read_line(&mut reply).unwrap();
        assert!(!reply.is_empty(), "the stand-in ended answering {command}");
        reply.trim().to_string()
    }

    /// Where its load request's binds came from, and the bytes it imported and read.
    fn load(&mut self) -> (String, u64, u64) {
        let reply = self.ask("load");
        let [origins, kept, store] = reply.split(' ').collect::<Vec<_>>()[..] else { panic!("{reply}") };
        (origins.into(), kept.parse().unwrap(), store.parse().unwrap())
    }

    /// `kill -9`, then waitpid.
    fn kill(mut self) -> u32 {
        let pid = self.pid();
        // SAFETY: a signal to this test's own child, not yet reaped.
        assert_eq!(unsafe { libc::kill(pid as i32, libc::SIGKILL) }, 0);
        self.child.wait().unwrap();
        pid
    }
}

impl Drop for StandIn {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A peak record's key.
fn key(peak: &Peak) -> &str {
    match peak {
        Peak::Request { key, .. } | Peak::Pass { key, .. } => key,
    }
}

/// Whether `pid`'s live row maps pages (classes S, N, D, W, P).
fn maps(gpu: &Gpu, pid: u32) -> bool {
    gpu.ledger.status().rows.iter().any(|row| row.pid == pid && row.live && row.classes.iter().skip(2).any(|bytes| *bytes > 0))
}

/// Whether a request runs on the GPU, by any row.
fn running(gpu: &Gpu) -> bool {
    gpu.ledger.status().rows.iter().any(|row| row.live && (row.remaining > 0 || row.unmeasured > 0))
}

/// Bytes the table holds as orphans (class O): kept pages no live process maps.
fn orphans(gpu: &Gpu) -> u64 {
    gpu.ledger.status().rows.iter().map(|row| row.classes[1]).sum()
}

#[test]
fn a_killed_executors_kept_pages_are_the_next_ones_and_fits_follows_measured_peaks() {
    let dir = tempfile::tempdir().unwrap();
    let (log, peaks) = (MemoryLog::at(dir.path().join("logs")), Peaks::at(dir.path().join("peaks")));
    let id = gpu_id("a");
    let gpu = Arc::new(Gpu::create("0", &[inventory(&id)], None, HostMode::Dedicated, &peaks, &log).expect("a ledger for an inventoried GPU"));
    let (woke, wakes) = mpsc::channel();
    let _events = memory::react(vec![memory::Source::Ledger(gpu.clone())], move |_| {
        let _ = woke.send(());
    })
    .unwrap();
    let set_bytes = spec(&id).layout.nbytes;

    // Granted at spawn, it warms the set onto its GPU: binds are hits there.
    let mut first = StandIn::spawn(&gpu, &id, "gpu");
    assert_eq!(first.load(), ("Hit,Hit,Hit".into(), 0, 0));
    assert!(maps(&gpu, first.pid()));

    // An unmeasured run fits nothing beside it.
    assert!(varena::Ledger::admit(&[&gpu.ledger], "unmeasured", PACKAGE, "8").unwrap(), "an idle GPU admits it");
    assert_eq!(first.ask("run unmeasured 8"), "passing");
    assert!(running(&gpu));
    assert!(!gpu.ledger.fits(PACKAGE, "8"));
    assert!(!gpu.ledger.fits("s5/other@1", "4"));
    while wakes.try_recv().is_ok() {}
    assert_eq!(first.ask(&format!("end {ROOM}")), "ended");
    gpu.ledger.undispatch("unmeasured");
    wakes.recv_timeout(BACKSTOP).expect("a recorded peak signals");
    assert!(gpu.ledger.peaks().iter().any(|peak| key(peak) == format!("{PACKAGE}|8")));

    // Measured, a second request of that shape fits beside a running one.
    assert!(varena::Ledger::admit(&[&gpu.ledger], "measured", PACKAGE, "8").unwrap());
    assert_eq!(first.ask("run measured 8"), "passing");
    assert!(gpu.ledger.fits(PACKAGE, "8"), "two measured requests fit on the card");
    assert_eq!(first.ask(&format!("end {ROOM}")), "ended");
    gpu.ledger.undispatch("measured");

    // kill -9, waitpid, reap: what it kept is the table's orphans, and the next executor of the
    // same weights imports them instead of copying.
    let pid = first.kill();
    while wakes.try_recv().is_ok() {}
    gpu.ledger.reap(pid);
    wakes.recv_timeout(BACKSTOP).expect("a reaped row signals");
    assert!(!gpu.ledger.status().rows.iter().any(|row| row.pid == pid && row.live));
    assert!(orphans(&gpu) >= set_bytes, "its kept pages are orphans");
    // One load path: its load imports what custody keeps, at any level; its binds are hits.
    let mut second = StandIn::spawn(&gpu, &id, "ram");
    assert_eq!(second.load(), ("Hit,Hit,Hit".into(), 0, 0), "imported at load, not copied");
    assert_eq!(orphans(&gpu), 0, "every orphan imported by fingerprint");
    assert!(maps(&gpu, second.pid()));

    // Peaks persist per GPU model, whole: another GPU of the model, and every restart after,
    // starts with exactly what was measured (a room's start, delta, slack, context and sites).
    peaks.save(&id.model, std::iter::once(&gpu.ledger));
    let sorted = |mut peaks: Vec<Peak>| {
        peaks.sort_by(|a, b| key(a).cmp(key(b)));
        peaks
    };
    let measured = sorted(gpu.ledger.peaks());
    assert!(measured.iter().any(|peak| matches!(peak, Peak::Pass { room, .. } if room.slack > 0)), "{measured:?}");
    for restart in ["b", "c"] {
        let other = gpu_id(restart);
        let restored = Gpu::create("0", &[inventory(&other)], None, HostMode::Dedicated, &Peaks::at(dir.path().join("peaks")), &log).unwrap();
        assert_eq!(sorted(restored.ledger.peaks()), measured, "restart {restart}: restored exactly");
        Peaks::at(dir.path().join("peaks")).save(&id.model, std::iter::once(&restored.ledger));
    }
    drop(second);
}

/// Two requests at `BIG`: the first measures its pass's room, the second its working set (more
/// than half the card).
fn measure(a: &mut StandIn, request: &dyn Fn(&str) -> String, admit: &dyn Fn(&str) -> bool, gpu: &Gpu) {
    for key in ["measure-room", "measure-set"] {
        let id = request(key);
        assert!(admit(&id), "an idle GPU admits it");
        assert_eq!(a.ask(&format!("run {id} {BIG}")), "passing");
        assert_eq!(a.ask(&format!("end {BIG_ROOM}")), "ended");
        gpu.ledger.undispatch(&id);
    }
    let set = gpu.ledger.peaks().into_iter().find_map(|peak| match peak {
        Peak::Request { key, bytes } if key == format!("{PACKAGE}|{BIG}") => Some(bytes),
        _ => None,
    });
    assert!(set.is_some_and(|bytes| bytes > gpu.ledger.status().capacity / 2), "{set:?}");
}

/// A request whose measured working set does not fit beside a running one waits for it to end,
/// woken by the ledger's events, and never runs beside it; a cancel ends a wait.
#[test]
fn a_request_that_does_not_fit_waits_until_the_running_one_ends() {
    let dir = tempfile::tempdir().unwrap();
    let (log, peaks) = (MemoryLog::at(dir.path().join("logs")), Peaks::at(dir.path().join("peaks")));
    let id = gpu_id("wait");
    let gpu = Arc::new(Gpu::create("0", &[inventory(&id)], None, HostMode::Dedicated, &peaks, &log).unwrap());
    let engine = Engine::open(&dir.path().join("engine")).unwrap();
    let woken = engine.clone();
    let _events = memory::react(vec![memory::Source::Ledger(gpu.clone())], move |_| woken.notify_activity()).unwrap();
    let request = |key: &str| engine.submit(key, Invocation::default()).unwrap().id;
    let admit = |id: &str| memory::admit(&engine, &[&gpu.ledger], id, PACKAGE, BIG).unwrap();
    let mut a = StandIn::spawn(&gpu, &id, "gpu");
    let a_pid = a.pid();
    let running = |gpu: &Gpu| gpu.ledger.status().rows.iter().any(|row| row.pid == a_pid && (row.remaining > 0 || row.unmeasured > 0));

    measure(&mut a, &request, &admit, &gpu);
    let second = request("second");
    assert!(admit(&second), "alone, it fits");
    assert_eq!(a.ask(&format!("run {second} {BIG}")), "passing");
    assert!(!gpu.ledger.fits(PACKAGE, BIG), "two do not fit");

    let third = request("third");
    std::thread::scope(|scope| {
        let (done, admitted) = mpsc::channel();
        let (gpu, third, admit) = (&gpu, &third, &admit);
        scope.spawn(move || {
            let ok = admit(third);
            done.send((ok, running(gpu))).unwrap();
        });
        engine.notify_activity(); // a wake that frees nothing
        assert!(admitted.try_recv().is_err(), "it waits while the second runs");
        assert_eq!(a.ask(&format!("end {BIG_ROOM}")), "ended");
        let (ok, overlapped) = admitted.recv_timeout(BACKSTOP).expect("the second's end wakes it");
        assert!(ok && !overlapped, "admitted once the second ended, never beside it");
    });
    gpu.ledger.undispatch(&third);

    // Waiting again behind a running one, it leaves when canceled.
    let fourth = request("fourth");
    assert!(admit(&fourth));
    assert_eq!(a.ask(&format!("run {fourth} {BIG}")), "passing");
    let fifth = request("fifth");
    std::thread::scope(|scope| {
        let (fifth, admit) = (&fifth, &admit);
        let waiting = scope.spawn(move || admit(fifth));
        engine.cancel(fifth, "test").unwrap();
        assert!(!waiting.join().unwrap(), "a cancel ends the wait, not admitted");
    });
    assert!(!gpu.ledger.status().dispatched.iter().any(|(request, _)| *request == fifth));
    assert_eq!(a.ask(&format!("end {BIG_ROOM}")), "ended");
}

/// A child call on its family's own lane is its family's: its parent, blocked on it between
/// passes, still counts its remaining working set, so `fits` refuses the child (waiting would
/// never end). Dispatched under its own key, the child runs and the parent completes.
#[test]
fn a_child_on_its_familys_lane_runs_while_its_parent_waits_on_it() {
    let dir = tempfile::tempdir().unwrap();
    let (log, peaks) = (MemoryLog::at(dir.path().join("logs")), Peaks::at(dir.path().join("peaks")));
    let id = gpu_id("family");
    let gpu = Arc::new(Gpu::create("0", &[inventory(&id)], None, HostMode::Dedicated, &peaks, &log).unwrap());
    let engine = Engine::open(&dir.path().join("engine")).unwrap();
    let request = |key: &str| engine.submit(key, Invocation::default()).unwrap().id;
    let admit = |id: &str| memory::admit(&engine, &[&gpu.ledger], id, PACKAGE, BIG).unwrap();
    let mut parent = StandIn::spawn(&gpu, &id, "gpu");
    measure(&mut parent, &request, &admit, &gpu);
    let call = request("parent");
    assert!(admit(&call));
    assert_eq!(parent.ask(&format!("open {call}")), "open");
    assert_eq!(parent.ask(&format!("pass {BIG}")), "passing");
    assert_eq!(parent.ask(&format!("done {BIG_ROOM}")), "done");

    // Between its two passes it calls a child on its own lane: beside its working set the child
    // does not fit, so TensorD admits it under the family (waiting would never end); the parent
    // yields while it waits, and varena makes the child's room.
    const CHILD: &str = "s5/child@1";
    assert!(!gpu.ledger.fits(CHILD, BIG) && !gpu.ledger.fits(PACKAGE, BIG));
    assert_eq!(parent.ask("yield"), "yielding");
    let mut child = StandIn::spawn(&gpu, &id, "ram");
    memory::dispatch(&[&gpu.ledger], "child", CHILD, BIG);
    assert_eq!(child.ask(&format!("run child {BIG}")), "passing");
    assert_eq!(child.ask(&format!("end {BIG_ROOM}")), "ended");
    gpu.ledger.undispatch("child");
    assert!(gpu.ledger.peaks().iter().any(|peak| key(peak) == format!("{CHILD}|{BIG}")), "its own key's peak");
    assert_eq!(parent.ask("resume"), "resumed");

    assert_eq!(parent.ask(&format!("pass {BIG}")), "passing");
    assert_eq!(parent.ask(&format!("done {BIG_ROOM}")), "done");
    assert_eq!(parent.ask("close"), "ended", "the parent completes");
    gpu.ledger.undispatch(&call);
}

/// The events thread wakes on a ledger's event, by the GPU's named wake, and ends with its pool.
#[test]
fn the_events_thread_wakes_on_a_ledgers_event_and_ends_with_its_pool() {
    let dir = tempfile::tempdir().unwrap();
    let (log, peaks) = (MemoryLog::at(dir.path().join("logs")), Peaks::at(dir.path().join("peaks")));
    let id = gpu_id("events");
    let gpu = Arc::new(Gpu::create("0", &[inventory(&id)], None, HostMode::Shared, &peaks, &log).unwrap());
    let (woke, wakes) = mpsc::channel();
    let events = memory::react(vec![memory::Source::Ledger(gpu.clone())], move |what| {
        let _ = woke.send(what);
    })
    .unwrap();
    gpu.ledger.fault(); // its kept pages go: the ledger signals
    assert_eq!(wakes.recv_timeout(BACKSTOP).unwrap(), memory::Woke::Ledger);
    drop(events);
    assert!(wakes.recv().is_err(), "the thread ended with its pool");
}

/// Review 1, item 1: a TensorD that ended without undispatching (a crash) leaves `dispatch/`
/// entries in the table, which an executor's live row keeps; the next TensorD's ledger drops
/// them, so the GPU reads idle again and admits.
#[test]
fn a_restarted_tensord_drops_the_dispatches_its_predecessor_left() {
    let dir = tempfile::tempdir().unwrap();
    let (log, peaks) = (MemoryLog::at(dir.path().join("logs")), Peaks::at(dir.path().join("peaks")));
    let id = gpu_id("stale");
    let gpu = Gpu::create("0", &[inventory(&id)], None, HostMode::Dedicated, &peaks, &log).unwrap();
    let mut alive = StandIn::spawn(&gpu, &id, "ram");
    alive.ask("load"); // its live row keeps the table past its TensorD
    gpu.ledger.dispatch("left-behind", PACKAGE, BIG).unwrap();
    assert!(!gpu.ledger.fits("s5/other@1", "4"), "a dispatched, unmeasured request: not idle");
    drop(gpu); // ended without undispatching
    let restarted = Gpu::create("0", &[inventory(&id)], None, HostMode::Dedicated, &peaks, &log).unwrap();
    assert!(restarted.ledger.status().rows.iter().any(|row| row.pid == alive.pid() && row.live), "the same table");
    assert!(restarted.ledger.status().dispatched.is_empty());
    assert!(restarted.ledger.fits("s5/other@1", "4"), "idle again: it admits");
}

/// Review 1, item 6: a standalone engine (no TensorD grant; its memory opens the GPU's table by
/// name) signals the GPU's wake too, so TensorD sees what it frees and records.
#[test]
fn tensord_wakes_on_a_standalone_engines_events() {
    let dir = tempfile::tempdir().unwrap();
    let (log, peaks) = (MemoryLog::at(dir.path().join("logs")), Peaks::at(dir.path().join("peaks")));
    let id = gpu_id("standalone");
    let gpu = Arc::new(Gpu::create("0", &[inventory(&id)], None, HostMode::Shared, &peaks, &log).unwrap());
    let (woke, wakes) = mpsc::channel();
    let _events = memory::react(vec![memory::Source::Ledger(gpu.clone())], move |what| {
        let _ = woke.send(what);
    })
    .unwrap();
    let mut engine = StandIn::standalone(&id, "ram");
    engine.ask("load");
    while wakes.try_recv().is_ok() {}
    assert_eq!(engine.ask(&format!("run standalone {BIG}")), "passing");
    assert_eq!(engine.ask(&format!("end {BIG_ROOM}")), "ended");
    assert_eq!(wakes.recv_timeout(BACKSTOP).expect("its peak and its freed room wake TensorD"), memory::Woke::Ledger);
    assert!(gpu.ledger.status().rows.iter().any(|row| row.pid == engine.pid() && row.live), "its row is on TensorD's table");
}
