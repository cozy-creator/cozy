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
/// `pass <shape>` (its next pass starts), `done <room>` (that pass measured at `room`), `yield
/// <family>` and `resume` (around a wait on a child call) and `close` (the request ends); `run <id> <shape>`
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
    let grant = (opened == "granted").then(|| {
        let table = platform::handles::recv_fd(&control).unwrap();
        Grant { table, row: platform::handles::recv_fd(&control).unwrap() }
    });
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
                    let pass = request.pass(set, "load", "1", Torch::default()).unwrap();
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
                    pass = Some(request.as_ref().unwrap().pass(set, "run", shape, Torch::default()).unwrap());
                    "passing".into()
                }
                ("done", room) => {
                    let room: u64 = room.parse().unwrap();
                    pass.take().unwrap().end(Measured { allocated_peak: room, slack: room / 8, ..Default::default() }, Torch::default());
                    "done".into()
                }
                ("yield", family) => {
                    _yielded = Some(request.as_ref().unwrap().yielding(family));
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
        StandIn::granted(gpu, id, level, None)
    }

    /// Granted at its spawn, as `uid` (the executor identity a root TensorD runs it as) or as
    /// this process.
    fn granted(gpu: &Gpu, id: &GpuId, level: &str, uid: Option<u32>) -> StandIn {
        let standin = StandIn::start(id, level, "granted", uid);
        let grant = gpu.ledger.grant(standin.pid()).unwrap();
        platform::handles::send_fd(&standin.control, &grant.table).unwrap();
        platform::handles::send_fd(&standin.control, &grant.row).unwrap();
        standin
    }

    /// An engine with no TensorD: its memory opens the GPU's table by name.
    fn standalone(id: &GpuId, level: &str) -> StandIn {
        StandIn::start(id, level, "named", None)
    }

    fn start(id: &GpuId, level: &str, opened: &str, uid: Option<u32>) -> StandIn {
        let (control, theirs) = UnixStream::pair().unwrap();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "executor_stand_in", "--ignored", "--nocapture"])
            .env("S5_STAND_IN", format!("{}|{}|{}|{level}|{opened}", id.uuid, id.model, id.memory))
            .stdin(Stdio::from(OwnedFd::from(theirs)))
            .stdout(Stdio::null());
        if let Some(uid) = uid {
            std::os::unix::process::CommandExt::uid(&mut command, uid);
            std::os::unix::process::CommandExt::gid(&mut command, uid);
        }
        let child = command.spawn().unwrap();
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
        Peak::Request { key, .. } | Peak::Compute { key, .. } | Peak::Host { key, .. } | Peak::Pass { key, .. } | Peak::Set { key, .. } => key,
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

    // Granted at spawn, it warms the set onto its GPU from the store (its first request's record
    // carries those fills); binds are hits there.
    let mut first = StandIn::spawn(&gpu, &id, "gpu");
    assert_eq!(first.load(), ("Store,Store,Store,Hit,Hit,Hit".into(), 0, set_bytes));
    assert!(maps(&gpu, first.pid()));

    // An unmeasured run fits nothing beside it.
    assert!(varena::Ledger::admit(&[&gpu.ledger], "unmeasured", PACKAGE, "8", None).unwrap(), "an idle GPU admits it");
    assert_eq!(first.ask("run unmeasured 8"), "passing");
    assert!(running(&gpu));
    assert!(!gpu.ledger.fits(PACKAGE, "8", None));
    assert!(!gpu.ledger.fits("s5/other@1", "4", None));
    while wakes.try_recv().is_ok() {}
    assert_eq!(first.ask(&format!("end {ROOM}")), "ended");
    gpu.ledger.undispatch("unmeasured");
    wakes.recv_timeout(BACKSTOP).expect("a recorded peak signals");
    assert!(gpu.ledger.peaks().iter().any(|peak| key(peak) == format!("{PACKAGE}|8")));

    // Measured, a second request of that shape fits beside a running one.
    assert!(varena::Ledger::admit(&[&gpu.ledger], "measured", PACKAGE, "8", None).unwrap());
    assert_eq!(first.ask("run measured 8"), "passing");
    assert!(gpu.ledger.fits(PACKAGE, "8", None), "two measured requests fit on the card");
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
    assert_eq!(second.load(), ("Kept,Kept,Kept,Hit,Hit,Hit".into(), set_bytes, 0), "imported at load, not copied");
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
/// than half the card) and its compute room beside its weights (what placement keeps free).
fn measure(a: &mut StandIn, request: &dyn Fn(&str) -> String, admit: &dyn Fn(&str) -> bool, gpu: &Gpu) {
    for key in ["measure-room", "measure-set"] {
        let id = request(key);
        assert!(admit(&id), "an idle GPU admits it");
        assert_eq!(a.ask(&format!("run {id} {BIG}")), "passing");
        assert_eq!(a.ask(&format!("end {BIG_ROOM}")), "ended");
        gpu.ledger.undispatch(&id);
    }
    let at = format!("{PACKAGE}|{BIG}");
    let peaks = gpu.ledger.peaks();
    let (mut set, mut compute) = (None, None);
    for peak in &peaks {
        match peak {
            Peak::Request { key, bytes } if *key == at => set = Some(*bytes),
            Peak::Compute { key, bytes } if *key == at => compute = Some(*bytes),
            _ => {}
        }
    }
    assert!(set.is_some_and(|bytes| bytes > gpu.ledger.status().capacity / 2), "{peaks:?}");
    assert!(compute.is_some_and(|bytes| bytes > 0 && Some(bytes) < set), "{peaks:?}");
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
    let admit = |id: &str| memory::admit(&engine, &[&gpu.ledger], id, PACKAGE, BIG, None).unwrap();
    let mut a = StandIn::spawn(&gpu, &id, "gpu");
    let a_pid = a.pid();
    let running = |gpu: &Gpu| gpu.ledger.status().rows.iter().any(|row| row.pid == a_pid && (row.remaining > 0 || row.unmeasured > 0));

    measure(&mut a, &request, &admit, &gpu);
    let second = request("second");
    assert!(admit(&second), "alone, it fits");
    assert_eq!(a.ask(&format!("run {second} {BIG}")), "passing");
    assert!(!gpu.ledger.fits(PACKAGE, BIG, None), "two do not fit");

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

/// A family's child call while its parent waits on it: the parent yields under its family, so
/// the child is admitted though both working sets exceed the card, and runs on the whole card;
/// a stranger is refused while the parent waits, and admitted once the parent is done.
#[test]
fn a_yielded_parents_child_is_admitted_and_a_stranger_waits() {
    const FAMILY: &str = "s5-family";
    let dir = tempfile::tempdir().unwrap();
    let (log, peaks) = (MemoryLog::at(dir.path().join("logs")), Peaks::at(dir.path().join("peaks")));
    let id = gpu_id("family");
    let gpu = Arc::new(Gpu::create("0", &[inventory(&id)], None, HostMode::Dedicated, &peaks, &log).unwrap());
    let engine = Engine::open(&dir.path().join("engine")).unwrap();
    let woken = engine.clone();
    let _events = memory::react(vec![memory::Source::Ledger(gpu.clone())], move |_| woken.notify_activity()).unwrap();
    let request = |key: &str| engine.submit(key, Invocation::default()).unwrap().id;
    let admit = |id: &str, family: Option<&str>| memory::admit(&engine, &[&gpu.ledger], id, PACKAGE, BIG, family).unwrap();
    let mut parent = StandIn::spawn(&gpu, &id, "gpu");
    measure(&mut parent, &request, &|id| admit(id, None), &gpu);
    let call = request("parent");
    assert!(admit(&call, Some(FAMILY)));
    assert_eq!(parent.ask(&format!("open {call}")), "open");
    assert_eq!(parent.ask(&format!("pass {BIG}")), "passing");
    assert_eq!(parent.ask(&format!("done {BIG_ROOM}")), "done");
    assert!(!gpu.ledger.fits(PACKAGE, BIG, Some(FAMILY)), "not yielded yet: its working set counts for its family too");

    // It calls a child and waits on it, yielded.
    assert_eq!(parent.ask(&format!("yield {FAMILY}")), "yielding");
    assert!(!gpu.ledger.fits(PACKAGE, BIG, None), "a stranger is refused while the parent waits");
    let child = request("child");
    assert!(admit(&child, Some(FAMILY)), "its family's child is admitted beside it");
    let stranger = request("stranger");
    std::thread::scope(|scope| {
        let (done, admitted) = mpsc::channel();
        let (stranger, admit) = (&stranger, &admit);
        scope.spawn(move || done.send(admit(stranger, None)).unwrap());
        let mut runner = StandIn::spawn(&gpu, &id, "ram");
        assert_eq!(runner.ask(&format!("run {child} {BIG}")), "passing", "the child's room is made on the whole card");
        assert_eq!(runner.ask(&format!("end {BIG_ROOM}")), "ended");
        gpu.ledger.undispatch(&child);
        engine.notify_activity();
        assert!(admitted.try_recv().is_err(), "the stranger still waits: the parent does");
        assert_eq!(parent.ask("resume"), "resumed", "its room came back once the child ended");
        assert_eq!(parent.ask(&format!("pass {BIG}")), "passing");
        assert_eq!(parent.ask(&format!("done {BIG_ROOM}")), "done");
        assert_eq!(parent.ask("close"), "ended", "the parent completes");
        gpu.ledger.undispatch(&call);
        assert!(admitted.recv_timeout(BACKSTOP).expect("the parent's end wakes it"), "then the stranger is admitted");
    });
    gpu.ledger.undispatch(&stranger);
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
    assert!(!gpu.ledger.fits("s5/other@1", "4", None), "a dispatched, unmeasured request: not idle");
    drop(gpu); // ended without undispatching
    let restarted = Gpu::create("0", &[inventory(&id)], None, HostMode::Dedicated, &peaks, &log).unwrap();
    assert!(restarted.ledger.status().rows.iter().any(|row| row.pid == alive.pid() && row.live), "the same table");
    assert!(restarted.ledger.status().dispatched.is_empty());
    assert!(restarted.ledger.fits("s5/other@1", "4", None), "idle again: it admits");
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

/// The executor identity a root TensorD launches executors as (`config.identity`): `nobody`.
const IDENTITY: u32 = 65534;

/// Review 1, item 8b: under `config.identity` a root TensorD grants its table to executors of
/// another uid: they open it, keep pages in TensorD's custody, a successor of that uid imports
/// them after a kill, and admission works as for TensorD's own uid. Run as root (the hosted run
/// copies this binary where `nobody` may run it).
#[test]
#[ignore = "needs root: a TensorD granting executors of another uid"]
fn a_root_tensord_grants_executors_of_its_identity() {
    // SAFETY: geteuid reads a process attribute.
    assert_eq!(unsafe { libc::geteuid() }, 0, "run as root");
    let dir = tempfile::tempdir().unwrap();
    let (log, peaks) = (MemoryLog::at(dir.path().join("logs")), Peaks::at(dir.path().join("peaks")));
    let id = gpu_id("identity");
    let gpu = Gpu::create("0", &[inventory(&id)], None, HostMode::Dedicated, &peaks, &log).unwrap();
    let mut first = StandIn::granted(&gpu, &id, "gpu", Some(IDENTITY));
    let set_bytes = spec(&id).layout.nbytes;
    assert_eq!(first.load(), ("Store,Store,Store,Hit,Hit,Hit".into(), 0, set_bytes), "warmed on its GPU");
    assert!(maps(&gpu, first.pid()), "its row is on TensorD's table");
    assert!(varena::Ledger::admit(&[&gpu.ledger], "identity", PACKAGE, "8", None).unwrap());
    assert_eq!(first.ask("run identity 8"), "passing");
    assert_eq!(first.ask(&format!("end {ROOM}")), "ended");
    gpu.ledger.undispatch("identity");
    assert!(gpu.ledger.peaks().iter().any(|peak| key(peak) == format!("{PACKAGE}|8")), "its peak reached TensorD's table");
    let pid = first.kill();
    gpu.ledger.reap(pid);
    assert!(orphans(&gpu) >= set_bytes, "its kept pages are in TensorD's custody");
    let mut second = StandIn::granted(&gpu, &id, "ram", Some(IDENTITY));
    assert_eq!(second.load(), ("Kept,Kept,Kept,Hit,Hit,Hit".into(), set_bytes, 0), "imported at load, not copied");
    assert_eq!(orphans(&gpu), 0);
}

/// Hosting (S5 r3): TensorD names the GPU of a 2-GPU group that keeps a placeable set. Beside
/// GPU 0's running request (its pass's room held, the package's measured working set) the set
/// has no room there and is homed on GPU 1; with GPU 0 idle, before and after its request is
/// measured, it stays there (place, measure, place again: one answer). Its bytes are what its
/// load recorded on the group (a loader on GPU 1).
#[test]
fn a_placeable_set_is_homed_on_gpu_1_only_while_gpu_0_has_no_room() {
    let dir = tempfile::tempdir().unwrap();
    let (log, peaks) = (MemoryLog::at(dir.path().join("logs")), Peaks::at(dir.path().join("peaks")));
    let (id0, id1) = (gpu_id("home-0"), gpu_id("home-1"));
    let gpu0 = Gpu::create("0", &[inventory(&id0)], None, HostMode::Dedicated, &peaks, &log).unwrap();
    let gpu1 = Gpu::create("0", &[inventory(&id1)], None, HostMode::Dedicated, &peaks, &log).unwrap();
    let set = format!("{PACKAGE}/c");
    let mut loader = StandIn::spawn(&gpu1, &id1, "ram");
    loader.ask("load");
    let pid = loader.kill();
    gpu1.ledger.reap(pid);
    assert!(gpu1.ledger.peaks().iter().any(|peak| matches!(peak, Peak::Set { key, .. } if *key == set)), "its load recorded its bytes");

    let engine = Engine::open(&dir.path().join("engine")).unwrap();
    let request = |key: &str| engine.submit(key, Invocation::default()).unwrap().id;
    let admit = |id: &str| memory::admit(&engine, &[&gpu0.ledger], id, PACKAGE, BIG, None).unwrap();
    let homes = || memory::homes(&[&gpu0.ledger, &gpu1.ledger], PACKAGE, "", None, &[set.as_str()], &log);
    let mut busy = StandIn::spawn(&gpu0, &id0, "gpu");
    assert_eq!(homes(), [0], "nothing measured runs on GPU 0: it keeps it");
    measure(&mut busy, &request, &admit, &gpu0);
    assert_eq!(homes(), [0], "measuring GPU 0's request changes nothing while it is idle");
    let call = request("busy");
    assert!(admit(&call));
    assert_eq!(busy.ask(&format!("run {call} {BIG}")), "passing");
    assert_eq!(homes(), [1], "no room beside GPU 0's running request: GPU 1 keeps it");
    assert_eq!(busy.ask(&format!("end {BIG_ROOM}")), "ended");
    gpu0.ledger.undispatch(&call);
    assert_eq!(homes(), [0], "with room on GPU 0 it stays there");
    assert_eq!(homes(), [0], "and the same placement answers the same");
}

/// The RAM tier's need is varena's host need over TensorD's GPUs: a call running unmeasured on
/// the host makes it None (the tier fills nothing beside it); once its package and shape were
/// measured, the same call's need is known.
#[test]
fn an_unmeasured_running_call_lets_the_tier_fill_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (log, peaks) = (MemoryLog::at(dir.path().join("logs")), Peaks::at(dir.path().join("peaks")));
    let id = gpu_id("need");
    let gpu = Arc::new(Gpu::create("0", &[inventory(&id)], None, HostMode::Dedicated, &peaks, &log).unwrap());
    let need = memory::need(vec![gpu.clone()]);
    let engine = Engine::open(&dir.path().join("engine")).unwrap();
    let mut a = StandIn::spawn(&gpu, &id, "gpu");
    assert_eq!(need(), Some(0), "nothing runs");
    for (key, measured) in [("first", false), ("again", true)] {
        let call = engine.submit(key, Invocation::default()).unwrap().id;
        assert!(memory::admit(&engine, &[&gpu.ledger], &call, PACKAGE, BIG, None).unwrap());
        assert_eq!(a.ask(&format!("run {call} {BIG}")), "passing");
        assert_eq!(need().is_some(), measured, "{key}: {:?}", need());
        assert_eq!(a.ask(&format!("end {BIG_ROOM}")), "ended");
        gpu.ledger.undispatch(&call);
        assert_eq!(need(), Some(0), "{key} ended");
        assert!(gpu.ledger.peaks().iter().any(|peak| matches!(peak, Peak::Host { key, .. } if *key == format!("{PACKAGE}|{BIG}"))));
    }
}
