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
use tensord::{
    machine::receipt,
    memory::{self, Gpu, MemoryLog, Peaks},
};
use varena::{
    layout::Layout, Fill, Fingerprint, GpuId, Grant, HostBackend, Level, Measured, Memory, PartId, Plan, SetSpec, Step, Torch,
};

const MIB: u64 = 1 << 20;
/// Three regions of one part each: a set the host backend maps page by page.
const PARTS: [&str; 3] = ["c/w0", "c/w1", "c/w2"];
const PACKAGE: &str = "s5/package@1";
/// The room the stand-in's pass measures.
const ROOM: u64 = 6 * MIB;

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
    Plan { steps: vec![Step { sets: vec![format!("{PACKAGE}/c")], passes: 1 }] }
}

/// An executor stand-in: run by `StandIn::spawn` with its control socket at stdin. It opens its
/// memory with the two descriptors that arrive first and loads the set at its level (`gpu`:
/// warmed, `ram`: in the RAM tier only), then answers one line per command: `load` (a request
/// binding every block: `<origins> <kept bytes> <store bytes>`), `run <id>` (a request whose
/// pass starts, unmeasured) and `end` (that pass measured, the request ended).
#[test]
#[ignore = "memory_ledger's tests run it as an executor"]
fn executor_stand_in() {
    let Ok(gpu) = std::env::var("S5_STAND_IN") else { return };
    let [uuid, model, bytes, level] = gpu.split('|').collect::<Vec<_>>()[..] else { panic!("{gpu}") };
    let gpu = GpuId { uuid: uuid.into(), model: model.into(), memory: bytes.parse().unwrap() };
    // SAFETY: the spawning test installs its private socket at the child's stdin.
    let control = unsafe { UnixStream::from_raw_fd(0) };
    let table = platform::handles::recv_fd(&control).unwrap();
    let events = platform::handles::recv_fd(&control).unwrap();
    let memory = Memory::open(gpu.clone(), Some(Grant { table, events }), Box::new(HostBackend::new(gpu.memory))).unwrap();
    let set = memory.load(spec(&gpu), if level == "gpu" { Level::Gpu } else { Level::Ram }).unwrap();
    let (mut out, lines) = (control.try_clone().unwrap(), BufReader::new(control).lines());
    let mut running = None;
    for line in lines.map_while(Result::ok) {
        let reply = match line.split_once(' ').unwrap_or((&line, "")) {
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
            ("run", id) => {
                let request = memory.request(id, plan()).unwrap();
                let pass = request.pass(set, "8", Default::default()).unwrap();
                running = Some((request, pass));
                "passing".into()
            }
            ("end", _) => {
                let (request, pass) = running.take().unwrap();
                pass.end(Measured { allocated_peak: ROOM, ..Default::default() }, Torch::default());
                request.end();
                "ended".into()
            }
            (other, _) => panic!("unknown command {other}"),
        };
        writeln!(out, "{reply}").unwrap();
    }
}

/// A stand-in executor, granted `gpu`'s ledger at its spawn.
struct StandIn {
    child: Child,
    control: UnixStream,
    replies: BufReader<UnixStream>,
}

impl StandIn {
    fn spawn(gpu: &Gpu, id: &GpuId, level: &str) -> StandIn {
        let (control, theirs) = UnixStream::pair().unwrap();
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "executor_stand_in", "--ignored", "--nocapture"])
            .env("S5_STAND_IN", format!("{}|{}|{}|{level}", id.uuid, id.model, id.memory))
            .stdin(Stdio::from(OwnedFd::from(theirs)))
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let grant = gpu.ledger.grant(child.id()).unwrap();
        platform::handles::send_fd(&control, &grant.table).unwrap();
        platform::handles::send_fd(&control, &grant.events).unwrap();
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

/// Bytes the table holds as orphans (class O): kept pages no live process maps.
fn orphans(gpu: &Gpu) -> u64 {
    gpu.status().rows.iter().map(|row| row.classes[1]).sum()
}

#[test]
fn a_killed_executors_kept_pages_are_the_next_ones_and_fits_follows_measured_peaks() {
    let dir = tempfile::tempdir().unwrap();
    let (log, peaks) = (MemoryLog::at(dir.path().join("logs")), Peaks::at(dir.path().join("peaks")));
    let id = gpu_id("a");
    let gpu = Gpu::create("0", &[inventory(&id)], None, &peaks, &log).expect("a ledger for an inventoried GPU");
    let (woke, wakes) = mpsc::channel();
    let _events = memory::react(&[&gpu.ledger], move || {
        let _ = woke.send(());
    })
    .unwrap();
    let set_bytes = spec(&id).layout.nbytes;

    // Granted at spawn, it warms the set onto its GPU: binds are hits there.
    let mut first = StandIn::spawn(&gpu, &id, "gpu");
    assert_eq!(first.load(), ("Hit,Hit,Hit".into(), 0, 0));
    assert!(gpu.maps(first.pid()));

    // An unmeasured run fits nothing beside it.
    assert!(varena::Ledger::admit(&[&gpu.ledger], "unmeasured", PACKAGE, "8").unwrap(), "an idle GPU admits it");
    assert_eq!(first.ask("run unmeasured"), "passing");
    assert!(gpu.running());
    assert!(!gpu.ledger.fits(PACKAGE, "8"));
    assert!(!gpu.ledger.fits("s5/other@1", "4"));
    while wakes.try_recv().is_ok() {}
    assert_eq!(first.ask("end"), "ended");
    gpu.ledger.undispatch("unmeasured");
    wakes.recv_timeout(BACKSTOP).expect("a recorded peak signals");
    assert!(gpu.ledger.peaks().iter().any(|peak| peak.key == format!("{PACKAGE}|8")));

    // Measured, a second request of that shape fits beside a running one.
    assert!(varena::Ledger::admit(&[&gpu.ledger], "measured", PACKAGE, "8").unwrap());
    assert_eq!(first.ask("run measured"), "passing");
    assert!(gpu.ledger.fits(PACKAGE, "8"), "two measured requests fit on the card");
    assert_eq!(first.ask("end"), "ended");
    gpu.ledger.undispatch("measured");

    // kill -9, waitpid, reap: what it kept is the table's orphans, and the next executor of the
    // same weights imports them instead of copying.
    let pid = first.kill();
    while wakes.try_recv().is_ok() {}
    gpu.ledger.reap(pid);
    wakes.recv_timeout(BACKSTOP).expect("a reaped row signals");
    assert!(!gpu.status().rows.iter().any(|row| row.pid == pid && row.live));
    assert!(orphans(&gpu) >= set_bytes, "its kept pages are orphans");
    // One load path: its load imports what custody keeps, at any level; its binds are hits.
    let mut second = StandIn::spawn(&gpu, &id, "ram");
    assert_eq!(second.load(), ("Hit,Hit,Hit".into(), 0, 0), "imported at load, not copied");
    assert_eq!(orphans(&gpu), 0, "every orphan imported by fingerprint");
    assert!(gpu.maps(second.pid()));

    // Peaks persist per GPU model: another GPU of the model starts with them.
    peaks.save(&id.model, std::iter::once(&gpu.ledger));
    let other = gpu_id("b");
    let restored = Gpu::create("0", &[inventory(&other)], None, &Peaks::at(dir.path().join("peaks")), &log).unwrap();
    assert!(restored.ledger.peaks().iter().any(|peak| peak.key == format!("{PACKAGE}|8")));
    drop(second);
}
