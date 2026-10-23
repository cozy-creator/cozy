//! TensorD hosting varena's lease manager (design varena-leases), with real processes as
//! executors: stand-ins (this binary's `lease_agent`, run as children) dial the GPU's listener and
//! lease chunks of the host-page backend. TensorD leases to its tenants alone, keeps the books,
//! logs every grant and exit, and retains landed weights only for the same trust domain.
#![cfg(unix)]
use std::{
    io::{BufRead, BufReader, Write},
    os::{
        fd::{FromRawFd, OwnedFd},
        unix::net::UnixStream,
    },
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};
use platform::{
    contain::host_pressure::HostMode,
    handles::Peer,
    process::{Birth, Exact},
};
use tensord::{
    machine::receipt,
    memory::{self, Gpu, Leases, MemoryLog, Peaks, Peers, Tenant, Tenants},
};
use varena::{
    lease::{agent::Attach, Agent, Hosting, Identity, Region},
    GpuId, HostBackend,
};

const MIB: u64 = 1 << 20;
/// A failure backstop only: every wait below ends on the event it waits for.
const BACKSTOP: Duration = Duration::from_secs(60);

fn gpu_id() -> GpuId {
    GpuId { uuid: format!("GPU-s5-lease-{}", uuid::Uuid::new_v4()), model: "S5 Host Pages".into(), memory: 64 * MIB }
}

fn inventory(id: &GpuId) -> receipt::Gpu {
    serde_json::from_value(serde_json::json!({"device_index": 0, "device_name": id.model, "device_uuid": id.uuid,
        "driver_version": "", "memory_bytes": id.memory, "pci_bus_id": ""}))
    .unwrap()
}

fn euid() -> u32 {
    platform::memory::euid()
}

/// A lease agent stand-in: run by `StandIn::start` with its control socket at stdin. It answers
/// one line per command: `attach` (`attached`, or its refusal's code),
/// `take <MiB> <fingerprint>` (one region's chunks, imported and mapped: `<chunks> <adopted>`) and
/// `land` (every taken chunk's fill landed: `landed`).
#[test]
#[ignore = "the lease tests run it as an executor"]
fn lease_agent() {
    let Ok(uuid) = std::env::var("S5_LEASE_AGENT") else { return };
    // SAFETY: the spawning test installs its private socket at the child's stdin.
    let control = unsafe { UnixStream::from_raw_fd(0) };
    let (mut out, lines) = (control.try_clone().unwrap(), BufReader::new(control).lines());
    let (mut agent, mut taken): (Option<Arc<Agent>>, Vec<varena::lease::Lease>) = (None, vec![]);
    for line in lines.map_while(Result::ok) {
        let reply = match line.split_once(' ').unwrap_or((&line, "")) {
            ("attach", _) => {
                let stream = varena::lease::connect(euid(), &uuid).unwrap().expect("TensorD serves the GPU's leases");
                match Agent::attach(stream, Arc::new(HostBackend::new(64 * MIB)), || {}).unwrap() {
                    Attach::Leased(attached) => {
                        agent = Some(attached);
                        "attached".into()
                    }
                    Attach::Refused { code, .. } => format!("{code:?}"),
                }
            }
            ("take", args) => {
                let (mib, fingerprint) = args.split_once(' ').unwrap();
                let agent = agent.as_ref().unwrap();
                let region = Region { fingerprint: fingerprint.parse().unwrap(), region: 0 };
                let leases = agent.take(mib.parse::<u64>().unwrap() * MIB, Some(region)).unwrap();
                agent.mapped(&leases.iter().map(|lease| lease.ticket).collect::<Vec<_>>()).unwrap();
                let adopted = leases.iter().filter(|lease| lease.adopted).count();
                let reply = format!("{} {adopted}", leases.len());
                taken.extend(leases);
                reply
            }
            ("land", _) => {
                let agent = agent.as_ref().unwrap();
                taken.iter().for_each(|lease| agent.landed(lease.ticket).unwrap());
                "landed".into()
            }
            (other, _) => panic!("unknown command {other}"),
        };
        writeln!(out, "{reply}").unwrap();
    }
}

/// An executor stand-in on `uuid`'s listener.
struct StandIn {
    child: Child,
    control: UnixStream,
    replies: BufReader<UnixStream>,
}

impl StandIn {
    fn start(uuid: &str) -> StandIn {
        let (control, theirs) = UnixStream::pair().unwrap();
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "lease_agent", "--ignored", "--nocapture"])
            .env("S5_LEASE_AGENT", uuid)
            .stdin(Stdio::from(OwnedFd::from(theirs)))
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let replies = BufReader::new(control.try_clone().unwrap());
        StandIn { child, control, replies }
    }

    /// A tenant of `generation`'s package, as TensorD registers an executor it started.
    fn tenant(uuid: &str, tenants: &Tenants, plan: &str, generation: &str) -> StandIn {
        let standin = StandIn::start(uuid);
        let exact = Exact::open(&Birth::of(standin.pid()).unwrap()).unwrap().unwrap();
        let tenant = Tenant { plan: plan.into(), generation: generation.into(), start: 0, exact };
        tenants.lock().unwrap().insert(standin.pid(), tenant);
        standin
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

    /// `kill -9`, then waitpid.
    fn kill(mut self) {
        // SAFETY: a signal to this test's own child, not yet reaped.
        assert_eq!(unsafe { libc::kill(self.pid() as i32, libc::SIGKILL) }, 0);
        self.child.wait().unwrap();
    }
}

impl Drop for StandIn {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The memory log's events of `kind`.
fn events(dir: &std::path::Path, kind: &str) -> Vec<serde_json::Value> {
    let log = std::fs::read_to_string(dir.join("memory.log")).unwrap_or_default();
    log.lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line.split_once(' ')?.1).ok())
        .filter(|event| event["event"] == kind)
        .collect()
}

#[test]
fn a_peer_is_a_live_tenant_running_as_a_machine_uid_and_nothing_its_package_starts() {
    let tenants = Tenants::default();
    let peers = Peers::new(tenants.clone(), vec![euid()], MemoryLog::default());
    // An executor and a process it started (its `sleep`), both ended below.
    let mut executor = Command::new("sh").args(["-c", "sleep 1000; :"]).spawn().unwrap();
    let started = Instant::now();
    let child = loop {
        if let Some(&pid) = platform::process::descendants(executor.id()).first() {
            break pid;
        }
        assert!(started.elapsed() < BACKSTOP, "the executor started nothing");
        std::thread::yield_now();
    };
    let peer = |pid: u32, uid: u32| Peer { pid, uid, gid: 0 };
    assert_eq!(peers.identify(&peer(executor.id(), euid())), None, "no tenant yet");
    let exact = Exact::open(&Birth::of(executor.id()).unwrap()).unwrap().unwrap();
    tenants.lock().unwrap().insert(executor.id(), Tenant { plan: "plan".into(), generation: "gen".into(), start: 2, exact });
    let identity = Identity { executor: "plan#2".into(), domain: format!("gen/{}", euid()) };
    assert_eq!(peers.identify(&peer(executor.id(), euid())), Some(identity.clone()));
    assert_eq!(peers.identify(&peer(child, euid())), None, "a process its package started, never reported");
    let exact = Exact::open(&Birth::of(child).unwrap()).unwrap().unwrap();
    tenants.lock().unwrap().insert(child, Tenant { plan: "plan".into(), generation: "gen".into(), start: 2, exact });
    assert_eq!(peers.identify(&peer(child, euid())), Some(identity), "a follower its rank 0 reported");
    assert_eq!(peers.identify(&peer(executor.id(), euid() + 1)), None, "another user");
    for pid in [child, executor.id()] {
        // SAFETY: a signal to the exact processes this test started and listed.
        assert_eq!(unsafe { libc::kill(pid as i32, libc::SIGKILL) }, 0);
    }
    executor.wait().unwrap();
    assert_eq!(peers.identify(&peer(executor.id(), euid())), None, "an ended tenant");
}

/// One GPU's manager hosted as TensorD hosts it: two executors of one package and one of
/// another lease a region, a stranger is refused, the books name each holder, every grant and
/// exit is a memory-log line, and an exited executor's landed region is adopted with no copy
/// by the next executor of its package, never by the other package's.
#[test]
fn tensord_leases_to_its_tenants_and_keeps_landed_weights_within_their_package() {
    let dir = tempfile::tempdir().unwrap();
    let (id, log) = (gpu_id(), MemoryLog::at(dir.path().to_path_buf()));
    let tenants = Tenants::default();
    let hosting: Arc<dyn Hosting> = Arc::new(Peers::new(tenants.clone(), vec![euid()], log.clone()));
    let (card, opened) = (id.memory, Arc::new(AtomicUsize::new(0)));
    let opens = opened.clone();
    let open = move || {
        opens.fetch_add(1, Ordering::Relaxed);
        Ok(Arc::new(HostBackend::new(card)) as Arc<dyn varena::Backend>)
    };
    let leases = Leases { hosting, open: Box::new(open) };
    let peaks = Peaks::at(dir.path().join("peaks"));
    let gpu = Arc::new(Gpu::create("0", &[inventory(&id)], None, HostMode::Dedicated, Some(leases), &peaks, &log).unwrap());
    let manager = gpu.manager().expect("TensorD manages the GPU's chunks").clone();
    let (woke, wakes) = mpsc::channel();
    let _events = memory::react(vec![memory::Source::Ledger(gpu.clone())], move |what| { let _ = woke.send(what); }).unwrap();

    let mut stranger = StandIn::start(&id.uuid);
    assert_eq!(stranger.ask("attach"), "NotRegistered", "a process TensorD did not start is refused");
    assert_eq!(opened.load(Ordering::Relaxed), 0, "no context before the first ask");

    let mut first = StandIn::tenant(&id.uuid, &tenants, "plan-a", "gen-a");
    assert_eq!(first.ask("attach"), "attached");
    assert_eq!(first.ask("take 8 7"), "1 0", "one fresh 8 MiB chunk");
    assert_eq!(opened.load(Ordering::Relaxed), 1, "the first ask opens the context");
    let books = manager.holdings();
    assert_eq!((books.executors.len(), books.executors[0].pid, books.executors[0].bytes), (1, first.pid(), 8 * MIB), "{books:?}");
    assert_eq!(first.ask("land"), "landed");
    let grants = events(dir.path(), "grant");
    assert!(grants.iter().any(|grant| grant["pid"] == first.pid() && grant["bytes"] == 8 * MIB), "{grants:?}");

    let pid = first.pid();
    first.kill();
    // The manager logs the reclaim, then rings.
    let (started, reclaimed) = (Instant::now(), |reclaim: &serde_json::Value| reclaim["pid"] == pid);
    while !events(dir.path(), "reclaim").iter().any(reclaimed) {
        wakes.recv_timeout(BACKSTOP.saturating_sub(started.elapsed())).expect("the manager reclaims an exited executor's leases");
    }
    let books = manager.holdings();
    assert_eq!((books.retained, books.leased, books.executors.len()), (8 * MIB, 0, 0), "its landed region is retained: {books:?}");
    let reclaims = events(dir.path(), "reclaim");
    assert!(reclaims.iter().any(|reclaim| reclaimed(reclaim) && reclaim["retained_bytes"] == 8 * MIB), "{reclaims:?}");

    let mut other = StandIn::tenant(&id.uuid, &tenants, "plan-b", "gen-b");
    assert_eq!(other.ask("attach"), "attached");
    assert_eq!(other.ask("take 8 7"), "1 0", "another package's executor never adopts it");
    let mut next = StandIn::tenant(&id.uuid, &tenants, "plan-a", "gen-a");
    assert_eq!(next.ask("attach"), "attached");
    assert_eq!(next.ask("take 8 7"), "1 1", "the next executor of its package adopts it, no copy");
    assert!(!events(dir.path(), "adopt").is_empty());
    let books = manager.holdings();
    assert_eq!((books.retained, books.executors.len()), (0, 2), "{books:?}");
    drop((stranger, other, next));
}
