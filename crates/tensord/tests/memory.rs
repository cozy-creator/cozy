//! TensorD hosting varena's memory manager (design varena-leases), with real processes as
//! executors: stand-ins (this binary's `lease_agent`, run as children) dial the GPU's listener,
//! allocate chunks of the host-page backend and report their facts. TensorD allocates to its
//! tenants alone, accounts every page, logs every allocation and exit, keeps landed weights in the
//! GPU page cache for the same trust domain only, admits and places requests by measured peaks,
//! and reads the host rule's calls from the agents' facts.
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
    execution::Engine,
    journal::Invocation,
    machine::receipt,
    memory::{self, Gpu, Leases, MemoryLog, Peaks, Peers, Tenant, Tenants},
};
use varena::{
    lease::{Agent, Attach, Hosting, Identity, Measured, Peak, Region},
    GpuId, HostBackend,
};

const MIB: u64 = 1 << 20;
/// A failure backstop only: every wait below ends on the event it waits for.
const BACKSTOP: Duration = Duration::from_secs(60);

/// A card big enough that the driver headroom H (512 MiB at least, never leased) leaves room; the
/// host backend takes only the pages it makes.
fn gpu_id() -> GpuId {
    GpuId { uuid: format!("GPU-s5-lease-{}", uuid::Uuid::new_v4()), model: "S5 Host Pages".into(), memory: 8 << 30 }
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
/// one line per command: `attach` (`attached`, or its refusal's code), `take <MiB> <fingerprint>`
/// (one region's chunks, imported and mapped: `<chunks> <adopted>`), `land` (every taken chunk's
/// fill landed: `landed`) and `measured <request> <working> <compute> <host>` (a request's peaks
/// at its end: `measured`).
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
                let leases = agent.take(mib.parse::<u64>().unwrap() * MIB, varena::lease::Kind::Weights, Some(region)).unwrap();
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
            ("measured", args) => {
                let [request, working, compute, host] = args.split(' ').collect::<Vec<_>>()[..] else { panic!("{args}") };
                let (working, compute, host) = (working.parse().unwrap(), compute.parse().unwrap(), host.parse().unwrap());
                agent.as_ref().unwrap().measured(Measured::Request { request: request.into(), working, compute, host }).unwrap();
                "measured".into()
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
        StandIn::grouped(uuid, None)
    }

    /// A stand-in in process group `group` (a rank 0's: its follower), or its own.
    fn grouped(uuid: &str, group: Option<u32>) -> StandIn {
        let (control, theirs) = UnixStream::pair().unwrap();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "lease_agent", "--ignored", "--nocapture"])
            .env("S5_LEASE_AGENT", uuid)
            .stdin(Stdio::from(OwnedFd::from(theirs)))
            .stdout(Stdio::null());
        if let Some(group) = group {
            std::os::unix::process::CommandExt::process_group(&mut command, group as i32);
        }
        let child = command.spawn().unwrap();
        let replies = BufReader::new(control.try_clone().unwrap());
        StandIn { child, control, replies }
    }

    /// A tenant of `generation`'s package, as TensorD registers an executor it started.
    fn tenant(uuid: &str, tenants: &Tenants, plan: &str, generation: &str) -> StandIn {
        let standin = StandIn::start(uuid);
        let exact = Exact::open(&Birth::of(standin.pid()).unwrap()).unwrap().unwrap();
        let tenant = Tenant { plan: plan.into(), generation: generation.into(), start: 0, exact };
        tenants.register([(standin.pid(), tenant)]);
        standin
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn ask(&mut self, command: &str) -> String {
        self.send(command);
        self.reply(command)
    }

    fn send(&mut self, command: &str) {
        writeln!(self.control, "{command}").unwrap();
    }

    fn reply(&mut self, command: &str) -> String {
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

/// A GPU hosted as TensorD hosts it: its manager over host pages (none: chunk-less), its peers
/// `tenants`.
fn hosted(id: &GpuId, tenants: &Tenants, chunks: bool, dir: &std::path::Path, log: &MemoryLog) -> Arc<Gpu> {
    let peers = Arc::new(Peers::new(tenants.clone(), vec![euid()], log.clone()));
    let card = id.memory;
    let open: Option<varena::lease::Opener> = chunks.then(|| Box::new(move || Ok(Arc::new(HostBackend::new(card)) as Arc<dyn varena::Backend>)) as varena::lease::Opener);
    let leases = Leases { peers, open, tenants: tenants.clone() };
    Arc::new(Gpu::create("0", &[inventory(id)], None, HostMode::Dedicated, leases, &Peaks::at(dir.join("peaks")), log).unwrap())
}

/// Waits until `done` holds: the manager reads an agent's reports on its own thread.
fn eventually(what: &str, done: impl Fn() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(started.elapsed() < BACKSTOP, "never: {what}");
        std::thread::yield_now();
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
    tenants.register([(executor.id(), Tenant { plan: "plan".into(), generation: "gen".into(), start: 2, exact })]);
    let identity = Identity { executor: "plan#2".into(), domain: format!("gen/{}", euid()) };
    assert_eq!(peers.identify(&peer(executor.id(), euid())), Some(identity.clone()));
    assert_eq!(peers.identify(&peer(child, euid())), None, "a process its package started, never reported");
    let exact = Exact::open(&Birth::of(child).unwrap()).unwrap().unwrap();
    tenants.register([(child, Tenant { plan: "plan".into(), generation: "gen".into(), start: 2, exact })]);
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
    let peers = Arc::new(Peers::new(tenants.clone(), vec![euid()], log.clone()));
    let (card, opened) = (id.memory, Arc::new(AtomicUsize::new(0)));
    let opens = opened.clone();
    let open = move || {
        opens.fetch_add(1, Ordering::Relaxed);
        Ok(Arc::new(HostBackend::new(card)) as Arc<dyn varena::Backend>)
    };
    let leases = Leases { peers, open: Some(Box::new(open)), tenants: tenants.clone() };
    let peaks = Peaks::at(dir.path().join("peaks"));
    let gpu = Arc::new(Gpu::create("0", &[inventory(&id)], None, HostMode::Dedicated, leases, &peaks, &log).unwrap());
    let manager = gpu.manager.clone();
    let (woke, wakes) = mpsc::channel();
    let _events = memory::react(vec![memory::Source::Gpu(gpu.clone())], move |what| { let _ = woke.send(what); }).unwrap();

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
    // The manager logs an allocation once its grant is sent.
    let allocated = |grant: &serde_json::Value| grant["pid"] == first.pid() && grant["bytes"] == 8 * MIB;
    eventually("its allocation is logged", || events(dir.path(), "allocate").iter().any(allocated));

    let pid = first.pid();
    first.kill();
    // The manager logs the exit, then rings.
    let (started, reclaimed) = (Instant::now(), |reclaim: &serde_json::Value| reclaim["pid"] == pid);
    while !events(dir.path(), "exit").iter().any(reclaimed) {
        wakes.recv_timeout(BACKSTOP.saturating_sub(started.elapsed())).expect("the manager reclaims an exited executor's leases");
    }
    let books = manager.holdings();
    assert_eq!((books.page_cache, books.allocated, books.executors.len()), (8 * MIB, 0, 0), "its landed region is in the GPU page cache: {books:?}");
    let reclaims = events(dir.path(), "exit");
    assert!(reclaims.iter().any(|reclaim| reclaimed(reclaim) && reclaim["cached_bytes"] == 8 * MIB), "{reclaims:?}");

    let mut other = StandIn::tenant(&id.uuid, &tenants, "plan-b", "gen-b");
    assert_eq!(other.ask("attach"), "attached");
    assert_eq!(other.ask("take 8 7"), "1 0", "another package's executor never adopts it");
    let mut next = StandIn::tenant(&id.uuid, &tenants, "plan-a", "gen-a");
    assert_eq!(next.ask("attach"), "attached");
    assert_eq!(next.ask("take 8 7"), "1 1", "the next executor of its package adopts it, no copy");
    eventually("its adoption is logged", || !events(dir.path(), "cached").is_empty());
    let books = manager.holdings();
    assert_eq!((books.page_cache, books.executors.len()), (0, 2), "{books:?}");
    drop((stranger, other, next));
}

/// Waits until `count` peers wait for their registration.
fn parked(tenants: &Tenants, count: usize) {
    let started = Instant::now();
    while tenants.parked() != count {
        assert!(started.elapsed() < BACKSTOP, "{} parked, not {count}", tenants.parked());
        std::thread::yield_now();
    }
}

/// A follower that dials before its rank 0 reports it waits, parked, and is admitted the moment
/// it is registered; one its group's Start ends without is refused with a typed reason. Nothing
/// redials and nothing times out: the Start's state decides. No more wait than the Start expects
/// followers, and a process that left rank 0's group (its own, as `setsid` makes) never waits.
#[test]
fn a_follower_dialing_before_it_is_reported_waits_and_is_admitted() {
    let dir = tempfile::tempdir().unwrap();
    let (id, log) = (gpu_id(), MemoryLog::at(dir.path().to_path_buf()));
    let tenants = Tenants::default();
    let _gpu = hosted(&id, &tenants, true, dir.path(), &log);
    // Rank 0: its own process group's leader, a registered executor whose Start forms the group.
    let mut command = Command::new("sh");
    command.args(["-c", "sleep 1000; :"]);
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    let mut leader = command.spawn().unwrap();
    let birth = Birth::of(leader.id()).unwrap();
    let tenant = |exact: Exact| Tenant { plan: "plan-g".into(), generation: "gen-g".into(), start: 0, exact };
    tenants.register([(leader.id(), tenant(Exact::open(&birth).unwrap().unwrap()))]);
    let forming = tenants.forming(&birth, 1);

    let mut early = StandIn::grouped(&id.uuid, Some(leader.id()));
    early.send("attach");
    parked(&tenants, 1);
    tenants.register([(early.pid(), tenant(Exact::open(&Birth::of(early.pid()).unwrap()).unwrap().unwrap()))]);
    assert_eq!(early.reply("attach"), "attached", "admitted at its registration");
    parked(&tenants, 0);

    let mut late = StandIn::grouped(&id.uuid, Some(leader.id()));
    late.send("attach");
    parked(&tenants, 1);
    let mut extra = StandIn::grouped(&id.uuid, Some(leader.id()));
    assert_eq!(extra.ask("attach"), "NotRegistered", "the Start expects one follower: one waits");
    let mut left = StandIn::grouped(&id.uuid, Some(0));
    assert_eq!(left.ask("attach"), "NotRegistered", "a process in a group of its own never waits");
    assert_eq!(tenants.parked(), 1);
    drop(forming);
    assert_eq!(late.reply("attach"), "NotRegistered", "the Start ended without it");
    parked(&tenants, 0);
    drop((early, late, extra, left));
    // SAFETY: a signal to the process group this test made (rank 0, its sleep, the stand-ins).
    assert_eq!(unsafe { libc::kill(-(leader.id() as i32), libc::SIGKILL) }, 0);
    leader.wait().unwrap();
}

const GIB: u64 = 1 << 30;
const PACKAGE: &str = "s5/package@1";
const BIG: &str = "big";

/// A request whose measured peak does not fit beside a dispatched one waits, woken by the
/// manager's events, and is admitted once that one ends; a cancel ends a wait. The peak is what
/// an agent measured under its dispatch's key.
#[test]
fn a_request_that_does_not_fit_waits_until_the_running_one_ends() {
    let dir = tempfile::tempdir().unwrap();
    let (id, log, tenants) = (gpu_id(), MemoryLog::at(dir.path().join("logs")), Tenants::default());
    let gpu = hosted(&id, &tenants, true, dir.path(), &log);
    let engine = Engine::open(&dir.path().join("engine")).unwrap();
    let woken = engine.clone();
    let _events = memory::react(vec![memory::Source::Gpu(gpu.clone())], move |_| woken.notify_activity()).unwrap();
    let request = |key: &str| engine.submit(key, Invocation::default()).unwrap().id;
    let admit = |id: &str| memory::admit(&engine, &[&*gpu.manager], id, PACKAGE, BIG).unwrap();
    let mut a = StandIn::tenant(&id.uuid, &tenants, "plan-a", "gen-a");
    assert_eq!(a.ask("attach"), "attached");

    let first = request("first");
    assert!(admit(&first), "unmeasured on an idle GPU, it runs");
    assert_eq!(a.ask(&format!("measured {first} {} 0 0", 5 * GIB)), "measured");
    eventually("its peak is recorded", || gpu.manager.peaks().iter().any(|p| matches!(p, Peak::Request { key, bytes } if *key == format!("{PACKAGE}|{BIG}") && *bytes == 5 * GIB)));
    gpu.manager.undispatch(&first);

    let second = request("second");
    assert!(admit(&second), "alone, it fits");
    assert!(!gpu.manager.fits(PACKAGE, BIG), "two do not fit");
    let third = request("third");
    std::thread::scope(|scope| {
        let (done, admitted) = mpsc::channel();
        let (third, admit) = (&third, &admit);
        scope.spawn(move || done.send(admit(third)).unwrap());
        engine.notify_activity(); // a wake that frees nothing
        assert!(admitted.try_recv().is_err(), "it waits while the second runs");
        gpu.manager.undispatch(&second);
        engine.notify_activity();
        assert!(admitted.recv_timeout(BACKSTOP).expect("the second's end wakes it"), "admitted once the second ended");
    });
    let fourth = request("fourth");
    std::thread::scope(|scope| {
        let (fourth, admit) = (&fourth, &admit);
        let waiting = scope.spawn(move || admit(fourth));
        engine.cancel(fourth, "test").unwrap();
        assert!(!waiting.join().unwrap(), "a cancel ends the wait, not admitted");
    });
    assert!(!gpu.manager.holdings().dispatched.iter().any(|(request, _)| *request == fourth));
}

/// Measured peaks persist per GPU model: saved from one manager and restored, exactly, into the
/// next of the model (a restarted TensorD, another GPU of the model).
#[test]
fn measured_peaks_are_restored_exactly_by_the_next_manager_of_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let (id, log, tenants) = (gpu_id(), MemoryLog::at(dir.path().join("logs")), Tenants::default());
    let gpu = hosted(&id, &tenants, true, dir.path(), &log);
    let peaks = Peaks::at(dir.path().join("peaks"));
    let measured = [Peak::Request { key: format!("{PACKAGE}|8"), bytes: 3 * GIB }, Peak::Set { key: "s5/package@1/unet".into(), bytes: 2 * GIB }];
    gpu.manager.restore(&measured).unwrap();
    peaks.save(&gpu.model, std::iter::once(&*gpu.manager));
    let sorted = |mut p: Vec<Peak>| { p.sort_by_key(|p| format!("{p:?}")); p };
    drop(gpu);
    let next = hosted(&GpuId { uuid: format!("{}-next", id.uuid), ..id.clone() }, &tenants, true, dir.path(), &log);
    assert_eq!(sorted(next.manager.peaks()), sorted(measured.to_vec()));
}

/// The host rule reads TensorD's own record of the running calls beside each GPU's measured host
/// peaks: an unmeasured call carries none, a measured one the largest any GPU measured.
#[test]
fn an_unmeasured_running_call_carries_no_host_peak() {
    let dir = tempfile::tempdir().unwrap();
    let (log, tenants, running) = (MemoryLog::at(dir.path().join("logs")), Tenants::default(), memory::Running::default());
    let gpus = [hosted(&gpu_id(), &tenants, true, dir.path(), &log), hosted(&gpu_id(), &tenants, true, dir.path(), &log)];
    let calls = || memory::host_calls(&running, &gpus);
    assert!(calls().is_empty(), "nothing runs");
    let call = memory::Call { pid: std::process::id(), package: PACKAGE.into(), shape: BIG.into() };
    running.lock().unwrap().insert("r".into(), call);
    assert_eq!(calls()[0].peak, None, "unmeasured");
    gpus[1].manager.restore(&[Peak::Host { key: format!("{PACKAGE}|{BIG}"), bytes: GIB }]).unwrap();
    assert_eq!((calls()[0].peak, calls()[0].pid), (Some(GIB), std::process::id()), "measured on GPU 1");
    running.lock().unwrap().remove("r");
    assert!(calls().is_empty(), "ended");
}

/// The events thread wakes on a manager's event and ends with its pool.
#[test]
fn the_events_thread_wakes_on_a_managers_event_and_ends_with_its_pool() {
    let dir = tempfile::tempdir().unwrap();
    let (id, log, tenants) = (gpu_id(), MemoryLog::at(dir.path().join("logs")), Tenants::default());
    let gpu = hosted(&id, &tenants, true, dir.path(), &log);
    let (woke, wakes) = mpsc::channel();
    let events = memory::react(vec![memory::Source::Gpu(gpu.clone())], move |what| {
        let _ = woke.send(what);
    })
    .unwrap();
    while wakes.recv_timeout(std::time::Duration::ZERO).is_ok() {}
    gpu.manager.fault(); // its page cache goes: the manager signals
    assert_eq!(wakes.recv_timeout(BACKSTOP).unwrap(), memory::Woke::Gpu);
    drop(events);
    while wakes.recv_timeout(BACKSTOP).is_ok() {}
    assert!(wakes.recv().is_err(), "the thread ended with its pool");
}

/// TensorD names the GPU of a 2-GPU group that keeps a placeable set: beside a request running on
/// GPU 0 (its measured peak) the set has no room there and goes to GPU 1; once that request ends
/// it stays on GPU 0. Its bytes are what a load measured on the group.
#[test]
fn a_placeable_set_is_homed_on_gpu_1_only_while_gpu_0_has_no_room() {
    let dir = tempfile::tempdir().unwrap();
    let (log, tenants) = (MemoryLog::at(dir.path().join("logs")), Tenants::default());
    let (id0, id1) = (gpu_id(), gpu_id());
    let (gpu0, gpu1) = (hosted(&id0, &tenants, true, dir.path(), &log), hosted(&id1, &tenants, true, dir.path(), &log));
    let set = "s5/package@1/unet";
    gpu1.manager.restore(&[Peak::Set { key: set.into(), bytes: 3 * GIB }]).unwrap();
    let homes = || memory::homes(&[&*gpu0.manager, &*gpu1.manager], PACKAGE, BIG, &[set], &log);
    // An executor on each GPU: its first allocation opens the GPU's chunks and their reading.
    let executors: Vec<StandIn> = [&id0, &id1].iter().map(|id| {
        let mut executor = StandIn::tenant(&id.uuid, &tenants, "plan-a", "gen-a");
        assert_eq!(executor.ask("attach"), "attached");
        assert_eq!(executor.ask("take 2 9"), "1 0");
        executor
    }).collect();
    eventually("both GPUs read their free bytes", || [&gpu0, &gpu1].iter().all(|gpu| gpu.manager.holdings().free > 0));
    assert_eq!(homes(), vec![0], "GPU 0 has room");
    gpu0.manager.restore(&[Peak::Request { key: "s5/other@1|8".into(), bytes: 6 * GIB }]).unwrap();
    gpu0.manager.dispatch("running", "s5/other@1", "8").unwrap();
    assert_eq!(homes(), vec![1], "beside the running request it has none");
    gpu0.manager.undispatch("running");
    assert_eq!(homes(), vec![0], "its room is back");
    drop(executors);
}

/// A GPU TensorD cannot create chunks on has a chunk-less manager: its executors are refused as
/// incompatible (they run standalone), and its requests are admitted all the same.
#[test]
fn a_chunkless_manager_refuses_its_agents_incompatible_and_still_admits() {
    let dir = tempfile::tempdir().unwrap();
    let (id, log, tenants) = (gpu_id(), MemoryLog::at(dir.path().join("logs")), Tenants::default());
    let gpu = hosted(&id, &tenants, false, dir.path(), &log);
    let mut a = StandIn::tenant(&id.uuid, &tenants, "plan-a", "gen-a");
    assert_eq!(a.ask("attach"), "Incompatible");
    assert!(varena::lease::Manager::admit(&[&*gpu.manager], "r", PACKAGE, BIG).unwrap());
}
