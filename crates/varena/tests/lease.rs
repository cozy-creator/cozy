//! Leases across real processes (varena-leases §8 P0): this process hosts the Manager over host
//! pages (memfds crossing by SCM_RIGHTS; on Windows sections DuplicateHandle'd into the agent)
//! on the GPU's own listener (an abstract socket, a named pipe); each agent is this binary as a
//! child that runs one command per stdin line and answers on stdout.

#[cfg(any(target_os = "linux", windows))]
mod cases {
    use std::io::{BufRead, BufReader, Write};
    use std::path::Path;
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::sync::atomic::{AtomicU64, Ordering::SeqCst};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use parking_lot::Mutex;
    use platform::handles::Peer;
    use tensorfs::io::Dst;
    use varena::backend::{Access, Backend, Va};
    use varena::lease::wire::{End, ToAgent};
    use varena::lease::{Agent, Attach, Holdings, Hosting, Identity, Lease, Manager, Refusal, Region};
    use varena::ledger::Mode;
    use varena::{GpuId, HostBackend};

    const MIB: u64 = 1 << 20;

    /// Only this test's children.
    struct Children(Mutex<Vec<u32>>);

    impl Hosting for Children {
        fn identify(&self, peer: &Peer) -> Option<Identity> {
            self.0.lock().contains(&peer.pid).then(|| Identity { executor: peer.pid.to_string(), domain: "test".into() })
        }
    }

    struct Host {
        manager: Arc<Manager>,
        uuid: String,
        children: Arc<Children>,
    }

    fn host(_dir: &Path, case: &str, capacity: u64) -> Host {
        let gpu = GpuId { uuid: format!("lease{}-{case}", std::process::id()), model: "host".into(), memory: capacity };
        let listener = varena::lease::listen(&gpu).unwrap().unwrap();
        let children = Arc::new(Children(Mutex::new(vec![])));
        let open = Box::new(move || Ok(Arc::new(HostBackend::new(capacity)) as Arc<dyn Backend>));
        let manager = Manager::create(&gpu, Some(capacity), Mode::Dedicated, open, listener, children.clone()).unwrap();
        varena::os::remove_table(&gpu.uuid);
        Host { manager, uuid: gpu.uuid, children }
    }

    /// The GPU's listener, as an agent dials it.
    fn dial(uuid: &str) -> platform::memory::Stream {
        varena::lease::connect(platform::memory::euid(), uuid).unwrap().unwrap()
    }

    struct Agentd {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
    }

    impl Agentd {
        fn start(host: &Host) -> Agentd {
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args(["agent", &host.uuid])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .unwrap();
            host.children.0.lock().push(child.id());
            let (stdin, stdout) = (child.stdin.take().unwrap(), BufReader::new(child.stdout.take().unwrap()));
            let mut a = Agentd { child, stdin, stdout };
            assert_eq!(a.call("attach"), "ok");
            a
        }

        fn send(&mut self, line: &str) {
            writeln!(self.stdin, "{line}").unwrap();
            self.stdin.flush().unwrap();
        }

        fn answer(&mut self) -> String {
            let mut line = String::new();
            self.stdout.read_line(&mut line).unwrap();
            line.trim().to_string()
        }

        fn call(&mut self, line: &str) -> String {
            self.send(line);
            self.answer()
        }

        fn kill(&mut self) {
            self.child.kill().unwrap();
            self.child.wait().unwrap();
        }
    }

    /// Until `done` holds of the manager's holdings; a test fails rather than hangs.
    fn until(host: &Host, done: impl Fn(&Holdings) -> bool) -> Holdings {
        let start = Instant::now();
        loop {
            let h = host.manager.holdings();
            if done(&h) {
                return h;
            }
            assert!(start.elapsed() < Duration::from_secs(20), "the manager never reached it: {h:?}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn region(n: u32) -> String {
        format!("{} {n}", 0xfeed_u64)
    }

    fn grant_maps_and_writes(dir: &Path) {
        let host = host(dir, "grant", 256 * MIB);
        let mut a = Agentd::start(&host);
        assert_eq!(a.call(&format!("take {} {}", 80 * MIB, region(1))), "ok 1 fresh verified");
        let h = until(&host, |h| h.leased == 80 * MIB);
        assert_eq!((h.created, h.retained, h.executors.len()), (80 * MIB, 0, 1));
    }

    /// A weights region leases as one chunk sized to its page, however big (§3): never split.
    fn a_weights_page_is_one_chunk_however_big(dir: &Path) {
        let host = host(dir, "big", 512 * MIB);
        let mut a = Agentd::start(&host);
        assert_eq!(a.call(&format!("take {} {}", 400 * MIB, region(1))), "ok 1 fresh verified");
        let h = until(&host, |h| h.leased == 400 * MIB);
        assert_eq!(h.created_chunks, 1, "{h:?}");
    }

    /// A running pass pins its chunks and asks for more: when only its own pins could give the
    /// room, it is refused at once, never left waiting on itself.
    fn an_ask_against_its_own_pins_is_refused(dir: &Path) {
        let host = host(dir, "own", 64 * MIB);
        let mut a = Agentd::start(&host);
        assert_eq!(a.call(&format!("take {} {}", 64 * MIB, region(1))), "ok 1 fresh verified");
        assert_eq!(a.call("pin"), "ok");
        assert!(a.call(&format!("take {} {}", 32 * MIB, region(2))).starts_with("refused"));
        assert_eq!(host.manager.holdings().leased, 64 * MIB);
    }

    fn an_unpinned_lease_is_revoked_for_another(dir: &Path) {
        let host = host(dir, "revoke", 128 * MIB);
        let (mut a, mut b) = (Agentd::start(&host), Agentd::start(&host));
        assert_eq!(a.call(&format!("take {} {}", 96 * MIB, region(1))), "ok 1 fresh verified");
        b.send(&format!("take {} {}", 64 * MIB, region(2)));
        a.wait_rang();
        // 32 MiB were free: A's one chunk is asked back.
        assert_eq!(a.call("revokes"), "released 1");
        assert_eq!(b.answer(), "ok 1 fresh verified");
        until(&host, |h| h.leased == 64 * MIB && h.created == 64 * MIB);
    }

    fn a_revoke_waits_for_the_pin(dir: &Path) {
        let host = host(dir, "pin", 128 * MIB);
        let (mut a, mut b) = (Agentd::start(&host), Agentd::start(&host));
        assert_eq!(a.call(&format!("take {} {}", 96 * MIB, region(1))), "ok 1 fresh verified");
        b.send(&format!("take {} {}", 64 * MIB, region(2)));
        a.wait_rang();
        // Pinned after the ask went out: the agent's CAS refuses the revoke, the asker waits.
        assert_eq!(a.call("pin"), "ok");
        assert_eq!(a.call("revokes"), "released 0");
        assert_eq!(host.manager.holdings().leased, 96 * MIB);
        assert_eq!(a.call("unpin"), "ok");
        assert_eq!(a.call("revokes"), "released 1");
        assert_eq!(b.answer(), "ok 1 fresh verified");
    }

    /// Only another executor's pinned chunks could give the room: they are asked, and the ask
    /// waits for their unpin.
    fn an_ask_waits_for_anothers_pins(dir: &Path) {
        let host = host(dir, "others", 64 * MIB);
        let (mut a, mut b) = (Agentd::start(&host), Agentd::start(&host));
        assert_eq!(a.call(&format!("take {} {}", 64 * MIB, region(1))), "ok 1 fresh verified");
        assert_eq!(a.call("pin"), "ok");
        b.send(&format!("take {} {}", 32 * MIB, region(2)));
        a.wait_rang();
        assert_eq!(a.call("revokes"), "released 0");
        assert_eq!(a.call("unpin"), "ok");
        assert_eq!(a.call("revokes"), "released 1");
        assert_eq!(b.answer(), "ok 1 fresh verified");
    }

    /// Two running passes, each pinned, each asking for the other's room: the second ask would
    /// close a wait cycle and is refused; the first is served once the second unpins.
    fn a_wait_cycle_is_refused(dir: &Path) {
        let host = host(dir, "cycle", 128 * MIB);
        let (mut a, mut b) = (Agentd::start(&host), Agentd::start(&host));
        assert_eq!(a.call(&format!("take {} {}", 64 * MIB, region(1))), "ok 1 fresh verified");
        assert_eq!(b.call(&format!("take {} {}", 64 * MIB, region(2))), "ok 1 fresh verified");
        assert_eq!(a.call("pin"), "ok");
        assert_eq!(b.call("pin"), "ok");
        a.send(&format!("take {} {}", 32 * MIB, region(3)));
        b.wait_rang();
        assert!(b.call(&format!("take {} {}", 32 * MIB, region(4))).starts_with("refused"));
        assert_eq!(b.call("unpin"), "ok");
        assert_eq!(b.call("revokes"), "released 1");
        assert_eq!(a.answer(), "ok 1 fresh verified");
    }

    /// Stall verdicts come only from the look's cadence and measured progress: back-to-back
    /// reads name nobody; an agent whose GPU work keeps completing (its revoke behind a long
    /// kernel's fence) is never named; one with neither a release nor GPU progress is.
    fn stalls_are_judged_on_the_looks_cadence(dir: &Path) {
        let host = host(dir, "stall", 128 * MIB);
        let (mut a, mut b) = (Agentd::start(&host), Agentd::start(&host));
        assert_eq!(a.call(&format!("take {} {}", 96 * MIB, region(1))), "ok 1 fresh verified");
        b.send(&format!("take {} {}", 64 * MIB, region(2)));
        a.wait_rang();
        for _ in 0..10_000 {
            assert!(host.manager.stalled().is_empty(), "a healthy agent named between looks");
        }
        // Eight looks' worth of a long fence: its GPU work keeps completing.
        let fenced = Instant::now();
        while fenced.elapsed() < Duration::from_secs(4) {
            assert_eq!(a.call("tick"), "ok");
            assert!(host.manager.stalled().is_empty(), "an agent whose GPU work moves was named");
            std::thread::sleep(Duration::from_millis(100));
        }
        let start = Instant::now();
        while !host.manager.stalled().contains(&a.child.id()) {
            assert!(start.elapsed() < Duration::from_secs(20), "an agent that never answers is never named");
            std::thread::sleep(Duration::from_millis(50));
        }
        assert_eq!(a.call("revokes"), "released 1");
        assert_eq!(b.answer(), "ok 1 fresh verified");
        let start = Instant::now();
        while !host.manager.stalled().is_empty() {
            assert!(start.elapsed() < Duration::from_secs(20), "a release clears the verdict");
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn a_killed_agents_leases_return_and_landed_weights_are_adopted(dir: &Path) {
        let host = host(dir, "kill", 256 * MIB);
        let mut a = Agentd::start(&host);
        assert_eq!(a.call(&format!("take {} {}", 80 * MIB, region(7))), "ok 1 fresh verified");
        assert_eq!(a.call(&format!("take {} {} unlanded", 16 * MIB, region(8))), "ok 1 fresh verified");
        a.kill();
        // Its landed region is retained; the region whose fill never landed is destroyed.
        let h = until(&host, |h| h.executors.is_empty());
        assert_eq!((h.leased, h.retained, h.created), (0, 80 * MIB, 80 * MIB));
        let mut c = Agentd::start(&host);
        assert_eq!(c.call(&format!("take {} {}", 80 * MIB, region(7))), "ok 1 adopted verified");
        let h = until(&host, |h| h.leased == 80 * MIB);
        assert_eq!((h.retained, h.created), (0, 80 * MIB), "adopted with no new chunk");
    }

    /// A process the host never registered dials the GPU's listener: refused before it is
    /// heard, and it holds nothing.
    fn a_stranger_is_refused(dir: &Path) {
        let host = host(dir, "stranger", 64 * MIB);
        let stranger = End::new(dial(&host.uuid));
        match stranger.recv::<ToAgent>().unwrap() {
            Some((ToAgent::Refuse { code: Refusal::NotRegistered, .. }, None)) => {}
            other => panic!("a stranger was answered {:?}", other.map(|(m, h)| (m, h.is_some()))),
        }
        assert!(stranger.recv::<ToAgent>().unwrap().is_none(), "the refused connection stays open");
        assert!(host.manager.holdings().executors.is_empty());
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if let [role, uuid] = args.as_slice() {
            if role == "agent" {
                return agent(uuid);
            }
        }
        let filter: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
        type Case = (&'static str, fn(&Path));
        let cases: Vec<Case> = vec![
            ("grant_maps_and_writes", grant_maps_and_writes),
            ("a_weights_page_is_one_chunk_however_big", a_weights_page_is_one_chunk_however_big),
            ("an_unpinned_lease_is_revoked_for_another", an_unpinned_lease_is_revoked_for_another),
            ("a_revoke_waits_for_the_pin", a_revoke_waits_for_the_pin),
            ("an_ask_waits_for_anothers_pins", an_ask_waits_for_anothers_pins),
            ("a_wait_cycle_is_refused", a_wait_cycle_is_refused),
            ("stalls_are_judged_on_the_looks_cadence", stalls_are_judged_on_the_looks_cadence),
            ("an_ask_against_its_own_pins_is_refused", an_ask_against_its_own_pins_is_refused),
            ("a_killed_agents_leases_return_and_landed_weights_are_adopted", a_killed_agents_leases_return_and_landed_weights_are_adopted),
            ("a_stranger_is_refused", a_stranger_is_refused),
        ];
        let mut failed = vec![];
        for (name, case) in cases {
            if !filter.is_empty() && !filter.iter().any(|f| name.contains(f.as_str())) {
                continue;
            }
            let dir = tempfile::tempdir().unwrap();
            print!("test {name} ... ");
            std::io::stdout().flush().unwrap();
            match std::panic::catch_unwind(|| case(dir.path())) {
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

    impl Agentd {
        /// Until the manager's Revoke doorbell reached this agent.
        fn wait_rang(&mut self) {
            let start = Instant::now();
            while self.call("rang") == "0" {
                assert!(start.elapsed() < Duration::from_secs(20), "no revoke reached the agent");
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    /// A byte of `region`'s chunk `index` at `offset`.
    fn pattern(region: u32, index: u32, offset: u64) -> u8 {
        ((u64::from(region) * 31 + u64::from(index) * 7 + offset / 4096) % 251) as u8
    }

    struct Held {
        lease: Lease,
        va: Va,
    }

    /// The child: an agent over host pages, mapping what it is granted at its own VAs.
    fn agent(uuid: &str) {
        let backend: Arc<dyn Backend> = Arc::new(HostBackend::new(1 << 40));
        let rang = Arc::new(AtomicU64::new(0));
        let mut agent: Option<Arc<Agent>> = None;
        let mut held: Vec<Held> = vec![];
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let line = line.unwrap();
            let words: Vec<&str> = line.split_whitespace().collect();
            let out = match words.as_slice() {
                ["attach"] => {
                    let r = rang.clone();
                    let stream = dial(uuid);
                    match Agent::attach(stream, backend.clone(), move || { r.fetch_add(1, SeqCst); }).unwrap() {
                        Attach::Leased(a) => {
                            agent = Some(a);
                            "ok".to_string()
                        }
                        Attach::Refused { code, why } => format!("refused {code:?} {why}"),
                    }
                }
                ["take", bytes, fingerprint, region, rest @ ..] => {
                    let unlanded = rest == ["unlanded"];
                    let a = agent.as_ref().unwrap();
                    let region: u32 = region.parse().unwrap();
                    let wanted = Region { fingerprint: fingerprint.parse().unwrap(), region };
                    match a.take(bytes.parse().unwrap(), Some(wanted)) {
                        Ok(leases) => {
                            let adopted = leases.iter().all(|l| l.adopted);
                            let n = leases.len();
                            let mut verified = true;
                            let mut vas = vec![];
                            for (index, lease) in leases.iter().enumerate() {
                                let va = backend.reserve(lease.phys.len).unwrap();
                                backend.map(va, 0, &lease.phys, Access::ReadWrite).unwrap();
                                let Dst::Host { ptr, len } = backend.target(va, 0, lease.phys.len) else { unreachable!() };
                                // SAFETY: the mapping is ours, `len` bytes long.
                                let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
                                if !lease.adopted {
                                    for (o, b) in bytes.iter_mut().enumerate() {
                                        *b = pattern(region, index as u32, o as u64);
                                    }
                                }
                                verified &= bytes.iter().enumerate().step_by(4093).all(|(o, b)| *b == pattern(region, index as u32, o as u64));
                                vas.push(va);
                            }
                            a.mapped(&leases.iter().map(|l| l.ticket).collect::<Vec<_>>()).unwrap();
                            if !unlanded {
                                leases.iter().for_each(|l| a.landed(l.ticket).unwrap());
                            }
                            held.extend(leases.into_iter().zip(vas).map(|(lease, va)| Held { lease, va }));
                            format!("ok {n} {} {}", if adopted { "adopted" } else { "fresh" }, if verified { "verified" } else { "corrupt" })
                        }
                        Err(error) => format!("refused {error}"),
                    }
                }
                ["pin"] => {
                    let a = agent.as_ref().unwrap();
                    held.iter().for_each(|h| assert!(a.pin(h.lease.ticket).unwrap()));
                    "ok".into()
                }
                ["unpin"] => {
                    let a = agent.as_ref().unwrap();
                    held.iter().for_each(|h| a.unpin(h.lease.ticket).unwrap());
                    "ok".into()
                }
                ["rang"] => rang.load(SeqCst).to_string(),
                ["tick"] => {
                    agent.as_ref().unwrap().progressed();
                    "ok".into()
                }
                ["revokes"] => {
                    let a = agent.as_ref().unwrap();
                    let mut released = 0;
                    let mut kept = vec![];
                    for h in held.drain(..) {
                        if a.asked(h.lease.ticket) && a.commit(h.lease.ticket).unwrap() {
                            backend.unmap(h.va, 0, h.va.len).unwrap();
                            backend.release(h.va);
                            a.release(h.lease.ticket, h.lease.phys).unwrap();
                            released += 1;
                        } else {
                            kept.push(h);
                        }
                    }
                    held = kept;
                    format!("released {released}")
                }
                _ => format!("unknown {line}"),
            };
            println!("{out}");
            std::io::stdout().flush().unwrap();
        }
    }
}

#[cfg(any(target_os = "linux", windows))]
fn main() {
    cases::main();
}

#[cfg(not(any(target_os = "linux", windows)))]
fn main() {
    println!("lease: macOS has no shareable host pages yet; skipped here");
}
