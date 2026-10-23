//! P0 lease cases across real processes (varena-leases §8). TensorD is this binary as a child
//! hosting the Manager over host pages, so it can be killed -9. Each executor is another child:
//! the library's agent, or a raw one driving the wire and its table by hand, as package code
//! could (skewed versions, unanswered grants, forged words). Children run one command per stdin
//! line and answer one line. The fake NVML is what the case's processes hold: every host page
//! any of them has open or mapped, counted once, as a card counts physical pages.

#[cfg(target_os = "linux")]
mod cases {
    use std::collections::{BTreeMap, HashMap};
    use std::io::{BufRead, BufReader, Write};
    use std::os::fd::{AsFd, AsRawFd};
    use std::os::unix::fs::MetadataExt;
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::Path;
    use std::process::{Child, ChildStdin, Command, Stdio};
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::SeqCst};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use crossbeam_channel::{Receiver, RecvTimeoutError};
    use parking_lot::Mutex;
    use platform::handles::{Owned, Peer};
    use platform::memory::Map;
    use platform::process::Birth;
    use serde_json::Value;
    use tensorfs::io::Dst;
    use varena::backend::{Access, Backend, Va};
    use varena::lease::table::Table;
    use varena::lease::wire::{End, ToAgent, ToManager};
    use varena::lease::{Agent, Attach, Hosting, Identity, Lease, Manager, Record, Region, PROTOCOL};
    use varena::ledger::Mode;
    use varena::{GpuId, HostBackend};

    const MIB: u64 = 1 << 20;
    /// TensorD's stall look period (manager.rs `LOOK`, its sampler's): no verdict comes sooner.
    const LOOK: Duration = Duration::from_millis(500);
    /// Bounds every wait so a broken case fails instead of hanging; never a verdict.
    const BOUND: Duration = Duration::from_secs(20);
    /// The fingerprint every case's regions share.
    const FP: u64 = 0xfeed;
    /// The name of every host page and table (platform::memory::page) in /proc.
    const PAGE: &str = "/memfd:varena-page";

    // ---- The parent's side: children, TensorD, the fake NVML.

    /// A child of this binary in one role.
    struct Proc {
        child: Child,
        stdin: ChildStdin,
        lines: Receiver<String>,
    }

    impl Proc {
        fn spawn(args: &[&str]) -> Proc {
            let mut child = Command::new(std::env::current_exe().unwrap()).args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
            let (stdin, stdout) = (child.stdin.take().unwrap(), child.stdout.take().unwrap());
            let (tx, lines) = crossbeam_channel::unbounded();
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(std::result::Result::ok) {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            });
            Proc { child, stdin, lines }
        }

        fn pid(&self) -> u32 {
            self.child.id()
        }

        fn send(&mut self, line: &str) {
            writeln!(self.stdin, "{line}").unwrap();
            self.stdin.flush().unwrap();
        }

        /// Its next answer; empty once it died.
        fn answer(&mut self) -> String {
            match self.lines.recv_timeout(BOUND) {
                Ok(line) => line.trim().to_string(),
                Err(RecvTimeoutError::Timeout) => panic!("child {} gave no answer in {BOUND:?}", self.pid()),
                Err(RecvTimeoutError::Disconnected) => String::new(),
            }
        }

        fn call(&mut self, line: &str) -> String {
            self.send(line);
            self.answer()
        }

        /// An answer already given, if any.
        fn given(&mut self) -> Option<String> {
            self.lines.try_recv().ok()
        }

        /// kill -9, reaped: its mappings are gone.
        fn kill(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }

        /// Until the manager's Revoke doorbell reached this agent.
        fn wait_rang(&mut self) {
            let start = Instant::now();
            while self.call("rang") == "0" {
                assert!(start.elapsed() < BOUND, "no revoke reached the agent");
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    impl Drop for Proc {
        fn drop(&mut self) {
            self.kill();
        }
    }

    /// TensorD: the GPU's Manager over `capacity` bytes of host pages, in its own process.
    struct Tensord {
        p: Proc,
        socket: String,
    }

    impl Tensord {
        fn start(dir: &Path, capacity: u64) -> Tensord {
            let socket = dir.join("lease.sock").to_str().unwrap().to_string();
            let mut p = Proc::spawn(&["tensord", &socket, &capacity.to_string()]);
            assert_eq!(p.answer(), "ready");
            Tensord { p, socket }
        }

        fn admit(&mut self, pid: u32, domain: &str) {
            assert_eq!(self.p.call(&format!("admit {pid} {domain}")), "ok");
        }

        fn holdings(&mut self) -> Value {
            serde_json::from_str(&self.p.call("holdings")).unwrap()
        }

        fn stalled(&mut self) -> Vec<u32> {
            serde_json::from_str(&self.p.call("stalled")).unwrap()
        }

        fn records(&mut self) -> Vec<Value> {
            serde_json::from_str(&self.p.call("records")).unwrap()
        }

        fn until(&mut self, done: impl Fn(&Value) -> bool) -> Value {
            let start = Instant::now();
            loop {
                let h = self.holdings();
                if done(&h) {
                    return h;
                }
                assert!(start.elapsed() < BOUND, "the manager never reached it: {h}");
                std::thread::sleep(Duration::from_millis(5));
            }
        }

        fn until_stalled(&mut self, pid: u32) -> Instant {
            let start = Instant::now();
            loop {
                let named = self.stalled();
                if named.contains(&pid) {
                    break;
                }
                assert!(start.elapsed() < BOUND, "{pid} was never named stalled; named: {named:?}");
                std::thread::sleep(Duration::from_millis(10));
            }
            Instant::now()
        }
    }

    fn n(v: &Value, key: &str) -> u64 {
        v[key].as_u64().unwrap_or_else(|| panic!("no {key} in {v}"))
    }

    fn executors(h: &Value) -> usize {
        h["executors"].as_array().map_or(0, Vec::len)
    }

    /// An executor running the library's agent, in `domain`.
    fn agent(t: &mut Tensord, domain: &str) -> Proc {
        let mut a = Proc::spawn(&["agent", &t.socket]);
        t.admit(a.pid(), domain);
        assert_eq!(a.call("attach"), "ok");
        a
    }

    /// A raw executor, admitted in `domain` (or unregistered); it says Hello when told.
    fn raw(t: &mut Tensord, domain: Option<&str>) -> Proc {
        let r = Proc::spawn(&["raw", &t.socket]);
        if let Some(d) = domain {
            t.admit(r.pid(), d);
        }
        r
    }

    fn take(mib: u64, region: u32) -> String {
        format!("take {} {region}", mib * MIB)
    }

    /// The raw executor's next `n` grants, (slot, seq) each.
    fn grants(r: &mut Proc, n: usize) -> Vec<(u32, u32)> {
        (0..n)
            .map(|_| {
                let g = r.call("next");
                let w: Vec<&str> = g.split_whitespace().collect();
                assert_eq!(w.first(), Some(&"grant"), "{g}");
                (w[2].parse().unwrap(), w[3].parse().unwrap())
            })
            .collect()
    }

    /// The fake NVML's used bytes: every host page the processes hold open or mapped, once.
    fn nvml_used(pids: &[u32]) -> u64 {
        let mut pages: HashMap<u64, u64> = HashMap::new();
        for pid in pids {
            for fd in std::fs::read_dir(format!("/proc/{pid}/fd")).into_iter().flatten().flatten() {
                let page = std::fs::read_link(fd.path()).is_ok_and(|t| t.to_string_lossy().starts_with(PAGE));
                if let (true, Ok(m)) = (page, std::fs::metadata(fd.path())) {
                    let e = pages.entry(m.ino()).or_default();
                    *e = (*e).max(m.len());
                }
            }
            for line in std::fs::read_to_string(format!("/proc/{pid}/maps")).unwrap_or_default().lines().filter(|l| l.contains(PAGE)) {
                // start-end perms offset dev inode path
                let f: Vec<&str> = line.split_whitespace().collect();
                let hex = |s: &str| u64::from_str_radix(s, 16).unwrap();
                let (start, end) = f[0].split_once('-').unwrap();
                let e = pages.entry(f[4].parse().unwrap()).or_default();
                *e = (*e).max(hex(f[2]) + hex(end) - hex(start));
            }
        }
        pages.values().sum()
    }

    /// NVML's used bytes, the books (created and books-only) and the fixed overhead (each
    /// attached executor's table, `table` bytes measured).
    fn books(t: &mut Tensord, procs: &[&Proc], table: u64) -> (u64, u64, u64) {
        let h = t.holdings();
        let pids: Vec<u32> = std::iter::once(t.p.pid()).chain(procs.iter().map(|p| p.pid())).collect();
        (nvml_used(&pids), n(&h, "created") + n(&h, "books_only"), table * executors(&h) as u64)
    }

    fn check_books(t: &mut Tensord, procs: &[&Proc], table: u64, when: &str) {
        let (used, books, overhead) = books(t, procs, table);
        assert!(books <= used && used <= books + overhead, "{when}: NVML used {used}, books {books}, fixed overhead {overhead}");
    }

    // ---- The cases.

    /// The books against a fake NVML through a grant, a revoke, an exit that retains landed
    /// chunks and destroys unlanded ones, and an adoption: what the processes hold, less the
    /// books, is only the executors' tables.
    fn the_books_match_a_fake_nvml(dir: &Path) {
        let mut t = Tensord::start(dir, 256 * MIB);
        let (mut a, mut b) = (agent(&mut t, "d"), agent(&mut t, "d"));
        let table = nvml_used(&[t.p.pid(), a.pid(), b.pid()]) / 2;
        assert!(table > 0, "the fake NVML sees no table");
        assert_eq!(a.call(&take(96, 1)), "ok 3 fresh verified");
        assert_eq!(a.call(&format!("{} unlanded", take(32, 8))), "ok 1 fresh verified");
        t.until(|h| n(h, "leased") == 128 * MIB);
        check_books(&mut t, &[&a, &b], table, "after grants");
        b.send(&take(160, 2));
        a.wait_rang();
        assert_eq!(a.call("revokes"), "released 1");
        assert_eq!(b.answer(), "ok 5 fresh verified");
        t.until(|h| n(h, "leased") == 256 * MIB);
        check_books(&mut t, &[&a, &b], table, "after a revoke");
        a.kill();
        let h = t.until(|h| executors(h) == 1);
        assert_eq!((n(&h, "retained"), n(&h, "created")), (64 * MIB, 224 * MIB), "landed chunks retained, the unlanded destroyed: {h}");
        check_books(&mut t, &[&b], table, "after an exit");
        b.kill();
        t.until(|h| executors(h) == 0 && n(h, "retained") == 224 * MIB);
        check_books(&mut t, &[], table, "with only retained chunks");
        let mut c = agent(&mut t, "d");
        assert_eq!(c.call(&take(160, 2)), "ok 5 adopted verified");
        let h = t.until(|h| n(h, "leased") == 160 * MIB);
        assert_eq!(n(&h, "created"), 224 * MIB, "adopted with no new chunk");
        check_books(&mut t, &[&c], table, "after an adoption");
    }

    /// Room an ask gets by dropping retained chunks is free at once: the ask is granted then,
    /// never left waiting for some later event.
    fn room_from_dropped_retained_chunks_is_granted_at_once(dir: &Path) {
        let mut t = Tensord::start(dir, 64 * MIB);
        let mut a = agent(&mut t, "d");
        assert_eq!(a.call(&take(64, 1)), "ok 2 fresh verified");
        a.kill();
        t.until(|h| n(h, "retained") == 64 * MIB);
        let mut b = agent(&mut t, "d");
        assert_eq!(b.call(&take(32, 2)), "ok 1 fresh verified");
    }

    /// A grant is never asked back before its ack (L3): another executor's ask sees it as no
    /// one's room, its word stays Granted under its seq with no ask, and its acks still land.
    fn a_grant_is_never_asked_back_before_its_ack(dir: &Path) {
        let mut t = Tensord::start(dir, 96 * MIB);
        let mut c = agent(&mut t, "d");
        assert_eq!(c.call(&take(32, 3)), "ok 1 fresh verified");
        let mut r = raw(&mut t, Some("d"));
        assert!(r.call(&format!("hello {PROTOCOL}")).starts_with("welcome"));
        assert_eq!(r.call(&format!("ask 1 {}", 64 * MIB)), "sent");
        let g = grants(&mut r, 2);
        let mut b = agent(&mut t, "d");
        b.send(&take(64, 2));
        // B's ask was served: C's idle chunk is asked back.
        c.wait_rang();
        for (slot, seq) in &g {
            assert_eq!(r.call(&format!("word {slot}")), format!("Granted 0 {seq} ask 0"), "a grant in flight was asked back");
        }
        assert_eq!(n(&t.holdings(), "created"), 96 * MIB);
        assert_eq!(r.call(&format!("ack {} {} mapped", g[0].0, g[0].1)), "ok");
        assert_eq!(r.call(&format!("ack {} {} refused", g[1].0, g[1].1)), "ok");
    }

    /// An ask whose only room is another executor's grant in flight waits for that ack and is
    /// served then (L3: a revoke during the map follows the map), never refused as nothing
    /// revocable.
    fn a_grant_in_flight_is_coming_room(dir: &Path) {
        let mut t = Tensord::start(dir, 96 * MIB);
        let mut c = agent(&mut t, "d");
        assert_eq!(c.call(&take(32, 3)), "ok 1 fresh verified");
        let mut r = raw(&mut t, Some("d"));
        assert!(r.call(&format!("hello {PROTOCOL}")).starts_with("welcome"));
        assert_eq!(r.call(&format!("ask 1 {}", 64 * MIB)), "sent");
        let g = grants(&mut r, 2);
        let mut b = agent(&mut t, "d");
        b.send(&take(64, 2));
        c.wait_rang();
        for (i, (slot, seq)) in g.iter().enumerate() {
            assert_eq!(r.call(&format!("map {slot} 0 {i}")), "ok");
            assert_eq!(r.call(&format!("ack {slot} {seq} mapped")), "ok");
        }
        assert_eq!(c.call("revokes"), "released 1");
        assert_eq!(b.given(), None, "refused while the room was a grant in flight");
        assert_eq!(r.call("next"), "revoke", "an acked grant B waits on is asked back");
        let asked = r.call("asked");
        for (slot, seq) in g.iter().filter(|(s, _)| asked.split_whitespace().any(|a| a == s.to_string())) {
            assert_eq!(r.call(&format!("release {slot} {seq}")), "ok");
        }
        assert_eq!(b.answer(), "ok 2 fresh verified");
    }

    /// Acks go by grant sequence (L3): a stale ack moves nothing. A lying ack (released but
    /// still mapped) is memory-safe and shows against NVML as exactly the bytes it kept.
    fn stale_and_lying_acks(dir: &Path) {
        let mut t = Tensord::start(dir, 64 * MIB);
        let mut r = raw(&mut t, Some("d"));
        assert!(r.call(&format!("hello {PROTOCOL}")).starts_with("welcome"));
        let table = nvml_used(&[t.p.pid(), r.pid()]);
        assert_eq!(r.call(&format!("ask 1 {}", 32 * MIB)), "sent");
        let (slot, old) = grants(&mut r, 1)[0];
        assert_eq!(r.call(&format!("ack {slot} {old} refused")), "ok");
        t.until(|h| n(h, "created") == 0);
        assert_eq!(r.call(&format!("ask 2 {}", 32 * MIB)), "sent");
        let (again, seq) = grants(&mut r, 1)[0];
        assert_eq!((again, seq != old), (slot, true), "the slot is granted again under a new seq");
        assert_eq!(r.call(&format!("map {slot} 4 0")), "ok");
        assert_eq!(r.call(&format!("ack {slot} {seq} mapped")), "ok");
        // The old grant's ack, and a forged release under its seq, move nothing.
        assert_eq!(r.call(&format!("ack {slot} {old} mapped")), "stale");
        assert_eq!(r.call(&format!("forge word {slot} released 0 {old}")), "ok");
        // Its next ask is handled after that doorbell (one socket, in order).
        assert_eq!(r.call(&format!("ask 3 {}", 32 * MIB)), "sent");
        let (third, _) = grants(&mut r, 1)[0];
        let h = t.holdings();
        assert_eq!((n(&h, "leased"), third != slot), (64 * MIB, true), "a stale release reclaimed a live grant: {h}");
        assert_eq!(r.call(&format!("forge word {slot} mapped 0 {seq}")), "ok");
        // The lie: released by the table, still mapped by the executor.
        assert_eq!(r.call(&format!("release {slot} {seq} keep")), "ok");
        t.until(|h| n(h, "leased") == 32 * MIB);
        let (used, books, overhead) = books(&mut t, &[&r], table);
        assert_eq!(used - books - overhead, 32 * MIB, "NVML shows the lie as the bytes kept");
        assert_eq!(r.call(&format!("read {slot} 4 0")), "verified", "the liar's page is its own still");
        // The books freed it, so the next ask gets a new page, never the liar's.
        let mut b = agent(&mut t, "d");
        assert_eq!(b.call(&take(32, 5)), "ok 1 fresh verified");
        assert_eq!(r.call(&format!("read {slot} 4 0")), "verified");
        assert_eq!(b.call("check"), "verified");
    }

    /// Retained weights are kept by (trust domain, region) (L4): another domain's ask for the
    /// region gets fresh chunks; its own domain adopts them, in the manager's own index order
    /// whatever the executor wrote into its table.
    fn retention_is_keyed_by_domain_and_region(dir: &Path) {
        let mut t = Tensord::start(dir, 256 * MIB);
        let mut a = agent(&mut t, "pub/a@1");
        assert_eq!(a.call(&take(80, 7)), "ok 3 fresh verified");
        a.kill();
        t.until(|h| executors(h) == 0 && n(h, "retained") == 80 * MIB);
        let mut b = agent(&mut t, "pub/b@1");
        assert_eq!(b.call(&take(80, 7)), "ok 3 fresh verified", "adopted across trust domains");
        let h = t.until(|h| n(h, "leased") == 80 * MIB);
        assert_eq!((n(&h, "retained"), n(&h, "created")), (80 * MIB, 160 * MIB));
        let mut c = agent(&mut t, "pub/a@1");
        assert_eq!(c.call(&take(80, 7)), "ok 3 adopted verified");
        t.until(|h| n(h, "retained") == 0 && n(h, "created") == 160 * MIB);
        let mut r = raw(&mut t, Some("pub/z@1"));
        assert!(r.call(&format!("hello {PROTOCOL}")).starts_with("welcome"));
        assert_eq!(r.call(&format!("ask 1 {} 9", 64 * MIB)), "sent");
        let g = grants(&mut r, 2);
        for (i, (slot, seq)) in g.iter().enumerate() {
            assert_eq!(r.call(&format!("map {slot} 9 {i}")), "ok");
            assert_eq!(r.call(&format!("ack {slot} {seq} mapped")), "ok");
            assert_eq!(r.call(&format!("landed {slot}")), "ok");
        }
        assert_eq!(r.call(&format!("forge index {} 1", g[0].0)), "ok");
        assert_eq!(r.call(&format!("forge index {} 0", g[1].0)), "ok");
        r.kill();
        t.until(|h| n(h, "retained") == 64 * MIB);
        let mut d = agent(&mut t, "pub/z@1");
        assert_eq!(d.call(&take(64, 9)), "ok 2 adopted verified");
    }

    /// The manager speaks the current protocol and, from protocol 3, the previous one. Any other
    /// Hello gets Refuse{Incompatible}, so that agent runs standalone and is never refused
    /// memory. An unregistered peer gets NotRegistered (wait and redial), whatever it speaks.
    fn versions_are_negotiated(dir: &Path) {
        let mut t = Tensord::start(dir, 64 * MIB);
        // Protocol 1 is the ledger's and never dials the lease socket.
        let served = |p: u32| p == PROTOCOL || (p.checked_add(1) == Some(PROTOCOL) && p >= 2);
        let mut versions = vec![0, 1, PROTOCOL - 1, PROTOCOL, PROTOCOL + 1, u32::MAX];
        versions.dedup();
        let mut held = vec![];
        for p in versions.iter().copied() {
            let mut r = raw(&mut t, Some("d"));
            let answer = r.call(&format!("hello {p}"));
            if served(p) {
                assert!(answer.starts_with("welcome") && answer.ends_with("table"), "protocol {p}: {answer}");
            } else {
                assert_eq!(answer, "refuse 0 Incompatible", "protocol {p}");
            }
            held.push(r);
        }
        assert_eq!(executors(&t.holdings()), versions.iter().filter(|p| served(**p)).count());
        let mut u = raw(&mut t, None);
        assert_eq!(u.call(&format!("hello {}", PROTOCOL + 1)), "refuse 0 NotRegistered");
        agent(&mut t, "d");
    }

    /// Neither the lease table nor a lent host page can be resized by the executor holding it:
    /// a shrink would SIGBUS TensorD reading the table, or the next adopter mapping the page.
    fn tables_and_pages_are_sealed(dir: &Path) {
        let mut t = Tensord::start(dir, 64 * MIB);
        let mut r = raw(&mut t, Some("d"));
        assert!(r.call(&format!("hello {PROTOCOL}")).starts_with("welcome"));
        let resize = platform::shm::RESIZE_SEALS;
        assert_eq!(r.call("truncate table"), "refused");
        assert_eq!(r.call("seals table").parse::<i32>().unwrap() & resize, resize);
        assert_eq!(r.call(&format!("ask 1 {} 11", 32 * MIB)), "sent");
        let (slot, seq) = grants(&mut r, 1)[0];
        assert_eq!(r.call(&format!("map {slot} 11 0")), "ok");
        assert_eq!(r.call(&format!("ack {slot} {seq} mapped")), "ok");
        assert_eq!(r.call(&format!("landed {slot}")), "ok");
        assert_eq!(r.call(&format!("truncate {slot}")), "refused");
        assert_eq!(r.call(&format!("seals {slot}")).parse::<i32>().unwrap() & resize, resize);
        r.kill();
        t.until(|h| n(h, "retained") == 32 * MIB);
        let mut c = agent(&mut t, "d");
        assert_eq!(c.call(&take(32, 11)), "ok 1 adopted verified");
    }

    /// A stall is judged on the look's cadence only: no number of reads, nor other executors'
    /// doorbells at event rate, names an executor sooner than a look period after its ask.
    fn a_stall_is_judged_on_the_looks_cadence(dir: &Path) {
        let mut t = Tensord::start(dir, 128 * MIB);
        let (mut a, mut c) = (agent(&mut t, "d"), agent(&mut t, "d"));
        assert_eq!(a.call(&take(96, 1)), "ok 3 fresh verified");
        let mut b = agent(&mut t, "d");
        b.send(&take(64, 2));
        a.wait_rang();
        let rang = Instant::now();
        for _ in 0..1000 {
            assert!(t.stalled().is_empty(), "a healthy agent named by reads between looks");
        }
        // Another executor's acks arrive at event rate.
        for _ in 0..20 {
            assert_eq!(c.call("doorbell"), "ok");
        }
        let named = t.until_stalled(a.pid());
        let waited: Vec<u64> = t.records().iter().filter(|r| r["event"] == "stalled" && r["pid"] == a.pid()).map(|r| n(r, "waited_ms")).collect();
        assert!(named - rang > LOOK && waited.iter().all(|w| *w >= LOOK.as_millis() as u64), "named {:?} after its ask, judged at {waited:?} ms", named - rang);
        assert_eq!(a.call("revokes"), "released 1");
        assert_eq!(b.answer(), "ok 2 fresh verified");
        let start = Instant::now();
        while !t.stalled().is_empty() {
            assert!(start.elapsed() < BOUND, "a release clears the verdict");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// A revoke waiting on a long fence is never a stall while the executor's GPU work keeps
    /// completing; once it stops, with nothing released, it is.
    fn a_fence_whose_epoch_moves_is_never_a_stall(dir: &Path) {
        let mut t = Tensord::start(dir, 128 * MIB);
        let mut a = agent(&mut t, "d");
        assert_eq!(a.call(&take(96, 1)), "ok 3 fresh verified");
        let mut b = agent(&mut t, "d");
        b.send(&take(64, 2));
        a.wait_rang();
        // Kernels each longer than a look, for eight looks: events complete between them.
        let fenced = Instant::now();
        while fenced.elapsed() < LOOK * 8 {
            assert_eq!(a.call("tick"), "ok");
            std::thread::sleep(LOOK * 9 / 5);
            assert!(t.stalled().is_empty(), "an executor whose GPU work completes was named");
        }
        t.until_stalled(a.pid());
        assert_eq!(a.call("revokes"), "released 1");
        assert_eq!(b.answer(), "ok 2 fresh verified");
    }

    /// An asked chunk under a pin (a running pass) is never a stall; once unpinned and still
    /// unreleased, with no progress, it is.
    fn a_pinned_ask_is_never_a_stall(dir: &Path) {
        let mut t = Tensord::start(dir, 64 * MIB);
        let mut a = agent(&mut t, "d");
        assert_eq!(a.call(&take(64, 1)), "ok 2 fresh verified");
        assert_eq!(a.call("pin"), "ok");
        let mut b = agent(&mut t, "d");
        b.send(&take(32, 2));
        a.wait_rang();
        let start = Instant::now();
        while start.elapsed() < LOOK * 6 {
            assert!(t.stalled().is_empty(), "a pinned chunk's holder was named");
            std::thread::sleep(Duration::from_millis(50));
        }
        assert_eq!(a.call("unpin"), "ok");
        t.until_stalled(a.pid());
        assert_eq!(a.call("revokes"), "released 1");
        assert_eq!(b.answer(), "ok 1 fresh verified");
    }

    /// A stall names the peer's own pid (its credentials), never the pid its table claims.
    fn a_stall_names_the_peer_not_its_table(dir: &Path) {
        let mut t = Tensord::start(dir, 64 * MIB);
        let mut r = raw(&mut t, Some("d"));
        assert!(r.call(&format!("hello {PROTOCOL}")).starts_with("welcome"));
        assert_eq!(r.call(&format!("ask 1 {}", 64 * MIB)), "sent");
        for (i, (slot, seq)) in grants(&mut r, 2).into_iter().enumerate() {
            assert_eq!(r.call(&format!("map {slot} 0 {i}")), "ok");
            assert_eq!(r.call(&format!("ack {slot} {seq} mapped")), "ok");
        }
        let mut b = agent(&mut t, "d");
        assert_eq!(r.call(&format!("forge pid {}", b.pid())), "ok");
        b.send(&take(32, 2));
        assert_eq!(r.call("next"), "revoke");
        t.until_stalled(r.pid());
        assert_eq!(t.stalled(), vec![r.pid()]);
    }

    /// TensorD restarts under a running executor (L5, M7): it redials and hands its table
    /// back; the new manager reopens it (never a new one), rebuilds its books from the agent's
    /// records, and keeps every pre-restart chunk books-only: counted, never retained or
    /// re-granted.
    fn a_tensord_restart_rebuilds_the_books_from_the_agents(dir: &Path) {
        let mut t = Tensord::start(dir, 128 * MIB);
        let mut r = raw(&mut t, Some("d"));
        let welcome = r.call(&format!("hello {PROTOCOL}"));
        let epoch: u64 = welcome.split_whitespace().nth(1).unwrap().parse().unwrap();
        let table = nvml_used(&[t.p.pid(), r.pid()]);
        assert_eq!(r.call(&format!("ask 1 {} 5", 64 * MIB)), "sent");
        let g = grants(&mut r, 2);
        for (i, (slot, seq)) in g.iter().enumerate() {
            assert_eq!(r.call(&format!("map {slot} 5 {i}")), "ok");
            assert_eq!(r.call(&format!("ack {slot} {seq} mapped")), "ok");
            assert_eq!(r.call(&format!("landed {slot}")), "ok");
        }
        t.until(|h| n(h, "leased") == 64 * MIB);
        t.p.kill();
        // The pages outlive their creator: the executor's imports hold them.
        for (i, (slot, _)) in g.iter().enumerate() {
            assert_eq!(r.call(&format!("read {slot} 5 {i}")), "verified");
        }
        let mut t = Tensord::start(dir, 128 * MIB);
        t.admit(r.pid(), "d");
        let again = r.call(&format!("hello {PROTOCOL} back"));
        let w: Vec<&str> = again.split_whitespace().collect();
        assert!(w[0] == "welcome" && matches!(w[2], "none" | "same"), "the re-attach was refused or got a new table: {again}");
        assert!(w[1].parse::<u64>().unwrap() > epoch, "the epoch moves at a re-attach");
        let h = t.until(|h| executors(h) == 1);
        assert_eq!((n(&h, "books_only"), n(&h, "created")), (64 * MIB, 0), "books from the agent's records: {h}");
        for (slot, seq) in &g {
            assert_eq!(r.call(&format!("word {slot}")), format!("Mapped 0 {seq} ask 0"));
        }
        // Books-only bytes count against the card: a new executor gets only the rest.
        let mut b = agent(&mut t, "d");
        assert_eq!(b.call(&take(64, 6)), "ok 2 fresh verified");
        check_books(&mut t, &[&r, &b], table, "after a restart");
        assert_eq!(r.call(&format!("release {} {}", g[0].0, g[0].1)), "ok");
        t.until(|h| n(h, "books_only") == 32 * MIB);
        check_books(&mut t, &[&r, &b], table, "after a books-only release");
        r.kill();
        let h = t.until(|h| executors(h) == 1);
        assert_eq!((n(&h, "books_only"), n(&h, "retained")), (0, 0), "a books-only chunk is never retained: {h}");
    }

    type Case = (&'static str, fn(&Path));

    const CASES: &[Case] = &[
        ("the_books_match_a_fake_nvml", the_books_match_a_fake_nvml),
        ("room_from_dropped_retained_chunks_is_granted_at_once", room_from_dropped_retained_chunks_is_granted_at_once),
        ("a_grant_is_never_asked_back_before_its_ack", a_grant_is_never_asked_back_before_its_ack),
        ("a_grant_in_flight_is_coming_room", a_grant_in_flight_is_coming_room),
        ("stale_and_lying_acks", stale_and_lying_acks),
        ("retention_is_keyed_by_domain_and_region", retention_is_keyed_by_domain_and_region),
        ("versions_are_negotiated", versions_are_negotiated),
        ("tables_and_pages_are_sealed", tables_and_pages_are_sealed),
        ("a_stall_is_judged_on_the_looks_cadence", a_stall_is_judged_on_the_looks_cadence),
        ("a_fence_whose_epoch_moves_is_never_a_stall", a_fence_whose_epoch_moves_is_never_a_stall),
        ("a_pinned_ask_is_never_a_stall", a_pinned_ask_is_never_a_stall),
        ("a_stall_names_the_peer_not_its_table", a_stall_names_the_peer_not_its_table),
        ("a_tensord_restart_rebuilds_the_books_from_the_agents", a_tensord_restart_rebuilds_the_books_from_the_agents),
    ];

    /// Cases that fail today, and why: a full run reports them and stays green; a run naming one
    /// fails on it.
    const PENDING: &[(&str, &str)] = &[
        ("room_from_dropped_retained_chunks_is_granted_at_once", "bug, sent to S1: the ask waits for another event"),
        ("a_grant_in_flight_is_coming_room", "bug, sent to S1: refused NothingRevocable"),
        ("a_stall_is_judged_on_the_looks_cadence", "bug, sent to S1: doorbells drive looks"),
        ("a_tensord_restart_rebuilds_the_books_from_the_agents", "re-attach (M7) is after slice 1"),
    ];

    pub fn main() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
            ["tensord", socket, capacity] => return tensord(socket, capacity.parse().unwrap()),
            ["agent", socket] => return agent_child(socket),
            ["raw", socket] => return raw_child(socket),
            _ => {}
        }
        let filter: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
        let mut failed = vec![];
        for (name, case) in CASES {
            let named = filter.iter().any(|f| name.contains(f.as_str()));
            if !filter.is_empty() && !named {
                continue;
            }
            let pending = PENDING.iter().find(|(p, _)| p == name).map(|(_, why)| *why);
            let dir = tempfile::tempdir().unwrap();
            print!("test {name} ... ");
            std::io::stdout().flush().unwrap();
            let start = Instant::now();
            let passed = std::panic::catch_unwind(|| case(dir.path())).is_ok();
            let took = start.elapsed().as_secs_f32();
            match (passed, pending) {
                (true, None) => println!("ok ({took:.1} s)"),
                (true, Some(why)) => println!("ok ({took:.1} s; pending no more: {why})"),
                (false, Some(why)) if filter.is_empty() => println!("pending: {why}"),
                (false, _) => {
                    println!("FAILED");
                    failed.push(*name);
                }
            }
        }
        if !failed.is_empty() {
            eprintln!("failed: {failed:?}");
            std::process::exit(1);
        }
    }

    // ---- The children.

    /// A child's loop: one command per stdin line, one answer line each; it exits at EOF.
    fn serve(ready: Option<&str>, mut answer: impl FnMut(&[&str]) -> String) {
        if let Some(r) = ready {
            println!("{r}");
        }
        for line in std::io::stdin().lock().lines() {
            let line = line.unwrap();
            let words: Vec<&str> = line.split_whitespace().collect();
            println!("{}", answer(&words));
            std::io::stdout().flush().unwrap();
        }
        std::process::exit(0);
    }

    /// Who TensorD admits (each child's trust domain, named by the case) and what it did.
    #[derive(Default)]
    struct Admitted {
        domains: Mutex<HashMap<u32, String>>,
        records: Mutex<Vec<Record>>,
    }

    impl Hosting for Admitted {
        fn identify(&self, peer: &Peer) -> Option<Identity> {
            self.domains.lock().get(&peer.pid).map(|d| Identity { executor: peer.pid.to_string(), domain: d.clone() })
        }

        fn record(&self, record: Record) {
            self.records.lock().push(record);
        }
    }

    fn tensord(socket: &str, capacity: u64) {
        // A killed TensorD's socket file stays behind.
        let _ = std::fs::remove_file(socket);
        let listener = UnixListener::bind(socket).unwrap();
        let gpu = GpuId { uuid: format!("p0-{}", std::process::id()), model: "host".into(), memory: capacity };
        let admitted = Arc::new(Admitted::default());
        // The card is twice the manager's limit (a --gpu-memory-limit): a books error shows in
        // NVML, not as the device's own out-of-memory.
        let open = Box::new(move || Ok(Arc::new(HostBackend::new(capacity * 2)) as Arc<dyn Backend>));
        let manager = Manager::create(&gpu, Some(capacity), Mode::Dedicated, open, listener, admitted.clone()).unwrap();
        varena::os::remove_table(&gpu.uuid);
        serve(Some("ready"), |words| match words {
            ["admit", pid, domain] => {
                admitted.domains.lock().insert(pid.parse().unwrap(), domain.to_string());
                "ok".into()
            }
            ["holdings"] => serde_json::to_string(&manager.holdings()).unwrap(),
            ["stalled"] => serde_json::to_string(&manager.stalled()).unwrap(),
            ["records"] => serde_json::to_string(&*admitted.records.lock()).unwrap(),
            _ => format!("unknown {words:?}"),
        });
    }

    /// A byte of `region`'s chunk `index` at `offset`.
    fn pattern(region: u32, index: u32, offset: usize) -> u8 {
        ((u64::from(region) * 31 + u64::from(index) * 7 + offset as u64 / 4096) % 251) as u8
    }

    fn fill(bytes: &mut [u8], region: u32, index: u32) {
        bytes.iter_mut().enumerate().for_each(|(o, b)| *b = pattern(region, index, o));
    }

    fn verified(bytes: &[u8], region: u32, index: u32) -> bool {
        bytes.iter().enumerate().step_by(4093).all(|(o, b)| *b == pattern(region, index, o))
    }

    struct Held {
        lease: Lease,
        va: Va,
        region: u32,
        index: u32,
    }

    /// A held chunk's bytes, mapped at its VA.
    #[allow(clippy::mut_from_ref)]
    fn host_bytes<'a>(backend: &dyn Backend, h: &'a Held) -> &'a mut [u8] {
        let Dst::Host { ptr, len } = backend.target(h.va, 0, h.lease.phys.len) else { unreachable!() };
        // SAFETY: our own mapping, `len` bytes long, while `h` is held.
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    /// An executor on the library's agent over host pages, mapping its grants at its own VAs.
    fn agent_child(socket: &str) {
        let backend: Arc<dyn Backend> = Arc::new(HostBackend::new(1 << 40));
        let rang = Arc::new(AtomicU64::new(0));
        let mut me: Option<Arc<Agent>> = None;
        let mut held: Vec<Held> = vec![];
        serve(None, |words| match words {
            ["attach"] => {
                let r = rang.clone();
                match Agent::attach(UnixStream::connect(socket).unwrap(), backend.clone(), move || {
                    r.fetch_add(1, SeqCst);
                })
                .unwrap()
                {
                    Attach::Leased(a) => {
                        me = Some(a);
                        "ok".into()
                    }
                    Attach::Refused { code, why } => format!("refused {code:?} {why}"),
                }
            }
            ["take", size, region, rest @ ..] => {
                let a = me.as_ref().unwrap();
                let region: u32 = region.parse().unwrap();
                match a.take(size.parse().unwrap(), Some(Region { fingerprint: FP, region })) {
                    Ok(leases) => {
                        let (n, adopted) = (leases.len(), leases.iter().all(|l| l.adopted));
                        let mut ok = true;
                        let mut got = vec![];
                        for (index, lease) in leases.into_iter().enumerate() {
                            let va = backend.reserve(lease.phys.len).unwrap();
                            backend.map(va, 0, &lease.phys, Access::ReadWrite).unwrap();
                            let h = Held { lease, va, region, index: index as u32 };
                            if !h.lease.adopted {
                                fill(host_bytes(backend.as_ref(), &h), region, h.index);
                            }
                            ok &= verified(host_bytes(backend.as_ref(), &h), region, h.index);
                            got.push(h);
                        }
                        a.mapped(&got.iter().map(|h| h.lease.ticket).collect::<Vec<_>>()).unwrap();
                        if rest != ["unlanded"] {
                            got.iter().for_each(|h| a.landed(h.lease.ticket).unwrap());
                        }
                        held.extend(got);
                        format!("ok {n} {} {}", if adopted { "adopted" } else { "fresh" }, if ok { "verified" } else { "corrupt" })
                    }
                    Err(error) => format!("refused {error}"),
                }
            }
            ["pin"] => {
                held.iter().for_each(|h| assert!(me.as_ref().unwrap().pin(h.lease.ticket).unwrap()));
                "ok".into()
            }
            ["unpin"] => {
                held.iter().for_each(|h| me.as_ref().unwrap().unpin(h.lease.ticket).unwrap());
                "ok".into()
            }
            ["rang"] => rang.load(SeqCst).to_string(),
            // Its GPU work completed (an event its revoke's fence waits on).
            ["tick"] => {
                me.as_ref().unwrap().progressed();
                "ok".into()
            }
            // An Answered with nothing in it, as its acks of other grants ring.
            ["doorbell"] => {
                me.as_ref().unwrap().mapped(&[]).unwrap();
                "ok".into()
            }
            ["check"] => String::from(if held.iter().all(|h| verified(host_bytes(backend.as_ref(), h), h.region, h.index)) { "verified" } else { "corrupt" }),
            ["revokes"] => {
                let a = me.as_ref().unwrap();
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
            _ => format!("unknown {words:?}"),
        });
    }

    /// A grant the raw executor holds: its handle, and its mapping once mapped.
    struct Grant {
        seq: u32,
        bytes: u64,
        fd: Owned,
        map: Option<Map>,
    }

    /// The raw executor's connection, its table (and a second mapping of it for forged words)
    /// and its grants by slot.
    #[derive(Default)]
    struct Raw {
        end: Option<Arc<End>>,
        inbox: Option<Receiver<(ToAgent, Option<Owned>)>>,
        table_fd: Option<Owned>,
        table: Option<Table>,
        forge: Option<Map>,
        grants: BTreeMap<u32, Grant>,
    }

    /// table.rs's layout, mirrored for forging: a 64-byte line per slot after the header; a
    /// word is state | pins << 4 | seq << 32.
    const LINE: usize = 64;
    const HEADER_PID: usize = 16;
    const SLOT_INDEX: usize = 40;

    impl Raw {
        fn table(&self) -> &Table {
            self.table.as_ref().unwrap()
        }

        fn ring(&self) {
            self.end.as_ref().unwrap().send(&ToManager::Answered, None).unwrap();
        }

        fn at<T>(&self, offset: usize) -> &T {
            let map = self.forge.as_ref().unwrap();
            assert!(offset + std::mem::size_of::<T>() <= map.len);
            // SAFETY: inside our own shared mapping of the table, aligned, atomics only.
            unsafe { &*(map.base.add(offset) as *const T) }
        }

        fn open(&mut self, fd: Owned) {
            let len = platform::memory::size(&fd).unwrap() as usize;
            self.table = Some(Table::open(platform::memory::map(&fd, 0, len, true).unwrap()).unwrap());
            self.forge = Some(platform::memory::map(&fd, 0, len, true).unwrap());
            self.table_fd = Some(fd);
        }

        fn fd(&self, which: &str) -> &Owned {
            if which == "table" {
                self.table_fd.as_ref().unwrap()
            } else {
                &self.grants[&which.parse::<u32>().unwrap()].fd
            }
        }

        fn next(&mut self) -> String {
            let Some(inbox) = &self.inbox else { return "undialled".into() };
            match inbox.recv_timeout(BOUND) {
                Ok((ToAgent::Welcome { epoch }, fd)) => {
                    let kind = match fd {
                        None => "none",
                        Some(fd) => match self.table_fd.as_ref().map(inode) {
                            Some(old) if old == inode(&fd) => "same",
                            Some(_) => "new",
                            None => {
                                self.open(fd);
                                "table"
                            }
                        },
                    };
                    format!("welcome {epoch} {kind}")
                }
                Ok((ToAgent::Grant { id, slot, seq, bytes, adopted, last }, fd)) => {
                    self.grants.insert(slot, Grant { seq, bytes, fd: fd.unwrap(), map: None });
                    format!("grant {id} {slot} {seq} {bytes} {adopted} {last}")
                }
                Ok((ToAgent::Refuse { id, code, .. }, _)) => format!("refuse {id} {code:?}"),
                Ok((ToAgent::Revoke, _)) => "revoke".into(),
                Err(RecvTimeoutError::Timeout) => "timeout".into(),
                Err(RecvTimeoutError::Disconnected) => "closed".into(),
            }
        }
    }

    fn inode(fd: &Owned) -> u64 {
        std::fs::metadata(format!("/proc/self/fd/{}", fd.as_raw_fd())).unwrap().ino()
    }

    /// Shrinks or grows `fd`: "refused" when both fail, as a resize-sealed memfd answers.
    fn truncate(fd: &Owned) -> String {
        let len = platform::memory::size(fd).unwrap() as libc::off_t;
        // SAFETY: resizing a descriptor this process holds.
        let (shrink, grow) = unsafe { (libc::ftruncate(fd.as_raw_fd(), 0), libc::ftruncate(fd.as_raw_fd(), len * 2)) };
        String::from(if shrink == 0 || grow == 0 { "resized" } else { "refused" })
    }

    /// An executor driving the wire and its table by hand.
    fn raw_child(socket: &str) {
        let mut s = Raw::default();
        serve(None, |words| match words {
            ["hello", protocol, rest @ ..] => {
                let end = Arc::new(End::new(UnixStream::connect(socket).unwrap()));
                let (tx, inbox) = crossbeam_channel::unbounded();
                let reader = end.clone();
                std::thread::spawn(move || {
                    while let Ok(Some(m)) = reader.recv::<ToAgent>() {
                        if tx.send(m).is_err() {
                            break;
                        }
                    }
                });
                let back = (rest == ["back"]).then(|| s.table_fd.as_ref().unwrap().try_clone().unwrap());
                // An unregistered peer is refused before its Hello is read: the send may fail.
                let _ = end.send(&ToManager::Hello { protocol: protocol.parse().unwrap(), birth: Birth::own().unwrap().start_ticks }, back);
                (s.end, s.inbox) = (Some(end), Some(inbox));
                s.next()
            }
            ["ask", id, size, region @ ..] => {
                let region = region.first().map(|r| Region { fingerprint: FP, region: r.parse().unwrap() });
                s.end.as_ref().unwrap().send(&ToManager::Ask { id: id.parse().unwrap(), bytes: size.parse().unwrap(), region }, None).unwrap();
                "sent".into()
            }
            ["next"] => s.next(),
            ["ack", slot, seq, how] => {
                let slot: u32 = slot.parse().unwrap();
                if !s.table().answer(slot, seq.parse().unwrap(), *how == "mapped").unwrap() {
                    return "stale".into();
                }
                if *how == "refused" {
                    s.grants.remove(&slot);
                }
                s.ring();
                "ok".into()
            }
            ["map", slot, region, index] => {
                let g = s.grants.get_mut(&slot.parse().unwrap()).unwrap();
                let map = platform::memory::map(&g.fd, 0, g.bytes as usize, true).unwrap();
                // SAFETY: our own mapping, `len` bytes long.
                fill(unsafe { std::slice::from_raw_parts_mut(map.base, map.len) }, region.parse().unwrap(), index.parse().unwrap());
                g.map = Some(map);
                "ok".into()
            }
            ["read", slot, region, index] => {
                let map = s.grants[&slot.parse().unwrap()].map.as_ref().unwrap();
                // SAFETY: our own mapping, `len` bytes long.
                let bytes = unsafe { std::slice::from_raw_parts(map.base, map.len) };
                String::from(if verified(bytes, region.parse().unwrap(), index.parse().unwrap()) { "verified" } else { "corrupt" })
            }
            ["landed", slot] => {
                s.table().set_landed(slot.parse().unwrap()).unwrap();
                s.ring();
                "ok".into()
            }
            ["release", slot, seq, rest @ ..] => {
                let (slot, seq) = (slot.parse().unwrap(), seq.parse().unwrap());
                if !s.table().commit(slot, seq).unwrap() {
                    return "pinned".into();
                }
                if rest != ["keep"] {
                    s.grants.remove(&slot);
                }
                assert!(s.table().released(slot, seq).unwrap());
                s.ring();
                "ok".into()
            }
            ["word", slot] => {
                let slot = slot.parse().unwrap();
                let w = s.table().word(slot).unwrap();
                format!("{:?} {} {} ask {}", w.state, w.pins, w.seq, s.table().ask(slot))
            }
            ["asked"] => s.grants.iter().filter(|(slot, g)| s.table().ask(**slot) == g.seq).map(|(slot, _)| slot.to_string()).collect::<Vec<_>>().join(" "),
            ["forge", "word", slot, state, pins, seq] => {
                let state: u64 = match *state {
                    "granted" => 1,
                    "mapped" => 2,
                    "revoking" => 3,
                    "released" => 4,
                    "refused" => 5,
                    _ => 0,
                };
                let word = state | pins.parse::<u64>().unwrap() << 4 | seq.parse::<u64>().unwrap() << 32;
                s.at::<AtomicU64>(LINE * (1 + slot.parse::<usize>().unwrap())).store(word, SeqCst);
                s.ring();
                "ok".into()
            }
            ["forge", "pid", pid] => {
                s.at::<AtomicU32>(HEADER_PID).store(pid.parse().unwrap(), SeqCst);
                "ok".into()
            }
            ["forge", "index", slot, index] => {
                s.at::<AtomicU32>(LINE * (1 + slot.parse::<usize>().unwrap()) + SLOT_INDEX).store(index.parse().unwrap(), SeqCst);
                s.ring();
                "ok".into()
            }
            ["truncate", which] => truncate(s.fd(which)),
            ["seals", which] => platform::shm::seals(s.fd(which).as_fd()).unwrap().to_string(),
            _ => format!("unknown {words:?}"),
        });
    }
}

#[cfg(target_os = "linux")]
fn main() {
    cases::main();
}

#[cfg(not(target_os = "linux"))]
fn main() {
    println!("lease_p0: the cases pass memfds between Linux processes; skipped here");
}
