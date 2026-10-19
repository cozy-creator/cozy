//! The ledger across real processes. A child is this test binary re-run as `child`: it holds
//! the parent's ledger at fd 9 and is driven over stdin, answering `@`-lines on stdout. CPU only,
//! no mocks: real memfds, OFD locks, futexes and `kill -9`.
#![cfg(target_os = "linux")]

use std::io::{BufRead, BufReader, Write};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, PoisonError, RwLock};

use varena::ledger::{Admit, Facts, Ledger, RowFacts, State, ROWS};

const FD: RawFd = 9;
/// The ledger's eventfd in a child, when its parent's ledger has one.
const EVENTS: RawFd = 10;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// Held across each child's fork: until it execs, the child holds a copy of every row lock this
/// process holds, so a row it drops meanwhile still reads live. `spawn` returns once it exec'd.
static FORKING: RwLock<()> = RwLock::new(());

struct Proc {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl Proc {
    fn spawn(ledger: &Ledger) -> Proc {
        let fd = ledger.fd().as_raw_fd();
        let events = ledger.events().map(|e| e.as_raw_fd());
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["child", "--exact", "--ignored", "--nocapture", "--test-threads=1", "-q"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());
        // SAFETY: only async-signal-safe calls between fork and exec.
        unsafe {
            // Both are copied above the targets first: the eventfd may itself be number 9.
            command.pre_exec(move || {
                let copies = [Some(fd), events].map(|from| from.map(|f| libc::fcntl(f, libc::F_DUPFD_CLOEXEC, 64)));
                for (copy, to) in copies.into_iter().zip([FD, EVENTS]) {
                    if copy.is_some_and(|c| c < 0 || libc::dup2(c, to) < 0) {
                        return Err(std::io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
        let forking = FORKING.read().unwrap_or_else(PoisonError::into_inner);
        let mut child = command.spawn().unwrap();
        drop(forking);
        let (input, output) = (child.stdin.take().unwrap(), BufReader::new(child.stdout.take().unwrap()));
        Proc { child, input, output }
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn send(&mut self, line: &str) {
        writeln!(self.input, "{line}").unwrap();
        self.input.flush().unwrap();
    }

    fn reply(&mut self) -> Vec<u64> {
        let mut line = String::new();
        loop {
            line.clear();
            assert!(self.output.read_line(&mut line).unwrap() > 0, "child {} exited", self.pid());
            if let Some(rest) = line.trim_end().strip_prefix('@') {
                return rest.split_whitespace().map(|w| w.parse().unwrap()).collect();
            }
        }
    }

    fn call(&mut self, line: &str) -> Vec<u64> {
        self.send(line);
        self.reply()
    }

    fn kill9(mut self) {
        self.child.kill().unwrap();
        self.child.wait().unwrap();
    }
}

fn publish(f: &Facts) -> String {
    let w = [
        f.context, f.resident, f.cold, f.kept, f.pinned, f.torch_reserved, f.torch_allocated, f.cache, f.idle,
        f.active_peak, f.last_use, f.waiting,
    ];
    format!("publish {}", w.map(|n| n.to_string()).join(" "))
}

fn row(ledger: &Ledger, pid: u32) -> RowFacts {
    ledger.snapshot().unwrap().into_iter().find(|r| r.pid == pid).unwrap()
}

fn free(ledger: &Ledger) -> u64 {
    ledger.capacity().saturating_sub(ledger.snapshot().unwrap().iter().map(RowFacts::held).sum::<u64>())
}

/// Every field `k`: a torn read shows as unequal fields.
fn counted(k: u64) -> Facts {
    Facts {
        context: k,
        resident: k,
        cold: k,
        kept: k,
        pinned: k,
        torch_reserved: k,
        torch_allocated: k,
        cache: k,
        idle: k,
        active_peak: k,
        last_use: k,
        waiting: k,
    }
}

/// The child: a process holding a row, driven line by line.
#[test]
#[ignore = "a child process the other tests drive"]
fn child() {
    // SAFETY: fd 9 is open only when a parent test spawned this process.
    if unsafe { libc::fcntl(FD, libc::F_GETFD) } < 0 {
        return;
    }
    // SAFETY: as above; borrowed for the call.
    // SAFETY: as above; fd 10 is the eventfd when the parent passed one.
    let events = (unsafe { libc::fcntl(EVENTS, libc::F_GETFD) } >= 0).then(|| unsafe { BorrowedFd::borrow_raw(EVENTS) });
    let ledger = Ledger::adopt(unsafe { BorrowedFd::borrow_raw(FD) }, events).unwrap();
    let (mut row, mut facts, mut created) = (None, Facts::default(), None);
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let args: Vec<&str> = line.split_whitespace().skip(1).collect();
        let w: Vec<u64> = args.iter().filter_map(|n| n.parse().ok()).collect();
        let reply = match line.split_whitespace().next().unwrap() {
            "claim" => {
                let r = ledger.claim(std::process::id()).unwrap();
                let out = vec![r.id() as u64, r.gen()];
                row = Some(Arc::new(r));
                out
            }
            "publish" => {
                let [context, resident, cold, kept, pinned, torch_reserved, torch_allocated, cache, idle, active_peak, last_use, waiting] =
                    w[..].try_into().unwrap();
                facts = Facts {
                    context, resident, cold, kept, pinned, torch_reserved, torch_allocated, cache, idle, active_peak, last_use, waiting,
                };
                row.as_ref().unwrap().publish(&facts);
                vec![]
            }
            // Waits for an ask, evicts the cold bytes asked, publishes and answers.
            "serve" => {
                let r = row.as_ref().unwrap();
                let want = r.wait_want();
                let freed = want.min(facts.cold);
                facts.cold -= freed;
                facts.resident -= freed;
                r.publish(&facts);
                r.ack(freed);
                vec![want, freed]
            }
            // A Rust thread answers its asks from its cold bytes; then its main thread is busy
            // (one long kernel) and never reads another line.
            "busy" => {
                let cold = facts.cold;
                row.as_ref().unwrap().serve(move |want| want.min(cold)).unwrap();
                println!("@");
                loop {
                    std::thread::park();
                }
            }
            // Its running requests: `id remaining` pairs, -1 for never measured.
            "running" => {
                let pairs = args.chunks(2).map(|p| (p[0].to_string(), p[1].parse().unwrap())).collect::<Vec<(String, i64)>>();
                row.as_ref().unwrap().running(&pairs).unwrap();
                vec![]
            }
            "setpeak" => {
                ledger.set_peak(args[0], args[1].parse().unwrap()).unwrap();
                vec![]
            }
            // The key TensorD dispatched the request under, as its hash; 0 when none.
            "dispatched" => vec![ledger.dispatched(args[0]).map_or(0, |key| fnv(&key))],
            "freed" => {
                ledger.freed();
                vec![]
            }
            "waitfreed" => vec![ledger.wait_freed(w[0])],
            // Creates a ledger of its own, holds a row in it, and names its memfd.
            "create" => {
                let (l, fd) = Ledger::create("GPU-creator", w[0]).unwrap();
                let r = l.claim(std::process::id()).unwrap();
                r.publish(&Facts { context: w[1], ..Facts::default() });
                let out = vec![fd.as_raw_fd() as u64];
                created = Some((l, fd, r));
                out
            }
            other => panic!("unknown command {other}"),
        };
        println!("@{}", reply.iter().map(u64::to_string).collect::<Vec<_>>().join(" "));
    }
    drop(created);
}

fn fnv(key: &str) -> u64 {
    key.bytes().fold(0xcbf29ce484222325, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// TensorD names a request's key; the executor reads it, and the peak it records under it
/// reaches every process. Two processes record keys at once; each stands.
#[test]
fn peaks_and_dispatches_cross_processes() {
    let (ledger, _fd) = Ledger::create("GPU-peaks", 24 * GIB).unwrap();
    let (mut a, mut b) = (Proc::spawn(&ledger), Proc::spawn(&ledger));
    let key = format!("plan-{}|{}", "e".repeat(64), "0123456789abcdef");
    ledger.dispatch("41", &key).unwrap();
    assert_eq!(a.call("dispatched 41"), [fnv(&key)]);
    assert_eq!(a.call("dispatched 42"), [0]);
    assert_eq!(ledger.peak(&key), None);
    a.send(&format!("setpeak {key} {}", 3 * GIB));
    b.send(&format!("setpeak plan-b|s {}", 5 * GIB));
    a.reply();
    b.reply();
    assert_eq!((ledger.peak(&key), ledger.peak("plan-b|s")), (Some(3 * GIB), Some(5 * GIB)));
    a.call(&format!("setpeak {key} {}", 2 * GIB));
    assert_eq!(ledger.peak(&key), Some(2 * GIB), "the latest measurement stands");
    let status = ledger.status().unwrap();
    assert_eq!(status.dispatched, [("41".to_string(), key.clone())]);
    assert_eq!(status.peaks.len(), 2);
    ledger.undispatch("41");
    assert_eq!(a.call("dispatched 41"), [0]);
    assert!(ledger.set_peak(&"k".repeat(121), 1).is_err());
}

/// `fits` (design §5.6): never beside a dead row or a running request never measured; one never
/// measured only alone; else its peak beside every row's unevictable bytes and every running
/// request's remaining peak, and a new process's context when it spawns one.
#[test]
fn fits_reserves_remaining_peaks_and_never_admits_beside_the_unmeasured_or_the_dead() {
    let (ledger, _fd) = Ledger::create("GPU-fits", 24 * GIB).unwrap();
    let mut a = Proc::spawn(&ledger);
    a.call("claim");
    a.call(&publish(&Facts { context: GIB, torch_allocated: 2 * GIB, torch_reserved: 2 * GIB, last_use: 1, ..Facts::default() }));
    assert!(ledger.fits("plan", Some("s"), false).unwrap(), "never measured, and nothing runs");
    a.call("running r1 -1");
    assert!(!ledger.fits("plan", Some("s"), false).unwrap(), "beside a request never measured");
    a.call(&format!("running r1 {}", 10 * GIB));
    assert_eq!(row(&ledger, a.pid()).running, [("r1".to_string(), 10 * GIB as i64)]);
    assert!(!ledger.fits("plan", Some("s"), false).unwrap(), "never measured beside a running request");
    ledger.set_peak("plan|s", 8 * GIB).unwrap();
    assert!(ledger.fits("plan", Some("s"), false).unwrap(), "3 + 10 + 8 of 24");
    ledger.set_peak("plan|wide", 12 * GIB).unwrap();
    assert!(!ledger.fits("plan", Some("wide"), false).unwrap(), "3 + 10 + 12 of 24");
    assert!(!ledger.fits("plan", None, false).unwrap(), "a shape not yet known takes the widest");
    ledger.set_peak("plan|wide", 21 * GIB / 2).unwrap();
    assert!(ledger.fits("plan", Some("wide"), false).unwrap(), "3 + 10 + 10.5 of 24");
    assert!(!ledger.fits("plan", Some("wide"), true).unwrap(), "and a new process's 1 GiB context");
    a.call("running");
    a.kill9();
    assert!(!ledger.fits("plan", Some("s"), false).unwrap(), "beside a dead row's bytes");
}

/// A process whose main thread is busy in one long kernel still answers asks, from a Rust
/// thread: the answer moves its progress, so it is never named stalled.
#[test]
fn a_busy_process_answers_asks_from_its_own_thread_and_is_never_stalled() {
    let (ledger, _fd) = Ledger::create("GPU-busy", 24 * GIB).unwrap();
    let mut busy = Proc::spawn(&ledger);
    busy.call("claim");
    busy.call(&publish(&Facts { resident: 4 * GIB, cold: 4 * GIB, last_use: 1, ..Facts::default() }));
    busy.call("busy");
    assert!(ledger.stalled().unwrap().is_empty(), "the first look records");
    for ask in 0..3 {
        let victims = ledger.ask(None, GIB).unwrap();
        assert_eq!(victims.len(), 1, "ask {ask}");
        ledger.wait(&victims[0]).unwrap();
        assert!(ledger.answered(&victims[0]).unwrap());
        assert!(ledger.stalled().unwrap().is_empty(), "ask {ask}: answered, so never stalled");
    }
    assert_eq!(row(&ledger, busy.pid()).acked, 3 * GIB);
    busy.kill9();
}

/// Freed bytes move the free generation, wake a process waiting on it, and signal the eventfd.
#[test]
fn freeing_wakes_waiters_and_signals_the_eventfd() {
    let (ledger, _fd) = Ledger::create("GPU-freed", 24 * GIB).unwrap();
    let events = ledger.events().unwrap().as_raw_fd();
    let drain = || {
        let mut count = [0u8; 8];
        // SAFETY: an 8-byte read from the nonblocking eventfd this ledger holds.
        let read = unsafe { libc::read(events, count.as_mut_ptr().cast(), 8) };
        (read == 8).then(|| u64::from_ne_bytes(count))
    };
    assert_eq!((ledger.freed_gen(), drain()), (0, None));
    let (mut waiter, mut freer) = (Proc::spawn(&ledger), Proc::spawn(&ledger));
    waiter.send("waitfreed 0");
    freer.call("freed");
    assert_eq!(waiter.reply(), [1]);
    assert_eq!((ledger.freed_gen(), drain()), (1, Some(1)));
    // A dead row's bytes coming back count too.
    freer.call("claim");
    freer.kill9();
    ledger.reap(ledger.snapshot().unwrap()[0].pid).unwrap();
    ledger.returned(ledger.snapshot().unwrap()[0].pid).unwrap();
    assert_eq!((ledger.freed_gen(), drain()), (2, Some(1)));
}

#[test]
fn two_processes_publish_and_a_third_reads_exact_snapshots() {
    let (ledger, _fd) = Ledger::create("GPU-snapshots", 24 * GIB).unwrap();
    let fa = Facts {
        context: 500 * MIB,
        resident: 10 * GIB,
        cold: 4 * GIB,
        kept: 0,
        pinned: GIB,
        torch_reserved: 3 * GIB,
        torch_allocated: 2 * GIB,
        cache: 0,
        idle: 64 * MIB,
        active_peak: 0,
        last_use: 100,
        waiting: 0,
    };
    let fb = Facts {
        context: 450 * MIB,
        resident: 2 * GIB,
        cold: 0,
        kept: 0,
        pinned: 2 * GIB,
        torch_reserved: GIB,
        torch_allocated: GIB,
        cache: 256 * MIB,
        idle: 0,
        active_peak: 3 * GIB,
        last_use: 200,
        waiting: 1,
    };
    let (mut a, mut b) = (Proc::spawn(&ledger), Proc::spawn(&ledger));
    let (ca, cb) = (a.call("claim"), b.call("claim"));
    assert_ne!(ca[0], cb[0]);
    a.call(&publish(&fa));
    b.call(&publish(&fb));

    // The reader is this process, through a mapping and description of its own.
    let reader = Ledger::adopt(ledger.fd(), None).unwrap();
    assert_eq!((reader.gpu_uuid().as_str(), reader.capacity()), ("GPU-snapshots", 24 * GIB));
    let rows = reader.snapshot().unwrap();
    assert_eq!(rows.len(), 2);
    let (ra, rb) = (row(&reader, a.pid()), row(&reader, b.pid()));
    assert_eq!((ra.slot as u64, ra.gen, ra.state, ra.facts, ra.want, ra.acked, ra.progress), (ca[0], ca[1], State::Live, fa, 0, 0, 1));
    assert_eq!((rb.slot as u64, rb.gen, rb.state, rb.facts, rb.want, rb.acked, rb.progress), (cb[0], cb[1], State::Live, fb, 0, 0, 1));
    // A: idle; its torch slack (1 GiB), idle chunks and cold weights come back on an ask.
    assert_eq!(ra.held(), 500 * MIB + 10 * GIB + 64 * MIB + 3 * GIB);
    assert_eq!(ra.unevictable(), 500 * MIB + 6 * GIB + 2 * GIB);
    assert_eq!(ra.reclaimable(), 64 * MIB + GIB + 4 * GIB);
    // B: a request is running toward its 3 GiB peak; only its cache is reclaimable.
    assert_eq!(rb.held(), 450 * MIB + 2 * GIB + 3 * GIB + 256 * MIB);
    assert_eq!(rb.unevictable(), 450 * MIB + 2 * GIB + 3 * GIB);
    assert_eq!(free(&reader), 24 * GIB - ra.held() - rb.held());

    // A later publish replaces the facts whole.
    let fa2 = Facts { cold: 0, last_use: 300, ..fa };
    a.call(&publish(&fa2));
    let ra = row(&reader, a.pid());
    assert_eq!((ra.facts, ra.progress), (fa2, 2));

    let (pa, pb) = (a.pid(), b.pid());
    a.kill9();
    b.kill9();
    assert!(reader.snapshot().unwrap().iter().all(|r| r.state == State::Dead));
    for pid in [pa, pb] {
        assert!(reader.reap(pid).unwrap() > 0);
        reader.returned(pid).unwrap();
    }
    assert!(ledger.snapshot().unwrap().is_empty());
}

#[test]
fn an_ask_reaches_another_process_which_frees_and_answers() {
    let (ledger, _fd) = Ledger::create("GPU-ask", 8 * GIB).unwrap();
    // A is this process, mid-request at its peak.
    let a = ledger.claim(std::process::id()).unwrap();
    a.publish(&Facts {
        context: 500 * MIB,
        torch_reserved: 5 * GIB,
        torch_allocated: 5 * GIB,
        active_peak: 6 * GIB,
        last_use: 50,
        ..Facts::default()
    });
    let mut b = Proc::spawn(&ledger);
    b.call("claim");
    b.call(&publish(&Facts { context: 400 * MIB, resident: 2 * GIB, cold: 1536 * MIB, last_use: 10, ..Facts::default() }));
    let before = row(&ledger, b.pid());

    b.send("serve");
    let victims = ledger.ask(Some(a.id()), GIB).unwrap();
    assert_eq!(victims.iter().map(|v| (v.pid, v.bytes)).collect::<Vec<_>>(), [(b.pid(), GIB)]);
    ledger.wait(&victims[0]).unwrap();
    assert!(ledger.answered(&victims[0]).unwrap());
    assert_eq!(b.reply(), [GIB, GIB]);

    let after = row(&ledger, b.pid());
    assert_eq!((after.want, after.acked), (0, GIB));
    assert!(after.progress > before.progress);
    assert_eq!((after.facts.cold, after.facts.resident), (512 * MIB, GIB));

    // An ask covers only what is there; the asker's own row is never a victim.
    let more = ledger.ask(Some(a.id()), 2 * GIB).unwrap();
    assert_eq!(more.iter().map(|v| (v.pid, v.bytes)).collect::<Vec<_>>(), [(b.pid(), 512 * MIB)]);
    assert!(!ledger.answered(&more[0]).unwrap());
    assert_eq!(row(&ledger, b.pid()).want, 512 * MIB);
    // What is already asked is not asked again.
    assert!(ledger.ask(Some(a.id()), GIB).unwrap().is_empty());
    b.send("serve");
    ledger.wait(&more[0]).unwrap();
    assert_eq!(b.reply(), [512 * MIB, 512 * MIB]);
    assert_eq!(row(&ledger, b.pid()).acked, GIB + 512 * MIB);
    b.kill9();
}

#[test]
fn a_killed_holder_is_dead_reaped_and_counted_until_its_bytes_return() {
    let (ledger, _fd) = Ledger::create("GPU-kill", 8 * GIB).unwrap();
    let mut b = Proc::spawn(&ledger);
    let slot = b.call("claim")[0] as u32;
    let fb = Facts {
        context: 512 * MIB,
        resident: 3 * GIB,
        cold: 3 * GIB,
        torch_reserved: 2 * GIB,
        torch_allocated: 2 * GIB,
        last_use: 1,
        ..Facts::default()
    };
    b.call(&publish(&fb));
    let held = 512 * MIB + 5 * GIB;
    assert_eq!(free(&ledger), 8 * GIB - held);
    assert_eq!(ledger.admit(2 * GIB).unwrap(), Admit::Now);
    let Admit::After { victims, short } = ledger.admit(3 * GIB).unwrap() else { panic!("3 GiB is not free") };
    assert_eq!((victims.iter().map(|v| (v.pid, v.bytes)).collect::<Vec<_>>(), short), (vec![(b.pid(), 512 * MIB)], 0));

    let pid = b.pid();
    b.kill9();
    let dead = row(&ledger, pid);
    assert_eq!((dead.state, dead.facts, dead.held(), dead.reclaimable()), (State::Dead, fb, held, 0));
    // A dead row's bytes are held and nobody can be asked for them.
    assert_eq!(ledger.admit(3 * GIB).unwrap(), Admit::After { victims: vec![], short: 512 * MIB });
    assert_eq!(ledger.admit(2 * GIB).unwrap(), Admit::Now);
    assert!(ledger.ask(None, GIB).unwrap().is_empty());
    assert!(ledger.stalled().unwrap().is_empty());

    assert_eq!(ledger.reap(pid).unwrap(), held);
    let reaping = row(&ledger, pid);
    assert_eq!((reaping.state, reaping.reaping, reaping.held()), (State::Reaping, held, held));
    assert_eq!(ledger.admit(3 * GIB).unwrap(), Admit::After { victims: vec![], short: 512 * MIB });
    assert_eq!(ledger.reap(pid).unwrap(), 0);
    // Its slot is not reused while reaping.
    let other = ledger.claim(1).unwrap();
    assert_ne!(other.id(), slot);
    drop(other);
    ledger.reap(1).unwrap();
    ledger.returned(1).unwrap();

    ledger.returned(pid).unwrap();
    assert!(ledger.snapshot().unwrap().is_empty());
    assert_eq!(ledger.admit(8 * GIB).unwrap(), Admit::Now);
    let again = ledger.claim(2).unwrap();
    assert_eq!((again.id(), again.gen()), (slot, dead.gen + 1));

    // A live process is never reaped.
    let me = ledger.claim(std::process::id()).unwrap();
    assert!(ledger.reap(std::process::id()).is_err());
    assert!(ledger.claim(std::process::id()).is_err(), "one live row per pid");
    assert_eq!(row(&ledger, me.pid()).state, State::Live);
}

#[test]
fn a_victim_that_never_progresses_is_stalled() {
    let (ledger, _fd) = Ledger::create("GPU-stall", 8 * GIB).unwrap();
    let mut b = Proc::spawn(&ledger);
    b.call("claim");
    b.call(&publish(&Facts { resident: 2 * GIB, cold: 2 * GIB, last_use: 5, ..Facts::default() }));
    let mut c = Proc::spawn(&ledger);
    c.call("claim");
    c.call(&publish(&Facts { resident: GIB, cold: GIB, last_use: 1, ..Facts::default() }));
    assert!(ledger.stalled().unwrap().is_empty());

    let victims = ledger.ask(None, 2 * GIB + 512 * MIB).unwrap();
    let asked: Vec<_> = victims.iter().map(|v| (v.pid, v.bytes)).collect();
    assert_eq!(asked, [(c.pid(), GIB), (b.pid(), GIB + 512 * MIB)], "least recently used first");
    assert_eq!(c.call("serve"), [GIB, GIB]);

    assert!(ledger.stalled().unwrap().is_empty(), "the first look records");
    assert_eq!(ledger.stalled().unwrap(), [b.pid()]);
    assert_eq!(ledger.stalled().unwrap(), [b.pid()]);
    // Another handle looks for itself.
    let other = Ledger::adopt(ledger.fd(), None).unwrap();
    assert!(other.stalled().unwrap().is_empty());
    // Any progress clears it until the next look without progress.
    b.call(&publish(&Facts { resident: 2 * GIB, cold: 2 * GIB, last_use: 6, ..Facts::default() }));
    assert!(ledger.stalled().unwrap().is_empty());
    assert_eq!(ledger.stalled().unwrap(), [b.pid()]);
    assert_eq!(b.call("serve"), [GIB + 512 * MIB, GIB + 512 * MIB]);
    assert!(ledger.stalled().unwrap().is_empty());
    assert!(victims.iter().all(|v| ledger.answered(v).unwrap()));
    b.kill9();
    c.kill9();
}

/// A warm-set (`keep`) row's cold bytes are asked for only after every other row's, however
/// long ago it was used; its idle bytes are not held back.
#[test]
fn kept_cold_bytes_are_asked_last() {
    let (ledger, _fd) = Ledger::create("GPU-kept", 24 * GIB).unwrap();
    let rows = [
        (30, Facts { resident: 2 * GIB, cold: 2 * GIB, kept: 2 * GIB, idle: 256 * MIB, last_use: 1, ..Facts::default() }),
        (31, Facts { resident: GIB, cold: GIB, last_use: 5, ..Facts::default() }),
        (32, Facts { resident: 3 * GIB, cold: 2 * GIB, kept: GIB, last_use: 9, ..Facts::default() }),
    ];
    let held: Vec<_> = rows.iter().map(|(pid, f)| {
        let row = ledger.claim(*pid).unwrap();
        row.publish(f);
        row
    }).collect();
    let order = |victims: &[varena::ledger::Victim]| victims.iter().map(|v| (v.pid, v.bytes)).collect::<Vec<_>>();
    let all_but_kept = 256 * MIB + GIB + GIB;
    let Admit::After { victims, short } = ledger.admit(24 * GIB - (6 * GIB + 256 * MIB) + all_but_kept).unwrap() else { panic!() };
    assert_eq!((order(&victims), short), (vec![(30, 256 * MIB), (31, GIB), (32, GIB)], 0));
    let victims = ledger.ask(None, all_but_kept + 2 * GIB + 512 * MIB).unwrap();
    assert_eq!(order(&victims), [(30, 256 * MIB + 2 * GIB), (31, GIB), (32, GIB + 512 * MIB)]);
    drop(held);
}

/// 11405: an idle executor kept 7.27 GiB in torch after a failed request. TensorD unmapped plane
/// bytes and answered Go(104 MB) to a 335 MB need.
#[test]
fn bytes_that_are_not_cold_never_admit_below_need_11405() {
    let kept = 727 * GIB / 100;
    let (ledger, _fd) = Ledger::create("GPU-11405", 400 * MIB + kept + 104_000_000).unwrap();
    let idle = ledger.claim(11405).unwrap();
    idle.publish(&Facts { context: 400 * MIB, torch_reserved: kept, torch_allocated: kept, last_use: 1, ..Facts::default() });
    let need = 335_000_000;
    let gap = need - free(&ledger);
    assert!(free(&ledger) < need);
    assert_eq!(ledger.admit(need).unwrap(), Admit::After { victims: vec![], short: gap });
    assert_eq!(ledger.admit(free(&ledger)).unwrap(), Admit::Now);

    // The same bytes as warm weights (a model in use) are no more free.
    idle.publish(&Facts { context: 400 * MIB, resident: kept, cold: 0, pinned: GIB, last_use: 2, ..Facts::default() });
    assert_eq!(ledger.admit(need).unwrap(), Admit::After { victims: vec![], short: gap });
    // Cold, they are asked for, and only what the gap needs.
    idle.publish(&Facts { context: 400 * MIB, resident: kept, cold: kept - GIB, pinned: GIB, last_use: 3, ..Facts::default() });
    let Admit::After { victims, short } = ledger.admit(need).unwrap() else { panic!("admitted beside held bytes") };
    assert_eq!((victims.iter().map(|v| (v.pid, v.bytes)).collect::<Vec<_>>(), short), (vec![(11405, gap)], 0));

    assert_eq!(free(&ledger), 104_000_000);
}

/// Whatever the rows hold, `Now` means need fits in what no row holds, and `After` names
/// victims and a shortfall that add up to the gap.
#[test]
fn admission_never_answers_below_need() {
    let capacity = 24 * GIB;
    let (ledger, _fd) = Ledger::create("GPU-sweep", capacity).unwrap();
    let rows = [
        Facts { context: 500 * MIB, resident: 8 * GIB, cold: 3 * GIB, pinned: GIB, torch_reserved: 2 * GIB, torch_allocated: GIB, idle: 128 * MIB, last_use: 7, ..Facts::default() },
        Facts { context: 400 * MIB, resident: 4 * GIB, pinned: 4 * GIB, torch_reserved: GIB, torch_allocated: GIB / 2, cache: GIB / 4, active_peak: 3 * GIB, last_use: 9, ..Facts::default() },
        Facts { context: 300 * MIB, resident: GIB, cold: GIB, torch_reserved: GIB / 3, torch_allocated: GIB / 5, last_use: 2, ..Facts::default() },
    ];
    let held: Vec<_> = rows.iter().enumerate().map(|(i, f)| {
        let row = ledger.claim(20 + i as u32).unwrap();
        row.publish(f);
        row
    }).collect();
    let free = free(&ledger);
    assert!(free > 0 && free < capacity);
    let reclaimable: u64 = ledger.snapshot().unwrap().iter().map(RowFacts::reclaimable).sum();
    for need in (0..=capacity).step_by(capacity as usize / 211) {
        match ledger.admit(need).unwrap() {
            Admit::Now => assert!(need <= free, "Now for {need} with {free} free"),
            Admit::After { victims, short } => {
                assert!(need > free);
                assert_eq!(victims.iter().map(|v| v.bytes).sum::<u64>() + short, need - free);
                assert_eq!(short, (need - free).saturating_sub(reclaimable));
            }
        }
    }
    drop(held);
}

/// 11399: a follower's cap came from a ~20 MiB load-time row while its GPU held ~7 GB. Here the
/// only numbers are each row's latest publish, read by another process; nothing derives a cap.
#[test]
fn rows_own_published_bytes_are_the_truth_11399() {
    let (ledger, _fd) = Ledger::create("GPU-11399", 8 * GIB).unwrap();
    let mut follower = Proc::spawn(&ledger);
    follower.call("claim");
    let loading = Facts { context: 20 * MIB, last_use: 1, ..Facts::default() };
    follower.call(&publish(&loading));
    assert_eq!(row(&ledger, follower.pid()).held(), 20 * MIB);
    assert_eq!(ledger.admit(7 * GIB).unwrap(), Admit::Now);

    let loaded = Facts { context: 512 * MIB, resident: 6 * GIB + 512 * MIB, pinned: 512 * MIB, last_use: 2, ..Facts::default() };
    follower.call(&publish(&loaded));
    let r = row(&ledger, follower.pid());
    assert_eq!((r.facts, r.held(), r.unevictable()), (loaded, 7 * GIB, 7 * GIB));
    assert_eq!(free(&ledger), GIB);
    assert_eq!(ledger.admit(GIB).unwrap(), Admit::Now);
    assert_eq!(ledger.admit(GIB + 1).unwrap(), Admit::After { victims: vec![], short: 1 });

    // Its request ends: the same weights are now cold, and an ask can get them.
    let done = Facts { cold: 6 * GIB, last_use: 3, ..loaded };
    follower.call(&publish(&done));
    let Admit::After { victims, short } = ledger.admit(7 * GIB).unwrap() else { panic!("7 GiB is not free") };
    assert_eq!((victims.iter().map(|v| (v.pid, v.bytes)).collect::<Vec<_>>(), short), (vec![(follower.pid(), 6 * GIB)], 0));
    follower.kill9();
}

/// Four writers publish and answer while four askers snapshot, ask and admit. Each publish
/// writes one value to every field, so a torn read shows as unequal fields.
#[test]
fn concurrent_publishes_and_asks_never_tear() {
    const PUBLISHES: u64 = 20_000;
    let (ledger, _fd) = Ledger::create("GPU-race", u64::MAX / 2).unwrap();
    let ledger = Arc::new(ledger);
    let done = Arc::new(AtomicBool::new(false));
    let asked: Arc<Vec<AtomicU64>> = Arc::new((0..ROWS).map(|_| AtomicU64::new(0)).collect());
    let rows: Vec<_> = (0..4).map(|i| Arc::new(ledger.claim(4_000_000 + i).unwrap())).collect();

    let writers: Vec<_> = rows
        .iter()
        .map(|row| {
            let row = row.clone();
            std::thread::spawn(move || {
                for k in 1..=PUBLISHES {
                    let f = counted(k);
                    row.publish(&f);
                    if row.want() > 0 {
                        row.ack(row.want());
                    }
                }
            })
        })
        .collect();
    let askers: Vec<_> = (0..4)
        .map(|_| {
            let (ledger, done, asked) = (ledger.clone(), done.clone(), asked.clone());
            std::thread::spawn(move || {
                let mut last = vec![0u64; ROWS as usize];
                let mut looks = 0u64;
                while !done.load(Relaxed) || looks == 0 {
                    looks += 1;
                    for r in ledger.snapshot().unwrap() {
                        let f = r.facts;
                        assert_eq!(f, counted(f.last_use), "torn row {}: {f:?}", r.slot);
                        assert!(f.last_use >= last[r.slot as usize], "row {} went back", r.slot);
                        last[r.slot as usize] = f.last_use;
                    }
                    for v in ledger.ask(None, 1000).unwrap() {
                        asked[v.slot as usize].fetch_add(v.bytes, Relaxed);
                    }
                    let _ = ledger.admit(1 << 40).unwrap();
                }
                looks
            })
        })
        .collect();
    for w in writers {
        w.join().unwrap();
    }
    done.store(true, Relaxed);
    let looks: u64 = askers.into_iter().map(|a| a.join().unwrap()).sum();
    assert!(looks > 0);
    for row in &rows {
        if row.want() > 0 {
            row.ack(row.want());
        }
        let r = ledger.snapshot().unwrap().into_iter().find(|r| r.slot == row.id()).unwrap();
        assert_eq!(r.facts, counted(PUBLISHES));
        assert_eq!((r.want, r.acked), (0, asked[row.id() as usize].load(Relaxed)), "row {}: every ask answered once", row.id());
    }
}

/// The creator dies; a process that holds the memfd adopts the table and reaps the creator's row.
#[test]
fn the_table_survives_its_creator() {
    let (parent, _fd) = Ledger::create("GPU-parent", GIB).unwrap();
    let mut creator = Proc::spawn(&parent);
    let fd = creator.call(&format!("create {} {}", 12 * GIB, 7 * MIB))[0];
    let path = std::ffi::CString::new(format!("/proc/{}/fd/{fd}", creator.pid())).unwrap();
    // SAFETY: opening a child's descriptor by its magic link.
    let raw = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
    assert!(raw >= 0, "{}", std::io::Error::last_os_error());
    // SAFETY: a fresh descriptor from the kernel.
    let held = unsafe { OwnedFd::from_raw_fd(raw) };
    let pid = creator.pid();
    creator.kill9();

    let ledger = Ledger::adopt(held.as_fd(), None).unwrap();
    assert_eq!((ledger.gpu_uuid().as_str(), ledger.capacity()), ("GPU-creator", 12 * GIB));
    let rows = ledger.snapshot().unwrap();
    assert_eq!(rows.iter().map(|r| (r.pid, r.state, r.facts.context)).collect::<Vec<_>>(), [(pid, State::Dead, 7 * MIB)]);
    assert_eq!(ledger.reap(pid).unwrap(), 7 * MIB);
    ledger.returned(pid).unwrap();
    let mine = ledger.claim(std::process::id()).unwrap();
    assert_eq!(mine.id(), rows[0].slot);
}

#[test]
fn only_a_sealed_varena_ledger_is_adopted() {
    let plain = platform::shm::memfd_named("not-a-ledger").unwrap();
    // SAFETY: sizing our own memfd.
    assert_eq!(unsafe { libc::ftruncate(plain.as_raw_fd(), 1 << 16) }, 0);
    assert!(Ledger::adopt(plain.as_fd(), None).is_err(), "unsealed");
    platform::shm::seal_with(plain.as_fd(), platform::shm::RESIZE_SEALS).unwrap();
    assert!(Ledger::adopt(plain.as_fd(), None).is_err(), "no ledger header");
    assert!(Ledger::create(&"x".repeat(65), GIB).is_err());

    let (ledger, fd) = Ledger::create("GPU-full", GIB).unwrap();
    let _no_fork = FORKING.write().unwrap_or_else(PoisonError::into_inner);
    let rows: Vec<_> = (0..ROWS).map(|i| ledger.claim(10 + i).unwrap()).collect();
    assert!(ledger.claim(9).is_err(), "every row is held");
    drop(rows);
    // Dropped rows read dead until reaped and returned.
    assert!(ledger.claim(9).is_err());
    assert_eq!(ledger.snapshot().unwrap().iter().filter(|r| r.state == State::Dead).count(), ROWS as usize);
    for pid in 10..10 + ROWS {
        ledger.reap(pid).unwrap();
        ledger.returned(pid).unwrap();
    }
    assert!(Ledger::adopt(fd.as_fd(), None).unwrap().claim(9).is_ok());
}
