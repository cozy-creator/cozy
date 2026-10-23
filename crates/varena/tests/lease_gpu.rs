//! Leases on one real GPU (varena-leases §8 P2 on Windows; the same driver facts on Linux): this
//! process hosts the Manager over its own CudaBackend, as TensorD does; each agent is this binary
//! as a child over its own CudaBackend, as an executor is. Chunks cross as WIN32 NT handles
//! DuplicateHandle'd into the agent (Linux: fds by SCM_RIGHTS). Every case needs a GPU, so a case
//! runs only when named after `--ignored`: `lease_gpu --ignored <case>`. A case prints
//! `FACT …` lines and `PASS <case> <GPU UUID>`; a failed one panics.
//!
//! Roles for the session proof (M8), started by its script in two logon sessions:
//! `serve <name>` hosts a manager that admits any peer of this user and exits once its first
//! executor has; `borrow <name>` attaches to it, takes a chunk, writes and reads it back.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use platform::handles::Peer;
use tensorfs::io::cuda;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;
use varena::backend::{Access, Backend, Va};
use varena::lease::{Agent, Attach, Hosting, Identity, Lease, Manager, Record, Region};
use varena::lease::Mode;
use varena::{CudaBackend, GpuId};

const MIB: u64 = 1 << 20;

/// Each driver call's durations in µs, by name: CudaBackend opens a span per call.
#[derive(Clone, Default)]
struct Timer(Arc<Mutex<BTreeMap<&'static str, Vec<u64>>>>);

impl<S: tracing::Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Timer {
    fn on_new_span(&self, _: &tracing::span::Attributes<'_>, id: &tracing::span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(Instant::now());
        }
    }

    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let Some(start) = span.extensions().get::<Instant>().copied() else { return };
        self.0.lock().entry(span.name()).or_default().push(start.elapsed().as_micros() as u64);
    }
}

impl Timer {
    fn install() -> Timer {
        let timer = Timer::default();
        tracing::subscriber::set_global_default(tracing_subscriber::registry().with(timer.clone())).unwrap();
        timer
    }

    /// `FACT <who> <label> <call> n=… p50=… p99=… µs` for each call so far; then forgets them.
    fn report(&self, who: &str, label: &str) -> Vec<String> {
        let mut calls = std::mem::take(&mut *self.0.lock());
        calls
            .iter_mut()
            .filter(|(name, _)| name.starts_with("cuMem"))
            .map(|(name, us)| {
                us.sort_unstable();
                let at = |q: usize| us[(us.len() - 1) * q / 100];
                format!("FACT {who} {label} {name} n={} p50={}us p99={}us", us.len(), at(50), at(99))
            })
            .collect()
    }
}

fn say(line: &str) {
    println!("{line}");
    std::io::stdout().flush().unwrap();
}

/// This process's budget and usage on the device (Windows), as a FACT line's tail.
fn budget(b: &dyn Backend) -> String {
    match b.budget().map(|budget| budget.read()) {
        Some(Ok((bytes, usage))) => format!("budget={}MiB usage={}MiB", bytes / MIB, usage / MIB),
        Some(Err(e)) => format!("budget-unread={e}"),
        None => "budget=none".into(),
    }
}

/// A chunk's byte at `offset`.
fn pattern(region: u32, index: usize, offset: usize) -> u8 {
    ((region as usize * 31 + index * 7 + offset / 4096) % 251) as u8
}

/// Copies between host and a mapped VA on the backend's copy stream, in its primary context.
fn copy(b: &CudaBackend, va: u64, host: &mut [u8], to_device: bool) {
    let (_, ctx) = cuda::primary_context(b.ordinal()).unwrap();
    let _in = cuda::CtxGuard::enter(ctx).unwrap();
    let d = cuda::driver().unwrap();
    let stream = b.stream() as cuda::CUstream;
    // SAFETY: `va` maps at least `host.len()` bytes; `host` is ours for the call; synchronized.
    let code = unsafe {
        if to_device {
            (d.memcpy_htod_async)(va, host.as_ptr().cast(), host.len(), stream)
        } else {
            (d.memcpy_dtoh_async)(host.as_mut_ptr().cast(), va, host.len(), stream)
        }
    };
    cuda::check("cuMemcpyAsync", code).unwrap();
    b.synchronize(b.stream()).unwrap();
}

struct Held {
    lease: Lease,
    va: Va,
}

/// An executor's side: an agent over its own CudaBackend.
struct Executor {
    backend: Arc<CudaBackend>,
    agent: Arc<Agent>,
    held: Vec<Held>,
}

impl Executor {
    fn attach(name: &str) -> Result<Executor, String> {
        let backend = Arc::new(CudaBackend::open(0).unwrap());
        let stream = varena::lease::connect(platform::memory::euid(), name).unwrap().unwrap();
        match Agent::attach(stream, backend.clone(), || {}).unwrap() {
            Attach::Leased(agent) => Ok(Executor { backend, agent, held: vec![] }),
            Attach::Refused { code, why } => Err(format!("{code:?} {why}")),
        }
    }

    /// `bytes` of `region`: imported, mapped at a VA of this process, written (fresh) or read
    /// back (adopted), acked landed. "ok <chunks> fresh|adopted verified|corrupt".
    fn take(&mut self, bytes: u64, region: u32, land: bool) -> String {
        let leases = match self.agent.take(bytes, varena::lease::Kind::Weights, Some(Region { fingerprint: 0xfeed, region })) {
            Ok(leases) => leases,
            Err(e) => return format!("refused {e}"),
        };
        let (n, adopted) = (leases.len(), leases.iter().all(|l| l.adopted));
        let mut verified = true;
        let tickets: Vec<_> = leases.iter().map(|l| l.ticket).collect();
        for (index, lease) in leases.into_iter().enumerate() {
            let va = self.backend.reserve(lease.phys.len).unwrap();
            self.backend.map(va, 0, &lease.phys, Access::ReadWrite).unwrap();
            let mut host: Vec<u8> = (0..lease.phys.len as usize).map(|o| pattern(region, index, o)).collect();
            if lease.adopted {
                let want = host.clone();
                copy(&self.backend, va.base, &mut host, false);
                verified &= host == want;
            } else {
                copy(&self.backend, va.base, &mut host, true);
            }
            self.held.push(Held { lease, va });
        }
        self.agent.mapped(&tickets).unwrap();
        if land {
            tickets.iter().for_each(|t| self.agent.landed(*t).unwrap());
        }
        format!("ok {n} {} {}", if adopted { "adopted" } else { "fresh" }, if verified { "verified" } else { "corrupt" })
    }

    /// Every held chunk unmapped and given back.
    fn give_all(&mut self) {
        for h in self.held.drain(..) {
            self.backend.unmap(h.va, 0, h.va.len).unwrap();
            self.backend.release(h.va);
            self.agent.give(h.lease.ticket, h.lease.phys).unwrap();
        }
    }
}

/// The agent role: one command per stdin line, one answer per line.
fn agent(name: &str) {
    let timer = Timer::install();
    let mut exec = Executor::attach(name).unwrap();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let words: Vec<&str> = line.split_whitespace().collect();
        let out = match words.as_slice() {
            ["take", bytes, region, rest @ ..] => exec.take(bytes.parse().unwrap(), region.parse().unwrap(), rest != ["unlanded"]),
            ["budget"] => budget(exec.backend.as_ref()),
            ["times", label] => timer.report("executor", label).join("\n") + "\nend",
            ["give"] => {
                exec.give_all();
                "ok".into()
            }
            _ => format!("unknown {line}"),
        };
        say(&out);
    }
}

/// Admits this test's children (or, `any`, every peer of this user) and keeps the grants.
struct Children {
    pids: Mutex<Vec<u32>>,
    any: bool,
    grants: Mutex<Vec<Record>>,
}

impl Hosting for Children {
    fn identify(&self, peer: &Peer) -> Option<Identity> {
        (self.any || self.pids.lock().contains(&peer.pid)).then(|| Identity { executor: peer.pid.to_string(), domain: "p2".into() })
    }

    fn record(&self, record: Record) {
        self.grants.lock().push(record);
    }
}

struct Host {
    manager: Arc<Manager>,
    backend: Arc<CudaBackend>,
    name: String,
    children: Arc<Children>,
    uuid: String,
}

/// A manager serving `name` (the listener agents dial) over this process's own context.
fn host(name: String, any: bool) -> Host {
    let backend = Arc::new(CudaBackend::open(0).unwrap());
    let card = backend.gpu().unwrap();
    let gpu = GpuId { uuid: name.clone(), model: card.model.clone(), memory: card.memory };
    let listener = varena::lease::listen(&gpu).unwrap().unwrap();
    let children = Arc::new(Children { pids: Mutex::default(), any, grants: Mutex::default() });
    let lent = backend.clone();
    let open = Box::new(move || Ok(lent.clone() as Arc<dyn Backend>));
    let manager = Manager::create(&gpu, None, Mode::Shared, Some(open), listener, children.clone()).unwrap();
    varena::os::remove_table(&gpu.uuid);
    Host { manager, backend, name, children, uuid: card.uuid }
}

struct Spawned {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Spawned {
    fn start(host: &Host) -> Spawned {
        let mut child = Command::new(std::env::current_exe().unwrap()).args(["agent", &host.name]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        host.children.pids.lock().push(child.id());
        let (stdin, stdout) = (child.stdin.take().unwrap(), BufReader::new(child.stdout.take().unwrap()));
        Spawned { child, stdin, stdout }
    }

    fn call(&mut self, line: &str) -> String {
        writeln!(self.stdin, "{line}").unwrap();
        self.stdin.flush().unwrap();
        let mut out = String::new();
        self.stdout.read_line(&mut out).unwrap();
        out.trim().to_string()
    }

    /// A multi-line answer, up to its `end` line.
    fn lines(&mut self, line: &str) -> Vec<String> {
        let mut out = vec![self.call(line)];
        while out.last().is_some_and(|l| l != "end") {
            let mut next = String::new();
            self.stdout.read_line(&mut next).unwrap();
            out.push(next.trim().to_string());
        }
        out.pop();
        out
    }
}

/// Until the manager's holdings satisfy `done`; the case fails rather than hangs.
fn until(host: &Host, done: impl Fn(&varena::lease::Holdings) -> bool) -> varena::lease::Holdings {
    let start = Instant::now();
    loop {
        let h = host.manager.holdings();
        if done(&h) {
            return h;
        }
        assert!(start.elapsed() < Duration::from_secs(60), "the manager never reached it: {h:?}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Q1/Q2: 2 MiB and 32 MiB chunks cross by handle, map and take writes in the executor; the
/// executor is killed and its landed chunks, retained by the manager, are adopted by a second
/// executor bit-exact with no copy and no new chunk; given back, the manager frees them all.
fn chunks_cross_and_adopt() {
    let timer = Timer::install();
    let host = host(format!("leasegpu{}-cross", std::process::id()), false);
    let mut a = Spawned::start(&host);
    let mut total = 0;
    for (label, size, region) in [("2MiB", 2 * MIB, 100), ("32MiB", 32 * MIB, 200)] {
        for n in 0..16 {
            assert_eq!(a.call(&format!("take {size} {}", region + n)), "ok 1 fresh verified");
            total += size;
        }
        timer.report("manager", label).iter().for_each(|l| say(l));
        a.lines(&format!("times {label}")).iter().for_each(|l| say(l));
    }
    let free_held = host.backend.free().unwrap();
    a.child.kill().unwrap();
    a.child.wait().unwrap();
    let h = until(&host, |h| h.executors.is_empty());
    assert_eq!((h.page_cache, h.created), (total, total), "every landed chunk retained");
    let mut b = Spawned::start(&host);
    for (size, region) in [(2 * MIB, 100), (32 * MIB, 200)] {
        for n in 0..16 {
            assert_eq!(b.call(&format!("take {size} {}", region + n)), "ok 1 adopted verified");
        }
    }
    let h = until(&host, |h| h.allocated == total);
    assert_eq!((h.page_cache, h.created), (0, total), "adopted with no new chunk");
    assert_eq!(b.call("give"), "ok");
    until(&host, |h| h.created == 0);
    say(&format!("FACT manager free with {} MiB held {} MiB; all given back {} MiB", total / MIB, free_held / MIB, host.backend.free().unwrap() / MIB));
    say(&format!("PASS chunks_cross_and_adopt {}", host.uuid));
}

/// Q3/Q4 (M8): whose WDDM budget and usage move when an executor maps chunks the manager
/// created: both processes' readings at each step.
fn budget_follows_whom() {
    let host = host(format!("leasegpu{}-budget", std::process::id()), false);
    if host.backend.budget().is_none() {
        return say("SKIP budget_follows_whom: no DXGI budget on this OS or device");
    }
    let mut a = Spawned::start(&host);
    let step = |a: &mut Spawned, what: &str| say(&format!("FACT {what}: manager {} | executor {}", budget(host.backend.as_ref()), a.call("budget")));
    step(&mut a, "attached");
    assert_eq!(a.call(&format!("take {} 1 unlanded", 1024 * MIB)), "ok 32 fresh verified");
    step(&mut a, "1 GiB created by the manager, mapped and written by the executor");
    assert_eq!(a.call("give"), "ok");
    until(&host, |h| h.created == 0);
    step(&mut a, "given back and destroyed");
    say(&format!("PASS budget_follows_whom {}", host.uuid));
}

/// Q4: whether page-locked host memory counts in this process's WDDM usage.
fn pinned_host_and_the_budget() {
    let backend = CudaBackend::open(0).unwrap();
    if backend.budget().is_none() {
        return say("SKIP pinned_host_and_the_budget: no DXGI budget on this OS or device");
    }
    let before = budget(&backend);
    let mut host = vec![1u8; (1024 * MIB) as usize];
    let (_, ctx) = cuda::primary_context(backend.ordinal()).unwrap();
    let _in = cuda::CtxGuard::enter(ctx).unwrap();
    let d = cuda::driver().unwrap();
    // SAFETY: our buffer, registered for this call's span and unregistered below.
    cuda::check("cuMemHostRegister", unsafe { (d.mem_host_register)(host.as_mut_ptr().cast(), host.len(), 0) }).unwrap();
    let pinned = budget(&backend);
    // SAFETY: the registration above.
    cuda::check("cuMemHostUnregister", unsafe { (d.mem_host_unregister)(host.as_mut_ptr().cast()) }).unwrap();
    say(&format!("FACT 1 GiB page-locked: before {before} | registered {pinned} | unregistered {}", budget(&backend)));
    say(&format!("PASS pinned_host_and_the_budget {}", backend.gpu().unwrap().uuid));
}

/// The stand-in proof's watcher: a line per budget event until stdin closes.
fn watch() {
    let backend = CudaBackend::open(0).unwrap();
    let Some(events) = backend.budget() else { return say("SKIP watch: no DXGI budget") };
    let start = Instant::now();
    let watcher = events.clone();
    let waiting = std::thread::spawn(move || {
        while watcher.wait() {
            let (bytes, usage) = watcher.read().unwrap_or((0, 0));
            say(&format!("EVENT {}ms budget={}MiB usage={}MiB", start.elapsed().as_millis(), bytes / MIB, usage / MIB));
        }
    });
    say(&format!("FACT watching: {}", budget(&backend)));
    for _ in std::io::stdin().lock().lines() {}
    events.close();
    waiting.join().unwrap();
}

/// M8: a manager in one logon session serving an executor in another.
fn serve(name: &str) {
    let h = host(name.into(), true);
    say(&format!("serving {} {}", h.name, h.uuid));
    let start = until(&h, |h| !h.executors.is_empty());
    say(&format!("FACT served pid {}", start.executors.iter().map(|e| e.pid.to_string()).collect::<Vec<_>>().join(",")));
    until(&h, |h| h.executors.is_empty() && h.created == 0);
    h.children.grants.lock().iter().for_each(|r| say(&format!("FACT record {}", serde_json::to_string(r).unwrap_or_default())));
    say("served");
}

fn borrow(name: &str) {
    match Executor::attach(name) {
        Ok(mut exec) => {
            let took = exec.take(64 * MIB, 1, false);
            exec.give_all();
            say(&format!("{} borrow {took}", if took == "ok 2 fresh verified" { "PASS" } else { "FAIL" }));
        }
        Err(why) => say(&format!("FAIL borrow refused {why}")),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["agent", name] => return agent(name),
        ["watch"] => return watch(),
        ["serve", name] => return serve(name),
        ["borrow", name] => return borrow(name),
        _ => {}
    }
    type Case = (&'static str, fn());
    let cases: [Case; 3] = [("chunks_cross_and_adopt", chunks_cross_and_adopt), ("budget_follows_whom", budget_follows_whom), ("pinned_host_and_the_budget", pinned_host_and_the_budget)];
    let ignored = args.iter().any(|a| a == "--ignored" || a == "--include-ignored");
    let named: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    for (name, case) in cases {
        if !ignored || (!named.is_empty() && !named.iter().any(|n| name.contains(n.as_str()))) {
            println!("test {name} ... ignored (a GPU case: --ignored {name})");
            continue;
        }
        case();
    }
}
