//! varena's OS seams on every OS (Linux, macOS, Windows), with no stored set (TensorFS's
//! sources are Unix's so far): host pages shared through an exported handle, the GPU's table
//! held by a row whose lock dies with its process, and the GPU's wake by name. The engines are
//! this binary's `seams_engine`, run as children; they open the table by name (standalone).

use std::io::{BufRead, BufReader, Read};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use varena::{Access, Backend, GpuId, HostBackend, Memory};

const MB: u64 = 1 << 20;
/// A failure backstop only: every wait below ends on the event it waits for.
const BACKSTOP: Duration = Duration::from_secs(60);

/// This binary's namespace for tables and service names (never a machine's).
fn prefix() -> String {
    format!("seams{}-", std::process::id())
}

/// A test's GPU, its table removed when the test ends.
struct Gpu(GpuId);

impl Gpu {
    fn new(case: &str) -> Gpu {
        assert!(varena::os::namespace(&prefix()));
        Gpu(GpuId { uuid: format!("s{}-{case}", std::process::id()), model: "host".into(), memory: 64 * MB })
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        varena::os::remove_table(&self.0.uuid);
    }
}

/// A standalone engine: opens its memory on the GPU's table by name, says `open <pid>`, and
/// holds it until its stdin ends.
#[test]
#[ignore = "the seams tests run it as an engine"]
fn seams_engine() {
    let Ok(spec) = std::env::var("SEAMS_ENGINE") else { return };
    let [prefix, uuid, memory] = spec.split('|').collect::<Vec<_>>()[..] else { panic!("{spec}") };
    assert!(varena::os::namespace(prefix));
    let gpu = GpuId { uuid: uuid.into(), model: "host".into(), memory: memory.parse().unwrap() };
    let mem = Memory::open(gpu.clone(), None, Box::new(HostBackend::new(gpu.memory))).unwrap();
    // Its own line: libtest has written `test seams_engine ... ` before the test runs.
    println!("\nopen {}", std::process::id());
    let _ = std::io::stdin().read_to_end(&mut vec![]);
    drop(mem);
}

/// An engine started by this test, open on `gpu`'s table; its output read to the end.
struct Engine {
    child: Child,
    out: BufReader<ChildStdout>,
}

impl Engine {
    fn start(gpu: &GpuId) -> Engine {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "seams_engine", "--ignored", "--nocapture", "--test-threads=1"])
            .env("SEAMS_ENGINE", format!("{}|{}|{}", prefix(), gpu.uuid, gpu.memory))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut out = BufReader::new(child.stdout.take().unwrap());
        let opened = out.by_ref().lines().map_while(Result::ok).find(|line| line.starts_with("open "));
        assert_eq!(opened, Some(format!("open {}", child.id())), "the engine opened its memory");
        Engine { child, out }
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Its stdin ends: it drops its memory and exits.
    fn finish(mut self) {
        drop(self.child.stdin.take());
        let _ = self.out.read_to_end(&mut vec![]);
        assert!(self.child.wait().unwrap().success());
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn live(ledger: &varena::Ledger, pid: u32) -> bool {
    ledger.status().rows.iter().any(|row| row.pid == pid && row.live)
}

/// A page made and filled here, exported, and imported again: a second mapping of the same bytes,
/// read-only, as custody hands kept pages over (a memfd, an `shm_open` object, a pagefile
/// section).
#[test]
fn host_pages_are_shared_through_an_exported_handle() {
    let backend = HostBackend::new(64 * MB);
    let len = backend.granularity().max(2 * MB);
    let (va, page) = (backend.reserve(len).unwrap(), backend.create(len).unwrap());
    backend.map(va, 0, &page, Access::ReadWrite).unwrap();
    let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
    // SAFETY: `va` maps `len` writable bytes of `page`, nothing else aliases them here.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), va.base as *mut u8, len as usize) };
    backend.protect(va, 0, len, Access::Read).unwrap();
    let imported = backend.import(backend.export(&page).unwrap(), len).unwrap();
    let again = backend.reserve(len).unwrap();
    backend.map(again, 0, &imported, Access::Read).unwrap();
    // SAFETY: `again` maps `len` readable bytes of the same memory.
    let seen = unsafe { std::slice::from_raw_parts(again.base as *const u8, len as usize) };
    assert!(seen == bytes.as_slice(), "the imported page holds the same bytes");
    for (at, p) in [(va, page), (again, imported)] {
        backend.unmap(at, 0, len).unwrap();
        backend.destroy(p);
        backend.release(at);
    }
}

/// A standalone engine opens the GPU's table by name: its row is live on TensorD's table, and
/// when it ends, TensorD's wake (by name) says so and its row is gone.
#[test]
fn a_standalone_engines_row_is_on_the_table_and_its_end_wakes_tensord() {
    let gpu = Gpu::new("wake");
    let ledger = varena::Ledger::create(&gpu.0, None).unwrap();
    let engine = Engine::start(&gpu.0);
    let pid = engine.pid();
    assert!(live(&ledger, pid));
    ledger.clear_events();
    let (woke, wakes) = mpsc::channel();
    std::thread::scope(|scope| {
        let ledger = &ledger;
        scope.spawn(move || woke.send(ledger.await_event().is_ok()).unwrap());
        engine.finish();
        assert!(wakes.recv_timeout(BACKSTOP).expect("its end wakes TensorD"), "the wake is read");
    });
    assert!(!live(&ledger, pid), "its row went with it");
}

/// A killed engine's row lock dies with it: after the kill and a reap, its row is no longer live
/// (an OFD lock, a flock, a LockFileEx range).
#[test]
fn a_killed_engines_row_dies_with_it() {
    let gpu = Gpu::new("kill");
    let ledger = varena::Ledger::create(&gpu.0, None).unwrap();
    let mut engine = Engine::start(&gpu.0);
    let pid = engine.pid();
    assert!(live(&ledger, pid));
    engine.child.kill().unwrap();
    engine.child.wait().unwrap();
    ledger.reap(pid);
    assert!(!live(&ledger, pid), "its row died with it");
}
