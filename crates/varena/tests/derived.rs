//! Derived weights on disk (varena-leases §6, M1/M4), over host pages: the write-behind writer
//! runs only at idle, never writes a page that went, and a refill reads the entry back through
//! TensorFS's read path, as a store fill reads.
#![cfg(unix)]

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tensorfs::io::source::IoTally;
use tensorfs::io::Dst;
use varena::derived::{pays, Hold, Job, Key, Refill, Writer};
use varena::{Access, Backend, Event, HostBackend, Va};

const MB: u64 = 1 << 20;

#[derive(Default)]
struct Holder {
    idle: AtomicBool,
    gone: AtomicBool,
    pins: AtomicUsize,
    written: Mutex<Vec<(Key, File)>>,
}

impl Hold for Holder {
    fn pin(&self, _: &Key) -> Option<Box<dyn Send>> {
        self.pins.fetch_add(1, SeqCst);
        (!self.gone.load(SeqCst)).then(|| Box::new(()) as Box<dyn Send>)
    }

    fn idle(&self) -> bool {
        self.idle.load(SeqCst)
    }

    fn written(&self, key: Key, entry: File) {
        self.written.lock().push((key, entry));
    }
}

fn pattern(len: u64, seed: u8) -> Vec<u8> {
    (0..len).map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed)).collect()
}

/// A mapped page holding `pattern(len, seed)`.
fn page(backend: &HostBackend, len: u64, seed: u8) -> Va {
    let va = backend.reserve(len).unwrap();
    let phys = backend.create(len).unwrap();
    backend.map(va, 0, &phys, Access::ReadWrite).unwrap();
    let bytes = pattern(len, seed);
    // SAFETY: the page is mapped read-write and `len` long.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), va.base as *mut u8, bytes.len()) };
    va
}

fn until(what: &str, f: impl Fn() -> bool) {
    let start = Instant::now();
    while !f() {
        assert!(start.elapsed() < Duration::from_secs(20), "never: {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn rig() -> (Arc<HostBackend>, Arc<Holder>, Writer) {
    let backend = Arc::new(HostBackend::new(256 * MB));
    let holder = Arc::new(Holder::default());
    let writer = Writer::new(backend.clone(), holder.clone()).unwrap();
    (backend, holder, writer)
}

/// M4: written only where deriving again costs more than writing and reading back.
#[test]
fn only_what_pays_is_written() {
    let gib = 1 << 30;
    assert!(pays(2_000_000_000, gib, 2e9, 4e9), "2 s to derive against 0.8 s of disk");
    assert!(!pays(500_000_000, gib, 2e9, 4e9), "0.5 s to derive against 0.8 s of disk");
    assert!(!pays(u64::MAX, gib, 0.0, 4e9), "no measured write bandwidth");
}

/// The store's key: fingerprint, region and the fill identity's digest, as S5's store names it.
#[test]
fn a_key_names_its_page_and_fill() {
    let k = Key::new(1, 7, "lora:a1b2@0.8|bake:sr|lane:8");
    let s = k.store();
    assert!(s.starts_with("0000000000000001/7/") && s.len() == "0000000000000001/7/".len() + 32, "{s}");
    assert_ne!(k, Key::new(1, 7, "lora:a1b2@0.9|bake:sr|lane:8"), "another strength is another entry");
}

/// Written only once the holder is idle; then a refill reads it back through the page cache.
#[test]
fn a_derived_page_is_written_at_idle_and_refills() {
    let (backend, holder, writer) = rig();
    let len = 4 * MB;
    let va = page(&backend, len, 3);
    let key = Key::new(0xabc, 2, "lora:x@0.8");
    writer.offer(Job { key, entry: tempfile::tempfile().unwrap(), va, ready: Event::DONE });
    writer.wake();
    assert_eq!(holder.pins.load(SeqCst), 0, "a write ran while a request was running");
    holder.idle.store(true, SeqCst);
    writer.wake();
    until("the page is written", || writer.pending() == 0 && holder.written.lock().len() == 1);
    assert!(writer.write_bps().is_some_and(|b| b > 0.0), "no write bandwidth measured");
    let (written, entry) = holder.written.lock().pop().unwrap();
    assert_eq!(written, key);
    let mut on_disk = vec![0u8; len as usize];
    entry.read_exact_at(&mut on_disk, 0).unwrap();
    assert!(on_disk == pattern(len, 3), "the entry is not the page's bytes");
    let refill = Refill::new(&key, entry, len);
    let mut back = vec![0u8; len as usize];
    // SAFETY: a host buffer of the entry's length, written only by this refill.
    unsafe { refill.read(Dst::Host { ptr: back.as_mut_ptr(), len: back.len() }, &IoTally::default()) }.unwrap();
    assert!(back == pattern(len, 3), "the refill is not the derived bytes");
}

/// M1: a write queued for a page that went (abandoned by its revoke, or gone when its turn
/// comes) is dropped, never written.
#[test]
fn a_write_never_outlives_its_page() {
    let (backend, holder, writer) = rig();
    let (first, second) = (Key::new(1, 0, "f"), Key::new(1, 1, "f"));
    writer.offer(Job { key: first, entry: tempfile::tempfile().unwrap(), va: page(&backend, 2 * MB, 1), ready: Event::DONE });
    writer.offer(Job { key: second, entry: tempfile::tempfile().unwrap(), va: page(&backend, 2 * MB, 2), ready: Event::DONE });
    writer.abandon(&first);
    holder.idle.store(true, SeqCst);
    writer.wake();
    until("the queue drains", || writer.pending() == 0 && holder.written.lock().len() == 1);
    assert_eq!(holder.written.lock()[0].0, second, "an abandoned page was written");
    holder.gone.store(true, SeqCst);
    let pins = holder.pins.load(SeqCst);
    writer.offer(Job { key: Key::new(1, 2, "f"), entry: tempfile::tempfile().unwrap(), va: page(&backend, 2 * MB, 3), ready: Event::DONE });
    until("the gone page's turn passes", || holder.pins.load(SeqCst) > pins && writer.pending() == 0);
    assert_eq!(holder.written.lock().len(), 1, "a page that went was written");
}
