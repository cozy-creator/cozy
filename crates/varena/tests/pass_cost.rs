//! A pass's cost on the hot path (S3's census: 938 µs per pass on an L40S against 14 µs on CPU):
//! a pass start and end read varena's own accounting, never the device's per-process figure
//! (NVML) or its free bytes. Each case prints `PASSCOST <backend> <µs per
//! pass> <driver calls per pass>`; the native one is for a card (`--ignored`).

use std::collections::BTreeMap;
use std::fs::File;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use platform::handles::Owned;
use tensorfs::format::ObjectRef;
use tensorfs::io::source::Source;
use tensorfs::io::{Dst, ItemSource, ObjectRange};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Access, Backend, CudaBackend, Event, Fill, Fingerprint, GpuId, HostBackend, Level, Memory, Physical, Plan, Result, SetSpec, Step, Va};

const PASSES: u32 = 200;

/// A backend counting its device calls.
struct Counted<B> {
    inner: B,
    calls: Arc<Mutex<BTreeMap<&'static str, u64>>>,
}

impl<B> Counted<B> {
    fn count(&self, call: &'static str) {
        *self.calls.lock().entry(call).or_default() += 1;
    }
}

impl<B: Backend> Backend for Counted<B> {
    fn capacity(&self) -> u64 { self.inner.capacity() }
    fn headroom(&self) -> u64 { self.inner.headroom() }
    fn free(&self) -> Result<u64> { self.count("free"); self.inner.free() }
    fn process_used(&self) -> Result<u64> { self.count("process_used"); self.inner.process_used() }
    fn opened_context(&self) -> u64 { self.inner.opened_context() }
    fn granularity(&self) -> u64 { self.inner.granularity() }
    fn reserve(&self, len: u64) -> Result<Va> { self.inner.reserve(len) }
    fn release(&self, va: Va) { self.inner.release(va) }
    fn create(&self, len: u64) -> Result<Physical> { self.count("create"); self.inner.create(len) }
    fn destroy(&self, page: Physical) -> Result<()> { self.count("destroy"); self.inner.destroy(page) }
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()> { self.count("map"); self.inner.map(va, offset, page, access) }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> Result<()> { self.count("protect"); self.inner.protect(va, offset, len, access) }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> Result<()> { self.count("unmap"); self.inner.unmap(va, offset, len) }
    fn export(&self, page: &Physical) -> Result<Owned> { self.count("export"); self.inner.export(page) }
    fn import(&self, handle: Owned, len: u64) -> Result<Physical> { self.count("import"); self.inner.import(handle, len) }
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst { self.inner.target(va, offset, len) }
    fn fence(&self) -> Result<Event> { self.count("fence"); self.inner.fence() }
    fn record(&self, stream: usize) -> Result<Event> { self.count("record"); self.inner.record(stream) }
    fn query(&self, event: Event) -> Result<bool> { self.count("query"); self.inner.query(event) }
    fn elapsed(&self, from: Event, to: Event) -> Result<Option<u64>> { self.inner.elapsed(from, to) }
    fn wait(&self, event: Event, stream: usize) -> Result<()> { self.count("wait"); self.inner.wait(event, stream) }
    fn synchronize(&self, stream: usize) -> Result<()> { self.count("synchronize"); self.inner.synchronize(stream) }
    fn finish(&self, event: Event) -> Result<()> { self.count("finish"); self.inner.finish(event) }
    fn retire(&self, event: Event) { self.count("retire"); self.inner.retire(event) }
    fn drain(&self) -> Result<()> { self.inner.drain() }
}

fn set(dir: &std::path::Path, uuid: &str, pages: u64) -> SetSpec {
    let p = REGION_ALIGN;
    let path = dir.join("m.bin");
    std::fs::write(&path, vec![7u8; (pages * p) as usize]).unwrap();
    let obj = ObjectRef { sha256: "obj-m".into(), length: pages * p };
    let regions = (0..pages)
        .map(|i| Region { offset: i * p, nbytes: p, span: p, items: vec![Item { offset: i * p, len: p, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: i * p, len: p }), part_end: true }] })
        .collect();
    let layout = Layout {
        parts: (0..pages).map(|i| Part { what: format!("t{i}"), region: i as u32, offset: i * p, nbytes: p }).collect(),
        regions,
        nbytes: pages * p,
        digest: format!("layout-m-{uuid}"),
        names: (0..pages).map(|i| vec![format!("t{i}")]).collect(),
    };
    let source = Source::files(vec![(obj, File::open(&path).unwrap())], false, 0);
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", uuid);
    SetSpec { name: "m".into(), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

/// Opens `g` over `backend` (standalone), loads a 2-page set
/// and opens a request; then times `PASSES` passes of it, each a start and an end, a bind of
/// each block between. Returns µs per pass (start and end alone) and the device calls per pass.
fn passes<B: Backend + 'static>(backend: B, g: &GpuId) -> (f64, BTreeMap<&'static str, f64>) {
    let calls = Arc::new(Mutex::new(BTreeMap::new()));
    let mem = Memory::open(g.clone(), Box::new(Counted { inner: backend, calls: calls.clone() })).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let m = mem.load(set(dir.path(), &g.uuid, 2), Level::Gpu).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["m".into()], passes: 1 }] }).unwrap();
    let warm = r.pass(m, "k").unwrap();
    warm.end(0);
    let mut spent = std::time::Duration::ZERO;
    let before = calls.lock().clone();
    for _ in 0..PASSES {
        let t = Instant::now();
        let pass = r.pass(m, "k").unwrap();
        spent += t.elapsed();
        let (a, b) = (mem.bind(m, 0, 0).unwrap(), mem.bind(m, 1, 0).unwrap());
        drop((a, b));
        let t = Instant::now();
        pass.end(0);
        spent += t.elapsed();
    }
    let after = calls.lock().clone();
    drop(r);
    drop(mem);
    varena::os::remove_table(&g.uuid);
    let per: BTreeMap<&'static str, f64> = after.iter().map(|(k, v)| (*k, (v - before.get(k).copied().unwrap_or(0)) as f64 / f64::from(PASSES))).filter(|(_, v)| *v > 0.0).collect();
    (spent.as_secs_f64() * 1e6 / f64::from(PASSES), per)
}

#[test]
fn a_pass_reads_no_device_figure() {
    assert!(varena::os::namespace(&format!("pc{}-", std::process::id())));
    let g = GpuId { uuid: format!("pc-host-{}", std::process::id()), model: "host".into(), memory: 64 * REGION_ALIGN };
    let (us, per) = passes(HostBackend::new(g.memory), &g);
    println!("PASSCOST host {us:.1} µs per pass; device calls per pass {per:?}");
    // Binds of resident blocks query their release events; nothing reads the device's figures.
    assert!(!per.contains_key("process_used") && !per.contains_key("free"), "a pass read the device's figures: {per:?}");
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_pass_reads_no_device_figure_on_a_card() {
    assert!(varena::os::namespace(&format!("pc{}-", std::process::id())));
    let b = CudaBackend::open(0).expect("a CUDA device");
    let g = GpuId { uuid: format!("{}-pc-{}", b.gpu().unwrap().uuid, std::process::id()), ..b.gpu().unwrap() };
    let (us, per) = passes(b, &g);
    println!("PASSCOST cuda {us:.1} µs per pass; device calls per pass {per:?}");
    assert!(!per.contains_key("process_used") && !per.contains_key("free"), "a pass read the device's figures: {per:?}");
}
