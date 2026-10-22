//! Step 2's watermark over host pages: a running set bigger than the card streams its upper blocks through slots.
// Its sets load from TensorFS sources, which are Unix's so far (`tensorfs::io::source`).
#![cfg(unix)]

use std::fs::File;
use std::sync::Arc;

use tensorfs::format::ObjectRef;
use tensorfs::io::source::Source;
use tensorfs::io::{ItemSource, ObjectRange};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use platform::handles::Owned;
use tensorfs::io::Dst;
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Access, Backend, Event, Fill, Fingerprint, GpuId, HostBackend, Level, Memory, Physical, Plan, SetSpec, Step, Va};

/// Host pages, counting the pages made (`create`) and mapped (`map`): a slot moves with a map and no create. The third
/// count is torch's bytes (activations): the driver has that much less free, and a create beyond it is refused.
struct Counted(HostBackend, Arc<[AtomicU64; 3]>);

impl Backend for Counted {
    fn capacity(&self) -> u64 { self.0.capacity() }
    fn free(&self) -> varena::Result<u64> { Ok(self.0.free()?.saturating_sub(self.1[2].load(Relaxed))) }
    fn process_used(&self) -> varena::Result<u64> { self.0.process_used() }
    fn opened_context(&self) -> u64 { self.0.opened_context() }
    fn granularity(&self) -> u64 { self.0.granularity() }
    fn reserve(&self, len: u64) -> varena::Result<Va> { self.0.reserve(len) }
    fn release(&self, va: Va) { self.0.release(va) }
    fn create(&self, len: u64) -> varena::Result<Physical> {
        if len > self.free()? {
            return Err(varena::Error::Device { op: "create", code: varena::backend::OUT_OF_MEMORY, name: "out of memory".into() });
        }
        self.1[0].fetch_add(1, Relaxed);
        self.0.create(len)
    }
    fn destroy(&self, page: Physical) -> varena::Result<()> { self.0.destroy(page) }
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> varena::Result<()> {
        self.1[1].fetch_add(1, Relaxed);
        self.0.map(va, offset, page, access)
    }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> varena::Result<()> { self.0.protect(va, offset, len, access) }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> varena::Result<()> { self.0.unmap(va, offset, len) }
    fn export(&self, page: &Physical) -> varena::Result<Owned> { self.0.export(page) }
    fn import(&self, handle: Owned, len: u64) -> varena::Result<Physical> { self.0.import(handle, len) }
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst { self.0.target(va, offset, len) }
    fn save(&self, va: Va, offset: u64, host: &mut [u8], after: Event) -> varena::Result<()> { self.0.save(va, offset, host, after) }
    fn fence(&self) -> varena::Result<Event> { self.0.fence() }
    fn record(&self, stream: usize) -> varena::Result<Event> { self.0.record(stream) }
    fn query(&self, event: Event) -> varena::Result<bool> { self.0.query(event) }
    fn elapsed(&self, from: Event, to: Event) -> varena::Result<Option<u64>> { self.0.elapsed(from, to) }
    fn wait(&self, event: Event, stream: usize) -> varena::Result<()> { self.0.wait(event, stream) }
    fn synchronize(&self, stream: usize) -> varena::Result<()> { self.0.synchronize(stream) }
    fn finish(&self, event: Event) -> varena::Result<()> { self.0.finish(event) }
    fn register_host(&self, ptr: u64, len: u64) -> varena::Result<()> { self.0.register_host(ptr, len) }
    fn unregister_host(&self, ptr: u64) -> varena::Result<()> { self.0.unregister_host(ptr) }
    fn retire(&self, event: Event) { self.0.retire(event) }
    fn drain(&self) -> varena::Result<()> { self.0.drain() }
    fn drain_save(&self) -> varena::Result<()> { self.0.drain_save() }
}

/// This binary's namespace for tables and service names (never a machine's), and the test's table removed when it ends.
struct Scope(String);

impl Scope {
    fn new(uuid: &str) -> Scope {
        assert!(varena::os::namespace(&format!("test{}-", std::process::id())));
        Scope(uuid.to_string())
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        varena::os::remove_table(&self.0);
        varena::tier::remove_claims();
    }
}

/// Block `b`'s bytes: its own on every page.
fn bytes(b: u64) -> Vec<u8> {
    (0..REGION_ALIGN).map(|i| (i as u8).wrapping_mul(31).wrapping_add(b as u8)).collect()
}

/// A stored set of `blocks` pages, each its own bytes.
fn set(dir: &std::path::Path, name: &str, blocks: u64, uuid: &str) -> SetSpec {
    let path = dir.join(format!("{name}.bin"));
    std::fs::write(&path, (0..blocks).flat_map(bytes).collect::<Vec<u8>>()).unwrap();
    let obj = ObjectRef { sha256: format!("obj-{name}"), length: blocks * REGION_ALIGN };
    let layout = Layout {
        parts: (0..blocks).map(|r| Part { what: format!("t{r}"), region: r as u32, offset: r * REGION_ALIGN, nbytes: REGION_ALIGN }).collect(),
        regions: (0..blocks)
            .map(|r| Region {
                offset: r * REGION_ALIGN,
                nbytes: REGION_ALIGN,
                span: REGION_ALIGN,
                items: vec![Item { offset: r * REGION_ALIGN, len: REGION_ALIGN, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: r * REGION_ALIGN, len: REGION_ALIGN }), part_end: true }],
            })
            .collect(),
        nbytes: blocks * REGION_ALIGN,
        digest: format!("layout-{name}-{uuid}"),
        names: (0..blocks).map(|r| vec![format!("t{r}")]).collect(),
    };
    let source = Source::files(vec![(obj, File::open(&path).unwrap())], false, 0);
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", uuid);
    SetSpec { name: format!("pkg@1/{name}"), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

/// A card of eight pages and a set of twenty: two passes bind every block in order, each with its own bytes. The
/// set streams through the slots once the card is full, and nothing is refused. In the second pass the upper blocks
/// come through slots that move (a map each, no page made), and the lower blocks stay resident.
#[test]
fn a_running_set_bigger_than_the_card_streams_through_slots() {
    let dir = tempfile::tempdir().unwrap();
    let memory = 8 * REGION_ALIGN;
    let g = GpuId { uuid: format!("stream-{}", std::process::id()), model: "host".into(), memory };
    let _scope = Scope::new(&g.uuid);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let counts = Arc::new([AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)]);
    let mem = Memory::open(g.clone(), None, Box::new(Counted(HostBackend::new(memory), counts.clone()))).unwrap();
    let blocks = 20;
    let s = mem.load(set(dir.path(), "big", blocks, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["pkg@1/big".into()], passes: 1 }] }).unwrap();
    let mut made = vec![];
    for pass in 0..2 {
        let (creates, maps) = (counts[0].load(Relaxed), counts[1].load(Relaxed));
        let p = r.pass(s, "fwd|1").unwrap();
        for b in 0..blocks as u32 {
            let pin = mem.bind(s, b, 0).unwrap_or_else(|e| panic!("pass {pass}, block {b}: {e}"));
            // SAFETY: a host page mapped at `ptr`, pinned for this read.
            let got = unsafe { std::slice::from_raw_parts(pin.ptr as *const u8, pin.len as usize) };
            assert!(got == bytes(u64::from(b)).as_slice(), "pass {pass}: block {b} has other bytes");
        }
        p.end(0);
        made.push((counts[0].load(Relaxed) - creates, counts[1].load(Relaxed) - maps));
    }
    let (creates, maps) = made[1];
    assert!(maps > 0 && creates == 0, "the second pass streamed through slots without making pages: {made:?}");
    let stats = r.end();
    assert!(!stats.make_room.is_empty(), "the card ran out and made room");
    assert_eq!(stats.leftover, 0);
}

/// Binds every block of `s` in order, each checked against its own bytes.
fn sweep(mem: &Memory, s: varena::SetId, blocks: u32) -> Result<(), String> {
    (0..blocks).try_for_each(|b| {
        let pin = mem.bind(s, b, 0).map_err(|e| format!("block {b}: {e}"))?;
        // SAFETY: a host page mapped at `ptr`, pinned for this read.
        let got = unsafe { std::slice::from_raw_parts(pin.ptr as *const u8, pin.len as usize) };
        if got == bytes(u64::from(b)).as_slice() { Ok(()) } else { Err(format!("block {b} has other bytes")) }
    })
}

/// The floor is the activations and one block (the verifier's W1). A 10-block set beside a 2-block activation (torch's
/// segment, claimed through the hook), with 1 to 4 blocks of room beside it.
/// - Three passes each bind every block with its own bytes, and nothing is refused.
/// - Then the activation grows into all the room but one block (the hook takes the slots down to one), and a fourth
///   pass still streams.
#[test]
fn the_floor_is_the_activations_and_one_block() {
    let dir = tempfile::tempdir().unwrap();
    let mut results = vec![];
    for spare in [4u64, 3, 2, 1] {
        let (activation, blocks) = (2, 10);
        let memory = (activation + spare) * REGION_ALIGN;
        let g = GpuId { uuid: format!("floor{spare}-{}", std::process::id()), model: "host".into(), memory };
        let _scope = Scope::new(&g.uuid);
        let _ledger = varena::Ledger::create(&g, None).unwrap();
        let counts = Arc::new([AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)]);
        let mem = Memory::open(g.clone(), None, Box::new(Counted(HostBackend::new(memory), counts.clone()))).unwrap();
        let s = mem.load(set(dir.path(), &format!("floor{spare}"), blocks, &g.uuid), Level::Ram).unwrap();
        let r = mem.request("r", Plan { steps: vec![Step { sets: vec![format!("pkg@1/floor{spare}")], passes: 1 }] }).unwrap();
        let segment = |bytes: u64| {
            let claim = mem.make_room(bytes).map_err(|e| format!("the hook, {bytes} B: {e}"))?;
            counts[2].fetch_add(bytes, Relaxed);
            Ok::<_, String>(claim.resident())
        };
        let pass = || -> Result<(), String> {
            let p = r.pass(s, "fwd|1").map_err(|e| e.to_string())?;
            let swept = sweep(&mem, s, blocks as u32);
            p.end(0);
            swept
        };
        let outcome = (|| {
            let _first = segment(activation * REGION_ALIGN)?;
            (0..3).try_for_each(|_| pass())?;
            let _grown = segment((spare - 1) * REGION_ALIGN)?;
            pass()
        })();
        results.push((spare, outcome));
        drop(r);
    }
    println!("(blocks of room beside the activation, outcome): {results:?}");
    assert!(results.iter().all(|(_, r)| r.is_ok()), "weights were refused above the floor: {results:?}");
}
