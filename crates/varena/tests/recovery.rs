#![cfg(unix)]

use platform::handles::Owned;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tensorfs::io::Dst;
use varena::arena::{Direct, Owner, Torch};
use varena::backend::Allowance;
use varena::{Access, Backend, Error, Event, HostBackend, Physical, Result, Va};

struct Faults {
    host: HostBackend,
    fail_in_at: AtomicU64,
    ins: AtomicU64,
    fail_map_at: AtomicU64,
    maps: AtomicU64,
    fail_unmap_at: AtomicU64,
    unmaps: AtomicU64,
    fail_out_at: AtomicU64,
    outs: AtomicU64,
}
impl Faults {
    fn new() -> Self {
        Self {
            host: HostBackend::new(64 << 20),
            fail_in_at: AtomicU64::new(0),
            ins: AtomicU64::new(0),
            fail_map_at: AtomicU64::new(0),
            maps: AtomicU64::new(0),
            fail_unmap_at: AtomicU64::new(0),
            unmaps: AtomicU64::new(0),
            fail_out_at: AtomicU64::new(0),
            outs: AtomicU64::new(0),
        }
    }
}
impl Backend for Faults {
    fn capacity(&self) -> u64 {
        self.host.capacity()
    }
    fn headroom(&self) -> u64 {
        0
    }
    fn free(&self) -> Result<u64> {
        self.host.free()
    }
    fn process_used(&self) -> Result<u64> {
        self.host.process_used()
    }
    fn opened_context(&self) -> u64 {
        0
    }
    fn granularity(&self) -> u64 {
        self.host.granularity()
    }
    fn reserve(&self, n: u64) -> Result<Va> {
        self.host.reserve(n)
    }
    fn release(&self, v: Va) {
        self.host.release(v)
    }
    fn create(&self, n: u64) -> Result<Physical> {
        self.host.create(n)
    }
    fn destroy(&self, p: Physical) -> Result<()> {
        self.host.destroy(p)
    }
    fn map(&self, v: Va, o: u64, p: &Physical, a: Access) -> Result<()> {
        if self.fail_map_at.load(Ordering::SeqCst) == self.maps.fetch_add(1, Ordering::SeqCst) + 1 {
            return Err(Error::Io("injected map failure".into()));
        }
        self.host.map(v, o, p, a)
    }
    fn protect(&self, v: Va, o: u64, n: u64, a: Access) -> Result<()> {
        self.host.protect(v, o, n, a)
    }
    fn unmap(&self, v: Va, o: u64, n: u64) -> Result<()> {
        if self.fail_unmap_at.load(Ordering::SeqCst)
            == self.unmaps.fetch_add(1, Ordering::SeqCst) + 1
        {
            return Err(Error::Io("injected unmap failure".into()));
        }
        self.host.unmap(v, o, n)
    }
    fn export(&self, p: &Physical) -> Result<Owned> {
        self.host.export(p)
    }
    fn import(&self, h: Owned, n: u64) -> Result<Physical> {
        self.host.import(h, n)
    }
    fn target(&self, v: Va, o: u64, n: u64) -> Dst {
        self.host.target(v, o, n)
    }
    fn copy_in(&self, v: Va, o: u64, b: &[u8]) -> Result<()> {
        if self.fail_in_at.load(Ordering::SeqCst) == self.ins.fetch_add(1, Ordering::SeqCst) + 1 {
            return Err(Error::Io("injected copy-in failure".into()));
        }
        self.host.copy_in(v, o, b)
    }
    fn copy_out(&self, v: Va, o: u64, b: &mut [u8], e: Event) -> Result<()> {
        let n = self.outs.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_out_at.load(Ordering::SeqCst) == n {
            return Err(Error::Io("injected copy-out failure".into()));
        }
        self.host.copy_out(v, o, b, e)
    }
    fn fence(&self) -> Result<Event> {
        self.host.fence()
    }
    fn record(&self, s: usize) -> Result<Event> {
        self.host.record(s)
    }
    fn query(&self, e: Event) -> Result<bool> {
        self.host.query(e)
    }
    fn elapsed(&self, a: Event, b: Event) -> Result<Option<u64>> {
        self.host.elapsed(a, b)
    }
    fn wait(&self, e: Event, s: usize) -> Result<()> {
        self.host.wait(e, s)
    }
    fn synchronize(&self, s: usize) -> Result<()> {
        self.host.synchronize(s)
    }
    fn finish(&self, e: Event) -> Result<()> {
        self.host.finish(e)
    }
    fn retire(&self, e: Event) {
        self.host.retire(e)
    }
    fn drain(&self) -> Result<()> {
        self.host.drain()
    }
}
fn allocator() -> (Arc<Faults>, Arc<Torch>) {
    let b = Arc::new(Faults::new());
    let budget = Arc::new(Allowance::new(64 << 20));
    let t = Torch::new(b.clone(), Arc::new(Direct::new(b.clone(), budget))).unwrap();
    (b, t)
}

fn populated(b: &Faults, t: &Torch) -> [u64; 2] {
    let _route = t.route(Owner::Region(1));
    let pointers = [
        t.alloc(1 << 20, 1, None).unwrap(),
        t.alloc(1 << 20, 2, None).unwrap(),
    ];
    for (i, &ptr) in pointers.iter().enumerate() {
        b.copy_in(Va { base: ptr, len: 16 }, 0, &[123 + i as u8; 16])
            .unwrap();
    }
    pointers
}

fn assert_contents(b: &Faults, pointers: &[u64; 2]) {
    for (i, &ptr) in pointers.iter().enumerate() {
        let mut actual = [0; 16];
        b.copy_out(Va { base: ptr, len: 16 }, 0, &mut actual, Event::DONE)
            .unwrap();
        assert_eq!(actual, [123 + i as u8; 16]);
    }
}

#[test]
fn eviction_copy_failure_preserves_every_mapping_and_byte() {
    for fail in [1, 2] {
        let (b, t) = allocator();
        let pointers = populated(&b, &t);
        b.fail_out_at.store(fail, Ordering::SeqCst);
        assert!(t.evict(Owner::Region(1), true).is_err());
        assert_eq!(t.mapped_by(Owner::Region(1)), 4 << 20);
        assert_eq!(b.process_used().unwrap(), 4 << 20);
        assert_contents(&b, &pointers);
        let mut saved = t.evict(Owner::Region(1), true).unwrap().unwrap();
        assert!(!t.make_resident(Owner::Region(1), &mut saved).unwrap());
        assert_contents(&b, &pointers);
    }
}

#[test]
fn restore_copy_failure_keeps_swap_and_rolls_back_every_mapping() {
    for fail in [1, 2] {
        let (b, t) = allocator();
        let pointers = populated(&b, &t);
        let mut saved = t.evict(Owner::Region(1), true).unwrap().unwrap();
        b.fail_in_at
            .store(b.ins.load(Ordering::SeqCst) + fail, Ordering::SeqCst);
        assert!(t.make_resident(Owner::Region(1), &mut saved).is_err());
        assert_eq!(saved.bytes, 4 << 20);
        assert!(saved.swapped());
        assert_eq!(t.mapped(), 0);
        assert_eq!(b.process_used().unwrap(), 0);
        assert!(!t.make_resident(Owner::Region(1), &mut saved).unwrap());
        assert_contents(&b, &pointers);
    }
}

#[test]
fn restore_map_failure_keeps_swap_and_rolls_back_prior_maps() {
    for fail in [1, 2] {
        let (b, t) = allocator();
        let pointers = populated(&b, &t);
        let mut saved = t.evict(Owner::Region(1), true).unwrap().unwrap();
        b.fail_map_at
            .store(b.maps.load(Ordering::SeqCst) + fail, Ordering::SeqCst);
        assert!(t.make_resident(Owner::Region(1), &mut saved).is_err());
        assert_eq!(saved.bytes, 4 << 20);
        assert!(saved.swapped());
        assert_eq!(t.mapped(), 0);
        assert_eq!(b.process_used().unwrap(), 0);
        assert!(!t.make_resident(Owner::Region(1), &mut saved).unwrap());
        assert_contents(&b, &pointers);
    }
}

#[test]
fn eviction_unmap_failure_retains_ownership_until_retry() {
    for fail in [1, 2] {
        let (b, t) = allocator();
        let pointers = populated(&b, &t);
        b.fail_unmap_at.store(fail, Ordering::SeqCst);
        let mut saved = t.evict(Owner::Region(1), true).unwrap().unwrap();
        assert_eq!(saved.bytes, 4 << 20);
        assert_eq!(t.mapped(), 2 << 20);
        assert_eq!(b.process_used().unwrap(), 2 << 20);
        assert!(!t.make_resident(Owner::Region(1), &mut saved).unwrap());
        assert_contents(&b, &pointers);
    }
}

#[test]
fn failed_restore_rollback_retains_stuck_mapping_and_saved_bytes() {
    let (b, t) = allocator();
    let pointers = populated(&b, &t);
    let mut saved = t.evict(Owner::Region(1), true).unwrap().unwrap();
    b.fail_in_at
        .store(b.ins.load(Ordering::SeqCst) + 2, Ordering::SeqCst);
    b.fail_unmap_at
        .store(b.unmaps.load(Ordering::SeqCst) + 1, Ordering::SeqCst);
    assert!(t.make_resident(Owner::Region(1), &mut saved).is_err());
    assert_eq!(saved.bytes, 4 << 20);
    assert!(saved.swapped());
    assert_eq!(t.mapped(), 2 << 20);
    assert_eq!(b.process_used().unwrap(), 2 << 20);
    assert!(!t.make_resident(Owner::Region(1), &mut saved).unwrap());
    assert_contents(&b, &pointers);
}

fn loaded_memory(name: &str) -> (varena::Memory, varena::SetId) {
    use tensorfs::io::source::Source;
    use varena::layout::{Item, ItemSource, Layout, Region};
    use varena::{Fill, Fingerprint, GpuId, Level, Memory, SetSpec};
    let memory = Memory::open(
        GpuId {
            uuid: name.into(),
            model: "host".into(),
            memory: 64 << 20,
        },
        Box::new(HostBackend::new(64 << 20)),
    )
    .unwrap();
    memory.serve_standalone_torch().unwrap();
    let layout = Arc::new(Layout {
        parts: vec![],
        names: vec![vec!["weight".into()]],
        nbytes: 32 << 20,
        digest: "audit-weight".into(),
        regions: vec![Region {
            offset: 0,
            nbytes: 16,
            span: 32 << 20,
            items: vec![Item {
                offset: 0,
                len: 16,
                source: ItemSource::Inline(vec![68; 16].into()),
                part_end: true,
            }],
        }],
    });
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "audit", "0/1", name);
    let set = memory
        .load(
            SetSpec {
                name: "audit/weight".into(),
                layout,
                source: Arc::new(Source::files(vec![], false, 0)),
                fingerprint,
            },
            Level::Gpu,
        )
        .unwrap();
    (memory, set)
}

#[test]
fn standalone_torch_reclaims_cold_weights_before_exhaustion() {
    let (memory, _) = loaded_memory("standalone-cold-weights");
    assert_eq!(memory.classes()[3], 32 << 20);
    let pointer = memory.torch_alloc(33 << 20, 0, None).unwrap();
    assert_eq!(memory.classes()[3], 0);
    memory.torch_free(pointer, 33 << 20, 0);
}

#[test]
fn standalone_torch_cannot_reclaim_a_pinned_weight() {
    let (memory, set) = loaded_memory("standalone-pinned-weights");
    let pin = memory.bind(set, 0, 0).unwrap();
    assert!(matches!(
        memory.torch_alloc(33 << 20, 0, None),
        Err(Error::Exhausted(_))
    ));
    assert_eq!(memory.classes()[6], 32 << 20);
    drop(pin);
}

#[test]
fn capture_cannot_enter_standalone_recovery() {
    let (memory, _) = loaded_memory("standalone-capture");
    let pool = varena::PoolId(41, 1);
    memory.pool_begin(pool, None);
    assert!(matches!(
        memory.torch_alloc(33 << 20, 0, Some(pool)),
        Err(Error::Exhausted(_))
    ));
    memory.pool_end(pool);
    memory.pool_release(pool);
    assert_eq!(memory.classes()[3], 32 << 20);
}
