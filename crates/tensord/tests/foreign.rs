//! VRAM yield, TensorD's side (vram-yield.md §3.2): its watch reads NVML (a fake reading here,
//! built from what the device really holds: the rows' bytes plus a foreign process's) and spends
//! each reading once on the ledger, whose asks reach a real row (varena's `Memory` over host
//! pages, in this process).
#![cfg(target_os = "linux")]
use std::{collections::BTreeMap, sync::Arc};

use platform::gpu::Reading;
use tensord::memory::ForeignWatch;
use varena::{
    layout::{Item, Layout, Part, Region, REGION_ALIGN},
    Fill, Fingerprint, GpuId, HostBackend, Ledger, Level, Memory, SetSpec,
};

const P: u64 = REGION_ALIGN;
/// A card H is 1 GiB of; only the pages made take host memory.
const CARD: u64 = 16 << 30;
const MIB: u64 = 1 << 20;

fn gpu(tag: &str) -> GpuId {
    GpuId { uuid: format!("GPU-foreign-{tag}-{}", uuid::Uuid::new_v4()), model: "host".into(), memory: CARD }
}

fn set(dir: &std::path::Path, name: &str, blocks: u64, uuid: &str) -> SetSpec {
    let path = dir.join(format!("{name}.bin"));
    std::fs::write(&path, vec![9u8; (blocks * P) as usize]).unwrap();
    let obj = tensorfs::format::ObjectRef { sha256: format!("obj-{name}"), length: blocks * P };
    let regions = (0..blocks)
        .map(|i| Region {
            offset: i * P,
            nbytes: P,
            span: P,
            items: vec![Item { offset: i * P, len: P, source: tensorfs::io::ItemSource::Object(tensorfs::io::ObjectRange { obj: obj.clone(), off: i * P, len: P }), part_end: true }],
        })
        .collect();
    let layout = Layout {
        parts: (0..blocks).map(|i| Part { what: format!("t{i}"), region: i as u32, offset: i * P, nbytes: P }).collect(),
        regions,
        nbytes: blocks * P,
        digest: format!("layout-{name}-{uuid}"),
        names: (0..blocks).map(|i| vec![format!("t{i}")]).collect(),
    };
    let source = tensorfs::io::source::Source::files(vec![(obj, std::fs::File::open(&path).unwrap())], false, 0);
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", uuid);
    SetSpec { name: name.into(), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

/// NVML's view: the rows hold what they hold, and each foreign process its bytes.
fn reading(ledger: &Ledger, foreign: &[(u32, u64)]) -> Reading {
    let used = ForeignWatch::held(ledger) + foreign.iter().map(|(_, b)| b).sum::<u64>();
    Reading { total: CARD, free: CARD - used, display: false, processes: Some(foreign.iter().copied().collect::<BTreeMap<_, _>>()) }
}

fn pages(ledger: &Ledger) -> u64 {
    ledger.status().rows.iter().map(|r| r.classes.iter().enumerate().filter(|(c, _)| *c != 1).map(|(_, b)| b).sum::<u64>()).sum()
}

/// A foreign process grows: while 2H stays free nothing is asked; past H free it asks for
/// 2H − free and the idle pages go; the next reading of the same process asks nothing again (each
/// deficit spent once).
#[test]
fn a_growing_foreign_process_is_given_the_excess_once() {
    let id = gpu("grow");
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::create(&id, None).unwrap();
    let mem = Memory::open(id.clone(), None, Box::new(HostBackend::new(CARD))).unwrap();
    mem.load(set(dir.path(), "idle", 4, &id.uuid), Level::Gpu).unwrap();
    let (h, game, own) = (varena::headroom(CARD), 4242, std::process::id());
    let mut watch = ForeignWatch::default();
    // The game leaves 2H free beside cozy: nothing asked.
    let small = CARD - ForeignWatch::held(&ledger) - 2 * h;
    let spent = watch.reading(&ledger, &reading(&ledger, &[(game, small)]), own).unwrap();
    assert_eq!((spent.freed, pages(&ledger)), (0, 4 * P), "{spent:?}");
    // It grows past H free: the idle set goes.
    let big = CARD - ForeignWatch::held(&ledger) - h / 2;
    let spent = watch.reading(&ledger, &reading(&ledger, &[(game, big)]), own).unwrap();
    assert_eq!((spent.freed, pages(&ledger)), (4 * P, 0), "{spent:?}");
    assert_eq!(spent.holder, Some((game, big)));
    // The same process again: the room it took is free, nothing more is asked.
    let spent = watch.reading(&ledger, &reading(&ledger, &[(game, big)]), own).unwrap();
    assert_eq!(spent.freed, 0, "{spent:?}");
    drop(mem);
}

/// A process present when sampling began never counts as new; one that appears and crosses
/// 256 MiB takes cozy's idle pages with no deficit, once.
#[test]
fn a_new_foreign_process_takes_idle_pages_once() {
    let id = gpu("new");
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::create(&id, None).unwrap();
    let mem = Memory::open(id.clone(), None, Box::new(HostBackend::new(CARD))).unwrap();
    mem.load(set(dir.path(), "idle", 3, &id.uuid), Level::Gpu).unwrap();
    let (desktop, game, own) = (100, 4242, std::process::id());
    let mut watch = ForeignWatch::default();
    let spent = watch.reading(&ledger, &reading(&ledger, &[(desktop, 300 * MIB)]), own).unwrap();
    assert!(!spent.fresh && spent.freed == 0, "the desktop was there first: {spent:?}");
    let spent = watch.reading(&ledger, &reading(&ledger, &[(desktop, 300 * MIB), (game, 100 * MIB)]), own).unwrap();
    assert!(!spent.fresh, "under 256 MiB: {spent:?}");
    let spent = watch.reading(&ledger, &reading(&ledger, &[(desktop, 300 * MIB), (game, 300 * MIB)]), own).unwrap();
    assert!(spent.fresh && spent.freed == 3 * P && pages(&ledger) == 0, "a new process crossed 256 MiB: {spent:?}");
    let spent = watch.reading(&ledger, &reading(&ledger, &[(desktop, 300 * MIB), (game, 400 * MIB)]), own).unwrap();
    assert!(!spent.fresh, "once per process: {spent:?}");
    drop(mem);
}

/// Sampling stops once nothing is held: executors measure foreign bytes themselves again.
#[test]
fn sampling_stops_when_nothing_is_held() {
    let id = gpu("stop");
    let ledger = Ledger::create(&id, None).unwrap();
    let mut watch = ForeignWatch::default();
    watch.reading(&ledger, &reading(&ledger, &[(4242, MIB)]), std::process::id()).unwrap();
    let foreign = ledger.status().foreign;
    watch.stop(&ledger).unwrap();
    // A standalone executor opened now measures for itself: its own figure, not the watch's.
    let mem = Memory::open(id.clone(), None, Box::new(HostBackend::new(CARD))).unwrap();
    drop(mem.make_room(0));
    assert_eq!(foreign, MIB);
    assert_ne!(ledger.status().foreign, MIB, "an executor still read the stopped watch's figure");
}
