//! Host-OOM, the RAM tier's side: a tier fill (a filled page's copy) goes only into measured
//! surplus. Its own binary: the namespace's tier is this test's alone.
//! Linux: the RAM tier's files are Linux's (elsewhere it is absent).
#![cfg(target_os = "linux")]

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use tensorfs::format::ObjectRef;
use tensorfs::io::source::Source;
use tensorfs::io::{ItemSource, ObjectRange};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Fill, Fingerprint, GpuId, HostBackend, Level, Memory, Plan, SetSpec};

fn set(dir: &Path, name: &str, uuid: &str) -> SetSpec {
    let path = dir.join(format!("{name}.bin"));
    std::fs::write(&path, vec![9u8; REGION_ALIGN as usize]).unwrap();
    let obj = ObjectRef { sha256: format!("obj-{name}"), length: REGION_ALIGN };
    let layout = Layout {
        parts: vec![Part { what: "t0".into(), region: 0, offset: 0, nbytes: REGION_ALIGN }],
        regions: vec![Region { offset: 0, nbytes: REGION_ALIGN, span: REGION_ALIGN, items: vec![Item { offset: 0, len: REGION_ALIGN, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: 0, len: REGION_ALIGN }), part_end: true }] }],
        nbytes: REGION_ALIGN,
        digest: format!("layout-{name}-{uuid}"),
        names: vec![vec!["t0".into()]],
    };
    let source = Source::files(vec![(obj, File::open(&path).unwrap())], false, 0);
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", uuid);
    SetSpec { name: format!("pkg@1/{name}"), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

/// Nothing fills beside an unmeasured call, nor past the live budget less what the running
/// calls keep free; into measured surplus it does.
#[test]
fn a_tier_copy_goes_only_into_measured_surplus() {
    assert!(varena::os::namespace(&format!("host{}-", std::process::id())));
    let dir = tempfile::tempdir().unwrap();
    let keep = Arc::new(Mutex::new(None::<u64>));
    let need: varena::Need = Arc::new({
        let keep = keep.clone();
        move || *keep.lock()
    });
    let _tier = varena::Tier::serve(varena::Pressure::Headroom, need).unwrap().expect("this binary's tier");
    let g = GpuId { uuid: format!("h{}", std::process::id()), model: "host".into(), memory: 8 << 20 };
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(g.memory))).unwrap();
    let saved = |name: &str| {
        mem.load(set(dir.path(), name, &g.uuid), Level::Gpu).unwrap();
        mem.settle();
        mem.request(name, Plan { steps: vec![] }).unwrap().end().moved.saved.bytes
    };
    assert_eq!(saved("unmeasured"), 0, "a call is unmeasured: nothing fills beside it");
    *keep.lock() = Some(u64::MAX / 2);
    assert_eq!(saved("short"), 0, "past the budget less what calls keep free");
    *keep.lock() = Some(0);
    assert_eq!(saved("room"), REGION_ALIGN, "measured surplus");
    drop((mem, ledger));
    varena::os::remove_table(&g.uuid);
}
