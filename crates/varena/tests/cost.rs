//! What varena itself costs per page, over host pages whose copies are small (each region
//! carries 4 KiB from the store; the rest is zeroed): a cold load of many pages, and binds that
//! each evict.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use tensorfs::format::ObjectRef;
use tensorfs::io::source::Source;
use tensorfs::io::{ItemSource, ObjectRange};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Fill, Fingerprint, GpuId, HostBackend, Level, Memory, Plan, SetSpec, Step, Torch};

const CARRIED: u64 = 4096;

fn set(dir: &Path, name: &str, pages: u64, uuid: &str) -> SetSpec {
    let path = dir.join(format!("{name}.bin"));
    std::fs::write(&path, vec![5u8; (pages * CARRIED) as usize]).unwrap();
    let obj = ObjectRef { sha256: format!("obj-{name}"), length: pages * CARRIED };
    let layout = Layout {
        parts: (0..pages).map(|r| Part { what: format!("t{r}"), region: r as u32, offset: r * REGION_ALIGN, nbytes: CARRIED }).collect(),
        regions: (0..pages)
            .map(|r| Region { offset: r * REGION_ALIGN, nbytes: CARRIED, span: REGION_ALIGN, items: vec![Item { offset: r * REGION_ALIGN, len: CARRIED, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: r * CARRIED, len: CARRIED }), part_end: true }] })
            .collect(),
        nbytes: pages * REGION_ALIGN,
        digest: format!("layout-{name}-{uuid}"),
        names: (0..pages).map(|r| vec![format!("t{r}")]).collect(),
    };
    let source = Source::files(vec![(obj, File::open(&path).unwrap())], false, 0);
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", uuid);
    SetSpec { name: format!("pkg@1/{name}"), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

/// Per page: a cold load of 1,000 pages, and 64 binds that each evict one.
#[test]
#[ignore = "a measurement: run alone, with --nocapture"]
fn per_page_cost() {
    assert!(varena::os::namespace(&format!("cost{}-", std::process::id())));
    let dir = tempfile::tempdir().unwrap();
    let pages = 1000;
    let g = GpuId { uuid: format!("c{}-load", std::process::id()), model: "host".into(), memory: pages * REGION_ALIGN };
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(g.memory))).unwrap();
    let spec = set(dir.path(), "load", pages, &g.uuid);
    let start = Instant::now();
    mem.load(spec, Level::Gpu).unwrap();
    let load = start.elapsed();
    println!("load: {pages} pages in {load:?}: {:.0} us/page", load.as_secs_f64() * 1e6 / pages as f64);
    drop((mem, ledger));
    varena::os::remove_table(&g.uuid);

    let binds = 64;
    let g = GpuId { uuid: format!("c{}-evict", std::process::id()), model: "host".into(), memory: 4 * REGION_ALIGN };
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(g.memory))).unwrap();
    let id = mem.load(set(dir.path(), "evict", binds, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["pkg@1/evict".into()], passes: 1 }] }).unwrap();
    let pass = r.pass(id, "fwd", "1x1", Torch::default()).unwrap();
    let start = Instant::now();
    for b in 0..binds as u32 {
        drop(mem.bind(id, b, 0).unwrap());
    }
    let evicting = start.elapsed();
    println!("evicting binds: {binds} in {evicting:?}: {:.0} us/bind", evicting.as_secs_f64() * 1e6 / binds as f64);
    drop(pass);
    drop((r, mem, ledger));
    varena::os::remove_table(&g.uuid);
}
