//! Fill throughput from a real file store, against the disk's ceiling measured on the same file
//! the same way (TensorFS's engine reading it whole, past the page cache).

use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use tensorfs::format::ObjectRef;
use tensorfs::io::source::Source;
use tensorfs::io::{Copy, Dst, ItemSource, ObjectRange, ReadOpts, Src};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Fill, Fingerprint, GpuId, HostBackend, Level, Memory, SetSpec};

const PAGES: u64 = 50;
/// 8 MB pages, 400 MB a set: the disk, not a call's setup, should set the time.
const PAGE: u64 = 4 * REGION_ALIGN;
const RUNS: usize = 3;

/// Out of the page cache, so the next read is the disk's.
fn uncached(path: &Path) {
    let f = File::open(path).unwrap();
    f.sync_all().unwrap();
    // SAFETY: a valid descriptor; advice only.
    assert_eq!(unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) }, 0);
    assert_eq!(platform::durable::cached(&f, 0, PAGES * PAGE), Some(false), "the file left the page cache");
}

/// The engine reading the file whole into host memory: GB/s.
fn ceiling(path: &Path) -> f64 {
    uncached(path);
    let f = File::open(path).unwrap();
    let mut buf = vec![0u8; (PAGES * PAGE) as usize];
    let start = Instant::now();
    // SAFETY: `buf` is ours for the call.
    unsafe { tensorfs::io::read(&[Copy { src: Src::File(&f, 0), dst: Dst::host(&mut buf) }], &ReadOpts::default()) }.unwrap();
    (PAGES * PAGE) as f64 / start.elapsed().as_secs_f64() / 1e9
}

/// A set of `PAGES` pages over the file; `run` makes its layout new to the RAM tier.
fn set(path: &Path, run: usize, uuid: &str) -> SetSpec {
    let length = PAGES * PAGE;
    let obj = ObjectRef { sha256: "obj-fill".into(), length };
    let layout = Layout {
        parts: (0..PAGES).map(|r| Part { what: format!("t{r}"), region: r as u32, offset: r * PAGE, nbytes: PAGE }).collect(),
        regions: (0..PAGES)
            .map(|r| Region { offset: r * PAGE, nbytes: PAGE, span: PAGE, items: vec![Item { offset: r * PAGE, len: PAGE, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: r * PAGE, len: PAGE }), part_end: true }] })
            .collect(),
        nbytes: length,
        digest: format!("layout-fill-{run}-{uuid}"),
        names: (0..PAGES).map(|r| vec![format!("t{r}")]).collect(),
    };
    let source = Source::files(vec![(obj, File::open(path).unwrap())], true, 0);
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", uuid);
    SetSpec { name: format!("pkg@1/fill{run}"), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

/// A cold load of the set to the device (host pages here): GB/s from the call to every page
/// filled.
fn load(path: &Path, run: usize) -> f64 {
    let g = GpuId { uuid: format!("f{}-{run}", std::process::id()), model: "host".into(), memory: 2 * PAGES * PAGE };
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(g.memory))).unwrap();
    let spec = set(path, run, &g.uuid);
    uncached(path);
    let start = Instant::now();
    mem.load(spec, Level::Gpu).unwrap();
    let rate = (PAGES * PAGE) as f64 / start.elapsed().as_secs_f64() / 1e9;
    drop(mem);
    varena::os::remove_table(&g.uuid);
    rate
}

/// The owner's bar: a cold load of 50 pages runs at ≥ 70% of the disk's measured ceiling.
#[test]
fn a_cold_load_fills_near_the_disk_ceiling() {
    assert!(varena::os::namespace(&format!("fill{}-", std::process::id())));
    let dir = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap();
    let path = dir.path().join("set.bin");
    let page: Vec<u8> = (0..PAGE).map(|i| (i % 251) as u8).collect();
    std::fs::write(&path, page.repeat(PAGES as usize)).unwrap();
    // Alternating, best of each: other work on the disk slows either alike.
    let (mut disk, mut fill) = (0f64, 0f64);
    for run in 0..RUNS {
        disk = disk.max(ceiling(&path));
        fill = fill.max(load(&path, run));
    }
    let share = fill / disk;
    println!("fill {fill:.2} GB/s of a {disk:.2} GB/s disk ceiling: {:.0}%", share * 100.0);
    assert!(share >= 0.7, "fills at {:.0}% of the disk ceiling", share * 100.0);
}
