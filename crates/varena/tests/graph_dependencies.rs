use std::sync::Arc;

use tensorfs::io::source::Source;
use varena::layout::{Item, ItemSource, Layout, Region};
use varena::{Fill, Fingerprint, GpuId, HostBackend, Level, Memory, PoolId, SetId, SetSpec};

const BLOCK: u64 = 4 << 20;

fn model(name: &str, blocks: u64, capacity: u64) -> (Memory, SetId) {
    let memory = Memory::open(
        GpuId { uuid: name.into(), model: "host".into(), memory: capacity },
        Box::new(HostBackend::new(capacity)),
    ).unwrap();
    memory.serve_standalone_torch().unwrap();
    let layout = Arc::new(Layout {
        parts: vec![],
        names: (0..blocks).map(|b| vec![format!("weight{b}")]).collect(),
        nbytes: blocks * BLOCK,
        digest: name.into(),
        regions: (0..blocks).map(|b| Region {
            offset: b * BLOCK, nbytes: 16, span: BLOCK,
            items: vec![Item { offset: b * BLOCK, len: 16, source: ItemSource::Inline(vec![68; 16].into()), part_end: true }],
        }).collect(),
    });
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, name, "0/1", name);
    let set = memory.load(SetSpec {
        name: name.into(), layout, source: Arc::new(Source::files(vec![], false, 0)), fingerprint,
    }, Level::Ram).unwrap();
    (memory, set)
}

#[test]
fn a_block_graph_fits_when_the_whole_model_does_not() {
    let (memory, set) = model("graph-subset", 3, 2 * BLOCK);
    let graph = memory.graph_region_blocks(&[(set, vec![1, 1]), (set, vec![1])], PoolId(1, 1)).unwrap();
    let records = memory.region_records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].bytes, BLOCK, "duplicate declarations pin one block once");
    assert_eq!(memory.classes()[6], BLOCK, "only the selected block is pinned");
    assert!(graph.acquire());
    graph.release(0).unwrap();
    graph.close();
    memory.settle();
    assert!(memory.graph_region(&[set], &[], PoolId(1, 2)).is_err());
}

#[test]
fn the_legacy_sets_and_retained_lists_do_not_duplicate_pins() {
    let (memory, set) = model("graph-dedup", 2, 4 * BLOCK);
    let graph = memory.graph_region(&[set, set], &[set], PoolId(2, 1)).unwrap();
    assert_eq!(memory.region_records()[0].bytes, 2 * BLOCK);
    graph.close();
}
