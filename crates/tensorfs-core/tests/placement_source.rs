//! Reused destinations expose only their selected payload and deterministic zero padding.
#![cfg(unix)]
use tensorfs_core::io::{
    source::{IoTally, Item, ItemSource, Mode, Source},
    Dst,
};

#[test]
fn reusing_a_destination_zeroes_gaps_and_preserves_sentinels() {
    let source = Source::files(Vec::new(), false, 0);
    let tally = IoTally::default();
    let mut bytes = vec![0xa5; 4099];
    let dst = Dst::Host {
        ptr: bytes[1..].as_mut_ptr(),
        len: 4097,
    };
    let first = [Item {
        offset: 0,
        len: 4097,
        source: ItemSource::Inline(vec![7; 4097].into()),
        part_end: true,
    }];
    // SAFETY: the disjoint selected destination is ours for this call.
    unsafe { source.read_items(&first, 0, dst, 2, Mode::Cache, &tally) }.unwrap();
    let next = [
        Item {
            offset: 0,
            len: 1,
            source: ItemSource::Inline(vec![1].into()),
            part_end: true,
        },
        Item {
            offset: 4096,
            len: 1,
            source: ItemSource::Inline(vec![2].into()),
            part_end: true,
        },
    ];
    // SAFETY: the same destination remains exclusively borrowed here.
    let stats = unsafe { source.read_items(&next, 0, dst, 2, Mode::Cache, &tally) }.unwrap();
    assert_eq!((bytes[0], bytes[4098]), (0xa5, 0xa5));
    assert_eq!((bytes[1], bytes[4097]), (1, 2));
    assert!(bytes[2..4097].iter().all(|byte| *byte == 0));
    assert_eq!(
        stats.bytes, 2,
        "padding is not counted as payload throughput"
    );
}

#[test]
fn overlapping_items_are_refused_before_any_destination_write() {
    let source = Source::files(Vec::new(), false, 0);
    let mut bytes = [0xa5; 4];
    let items = [
        Item {
            offset: 0,
            len: 4,
            source: ItemSource::Inline(vec![1; 4].into()),
            part_end: true,
        },
        Item {
            offset: 3,
            len: 1,
            source: ItemSource::Inline(vec![2].into()),
            part_end: true,
        },
    ];
    // SAFETY: the destination is exclusively ours; invalid item overlap is refused before I/O.
    let error = unsafe {
        source.read_items(
            &items,
            0,
            Dst::host(&mut bytes),
            2,
            Mode::Cache,
            &IoTally::default(),
        )
    }
    .unwrap_err();
    assert!(error.to_string().contains("overlap"));
    assert_eq!(bytes, [0xa5; 4]);
}
