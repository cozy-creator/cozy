//! Reused destinations expose only their selected payload and deterministic zero padding.
#![cfg(unix)]
use tensorfs::io::{
    source::{IoTally, Item, ItemSource, Mode, Source},
    Dst,
};

#[cfg(target_os = "linux")]
#[test]
fn exports_only_selected_objects_as_owned_read_only_files() {
    use std::io::{Read, Seek, Write};
    use std::os::fd::{AsFd, AsRawFd};
    use tensorfs::{format::ObjectRef, io::ObjectRange};
    let object = ObjectRef::of(b"verified data");
    let other = ObjectRef::of(b"unselected");
    let mut file = std::fs::File::from(platform::shm::memfd_named("source-export").unwrap());
    file.write_all(b"verified data").unwrap();
    let source = Source::files(
        vec![
            (object.clone(), file),
            (
                other,
                platform::shm::frozen("not-exported", b"unselected").unwrap(),
            ),
        ],
        false,
        0,
    );
    let item = Item {
        offset: 0,
        len: 4,
        part_end: true,
        source: ItemSource::Object(ObjectRange {
            obj: object.clone(),
            off: 3,
            len: 4,
        }),
    };
    let inline = Item {
        offset: 8,
        len: 1,
        part_end: true,
        source: ItemSource::Inline(vec![7].into()),
    };
    assert!(source
        .export_files(std::slice::from_ref(&inline))
        .unwrap()
        .is_empty());
    let mut exported = source.export_files(&[item.clone(), item, inline]).unwrap();
    assert_eq!(exported.len(), 1);
    let (exported_object, mut file) = exported.pop().unwrap();
    assert_eq!(exported_object, object);
    assert_eq!(
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) } & libc::O_ACCMODE,
        libc::O_RDONLY
    );
    let competing = std::fs::File::from(platform::shm::reopen(file.as_fd(), true).unwrap());
    drop(source);
    assert!(
        competing.try_lock().is_err(),
        "export retains the read lock after its source drops"
    );
    file.rewind().unwrap();
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();
    assert_eq!(
        bytes, b"verified data",
        "exports the complete verified object"
    );
    assert!(file.write_all(b"bad").is_err());
    drop(file);
    competing.try_lock().unwrap();
}

#[test]
fn export_refuses_missing_files_and_invalid_selected_ranges() {
    use tensorfs::{format::ObjectRef, io::ObjectRange};
    let source = Source::files(Vec::new(), false, 0);
    let mut item = Item {
        offset: 0,
        len: 4,
        part_end: true,
        source: ItemSource::Object(ObjectRange {
            obj: ObjectRef::of(b"data"),
            off: 1,
            len: 4,
        }),
    };
    assert!(source
        .export_files(&[item.clone()])
        .unwrap_err()
        .to_string()
        .contains("range"));
    item.source = ItemSource::Object(ObjectRange {
        obj: ObjectRef::of(b"data"),
        off: 0,
        len: 4,
    });
    assert!(source
        .export_files(&[item])
        .unwrap_err()
        .to_string()
        .contains("no verified file"));
}

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
