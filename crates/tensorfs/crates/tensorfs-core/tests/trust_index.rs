//! Concurrent writers in one process never share a temp name (v1 store).

use std::fs;
use std::path::PathBuf;

use tensorfs_core::ids::ObjectRef;
use tensorfs_core::store::{Fault, Store};

fn temporary(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "tensorfs-trust-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&root);
    root
}

fn body(index: usize) -> Vec<u8> {
    let mut bytes = vec![(index % 251) as u8; 4096 + index];
    bytes[..8].copy_from_slice(&(index as u64).to_le_bytes());
    bytes
}

/// Many writers in one process mint temp names at once; every object lands (a temp-name
/// collision fails a `create_new` with "File exists").
#[test]
fn concurrent_writers_in_one_process_never_share_a_temp_name() {
    let root = temporary("temp-names");
    let store = Store::init(&root).unwrap();
    let written: Vec<ObjectRef> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..16)
            .map(|worker| {
                let store = &store;
                scope.spawn(move || {
                    (0..16)
                        .map(|i| {
                            let bytes = body(worker * 16 + i);
                            let expect = ObjectRef::of(&bytes);
                            store
                                .put_stream(&mut bytes.as_slice(), Some(&expect), &Fault::default())
                                .unwrap();
                            expect
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().unwrap())
            .collect()
    });
    assert_eq!(written.len(), 16 * 16);
    for object in &written {
        assert_eq!(
            fs::read(store.object_path(&object.sha256)).unwrap().len() as u64,
            object.length
        );
    }
    let _ = fs::remove_dir_all(root);
}
