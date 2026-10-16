//! A read lease needs one descriptor per object. A low soft RLIMIT_NOFILE is raised toward
//! the hard limit rather than refused; only a hard limit too small refuses.
use std::path::PathBuf;

use rustix::process::{getrlimit, setrlimit, Resource, Rlimit};
use tensorfs_core::err::Code;
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::manifest::Manifest;
use tensorfs_core::meta::Meta;
use tensorfs_core::read;
use tensorfs_core::store::{Fault, Store};

const OBJECTS: usize = 300;

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-fd-headroom-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}

#[test]
fn a_low_soft_limit_is_raised_and_only_the_hard_limit_refuses() {
    let root = temporary("lease");
    let store = Store::init(&root).unwrap();
    let mut files = Vec::with_capacity(OBJECTS);
    for index in 0..OBJECTS {
        let body = format!("object {index}").into_bytes();
        let object = ObjectRef::of(&body);
        store
            .put_stream(&mut body.as_slice(), Some(&object), &Fault::default())
            .unwrap();
        files.push((format!("f/{index:04}"), object));
    }
    let manifest = Manifest::from_files(files).unwrap();
    store.put_manifest(&manifest).unwrap();
    let meta = Meta::open(&store).unwrap();

    let original = getrlimit(Resource::Nofile);
    let hard = original.maximum.unwrap_or(u64::MAX);
    assert!(
        hard >= 1024,
        "this proof needs a hard limit of at least 1024"
    );
    setrlimit(
        Resource::Nofile,
        Rlimit {
            current: Some(128),
            maximum: original.maximum,
        },
    )
    .unwrap();
    let (lease, _) = read::acquire_manifest(&store, &meta, &manifest).unwrap();
    assert!(getrlimit(Resource::Nofile).current.unwrap() > OBJECTS as u64);
    lease.release(&meta).unwrap();

    // A hard limit below the set is a real shortage.
    setrlimit(
        Resource::Nofile,
        Rlimit {
            current: Some(128),
            maximum: Some(128),
        },
    )
    .unwrap();
    assert_eq!(
        read::acquire_manifest(&store, &meta, &manifest)
            .err()
            .unwrap()
            .code,
        Code::FD_HEADROOM
    );
    let _ = std::fs::remove_dir_all(root);
}
