//! Public Writer identities remain usable for the same cache lifetime as their headers.
#![cfg(feature = "store")]

use std::fs::{self, File, FileTimes};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use tensorfs::format::{Dtype, ObjectRef};
use tensorfs::{Client, Config, Open, WriterOptions};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
                "writer-tree-cache-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&root) {
                Ok(()) => return Self(root),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("create owned test root: {error}"),
            }
        }
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn client(cache: &Path) -> Client {
    let mut config = Config::from_env();
    config.cache_dir = cache.to_path_buf();
    config.endpoint = "http://127.0.0.1:1".into();
    config.mirror = None;
    config.token = None;
    Client::new(config).unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn returned_writer_tree_survives_gc_for_the_header_cache_lifetime() {
    const WEIGHT: &[u8] = &[3, 7, 11, 19];
    const ASSET: &[u8] = b"tiny writer asset";
    let scratch = Scratch::new();
    let cache = scratch.0.join("cache");
    let producer = client(&cache);
    let mut writer = producer
        .writer(WriterOptions {
            variant: Some("u8"),
            ..WriterOptions::default()
        })
        .unwrap();
    {
        let mut model = writer
            .component("model", "test.Model", Default::default(), None)
            .unwrap();
        model.put("w", Dtype::U8, &[4], &mut &WEIGHT[..]).unwrap();
        model.asset("vocab.bin", ASSET).unwrap();
    }
    let checkpoint = writer.commit().await.unwrap();
    let header = checkpoint.id().clone();
    let tree = checkpoint
        .tree()
        .expect("Writer returned no tree identity")
        .clone();
    let payload: Vec<_> = checkpoint.header().objects().cloned().collect();
    assert_eq!(checkpoint.variant(), Some("u8"));
    assert_eq!(
        checkpoint.component("model").unwrap().read("w").unwrap(),
        WEIGHT
    );
    assert!(cache.join(tree.key()).exists());
    drop(checkpoint);
    drop(writer);
    drop(producer);

    // An ordinary raw operator runs the real collector without pressure or TTL pin removal.
    // Age actual owned files beyond the default one-hour grace, while their cache recency
    // remains fresh and well inside the seven-day library cache lifetime.
    let operator = tensorfs::store::Store::open(&cache).unwrap();
    let stale = operator
        .put_bytes(b"unrooted stale positive GC control")
        .unwrap();
    let old = SystemTime::now() - tensorfs::store::GRACE - Duration::from_secs(60);
    for object in [&header, &tree, &stale].into_iter().chain(&payload) {
        let path = cache.join(object.key());
        File::open(&path)
            .unwrap()
            .set_times(FileTimes::new().set_accessed(old).set_modified(old))
            .unwrap();
        assert!(
            fs::metadata(path).unwrap().modified().unwrap()
                < SystemTime::now() - tensorfs::store::GRACE
        );
    }
    operator.gc().unwrap();
    assert!(
        !cache.join(stale.key()).exists(),
        "positive GC control did not evict"
    );
    assert_eq!(
        ObjectRef::of(&fs::read(cache.join(header.key())).unwrap()),
        header,
        "fresh cache header did not survive ordinary GC"
    );
    for object in &payload {
        assert_eq!(
            &ObjectRef::of(&fs::read(cache.join(object.key())).unwrap()),
            object
        );
    }
    assert!(
        cache.join(tree.key()).exists(),
        "returned Writer tree expired at GRACE while its header cache pin is still fresh"
    );
    drop(operator);

    let reopened = client(&cache)
        .open(
            &tree.id(),
            &Open {
                local_files_only: true,
                ..Open::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reopened.id(), &header);
    assert_eq!(reopened.tree(), Some(&tree));
    assert_eq!(reopened.variant(), Some("u8"));
    assert_eq!(
        reopened.component("model").unwrap().read("w").unwrap(),
        WEIGHT
    );
    assert_eq!(
        reopened
            .component("model")
            .unwrap()
            .asset("vocab.bin")
            .unwrap(),
        ASSET
    );
}
