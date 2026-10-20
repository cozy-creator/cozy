//! Upload control shares the public uploader: retained bytes survive a refused final gate.
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Mutex,
    },
};
use tensorfs::{
    fetch::{upload_tree, upload_tree_with, Client, Endpoint, SourcePolicy, UploadControl},
    format::{Dtype, ObjectRef, Tree},
    store2::Store,
    writer::{Data, Source},
};

struct Fixture {
    root: PathBuf,
    store: Store,
    endpoint: Endpoint,
    tree: ObjectRef,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "tfs-upload-control-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let store = Store::open(root.join("source")).unwrap();
        let mut writer = store.writer(None, None).unwrap();
        let bytes = vec![27; 16384];
        writer
            .get("model")
            .put(
                "weight",
                &Dtype::U8,
                &[bytes.len() as u64],
                Data::Plain(Source::Read(&mut &bytes[..])),
            )
            .unwrap();
        let header = writer.commit().unwrap();
        let mut tree = Tree::default();
        tree.set_variant("f32", header);
        let tree = store.put_tree(&tree).unwrap();
        let client = Client::new(SourcePolicy {
            any_host: true,
            allow_local: true,
            ..Default::default()
        });
        let endpoint = Endpoint::new(
            &client,
            &format!("file://{}", root.join("endpoint").display()),
            None,
        )
        .unwrap();
        Self {
            root,
            store,
            endpoint,
            tree,
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

#[test]
fn a_refused_commit_keeps_admitted_blobs_for_the_same_public_uploader_to_retry() {
    let f = Fixture::new();
    let runtime = tensorfs::fetch::runtime().unwrap();
    let commits = AtomicUsize::new(0);
    let progress = Mutex::new(Vec::new());
    let control = UploadControl {
        cancelled: &|| false,
        progress: &|done, total| progress.lock().unwrap().push((done, total)),
        commit: &|| {
            commits.fetch_add(1, Ordering::Relaxed);
            false
        },
    };
    assert!(runtime
        .block_on(upload_tree_with(
            &f.endpoint,
            "owner/model",
            &f.tree,
            &f.store,
            None,
            &control
        ))
        .is_err());
    assert_eq!(commits.load(Ordering::Relaxed), 1);
    assert!(runtime
        .block_on(f.endpoint.repo("owner/model"))
        .unwrap()
        .is_none());
    assert!(runtime.block_on(f.endpoint.has(&f.tree)).unwrap());
    let rows = progress.lock().unwrap();
    assert_eq!(rows.first().unwrap().0, 0);
    assert_eq!(rows.last().unwrap().0, rows.last().unwrap().1);
    assert!(rows
        .windows(2)
        .all(|w| w[0].0 <= w[1].0 && w[0].1 == w[1].1));
    let retry = runtime
        .block_on(upload_tree(
            &f.endpoint,
            "owner/model",
            &f.tree,
            &f.store,
            None,
        ))
        .unwrap();
    assert_eq!(retry.sent, 0);
    assert!(
        !retry.converged,
        "deduplicated blobs do not mean the ref was already published"
    );
    assert!(retry.repo.checkpoints.contains(&f.tree));
    let again = runtime
        .block_on(upload_tree(
            &f.endpoint,
            "owner/model",
            &f.tree,
            &f.store,
            None,
        ))
        .unwrap();
    assert!(again.converged);
    assert_eq!(again.sent, 0);
}

#[test]
fn cancellation_before_the_gate_never_commits_and_cancel_after_acceptance_cannot_interrupt_it() {
    let f = Fixture::new();
    let runtime = tensorfs::fetch::runtime().unwrap();
    let canceled = AtomicBool::new(false);
    let commits = AtomicUsize::new(0);
    let control = UploadControl {
        cancelled: &|| canceled.load(Ordering::Acquire),
        progress: &|done, _| {
            if done > 0 {
                canceled.store(true, Ordering::Release);
            }
        },
        commit: &|| {
            commits.fetch_add(1, Ordering::Relaxed);
            true
        },
    };
    assert!(runtime
        .block_on(upload_tree_with(
            &f.endpoint,
            "owner/model",
            &f.tree,
            &f.store,
            None,
            &control
        ))
        .is_err());
    assert_eq!(commits.load(Ordering::Relaxed), 0);
    assert!(runtime
        .block_on(f.endpoint.repo("owner/model"))
        .unwrap()
        .is_none());
    canceled.store(false, Ordering::Release);
    let control = UploadControl {
        cancelled: &|| canceled.load(Ordering::Acquire),
        progress: &|_, _| {},
        commit: &|| {
            commits.fetch_add(1, Ordering::Relaxed);
            canceled.store(true, Ordering::Release);
            true
        },
    };
    let uploaded = runtime
        .block_on(upload_tree_with(
            &f.endpoint,
            "owner/model",
            &f.tree,
            &f.store,
            None,
            &control,
        ))
        .unwrap();
    assert!(canceled.load(Ordering::Acquire));
    assert_eq!(commits.load(Ordering::Relaxed), 1);
    assert!(uploaded.repo.checkpoints.contains(&f.tree));
}
