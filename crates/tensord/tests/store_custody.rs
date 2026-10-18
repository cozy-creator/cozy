//! Write objects keep their pins while an unfinished run names them; the store's
//! GC takes them only once released.
use tensord::{
    execution::Engine,
    journal::{InputFile, Invocation, State},
    objects::Objects,
};
use std::{fs, path::PathBuf, sync::Arc, time::Duration};
use tensord::storage::{self, Store};
use tensorfs_core::format::ObjectRef;

struct Area {
    root: PathBuf,
    store: Arc<Store>,
    engine: Arc<Engine>,
    objects: Objects,
}
impl Area {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("cm-custody-{}", uuid::Uuid::new_v4()));
        let store = Arc::new(storage::open(&root.join("store")).unwrap().as_ref().clone().with_grace(Duration::ZERO));
        let engine = Engine::open(&root.join("state")).unwrap();
        let objects = Objects::new(&root.join("writes"), store.clone(), engine.clone()).unwrap();
        Self {
            root,
            store,
            engine,
            objects,
        }
    }
    fn write(&self, bytes: &[u8]) -> ObjectRef {
        let object = ObjectRef::of(bytes);
        if let Some(upload) = self.objects.upload("alice", &object.id(), object.length).unwrap() {
            upload.write_at(0, bytes).unwrap();
            self.objects.settle(&upload).unwrap();
        }
        object
    }
    fn present(&self, object: &ObjectRef) -> bool {
        storage::present(&self.store, object)
    }
    /// A release of every pin nothing needs now, then the store's GC.
    fn collect(&self) -> usize {
        std::thread::sleep(Duration::from_millis(5));
        let released = self.objects.release(Duration::ZERO).unwrap();
        self.store.gc().unwrap();
        released
    }
}
impl Drop for Area {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
fn invocation(object: &ObjectRef) -> Invocation {
    Invocation {
        package: "audit/package".into(),
        input: serde_json::json!({}),
        inputs: vec![InputFile {
            input_id: "file".into(),
            digest: object.id(),
            length: object.length,
            media_type: "application/octet-stream".into(),
            order: 0,
        }],
        ..Default::default()
    }
}

#[test]
fn an_acknowledged_write_survives_gc_until_its_run_ends() {
    let area = Area::new();
    let object = area.write(b"accepted run input");
    // Pinned before the journal row: the store's GC alone never takes it.
    area.store.gc().unwrap();
    assert!(area.present(&object));
    let run = area.engine.submit("queued", invocation(&object)).unwrap();
    assert_eq!(run.state, State::Queued);
    assert_eq!(area.collect(), 0);
    area.engine.cancel(&run.id, "alice").unwrap();
    assert_eq!(area.collect(), 1);
    assert!(!area.present(&object));
}

#[test]
fn unknown_and_adopted_dependencies_stay_until_their_run_ends() {
    let area = Area::new();
    let input = area.write(b"unknown-state dependency");
    let unknown = area.engine.submit("unknown", invocation(&input)).unwrap();
    // A state a newer machine wrote is unfinished to this one.
    let db = rusqlite::Connection::open(area.root.join("state/executions.sqlite3")).unwrap();
    db.execute(
        "UPDATE executions SET state='future-state' WHERE id=?1",
        [&unknown.id],
    )
    .unwrap();
    assert_eq!(area.collect(), 0);
    assert!(area.present(&input));

    let parent = area
        .engine
        .submit(
            "parent",
            Invocation {
                package: "audit/caller".into(),
                input: serde_json::json!({}),
                ..Default::default()
            },
        )
        .unwrap();
    let output = b"child result its parent holds";
    let adopted = ObjectRef::of(output);
    let file = area.root.join("child-output");
    fs::write(&file, output).unwrap();
    area.objects
        .adopt("alice", &parent.id, &file, &adopted)
        .unwrap();
    assert_eq!(area.collect(), 0);
    assert!(area.present(&adopted));
    area.engine.cancel(&parent.id, "alice").unwrap();
    assert_eq!(area.collect(), 1);
    assert!(!area.present(&adopted) && area.present(&input));
}

#[test]
fn unsubmitted_writes_and_crash_orphans_age_out() {
    let area = Area::new();
    let unsubmitted = area.write(b"write never submitted");
    // Within the TTL an acknowledged Write waits for its run.
    assert_eq!(area.objects.release(tensord::reclaim::TTL).unwrap(), 0);
    // A pin written just before a crash, with no journal row.
    let orphan = area.store.put_bytes(b"death between pin and journal").unwrap();
    storage::hold(&area.store, &storage::pin::object(&orphan.sha256), vec![orphan.clone()], None).unwrap();
    assert_eq!(
        area.objects.release(tensord::reclaim::IDLE).unwrap(),
        0
    );
    assert_eq!(area.collect(), 2);
    assert!(!area.present(&unsubmitted) && !area.present(&orphan));
}
