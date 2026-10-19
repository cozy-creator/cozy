//! `weights_writer` requests over the derive channel, on a real store.
use super::*;
use crate::frame::{Channel, Frame as Wire};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};

struct Fixture {
    root: PathBuf,
    owner: Weights,
    grant: Grant,
    current: Arc<AtomicBool>,
    output: Frame,
}

impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("cozy-weights-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let store = crate::storage::open(&root.join("store")).unwrap();
        let current = Arc::new(AtomicBool::new(true));
        let state = current.clone();
        let grant = Grant::new(
            "owner",
            "run",
            &root,
            BTreeMap::new(),
            [("model".into(), 8)].into(),
            None,
            GrantAuthority {
                current: Arc::new(move || state.load(Ordering::SeqCst)),
                bind_output: Arc::new(|_, _| Ok(())),
                fetch: None,
                replay: None,
            },
        );
        let derivation = serde_json::to_vec(&json!({"base": null, "sources": [], "max_new_bytes": 8})).unwrap();
        std::fs::write(root.join("weights-model-derivation.canonical"), &derivation).unwrap();
        let output = Frame {
            operation: "output".into(),
            output_slot: "model".into(),
            length: derivation.len() as u64,
            ..Default::default()
        };
        Self { owner: Weights::new(store).unwrap(), root, grant, current, output }
    }

    /// The output's channel and its transaction.
    fn open(&self) -> (String, Channel<platform::ipc::LocalStream>) {
        let (answer, channel, _) = self.owner.answer(&self.grant, &self.output);
        assert!(answer.ok, "{answer:?}");
        #[cfg(unix)]
        let raw = std::os::fd::IntoRawFd::into_raw_fd(channel.unwrap()) as i64;
        #[cfg(windows)]
        let raw = std::os::windows::io::IntoRawHandle::into_raw_handle(channel.unwrap()) as i64;
        // SAFETY: the connected stream the answer handed over, now this test's.
        let stream = unsafe { platform::ipc::adopt(raw) };
        (answer.transaction, Channel(stream))
    }

    fn adopt(&self, transaction: &str, manifest: &Value) -> (Answer, Option<Adopted>) {
        let relayed = serde_json::to_vec(&json!({"transaction_id": transaction, "manifest": manifest})).unwrap();
        std::fs::write(self.root.join("weights-model-native-receipt.canonical"), &relayed).unwrap();
        let frame = Frame {
            operation: "adopt".into(),
            output_slot: "model".into(),
            transaction: transaction.into(),
            length: relayed.len() as u64,
            ..Default::default()
        };
        let (answer, _, adopted) = self.owner.answer(&self.grant, &frame);
        (answer, adopted)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn ask(channel: &mut Channel<platform::ipc::LocalStream>, request: Value) -> Value {
    channel.send(&request).unwrap();
    match channel.receive().unwrap() {
        Some(Wire::Json(answer)) => answer,
        other => panic!("{other:?}"),
    }
}

fn put(channel: &mut Channel<platform::ipc::LocalStream>) {
    channel.send(&json!({"op": "put", "component": "model", "key": "weight", "dtype": "F32", "shape": [2], "length": 8})).unwrap();
    channel.data(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    assert_eq!(channel.receive().unwrap(), Some(Wire::Json(json!({"result": null}))));
}

#[test]
fn an_output_resumes_commits_and_is_adopted_by_the_stores_record() {
    let f = Fixture::new();
    let (transaction, mut first) = f.open();
    put(&mut first);
    drop(first);
    // The same derivation reopens the same transaction, which kept what it journaled.
    let (again, mut second) = f.open();
    assert_eq!(again, transaction);
    assert_eq!(ask(&mut second, json!({"op": "has", "component": "model", "key": "weight"})), json!({"result": true}));
    let manifest = ask(&mut second, json!({"op": "commit"}))["result"]["manifest"].clone();
    // A receipt that names other bytes is refused; the store's record decides.
    let (refused, _) = f.adopt(&transaction, &json!({"sha256": "0".repeat(64), "length": 1}));
    assert_eq!(refused.code, "weights_receipt_mismatch");
    let (answer, adopted) = f.adopt(&transaction, &manifest);
    assert!(answer.ok, "{answer:?}");
    let adopted = adopted.unwrap();
    assert_eq!(adopted.manifest.sha256, manifest["sha256"].as_str().unwrap());
    let checkpoint = f.owner.store.checkpoint(&adopted.manifest).unwrap();
    assert_eq!(checkpoint.read("model", "weight").unwrap(), [1, 2, 3, 4, 5, 6, 7, 8]);
    // Cancel relinquishes only what is unadopted: the local model stays through GC.
    f.owner.cancel_output(&transaction).unwrap();
    f.owner.store.as_ref().clone().with_grace(std::time::Duration::ZERO).gc().unwrap();
    let name = format!("output-{}", &sha256::hex_digest(b"owner\0run\0model")[..40]);
    assert_eq!(storage::local(&f.owner.store, &name).unwrap(), Some(adopted.manifest));
}

#[test]
fn an_unadopted_output_goes_with_its_cancel_and_a_stale_attempt_is_refused() {
    let f = Fixture::new();
    let (transaction, mut channel) = f.open();
    put(&mut channel);
    let manifest = ask(&mut channel, json!({"op": "commit"}))["result"]["manifest"].clone();
    f.current.store(false, Ordering::SeqCst);
    let (stale, _) = f.adopt(&transaction, &manifest);
    assert_eq!(stale.code, "weights_writer_closed");
    let (answer, _, _) = f.owner.answer(&f.grant, &f.output);
    assert_eq!(answer.code, "weights_writer_closed");
    f.owner.cancel_output(&transaction).unwrap();
    assert_eq!(derive::committed(&f.owner.store, &transaction).unwrap(), None);
}

#[test]
fn a_source_or_output_outside_the_grant_is_refused() {
    let f = Fixture::new();
    let source = Frame { operation: "source".into(), manifest: format!("sha256:{}", "a".repeat(64)), ..Default::default() };
    assert_eq!(f.owner.answer(&f.grant, &source).0.code, "weights_source_ungranted");
    let other = Frame { output_slot: "other".into(), ..f.output.clone() };
    assert_eq!(f.owner.answer(&f.grant, &other).0.code, "weights_output_ungranted");
    let over = serde_json::to_vec(&json!({"base": null, "max_new_bytes": 9})).unwrap();
    std::fs::write(f.root.join("weights-model-derivation.canonical"), &over).unwrap();
    let bigger = Frame { length: over.len() as u64, ..f.output.clone() };
    assert_eq!(f.owner.answer(&f.grant, &bigger).0.code, "weights_writer_ungranted");
}
