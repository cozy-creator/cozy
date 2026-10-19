use super::*;
use cozy_platform::ipc::LocalStream;
use std::time::Duration;
use tensorfs_core::format::Value as Meta;

const TX: &str = "sha256:1111111111111111111111111111111111111111111111111111111111111111";

fn store() -> (Store, std::path::PathBuf) {
    let root = std::env::temp_dir().join(format!("derive-{}", uuid::Uuid::new_v4()));
    (Store::open(&root).unwrap().with_grace(Duration::ZERO), root)
}

pub(crate) fn f32s(n: usize, seed: f32) -> Vec<u8> {
    (0..n).flat_map(|i| (seed + i as f32 * 0.25).to_le_bytes()).collect()
}

/// A real checkpoint through the writer: one component of F32 tensors and its config.
pub(crate) fn checkpoint(store: &Store, key: &str, component: &str, tensors: &[(&str, Vec<u64>, Vec<u8>)]) -> ObjectRef {
    let mut writer = store.writer(None, Some(key)).unwrap();
    for (name, shape, bytes) in tensors {
        writer.put(component, name, &Dtype::F32, shape, Data::Plain(Source::Read(&mut bytes.as_slice()))).unwrap();
    }
    writer.file(&format!("{component}/config.json"), b"{\"a\":1}").unwrap();
    writer.set_metadata("normalized", Meta::Bool(true));
    writer.commit().unwrap()
}

/// The objects a tensor's bytes live in.
fn objects(checkpoint: &Checkpoint, component: &str, key: &str) -> Vec<ObjectRef> {
    chunks(checkpoint, component, key).unwrap().into_iter().cloned().collect()
}

struct Client(Channel<LocalStream>);

impl Client {
    fn ask(&mut self, request: Value) -> Value {
        self.0.send(&request).unwrap();
        self.answer()
    }

    fn answer(&mut self) -> Value {
        match self.0.receive().unwrap() {
            Some(Frame::Json(answer)) => answer,
            other => panic!("expected an answer, got {other:?}"),
        }
    }

    fn bytes(&mut self, mut n: u64) -> Vec<u8> {
        let mut out = vec![];
        while n > 0 {
            let Some(Frame::Bytes(bytes)) = self.0.receive().unwrap() else { panic!("expected bytes") };
            n -= bytes.len() as u64;
            out.extend(bytes);
        }
        out
    }

    fn read(&mut self, request: Value) -> Vec<u8> {
        let answer = self.ask(request);
        let length = answer["result"]["length"].as_u64().unwrap_or_else(|| panic!("{answer}"));
        self.bytes(length)
    }
}

fn serve(store: &Store, transaction: &str, derivation: &Derivation, client: impl FnOnce(&mut Client)) -> io::Result<Option<ObjectRef>> {
    let (near, far) = cozy_platform::ipc::pair().unwrap();
    let (current, ensure) = (|| true, |_: &[ObjectRef]| Ok(()));
    let hooks = Hooks { current: &current, ensure: &ensure };
    std::thread::scope(|scope| {
        let served = scope.spawn(|| serve_output(store, transaction, derivation, far, &hooks));
        client(&mut Client(Channel(near)));
        served.join().unwrap()
    })
}

fn derivation(sources: Vec<ObjectRef>, max_new_bytes: u64) -> Derivation {
    Derivation { base: None, sources, order: vec![], max_new_bytes }
}

/// A derive channel writes plain and encoded tensors, renamed copies and files, answers reads
/// in the middle of a write, and commits a checkpoint that reads back bit-identical, its copied
/// large part on the source's own objects.
#[test]
fn a_derive_channel_writes_a_checkpoint_that_reads_back() {
    let (store, root) = store();
    let large = f32s(1024 * 4096, 1.0);
    let bias = f32s(64, -3.0);
    let source = checkpoint(&store, "source", "unet", &[("a.weight", vec![1024, 4096], large.clone()), ("b.bias", vec![64], bias.clone())]);
    let from = source.id();
    let x = f32s(256, 7.0);
    let (weight, scale) = ((0..16u8).collect::<Vec<_>>(), 0.5f32.to_le_bytes());
    let committed_at = serve(&store, TX, &derivation(vec![source.clone()], 1 << 20), |c| {
        assert_eq!(c.ask(json!({"op": "has", "component": "unet", "key": "x"}))["result"], false);
        c.0.send(&json!({"op": "put", "component": "unet", "key": "x", "dtype": "F32", "shape": [256], "length": 1024})).unwrap();
        c.0.data(&x[..500]).unwrap();
        let read = c.read(json!({"op": "read", "from": from, "component": "unet", "key": "b.bias", "offset": 4, "length": 8}));
        assert_eq!(read, bias[4..12]);
        c.0.data(&x[500..]).unwrap();
        assert_eq!(c.answer()["result"], Value::Null);
        let put = json!({"op": "put", "component": "unet", "key": "q", "dtype": "F32", "shape": [16],
            "encoding": {"scheme": "fp8.tensorwise", "params": {"scale_dtype": "F32"}},
            "parts": [{"role": "weight", "dtype": "F8_E4M3", "shape": [16], "length": 16}, {"role": "scale", "dtype": "F32", "shape": [], "length": 4}]});
        c.0.send(&put).unwrap();
        c.0.data(&weight).unwrap();
        c.0.data(&scale).unwrap();
        assert_eq!(c.answer()["result"], Value::Null);
        let copy = json!({"op": "copy", "component": "unet", "key": "a.base_layer.weight", "from": from, "from_key": "a.weight"});
        assert_eq!(c.ask(copy)["result"], Value::Null);
        c.0.send(&json!({"op": "file", "path": "unet/config.json", "length": 2})).unwrap();
        c.0.data(b"{}").unwrap();
        assert_eq!(c.answer()["result"], Value::Null);
        let ungranted = c.ask(json!({"op": "read", "from": TX, "component": "unet", "key": "b.bias"}));
        assert_eq!(ungranted["error"], "ungranted", "a refused read keeps the channel");
        let part = c.read(json!({"op": "read", "from": from, "component": "unet", "key": "a.weight", "offset": 1 << 20, "length": 3 << 20}));
        assert_eq!(part, large[1 << 20..4 << 20]);
        let manifest = c.ask(json!({"op": "commit"}))["result"]["manifest"].clone();
        assert_eq!(manifest["sha256"].as_str().unwrap().len(), 64);
    });
    let manifest = committed_at.unwrap().unwrap();
    let view = store.checkpoint(&manifest).unwrap();
    assert_eq!(view.read("unet", "x").unwrap(), x);
    assert_eq!(view.read("unet", "a.base_layer.weight").unwrap(), large);
    assert_eq!(view.file("unet/config.json").unwrap(), b"{}");
    assert!(view.tensor("unet", "a.weight").is_err(), "only what was written or copied");
    let Body::Encoded { encoding, parts } = &view.tensor("unet", "q").unwrap().body else { panic!("q is encoded") };
    assert_eq!(encoding.scheme, "fp8.tensorwise");
    let mut got = vec![0; 16];
    view.bytes_into(&parts.iter().find(|p| p.role == "weight").unwrap().at(), &mut got).unwrap();
    assert_eq!(got, weight);
    let held = store.checkpoint(&source).unwrap();
    assert_eq!(objects(&view, "unet", "a.base_layer.weight"), objects(&held, "unet", "a.weight"), "a large copy keeps its objects");

    assert_eq!(committed(&store, TX).unwrap(), Some(manifest.clone()));
    assert_eq!(adopt(&store, TX, "output-x").unwrap(), manifest);
    assert_eq!(crate::storage::local(&store, "output-x").unwrap(), Some(manifest.clone()));
    cancel(&store, TX).unwrap();
    assert_eq!(committed(&store, TX).unwrap(), None);
    store.gc().unwrap();
    assert!(store.checkpoint(&manifest).unwrap().read("unet", "x").is_ok(), "the adopted ref keeps it");
    let _ = std::fs::remove_dir_all(root);
}

/// A channel closed before its commit leaves the writer's journal: the next attempt finds its
/// tensors written. A committed output replays its manifest, and new bytes past the grant are
/// refused.
#[test]
fn an_output_resumes_replays_and_keeps_its_bound() {
    let (store, root) = store();
    let (big, small) = (f32s(4 << 20, 2.0), f32s(32, 5.0));
    let put = |c: &mut Client, key: &str, bytes: &[u8]| {
        let n = bytes.len() as u64;
        c.0.send(&json!({"op": "put", "component": "vae", "key": key, "dtype": "F32", "shape": [n / 4], "length": n})).unwrap();
        c.0.data(bytes).unwrap();
        c.answer()
    };
    let plain = derivation(vec![], 1 << 30);
    let first = serve(&store, TX, &plain, |c| {
        assert_eq!(put(c, "big", &big)["result"], Value::Null);
        assert_eq!(put(c, "small", &small)["result"], Value::Null);
    });
    assert_eq!(first.unwrap(), None, "closed before its commit");
    let mut manifest = Value::Null;
    serve(&store, TX, &plain, |c| {
        for key in ["big", "small"] {
            assert_eq!(c.ask(json!({"op": "has", "component": "vae", "key": key}))["result"], true, "{key} was journaled");
        }
        assert_eq!(c.ask(json!({"op": "has", "component": "vae", "key": "other"}))["result"], false);
        manifest = c.ask(json!({"op": "commit"}))["result"]["manifest"].clone();
    })
    .unwrap();
    let replayed = serve(&store, TX, &plain, |c| {
        assert_eq!(c.ask(json!({"op": "has", "component": "vae", "key": "other"}))["result"], true);
        assert_eq!(put(c, "other", &small)["result"], Value::Null, "a replayed write is taken and dropped");
        assert_eq!(c.ask(json!({"op": "commit"}))["result"]["manifest"], manifest);
    });
    let replayed = replayed.unwrap().unwrap();
    assert_eq!(store.checkpoint(&replayed).unwrap().read("vae", "big").unwrap(), big);
    assert!(store.checkpoint(&replayed).unwrap().tensor("vae", "other").is_err());

    let other = "sha256:2222222222222222222222222222222222222222222222222222222222222222";
    let refused = serve(&store, other, &derivation(vec![], 100), |c| {
        let answer = c.ask(json!({"op": "put", "component": "vae", "key": "small", "dtype": "F32", "shape": [32], "length": 128}));
        assert_eq!(answer["error"], "bound_exceeded", "{answer}");
    });
    assert!(refused.is_err());
    let _ = std::fs::remove_dir_all(root);
}

/// A source channel answers its header, files and bytes; an object it lacks is made present
/// through the hook before it is read.
#[test]
fn a_source_channel_reads_and_fetches_what_it_lacks() {
    let (store, root) = store();
    let large = f32s(1024 * 4096, 9.0);
    let source = checkpoint(&store, "source", "unet", &[("a.weight", vec![1024, 4096], large.clone()), ("b.bias", vec![8], f32s(8, 1.0))]);
    let held = store.checkpoint(&source).unwrap();
    let lost = objects(&held, "unet", "a.weight").remove(0);
    let saved = store.read(&lost).unwrap();
    std::fs::remove_file(store.path(&lost).unwrap()).unwrap();
    let asked = std::sync::Mutex::new(vec![]);
    let current = || true;
    let ensure = |absent: &[ObjectRef]| {
        asked.lock().unwrap().extend(absent.iter().cloned());
        store.put(saved.as_slice(), Some(&absent[0])).map(drop).map_err(storage::io_error)
    };
    let hooks = Hooks { current: &current, ensure: &ensure };
    let (near, far) = cozy_platform::ipc::pair().unwrap();
    std::thread::scope(|scope| {
        let served = scope.spawn(|| serve_source(&store, &source, far, &hooks));
        let mut c = Client(Channel(near));
        let inspected = c.ask(json!({"op": "inspect"}))["result"].clone();
        assert_eq!(inspected["manifest"]["sha256"], source.sha256.as_str());
        assert!(inspected["files"]["unet/config.json"].is_object());
        let header = held.manifest().header().unwrap().clone();
        assert_eq!(c.read(json!({"op": "file", "path": tensorfs_core::format::HEADER_PATH})), store.read(&header).unwrap());
        assert_eq!(c.ask(json!({"op": "nope"}))["error"], "invalid_request");
        assert_eq!(c.read(json!({"op": "file", "path": "unet/config.json"})), b"{\"a\":1}");
        assert_eq!(c.read(json!({"op": "read", "component": "unet", "key": "a.weight"})), large);
        drop(c);
        served.join().unwrap().unwrap();
    });
    assert_eq!(*asked.lock().unwrap(), vec![lost]);
    let _ = std::fs::remove_dir_all(root);
}

/// An adapter view renames the base projection, grafts the factors under their adapter's name
/// and carries the rest; every tensor reads back bit-identical and the large base part keeps its
/// objects. A held view is reused; a collected one composes again identically.
#[test]
fn an_adapter_view_grafts_factors_onto_the_base() {
    let (store, root) = store();
    let (weight, bias, other) = (f32s(2048 * 2048, 0.5), f32s(2048, 1.5), f32s(8, 2.5));
    let base = checkpoint(&store, "base", "unet", &[("proj.weight", vec![2048, 2048], weight.clone()), ("proj.bias", vec![2048], bias.clone()), ("other.weight", vec![8], other.clone())]);
    let (a, b) = (f32s(2 * 2048, 0.125), f32s(2048 * 2, 0.75));
    let lora = checkpoint(&store, "lora", "adapter", &[("proj.lora_A.weight", vec![2, 2048], a.clone()), ("proj.lora_B.weight", vec![2048, 2], b.clone()), ("proj.alpha", vec![], 4.0f32.to_le_bytes().to_vec())]);
    let selection = [Selection { manifest: lora.clone(), component: String::new(), source_component: String::new(), strength: 1.0 }];
    let view = compose(&store, &base, &selection).unwrap();
    let composed = store.checkpoint(&view.manifest).unwrap();
    for (key, bytes) in [("proj.base_layer.weight", &weight), ("proj.base_layer.bias", &bias), ("other.weight", &other),
        ("proj.lora_A.adapter_0.weight", &a), ("proj.lora_B.adapter_0.weight", &b)] {
        assert_eq!(&composed.read("unet", key).unwrap(), bytes, "{key}");
    }
    assert!(composed.tensor("unet", "proj.weight").is_err());
    let held = store.checkpoint(&base).unwrap();
    assert_eq!(objects(&composed, "unet", "proj.base_layer.weight"), objects(&held, "unet", "proj.weight"));
    assert_eq!(composed.file("unet/config.json").unwrap(), b"{\"a\":1}");
    let graph: Value = serde_json::from_slice(&composed.file(GRAPH_FILE).unwrap()).unwrap();
    assert_eq!(graph["layers"][0]["alpha"], 4.0);
    assert_eq!(graph["layers"][0]["component"], "unet");
    assert_eq!(compose(&store, &base, &selection).unwrap(), view, "a held view is reused");
    let nested = compose(&store, &view.manifest, &selection).unwrap_err();
    assert!(nested.to_string().starts_with("adapter_nested"), "{nested}");

    store.unpin(&view.pin).unwrap();
    store.gc().unwrap();
    assert!(!store.reuse(&view.manifest).unwrap(), "the view went with its pin");
    assert!(store.checkpoint(&base).unwrap().read("unet", "proj.weight").is_ok(), "the base did not");
    assert_eq!(compose(&store, &base, &selection).unwrap(), view, "composed again, identically");
    let _ = std::fs::remove_dir_all(root);
}
