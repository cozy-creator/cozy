//! The machine's downloader and the executor's fetch channel against a real static origin: a
//! store's own tree (`refs/`, `blobs/`) served by `python3 -m http.server`.
use platform::ipc::LocalStream;
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Arc,
};

use serde_json::{json, Value};
use tensorfs_core::{
    fetch::HttpOrigin,
    format::{Dtype, ObjectRef, Release, Repo},
    store2::Store,
    writer::{Data, Source},
};

use super::{priority, serve, Grant, Machine, Remote};
use crate::{
    frame::{Channel, Frame},
    stop::Stop,
};

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn scratch(name: &str) -> Scratch {
    let dir = std::env::temp_dir().join(format!("td-origin-{name}-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

/// `python3 -m http.server` over `root` on loopback; killed on drop.
struct Python {
    child: Child,
    base: String,
}

impl Python {
    fn serve(root: &Path) -> Python {
        let mut child = Command::new("python3")
            .args(["-u", "-m", "http.server", "0", "--bind", "127.0.0.1", "--directory"])
            .arg(root)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("python3");
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
        // "Serving HTTP on 127.0.0.1 port 41234 (http://127.0.0.1:41234/) ..."
        let port: u16 = line.split(" port ").nth(1).and_then(|r| r.split_whitespace().next()).unwrap().parse().unwrap();
        Python { child, base: format!("http://127.0.0.1:{port}") }
    }
}

impl Drop for Python {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn bytes(n: usize, seed: u8) -> Vec<u8> {
    (0..n).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect()
}

/// A checkpoint of two components (a 17 MiB tensor on the grid, small packed ones, a config),
/// published in the store's refs as `acme/model@1.0` lane `bf16`.
fn checkpoint(store: &Store) -> ObjectRef {
    let mut writer = store.writer(None, Some("origin-test")).unwrap();
    let f32 = Dtype::parse("F32");
    for (component, key, n, seed) in [("unet", "big.weight", 17 << 20, 1), ("unet", "a.bias", 4096, 2), ("vae", "z.weight", 8192, 3)] {
        let data = bytes(n, seed);
        writer.put(component, key, &f32, &[(n / 4) as u64], Data::Plain(Source::Read(&mut &data[..]))).unwrap();
    }
    writer.file("unet/config.json", br#"{"layers":2}"#).unwrap();
    let manifest = writer.commit().unwrap();
    let release = Release { version: "1.0".into(), lanes: BTreeMap::from([("bf16".into(), manifest.clone())]), ..Default::default() };
    store.update_ref("acme/model", |_| Some(Repo { name: "acme/model".into(), releases: vec![release], checkpoints: vec![] })).unwrap();
    manifest
}

fn remote(machine: &Machine, base: &str) -> Remote {
    let client = machine.client(None, true).unwrap();
    Remote { origin: Arc::new(HttpOrigin::new(&client, base).unwrap()), trust: None, local: true }
}

fn same(a: &Store, b: &Store, object: &ObjectRef) -> bool {
    b.reuse(object).unwrap() && a.read(object).unwrap() == b.read(object).unwrap()
}

#[test]
fn a_pull_takes_the_components_asked_then_the_rest_bit_identical() {
    let (dir, there) = (scratch("pull"), scratch("pull-there"));
    let origin = Arc::new(Store::open(there.0.join("store")).unwrap());
    let manifest = checkpoint(&origin);
    let python = Python::serve(origin.root());
    let store = Arc::new(Store::open(dir.0.join("store")).unwrap());
    let machine = Machine::new(store.clone(), None).unwrap();
    let remote = remote(&machine, &python.base);
    let closure = origin.closure(&manifest).unwrap();

    let pulled = machine.pull(&remote, &manifest, Some(&["vae".to_string()]), priority::NOW, None, &Stop::default()).unwrap();
    let checkpoint = store.checkpoint(&manifest).unwrap();
    assert_eq!(checkpoint.read("vae", "z.weight").unwrap(), bytes(8192, 3));
    assert!(pulled.landed.origin > 0);
    let absent: Vec<&ObjectRef> = closure.iter().filter(|o| !store.reuse(o).unwrap()).collect();
    assert!(!absent.is_empty(), "the unet's chunks were not asked for");

    let again = machine.pull(&remote, &manifest, None, priority::NOW, None, &Stop::default()).unwrap();
    assert_eq!(again.objects, closure.len());
    assert!(closure.iter().all(|o| same(&origin, &store, o)));
    assert_eq!(store.checkpoint(&manifest).unwrap().read("unet", "big.weight").unwrap(), bytes(17 << 20, 1));

    let stopped = Stop::default();
    stopped.cancel();
    let fresh = Arc::new(Store::open(dir.0.join("fresh")).unwrap());
    let other = Machine::new(fresh, None).unwrap();
    let refused = other.pull(&self::remote(&other, &python.base), &manifest, None, priority::NOW, None, &stopped);
    assert_eq!(refused.unwrap_err().detail, "stopped");
}

fn ask(channel: &mut Channel<LocalStream>, request: Value) -> Value {
    channel.send(&request).unwrap();
    match channel.receive().unwrap() {
        Some(Frame::Json(answer)) => answer,
        other => panic!("{other:?}"),
    }
}

fn rows(objects: &[ObjectRef]) -> Value {
    objects.iter().map(|o| json!({"sha256": o.sha256, "length": o.length})).collect()
}

#[test]
fn the_fetch_channel_schedules_waits_and_answers_presence_for_its_checkpoints_only() {
    let (dir, there) = (scratch("channel"), scratch("channel-there"));
    let origin = Arc::new(Store::open(there.0.join("store")).unwrap());
    let manifest = checkpoint(&origin);
    let python = Python::serve(origin.root());
    let store = Arc::new(Store::open(dir.0.join("store")).unwrap());
    let machine = Machine::new(store.clone(), None).unwrap();
    let remote = remote(&machine, &python.base);
    let closure = origin.closure(&manifest).unwrap();
    let id = format!("sha256:{}", manifest.sha256);

    let (client, server) = platform::ipc::pair().unwrap();
    let grant = Grant { run: 7, checkpoints: vec![(manifest.clone(), remote)], stop: Stop::default() };
    let served = {
        let machine = machine.clone();
        std::thread::spawn(move || serve(&machine, grant, server))
    };
    let mut channel = Channel(client);
    let present = ask(&mut channel, json!({"id": 1, "op": "present", "manifest": id, "objects": rows(&closure)}));
    assert_eq!(present, json!({"id": 1, "result": vec![false; closure.len()]}));
    let scheduled = ask(&mut channel, json!({"id": 2, "op": "schedule", "manifest": id, "objects": rows(&closure[..1]), "rank": 3}));
    assert_eq!(scheduled, json!({"id": 2, "result": null}));
    let waited = ask(&mut channel, json!({"id": 3, "op": "wait", "manifest": id, "objects": rows(&closure)}));
    assert_eq!(waited, json!({"id": 3, "result": null}));
    assert!(closure.iter().all(|o| same(&origin, &store, o)));
    let present = ask(&mut channel, json!({"id": 4, "op": "present", "manifest": id, "objects": rows(&closure)}));
    assert_eq!(present["result"], json!(vec![true; closure.len()]));

    let other = format!("sha256:{}", "ab".repeat(32));
    let refused = ask(&mut channel, json!({"id": 5, "op": "wait", "manifest": other, "objects": []}));
    assert_eq!(refused["error"], "SOURCE_NOT_ALLOWED");
    let unknown = ask(&mut channel, json!({"id": 6, "op": "present", "manifest": id, "objects": [], "rank": 1}));
    assert_eq!((unknown["id"].clone(), unknown["error"].clone()), (json!(6), json!("UNKNOWN_FIELD")));
    drop(channel);
    served.join().unwrap().unwrap();
}

#[test]
fn priorities_keep_v1s_order() {
    assert!(priority::NOW < priority::next(0, 0));
    assert!(priority::next(1, u32::MAX) < priority::next(2, 0));
    assert!(priority::next(u64::MAX, u32::MAX) < priority::BEHIND);
}
