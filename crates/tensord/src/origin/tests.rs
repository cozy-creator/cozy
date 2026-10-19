//! The machine's downloader against a real static origin: a store's own tree (`refs/`,
//! `blobs/`) served by `python3 -m http.server`.
use std::{
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Arc,
};

use tensorfs_core::{
    fetch::Endpoint,
    format::{Dtype, ObjectRef, Repo, Tree},
    store2::Store,
    writer::{Data, Source},
};

use super::{priority, Machine, Remote};
use crate::stop::Stop;

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

/// A checkpoint of two components (a 17 MiB tensor on the grid, small packed ones, a tokenizer),
/// variant `bf16` of `acme/model@1.0` in the store's refs: its header.
fn checkpoint(store: &Store) -> ObjectRef {
    let mut writer = store.writer(None, Some("origin-test")).unwrap();
    let f32 = Dtype::parse("F32");
    for (component, key, n, seed) in [("unet", "big.weight", 17 << 20, 1), ("unet", "a.bias", 4096, 2), ("vae", "z.weight", 8192, 3)] {
        let data = bytes(n, seed);
        writer.get(component).put(key, &f32, &[(n / 4) as u64], Data::Plain(Source::Read(&mut &data[..]))).unwrap();
    }
    writer.get("unet").asset("tokenizer.json", br#"{"layers":2}"#).unwrap();
    let header = writer.commit().unwrap();
    let mut tree = Tree::default();
    tree.set_variant("bf16", header.clone());
    let mut repo = Repo::new("acme/model");
    repo.set_release("1.0", store.put_tree(&tree).unwrap());
    store.update_ref("acme/model", |_| Some(repo)).unwrap();
    header
}

fn remote(machine: &Machine, base: &str) -> Remote {
    let client = machine.client(None, true).unwrap();
    Remote { origin: Arc::new(Endpoint::new(&client, base, None).unwrap()), trust: None, local: true }
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

#[test]
fn priorities_keep_v1s_order() {
    assert!(priority::NOW < priority::next(0, 0));
    assert!(priority::next(1, u32::MAX) < priority::next(2, 0));
    assert!(priority::next(u64::MAX, u32::MAX) < priority::BEHIND);
}
