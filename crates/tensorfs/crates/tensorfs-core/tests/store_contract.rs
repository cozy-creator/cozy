//! One Store, opened by both builds of this tree: this library, as the `cozy` binary links it,
//! and the `tensorfs` wheel's extension in another process. They record and judge the same
//! Store contract and give the same answers about the same objects.
//!
//! `TENSORFS_TEST_PYTHON` names an interpreter with the wheel built from this tree installed
//! (dev-build's `tensorfs` build). Without it the test skips, except on a hosted runner.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tensorfs_core::contract::{self, Contract};
use tensorfs_core::err::Code;
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::meta::now_nanos_unique;
use tensorfs_core::store::{Fault, Store};
use tensorfs_core::{CLAIM_PROTOCOL, STORE_FORMAT};

/// argv: verb, root, then the verb's arguments. Prints one JSON answer, or the refusal's code.
const WHEEL: &str = r#"
import json, sys
import tensorfs, tensorfs._ext as ext
verb, root, *rest = sys.argv[1:]
try:
    if verb == "ensure":
        tensorfs.Store.ensure(root)
        answer = {}
    elif verb == "answers":
        rust_id, new_file, old_file = rest
        store = tensorfs.Store.open(root)
        put, again = store.put_file(new_file), store.put_file(old_file)
        answer = {
            "store_format": ext.STORE_FORMAT,
            "claim_protocol": ext.CLAIM_PROTOCOL,
            "contains": store.contains(rust_id),
            "bytes": store.document(rust_id).hex(),
            "put": [put["id"], put["length"], put["admitted"]],
            "again": [again["id"], again["length"], again["admitted"]],
        }
    else:
        tensorfs.Store.open(root)
        answer = {}
except tensorfs.errors.Refusal as refusal:
    answer = {"refused": refusal.code}
print(json.dumps(answer))
"#;

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "tensorfs-store-contract-{}-{}",
            std::process::id(),
            now_nanos_unique()
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn python() -> Option<String> {
    match std::env::var("TENSORFS_TEST_PYTHON") {
        Ok(python) => Some(python),
        Err(_) if std::env::var_os("GITHUB_ACTIONS").is_some() => {
            panic!("TENSORFS_TEST_PYTHON must name an interpreter with this tree's tensorfs wheel")
        }
        Err(_) => {
            eprintln!("skipped: TENSORFS_TEST_PYTHON names no interpreter with the tensorfs wheel");
            None
        }
    }
}

fn wheel(python: &str, verb: &str, root: &Path, rest: &[&str]) -> serde_json::Value {
    let out = Command::new(python)
        .args(["-I", "-c", WHEEL, verb])
        .arg(root)
        .args(rest)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "wheel {verb}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

fn record(root: &Path, contract: Contract) {
    fs::write(
        root.join(contract::RECORD),
        format!(
            "store-format {}\nclaim-protocol {}\n",
            contract.store_format, contract.claim_protocol
        ),
    )
    .unwrap();
}

fn refused(result: tensorfs_core::err::Result<Store>) -> Code {
    result.expect_err("the Store must refuse").code
}

#[test]
fn the_binary_and_the_wheel_share_one_store() {
    let Some(python) = python() else { return };
    let scratch = Scratch::new();

    // The library makes the Store and an object; the wheel reads it and adds one.
    let root = scratch.0.join("shared");
    let store = Store::ensure(&root).unwrap();
    assert_eq!(contract::recorded(&root).unwrap(), Some(Contract::OURS));
    let (old, new) = (scratch.0.join("old"), scratch.0.join("new"));
    fs::write(&old, b"written through the cozy binary's TensorFS").unwrap();
    fs::write(&new, b"written through the tensorfs wheel").unwrap();
    let put = store.put_file(&old, None, &Fault::default()).unwrap();
    let id = put.obj.id();
    let answers = wheel(
        &python,
        "answers",
        &root,
        &[&id, new.to_str().unwrap(), old.to_str().unwrap()],
    );
    let theirs = ObjectRef::of(&fs::read(&new).unwrap());
    assert_eq!(answers["store_format"], STORE_FORMAT);
    assert_eq!(answers["claim_protocol"], CLAIM_PROTOCOL);
    assert_eq!(answers["contains"], true);
    assert_eq!(answers["bytes"], hex(&fs::read(&old).unwrap()));
    assert_eq!(
        answers["put"],
        serde_json::json!([theirs.id(), theirs.length, true])
    );
    assert_eq!(
        answers["again"],
        serde_json::json!([id, put.obj.length, false])
    );
    assert!(store.contains(&theirs.sha256));
    assert_eq!(
        store.read_range(&theirs.sha256, 0, theirs.length).unwrap(),
        fs::read(&new).unwrap()
    );
    assert!(
        !store
            .put_file(&new, Some(&theirs), &Fault::default())
            .unwrap()
            .admitted
    );
    assert_eq!(contract::recorded(&root).unwrap(), Some(Contract::OURS));

    // A Store made before the record existed is adopted, and so is one a crash cut short.
    let older = scratch.0.join("older");
    wheel(&python, "ensure", &older, &[]);
    fs::remove_file(older.join(contract::RECORD)).unwrap();
    Store::open(&older).unwrap();
    assert_eq!(contract::recorded(&older).unwrap(), Some(Contract::OURS));
    assert_eq!(wheel(&python, "open", &older, &[]), serde_json::json!({}));
    let torn = scratch.0.join("torn");
    wheel(&python, "ensure", &torn, &[]);
    fs::write(torn.join(contract::RECORD), "store-format 1\nclaim-pro").unwrap();
    Store::open(&torn).unwrap();
    assert_eq!(contract::recorded(&torn).unwrap(), Some(Contract::OURS));

    // Another store format refuses both builds, the owner included.
    let format = scratch.0.join("format");
    wheel(&python, "ensure", &format, &[]);
    record(
        &format,
        Contract {
            store_format: STORE_FORMAT + 1,
            ..Contract::OURS
        },
    );
    assert_eq!(refused(Store::open(&format)), Code::STORE_ERA);
    assert_eq!(refused(Store::ensure_owned(&format)), Code::STORE_ERA);
    assert_eq!(wheel(&python, "open", &format, &[])["refused"], "STORE_ERA");

    // Another claim protocol refuses both, until the owner records its own.
    let claim = scratch.0.join("claim");
    wheel(&python, "ensure", &claim, &[]);
    record(
        &claim,
        Contract {
            claim_protocol: CLAIM_PROTOCOL + 1,
            ..Contract::OURS
        },
    );
    assert_eq!(wheel(&python, "open", &claim, &[])["refused"], "STORE_ERA");
    assert_eq!(refused(Store::open(&claim)), Code::STORE_ERA);
    let owner = Store::ensure_owned(&claim).unwrap();
    assert_eq!(contract::recorded(&claim).unwrap(), Some(Contract::OURS));
    drop(owner);
    assert_eq!(wheel(&python, "open", &claim, &[]), serde_json::json!({}));
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
