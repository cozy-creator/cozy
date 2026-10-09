//! `tfs ensure --origin` (th-243): a public checkpoint read anonymously by digest from a REAL
//! loopback object origin, into a REAL Store, through the shipped binary. No hub exists in
//! these proofs: every request the run makes is logged, and each is a GET of
//! `/sha256/<hex>` with no credential.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use tensorfs_core::dtype::Dtype;
use tensorfs_core::header::{Body, Closure, Header, Part, Tensor};
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::jcs::{self, Json};
use tensorfs_core::manifest::{Draft, Entry};
use tensorfs_core::registry;
use tensorfs_core::store::Store;

const TFS: &str = env!("CARGO_BIN_EXE_tfs");

struct Checkpoint {
    bodies: HashMap<String, Vec<u8>>,
    manifest: ObjectRef,
    blobs: Vec<ObjectRef>,
}

fn checkpoint(sizes: &[usize], salt: u8) -> Checkpoint {
    let spec = registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "plain/1")
        .unwrap()
        .spec;
    let (mut bodies, mut tensors, mut blobs) = (HashMap::new(), Vec::new(), Vec::new());
    for (index, size) in sizes.iter().copied().enumerate() {
        let body: Vec<u8> = (0..size)
            .map(|at| ((at * 31 + index * 7 + salt as usize) % 251) as u8)
            .collect();
        let object = ObjectRef::of(&body);
        let shape = vec![(size / 4) as u64];
        tensors.push((
            format!("blocks.{index}.weight"),
            Tensor {
                dtype: Dtype::F32,
                shape: shape.clone(),
                encoding: spec.object_id(),
                parts: vec![(
                    "value".into(),
                    Part {
                        dtype: Dtype::F32,
                        shape,
                        body: Body::Segments(vec![object.clone()]),
                    },
                )],
            },
        ));
        blobs.push(object.clone());
        bodies.insert(object.sha256.clone(), body);
    }
    let header = Header {
        configs: Vec::new(),
        assets: Vec::new(),
        encodings: vec![spec],
        components: vec![("model".into(), tensors)],
    };
    header.validate(&Closure::default()).unwrap();
    let header_bytes = header.canonical_bytes().unwrap();
    let header_ref = ObjectRef::of(&header_bytes);
    bodies.insert(header_ref.sha256.clone(), header_bytes);
    let manifest = Draft {
        entries: vec![("model.cozytensors".into(), Entry::CozyTensors(header_ref))],
    }
    .seal()
    .unwrap()
    .canonical_bytes();
    let manifest_ref = ObjectRef::of(&manifest);
    bodies.insert(manifest_ref.sha256.clone(), manifest);
    Checkpoint {
        bodies,
        manifest: manifest_ref,
        blobs,
    }
}

/// How the origin answers one object: as stored, withdrawn (404), or with other bytes.
#[derive(Clone, Copy)]
enum Serve {
    Stored,
    Withdrawn,
    Tampered,
}

/// One request the origin saw: method, path, Range, and whether it carried a credential.
type Seen = (String, String, Option<String>, bool);

struct Origin {
    base: String,
    /// Every request: method, path, Range, and whether it carried a credential.
    log: Arc<Mutex<Vec<Seen>>>,
}

impl Origin {
    fn requests(&self) -> Vec<Seen> {
        self.log.lock().unwrap().clone()
    }
}

/// A public bucket: `GET /sha256/<hex>`, ranged when asked, nothing else.
fn origin(checkpoint: Arc<Checkpoint>, rules: HashMap<String, Serve>) -> Origin {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let log: Arc<Mutex<Vec<_>>> = Arc::default();
    let served = Arc::clone(&log);
    let rules = Arc::new(rules);
    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let Ok(mut socket) = socket else { return };
            let (checkpoint, log, rules) = (
                Arc::clone(&checkpoint),
                Arc::clone(&served),
                Arc::clone(&rules),
            );
            std::thread::spawn(move || {
                let Some((method, path, headers)) = request(&mut socket) else {
                    return;
                };
                let range = headers.get("range").cloned();
                let credential =
                    headers.contains_key("authorization") || headers.contains_key("dpop");
                log.lock()
                    .unwrap()
                    .push((method.clone(), path.clone(), range.clone(), credential));
                let hex = path.strip_prefix("/sha256/").unwrap_or_default();
                let body = match (method.as_str(), checkpoint.bodies.get(hex)) {
                    ("GET", Some(body)) => match rules.get(hex).copied().unwrap_or(Serve::Stored) {
                        Serve::Stored => Some(body.clone()),
                        Serve::Withdrawn => None,
                        Serve::Tampered => Some(body.iter().map(|b| b ^ 0xff).collect()),
                    },
                    _ => None,
                };
                let Some(body) = body else {
                    let _ = socket.write_all(
                        b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                    );
                    return;
                };
                let (status, start, end) =
                    match range.as_deref().and_then(|r| r.strip_prefix("bytes=")) {
                        Some(spec) => {
                            let (a, b) = spec.split_once('-').unwrap();
                            let start: usize = a.parse().unwrap();
                            let end = b
                                .parse::<usize>()
                                .map_or(body.len() - 1, |b| b.min(body.len() - 1));
                            ("206 Partial Content", start, end)
                        }
                        None => ("200 OK", 0, body.len() - 1),
                    };
                let mut head = format!(
                    "HTTP/1.1 {status}\r\ncontent-length: {}\r\ncache-control: public, max-age=31536000, immutable\r\nconnection: close\r\n",
                    end + 1 - start
                );
                if status.starts_with("206") {
                    head.push_str(&format!(
                        "content-range: bytes {start}-{end}/{}\r\n",
                        body.len()
                    ));
                }
                head.push_str("\r\n");
                let _ = socket.write_all(head.as_bytes());
                let _ = socket.write_all(&body[start..=end]);
            });
        }
    });
    Origin { base, log }
}

fn request(stream: &mut impl Read) -> Option<(String, String, HashMap<String, String>)> {
    let (mut collected, mut buf) = (Vec::new(), [0u8; 4096]);
    let split = loop {
        if let Some(at) = collected.windows(4).position(|w| w == b"\r\n\r\n") {
            break at;
        }
        let n = stream.read(&mut buf).ok()?;
        if n == 0 {
            return None;
        }
        collected.extend_from_slice(&buf[..n]);
    };
    let head = String::from_utf8_lossy(&collected[..split]).to_string();
    let mut lines = head.lines();
    let mut first = lines.next()?.split_whitespace();
    let (method, target) = (first.next()?.to_string(), first.next()?.to_string());
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    Some((method, target, headers))
}

fn temporary(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "tfs-origin-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn store(dir: &Path) -> PathBuf {
    let root = dir.join("store");
    Store::init(&root).unwrap();
    root
}

/// The last JSON line `tfs ensure --origin` printed, and whether it exited 0.
fn ensure(root: &Path, refspec: &str, origin: &str) -> (bool, Json, String) {
    let output = Command::new(TFS)
        .args([
            "ensure",
            root.to_str().unwrap(),
            refspec,
            "--origin",
            origin,
            "--sample-seconds",
            "0.05",
        ])
        .env("TFS_CREDENTIAL", "bearer must-never-be-sent")
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let last = stdout
        .lines()
        .last()
        .unwrap_or_else(|| panic!("no output: {}", String::from_utf8_lossy(&output.stderr)));
    (
        output.status.success(),
        jcs::parse(last.as_bytes(), 1 << 20).unwrap(),
        stdout,
    )
}

fn field(line: &Json, name: &str) -> String {
    let Json::Obj(fields) = line else {
        panic!("{line:?}")
    };
    match fields.iter().find(|(k, _)| k == name).map(|(_, v)| v) {
        Some(Json::Str(s)) => s.clone(),
        Some(Json::Num(n)) => format!("{n}"),
        other => panic!("{name} is {other:?}"),
    }
}

#[test]
fn a_public_checkpoint_comes_from_the_origin_by_digest_and_asks_no_hub() {
    let dir = temporary("cold-warm");
    let root = store(&dir);
    // One tensor larger than a download chunk (8 MiB): it must arrive in ranged pieces.
    let checkpoint = Arc::new(checkpoint(&[64 << 10, 9 << 20], 3));
    let origin = origin(Arc::clone(&checkpoint), HashMap::new());
    let refspec = format!("acme/model@{}", checkpoint.manifest.id());

    let (ok, result, stdout) = ensure(&root, &refspec, &origin.base);
    assert!(ok, "{stdout}");
    assert_eq!(field(&result, "event"), "ensure.result");
    assert_eq!(field(&result, "manifest"), checkpoint.manifest.id());
    let total: usize = checkpoint.bodies.values().map(Vec::len).sum();
    assert_eq!(field(&result, "bytes_total"), total.to_string());
    let store = Store::open(&root).unwrap();
    for (hex, body) in &checkpoint.bodies {
        if *hex != checkpoint.manifest.sha256 {
            assert_eq!(
                std::fs::read(store.object_path(hex)).unwrap(),
                *body,
                "{hex} landed wrong"
            );
        }
    }
    let requests = origin.requests();
    assert!(
        requests
            .iter()
            .all(|(method, path, _, credential)| method == "GET"
                && path
                    .strip_prefix("/sha256/")
                    .is_some_and(|hex| checkpoint.bodies.contains_key(hex))
                && !credential),
        "the run asked something other than anonymous digest GETs: {requests:?}"
    );
    let large = &checkpoint.blobs[1].sha256;
    let pieces: Vec<_> = requests
        .iter()
        .filter(|(_, path, _, _)| path.ends_with(large.as_str()))
        .collect();
    assert!(
        pieces.len() >= 2 && pieces.iter().all(|(_, _, range, _)| range.is_some()),
        "the large object was not fetched in ranged pieces: {pieces:?}"
    );

    // Held whole and named by digest: the next ensure makes no request at all.
    let before = origin.requests().len();
    let (ok, _, stdout) = ensure(&root, &refspec, &origin.base);
    assert!(ok, "{stdout}");
    assert_eq!(
        origin.requests().len(),
        before,
        "a held public checkpoint asked the origin"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_withdrawn_object_refuses_and_tampered_bytes_never_enter_the_store() {
    let checkpoint = Arc::new(checkpoint(&[32 << 10, 32 << 10], 9));
    let refspec = format!("acme/model@{}", checkpoint.manifest.id());
    for (serve, label) in [
        (Serve::Withdrawn, "withdrawn"),
        (Serve::Tampered, "tampered"),
    ] {
        let dir = temporary(label);
        let root = store(&dir);
        let victim = checkpoint.blobs[0].sha256.clone();
        let origin = origin(
            Arc::clone(&checkpoint),
            HashMap::from([(victim.clone(), serve)]),
        );
        let (ok, refusal, stdout) = ensure(&root, &refspec, &origin.base);
        assert!(!ok, "{label}: {stdout}");
        assert_eq!(field(&refusal, "event"), "ensure.refused", "{stdout}");
        assert!(
            !Store::open(&root).unwrap().contains(&victim),
            "{label} bytes entered the Store"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
    // A withdrawn manifest: nothing to walk.
    let dir = temporary("withdrawn-manifest");
    let root = store(&dir);
    let origin = origin(
        Arc::clone(&checkpoint),
        HashMap::from([(checkpoint.manifest.sha256.clone(), Serve::Withdrawn)]),
    );
    let (ok, refusal, stdout) = ensure(&root, &refspec, &origin.base);
    assert!(
        !ok && field(&refusal, "event") == "ensure.refused",
        "{stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_public_read_names_its_checkpoint_by_digest() {
    let dir = temporary("unpinned");
    let root = store(&dir);
    let checkpoint = Arc::new(checkpoint(&[4 << 10], 1));
    let origin = origin(Arc::clone(&checkpoint), HashMap::new());
    let (ok, refusal, stdout) = ensure(&root, "acme/model@r1", &origin.base);
    assert!(!ok, "{stdout}");
    assert_eq!(field(&refusal, "code"), "MALFORMED_DIGEST", "{stdout}");
    assert!(
        origin.requests().is_empty(),
        "an unpinned public read asked the origin"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A component-narrowed ensure (the manifest, then the header, then the component's parts)
/// reads from the origin the same way and asks nothing else.
#[test]
fn a_named_component_comes_from_the_origin_too() {
    let dir = temporary("component");
    let root = store(&dir);
    let checkpoint = Arc::new(checkpoint(&[16 << 10, 16 << 10], 5));
    let origin = origin(Arc::clone(&checkpoint), HashMap::new());
    let refspec = format!("acme/model@{}", checkpoint.manifest.id());
    let output = Command::new(TFS)
        .args([
            "ensure",
            root.to_str().unwrap(),
            &refspec,
            "--origin",
            &origin.base,
        ])
        .args(["--component", "model", "--sample-seconds", "0.05"])
        .env("TFS_CREDENTIAL", "")
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success(), "{stdout}");
    let store = Store::open(&root).unwrap();
    assert!(
        checkpoint
            .blobs
            .iter()
            .all(|blob| store.contains(&blob.sha256)),
        "{stdout}"
    );
    let requests = origin.requests();
    assert!(
        requests
            .iter()
            .all(|(method, path, _, credential)| method == "GET"
                && path.starts_with("/sha256/")
                && !credential),
        "{requests:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
