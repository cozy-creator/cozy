//! `Checkpoint::open_file` on real folders and stores: the bytes equal an independent
//! concatenation of the header file and its object files, read in the header's own CBOR order;
//! seeks land anywhere; a thin folder fetches what it lacks; a closed file holds nothing.
#![cfg(all(unix, feature = "store"))]

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tensorfs::format::ClassRef;
use tensorfs::format::{Body, Component, Dtype, Header, Location, ObjectRef, Tensor};
use tensorfs::raw::block_on;
use tensorfs::store::Store;
use tensorfs::{CheckpointFile, Client, Config, Failure, Kind, Open};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "checkpoint-file-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn client(cache: &Path, endpoint: &str) -> Client {
    let mut config = Config::from_env();
    config.cache_dir = cache.to_path_buf();
    config.endpoint = endpoint.to_owned();
    config.mirror = None;
    Client::new(config).unwrap()
}

fn open(client: &Client, what: &str, local_files_only: bool) -> tensorfs::Checkpoint {
    let options = Open {
        local_files_only,
        ..Open::default()
    };
    block_on(client.open(what, &options)).unwrap()
}

fn noise(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

/// A folder: chunks of awkward lengths, assets whose CBOR order differs from name order, and an
/// asset cited twice. Returns the folder and every object's bytes.
fn fixture(root: &Path) -> (PathBuf, Vec<Vec<u8>>) {
    let folder = root.join("folder");
    fs::create_dir_all(folder.join("chunks")).unwrap();
    let mut objects = Vec::new();
    let mut put = |bytes: Vec<u8>| {
        let object = ObjectRef::of(&bytes);
        fs::write(folder.join("chunks").join(&object.sha256), &bytes).unwrap();
        objects.push(bytes);
        object
    };
    let chunks: Vec<ObjectRef> = [5000, 1, 4096, 12289]
        .iter()
        .enumerate()
        .map(|(i, n)| put(noise(i as u64 + 1, *n)))
        .collect();
    let vocab = put(b"{\"vocab\": 1}".to_vec());
    let tokenizer = put(noise(9, 4097));
    let merges = put(b"#version: 0.2\na b\n".to_vec());
    let tensor = |key: &str, chunk: usize| Tensor {
        key: key.into(),
        dtype: Dtype::U8,
        shape: vec![chunks[chunk].length],
        body: Body::Data(Location {
            chunk: chunk as u64,
            offset: 0,
            length: chunks[chunk].length,
        }),
    };
    let header = Header::new(
        chunks.clone(),
        None,
        vec![
            Component {
                name: "text_encoder".into(),
                class: ClassRef::new("test", "1.0", "Text"),
                tensors: vec![tensor("a", 0), tensor("b", 1)],
                assets: BTreeMap::from([
                    ("vocab.json".into(), vocab.clone()),
                    ("tokenizer.json".into(), tokenizer),
                    ("merges.txt".into(), merges),
                ]),
                ..Component::default()
            },
            Component {
                name: "unet".into(),
                tensors: vec![tensor("c", 2), tensor("d", 3)],
                assets: BTreeMap::from([("a.bin".into(), vocab)]),
                ..Component::default()
            },
        ],
        BTreeMap::new(),
    )
    .unwrap()
    .to_cbor()
    .unwrap();
    fs::write(folder.join("model.cozytensors"), &header).unwrap();
    objects.push(header);
    (folder, objects)
}

/// `[sha256, length]` as the header spells an object.
fn object(d: &mut minicbor::Decoder) -> String {
    assert_eq!(d.array().unwrap(), Some(2));
    let digest = d
        .bytes()
        .unwrap()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    d.u64().unwrap();
    digest
}

/// The file built from the header file's bytes alone: the header, then each chunk and each
/// asset citation in encoded order, each at the next multiple of 4096. Also each object's
/// offset and every asset name in file order.
fn expected(folder: &Path) -> (Vec<u8>, Vec<u64>, Vec<String>) {
    let header = fs::read(folder.join("model.cozytensors")).unwrap();
    let mut d = minicbor::Decoder::new(&header);
    d.array().unwrap();
    d.skip().unwrap();
    let mut objects = Vec::new();
    for _ in 0..d.array().unwrap().unwrap() {
        objects.push(object(&mut d));
    }
    d.skip().unwrap();
    d.skip().unwrap();
    let mut names = Vec::new();
    for _ in 0..d.array().unwrap().unwrap() {
        let fields = d.array().unwrap().unwrap();
        for _ in 0..4 {
            d.skip().unwrap();
        }
        for _ in 0..d.map().unwrap().unwrap() {
            names.push(d.str().unwrap().to_owned());
            objects.push(object(&mut d));
        }
        for _ in 5..fields {
            d.skip().unwrap();
        }
    }
    let (mut out, mut starts) = (header.clone(), Vec::new());
    for digest in objects {
        out.resize(out.len().next_multiple_of(4096), 0);
        starts.push(out.len() as u64);
        out.extend(fs::read(folder.join("chunks").join(digest)).unwrap());
    }
    (out, starts, names)
}

fn all(file: &mut CheckpointFile) -> Vec<u8> {
    let mut out = Vec::new();
    file.read_to_end(&mut out).unwrap();
    out
}

fn up_to(file: &mut CheckpointFile, n: usize) -> Vec<u8> {
    let mut out = Vec::new();
    file.by_ref().take(n as u64).read_to_end(&mut out).unwrap();
    out
}

/// Every unrooted, unheld object in `cache` goes.
fn collect(cache: &Path) -> u64 {
    let store = Store::open(cache).unwrap().with_grace(Duration::ZERO);
    store.collect(u64::MAX, Duration::ZERO).unwrap().objects
}

fn present(cache: &Path, objects: &[Vec<u8>]) -> Vec<bool> {
    objects
        .iter()
        .map(|b| cache.join(ObjectRef::of(b).key()).exists())
        .collect()
}

/// A store holding the fixture's objects, unrooted; its header's id.
fn stored(cache: &Path, objects: &[Vec<u8>]) -> String {
    let store = Store::open(cache).unwrap();
    let ids: Vec<ObjectRef> = objects
        .iter()
        .map(|b| store.put_bytes(b).unwrap())
        .collect();
    ids.last().unwrap().id()
}

#[test]
fn folder_and_store_read_as_the_independent_concatenation() {
    let scratch = Scratch::new();
    let (folder, objects) = fixture(&scratch.0);
    let (want, starts, names) = expected(&folder);
    assert_eq!(
        names,
        ["merges.txt", "vocab.json", "tokenizer.json", "a.bin"],
        "encoded order"
    );
    assert_eq!(
        starts.len(),
        8,
        "4 chunks and 4 asset citations, the shared one twice"
    );
    let ck = open(
        &client(&scratch.0.join("cache"), "http://127.0.0.1:9"),
        folder.to_str().unwrap(),
        true,
    );
    let mut file = ck.open_file().unwrap();
    assert_eq!(file.len(), want.len() as u64);
    assert_eq!(all(&mut file), want);

    let store = scratch.0.join("store");
    let header = stored(&store, &objects);
    let ck = open(&client(&store, "http://127.0.0.1:9"), &header, true);
    assert_eq!(all(&mut ck.open_file().unwrap()), want);
}

#[test]
fn seeks_cross_chunks_padding_and_the_end() {
    let scratch = Scratch::new();
    let (folder, _) = fixture(&scratch.0);
    let (want, starts, _) = expected(&folder);
    let len = want.len() as u64;
    let ck = open(
        &client(&scratch.0.join("cache"), "http://127.0.0.1:9"),
        folder.to_str().unwrap(),
        true,
    );
    let mut file = ck.open_file().unwrap();
    for start in &starts {
        assert_eq!(file.seek(SeekFrom::Start(start - 3)).unwrap(), start - 3);
        assert_eq!(
            up_to(&mut file, 7),
            want[*start as usize - 3..*start as usize + 4]
        );
    }
    assert_eq!(file.seek(SeekFrom::End(-5)).unwrap(), len - 5);
    assert_eq!(up_to(&mut file, 100), want[want.len() - 5..]);
    assert_eq!(file.seek(SeekFrom::Current(10)).unwrap(), len + 10);
    assert_eq!(up_to(&mut file, 5), b"");
    let error = file
        .seek(SeekFrom::Current(-(len as i64) - 11))
        .unwrap_err();
    assert_eq!(Failure::from(error).kind, Kind::Invalid);
    assert_eq!(
        file.stream_position().unwrap(),
        len + 10,
        "a refused seek stays put"
    );

    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = |n: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % n
    };
    let mut pos = file.seek(SeekFrom::Start(0)).unwrap();
    for _ in 0..2000 {
        let by = next(len + len / 2) as i64 - len as i64 / 2;
        let (to, target) = match next(3) {
            0 => (SeekFrom::Start(by.unsigned_abs()), Some(by.unsigned_abs())),
            1 => (SeekFrom::Current(by), pos.checked_add_signed(by)),
            _ => (SeekFrom::End(by), len.checked_add_signed(by)),
        };
        let Some(target) = target else {
            assert!(file.seek(to).is_err());
            continue;
        };
        pos = file.seek(to).unwrap();
        assert_eq!(pos, target);
        let size = next(3 * 4096) as usize;
        let mut buf = vec![0xa5; size];
        let n = file.read(&mut buf).unwrap();
        let from = (pos as usize).min(want.len());
        assert_eq!(
            &buf[..n],
            &want[from..(from + size).min(want.len())],
            "{pos} {size}"
        );
        pos += n as u64;
    }
}

#[test]
fn a_thin_folder_fetches_what_it_lacks_and_offline_refuses() {
    let scratch = Scratch::new();
    let (folder, objects) = fixture(&scratch.0);
    let (want, starts, _) = expected(&folder);
    let mirror = format!("file://{}", scratch.0.join("mirror").display());
    let producer = client(&scratch.0.join("producer"), &mirror);
    block_on(producer.upload_folder(&"acme/file@1".parse().unwrap(), &folder, None)).unwrap();

    let thin = scratch.0.join("thin");
    fs::create_dir_all(thin.join("chunks")).unwrap();
    fs::copy(
        folder.join("model.cozytensors"),
        thin.join("model.cozytensors"),
    )
    .unwrap();
    let first = ObjectRef::of(&objects[0]).sha256;
    fs::copy(
        folder.join("chunks").join(&first),
        thin.join("chunks").join(&first),
    )
    .unwrap();

    let offline = open(
        &client(&scratch.0.join("offline"), &mirror),
        thin.to_str().unwrap(),
        true,
    );
    let mut file = offline.open_file().unwrap();
    assert_eq!(
        up_to(&mut file, starts[1] as usize),
        want[..starts[1] as usize]
    );
    let error = file.read(&mut [0; 16]).unwrap_err();
    assert_eq!(Failure::from(error).kind, Kind::NotFound);

    let cache = scratch.0.join("online");
    let online = open(&client(&cache, &mirror), thin.to_str().unwrap(), false);
    assert_eq!(all(&mut online.open_file().unwrap()), want);
    assert_eq!(
        present(&cache, &objects[1..7]),
        [true; 6],
        "fetched into the cache"
    );
}

#[test]
fn a_closed_file_holds_nothing() {
    let scratch = Scratch::new();
    let (folder, objects) = fixture(&scratch.0);
    let (want, _, _) = expected(&folder);
    let cache = scratch.0.join("store");
    let header = stored(&cache, &objects);
    let ck = open(&client(&cache, "http://127.0.0.1:9"), &header, true);
    let mut file = ck.open_file().unwrap();
    drop(ck);
    collect(&cache);
    assert!(
        present(&cache, &objects).iter().all(|p| *p),
        "the open file holds its objects"
    );
    assert_eq!(all(&mut file), want);
    drop(file);
    assert_eq!(collect(&cache), objects.len() as u64);
    assert!(
        present(&cache, &objects).iter().all(|p| !p),
        "nothing stays held"
    );
}

/// The Python surface on the same fixture, with the store collected while it holds a file.
#[test]
fn python_checkpoint_file() {
    let Ok(python) = std::env::var("TENSORFS_TEST_PYTHON") else {
        eprintln!("skipped: TENSORFS_TEST_PYTHON is required for the Python binding proof");
        return;
    };
    let scratch = Scratch::new();
    let (folder, objects) = fixture(&scratch.0);
    let (want, starts, _) = expected(&folder);
    let expected = scratch.0.join("expected.bin");
    fs::write(&expected, &want).unwrap();
    let store = scratch.0.join("store");
    let header = stored(&store, &objects);
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tensorfs-py/scripts/checkpoint_file.py");
    let mut child = Command::new(python)
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let request = serde_json::json!({
        "folder": folder, "cache": scratch.0.join("cache"), "store": store, "header": header,
        "expected": expected, "starts": starts,
    });
    writeln!(stdin, "{request}").unwrap();
    let mut said = |what: &str| {
        let mut line = String::new();
        stdout.read_line(&mut line).unwrap();
        assert_eq!(line.trim(), what, "the Python proof stopped early");
    };
    said("held");
    collect(&store);
    assert!(
        present(&store, &objects).iter().all(|p| *p),
        "Python's open file holds its objects"
    );
    writeln!(stdin).unwrap();
    said("closed");
    collect(&store);
    assert!(
        present(&store, &objects).iter().all(|p| !p),
        "a closed Python file holds nothing"
    );
    writeln!(stdin).unwrap();
    assert!(child.wait().unwrap().success());
}
