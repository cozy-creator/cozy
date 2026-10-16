//! A pod with less RAM than one object still moves it. The object here is 128 MiB; the
//! transfer's peak resident memory must stay far below it in both directions. CI also runs
//! this binary inside a 64 MiB cgroup.
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tensorfs_core::fetch::{DeliveryGrant, Destination};
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::sha256::{hex, Sha256};
use tensorfs_core::store::{Fault, Store};
use tensorfs_core::transport::{
    fetch_ranged, push_object, Anonymous, Deadline, Ledger, Ranged, SourcePolicy, UploadGrant,
};

const OBJECT: u64 = 128 << 20;

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-transfer-memory-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}

/// Deterministic bytes, generated as they are read: nothing here holds the object.
struct Pattern {
    at: u64,
    end: u64,
}

impl Read for Pattern {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = buf.len().min((self.end - self.at) as usize) / 8 * 8;
        for (i, word) in buf[..n].chunks_exact_mut(8).enumerate() {
            let at = (self.at / 8 + i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            word.copy_from_slice(&at.to_le_bytes());
        }
        self.at += n as u64;
        Ok(n)
    }
}

fn peak_rss_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("VmHWM:")).unwrap();
    line.split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<u64>()
        .unwrap()
        * 1024
}

/// A minimal object store: PUT hashes the body it receives, GET streams `serve` back.
fn origin(serve: PathBuf, received: Arc<Mutex<Option<ObjectRef>>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let mut socket = socket.unwrap();
            let mut reader = BufReader::new(socket.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let method = line.split_whitespace().next().unwrap_or("").to_string();
            let mut length = 0u64;
            loop {
                let mut header = String::new();
                reader.read_line(&mut header).unwrap();
                if header.trim().is_empty() {
                    break;
                }
                if let Some((name, value)) = header.split_once(':') {
                    if name.eq_ignore_ascii_case("content-length") {
                        length = value.trim().parse().unwrap();
                    }
                }
            }
            if method == "PUT" {
                let mut hash = Sha256::new();
                let mut left = length;
                let mut buf = vec![0u8; 64 << 10];
                while left > 0 {
                    let n = reader
                        .read(&mut buf[..(left as usize).min(64 << 10)])
                        .unwrap();
                    assert!(n > 0, "upload ended early");
                    hash.update(&buf[..n]);
                    left -= n as u64;
                }
                *received.lock().unwrap() = Some(ObjectRef {
                    sha256: hex(&hash.finish()),
                    length,
                });
                socket
                    .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                    .unwrap();
            } else {
                let mut file = File::open(&serve).unwrap();
                let size = file.metadata().unwrap().len();
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\ncontent-length: {size}\r\nconnection: close\r\n\r\n"
                        )
                        .as_bytes(),
                    )
                    .unwrap();
                std::io::copy(&mut file, &mut socket).unwrap();
            }
        }
    });
    base
}

#[test]
fn an_object_larger_than_the_memory_budget_uploads_and_downloads() {
    let root = temporary("root");
    let store = Store::init(&root).unwrap();
    let object = store
        .put_stream(&mut Pattern { at: 0, end: OBJECT }, None, &Fault::default())
        .unwrap()
        .obj;
    assert_eq!(object.length, OBJECT);

    let received = Arc::new(Mutex::new(None));
    let base = origin(store.blob_path(&object.sha256), Arc::clone(&received));
    let policy = SourcePolicy {
        allowed_hosts: vec!["127.0.0.1".into()],
        allow_local: true,
        max_redirects: 0,
        ..Default::default()
    };

    let pushed = push_object(
        &store,
        &object.sha256,
        false,
        &UploadGrant::to(&format!("{base}/upload")),
        &policy,
        &Anonymous,
        Deadline::none(),
        &Ledger::new(),
        "",
    )
    .unwrap();
    assert_eq!(pushed.bytes_sent, OBJECT);
    assert_eq!(received.lock().unwrap().clone(), Some(object.clone()));

    let other = temporary("other");
    let destination = Store::init(&other).unwrap();
    let grant = DeliveryGrant {
        session: "transfer-memory".into(),
        object: object.clone(),
        destination: Destination::Blob,
    };
    fetch_ranged(
        &destination,
        &grant,
        &format!("{base}/object"),
        &policy,
        &Anonymous,
        Deadline::none(),
        &Ledger::new(),
        Ranged::default(),
        None,
    )
    .unwrap();
    assert!(destination.contains(&object.sha256));

    let peak = peak_rss_bytes();
    println!("peak RSS {peak} B for a {OBJECT} B object");
    assert!(
        peak < OBJECT / 4,
        "peak RSS {peak} B moving a {OBJECT} B object; a transfer must not hold the object"
    );
    let _ = std::fs::remove_dir_all(root);
    let _ = std::fs::remove_dir_all(other);
}
