//! A release's files sent to a real object store over real sockets: a store that drops the
//! connection mid-body, answers 5xx, refuses, or races five identical conditional PUTs to one key
//! (owner-day run 39068104998). The store keeps what it is sent and answers `If-None-Match: *` with
//! 412 for a key it holds, as S3 does.
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use serde_json::json;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

use super::*;

/// What the store does with one connection's request, by its order of arrival.
#[derive(Clone, Copy)]
enum Turn {
    /// Reads this much of the body, then closes the connection unanswered.
    Drop(usize),
    /// Reads the body and answers this status, keeping nothing.
    Answer(u16),
    /// Reads the body and keeps it: 200, or 412 for a held key under `If-None-Match: *`.
    Keep,
}

struct Store {
    url: String,
    held: Arc<Mutex<BTreeMap<String, Vec<u8>>>>,
    connections: Arc<AtomicUsize>,
}

async fn store(turn: fn(usize) -> Turn) -> Store {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (held, connections) = (
        Arc::<Mutex<BTreeMap<_, _>>>::default(),
        Arc::<AtomicUsize>::default(),
    );
    let (keeps, counts) = (held.clone(), connections.clone());
    tokio::spawn(async move {
        loop {
            let (socket, _) = listener.accept().await.unwrap();
            let arrival = counts.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(serve(socket, turn(arrival), keeps.clone()));
        }
    });
    Store {
        url,
        held,
        connections,
    }
}

async fn serve(mut socket: TcpStream, turn: Turn, held: Arc<Mutex<BTreeMap<String, Vec<u8>>>>) {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        if socket.read(&mut byte).await.unwrap_or(0) == 0 {
            return;
        }
        head.push(byte[0]);
    }
    let head = String::from_utf8_lossy(&head).into_owned();
    let key = head.split_whitespace().nth(1).unwrap_or("/").to_owned();
    let header = |name: &str| {
        head.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name)
                .then(|| value.trim().to_owned())
        })
    };
    let length = header("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0usize);
    let conditional = header("if-none-match").as_deref() == Some("*");
    let mut body = vec![0; length];
    let status = match turn {
        Turn::Drop(at) => {
            let _ = socket.read_exact(&mut body[..at.min(length)]).await;
            return;
        }
        Turn::Answer(status) => {
            let _ = socket.read_exact(&mut body).await;
            status
        }
        Turn::Keep => {
            if socket.read_exact(&mut body).await.is_err() {
                return;
            }
            let mut held = held.lock().unwrap();
            match conditional && held.contains_key(&key) {
                true => 412,
                false => {
                    held.insert(key, body);
                    200
                }
            }
        }
    };
    let answer = format!("HTTP/1.1 {status} S\r\ncontent-length: 0\r\nconnection: close\r\n\r\n");
    let _ = socket.write_all(answer.as_bytes()).await;
}

/// A release file of `length` bytes no store holds yet, and its grant on `store` at `key`.
fn file(dir: &Path, name: &str, length: usize) -> (File, Vec<u8>) {
    let bytes: Vec<u8> = (0..length as u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    let local = dir.join(name);
    fs::write(&local, &bytes).unwrap();
    let digest = format!("sha256:{}", hex::encode(Sha256::digest(&bytes)));
    let file = File {
        path: name.into(),
        local,
        digest,
        length: length as i64,
    };
    (file, bytes)
}

fn target(store: &Store, key: &str) -> types::PresignedUpload {
    serde_json::from_value(json!({"url": format!("{}/{key}", store.url),
        "required_headers": {"if-none-match": "*"}}))
    .unwrap()
}

const LENGTH: usize = 6 << 20;

#[tokio::test]
async fn a_send_dropped_mid_body_goes_again_and_lands_whole() {
    let store = store(|arrival| match arrival {
        0 => Turn::Drop(1 << 20),
        _ => Turn::Keep,
    })
    .await;
    let dir = tempfile::tempdir().unwrap();
    let (file, bytes) = file(dir.path(), "tiny-1.0.0.tar.gz", LENGTH);
    let sent = put(
        &reqwest::Client::new(),
        &file.path,
        &file.local,
        &target(&store, "k"),
    )
    .await;
    assert_eq!(sent.unwrap(), LENGTH as u64);
    assert_eq!(store.connections.load(Ordering::SeqCst), 2);
    assert!(
        store.held.lock().unwrap()["/k"] == bytes,
        "the store holds other bytes"
    );
}

/// Owner-day run 39068104998: five publishes put identical bytes to one content-addressed key at
/// once, and the store dropped them unanswered. Each goes again: one lands, the rest find it held.
#[tokio::test]
async fn five_identical_puts_to_one_key_all_settle() {
    let store = store(|arrival| match arrival {
        0..5 => Turn::Drop(usize::MAX),
        _ => Turn::Keep,
    })
    .await;
    let dir = tempfile::tempdir().unwrap();
    let (file, bytes) = file(dir.path(), "h3_standin-1.0.0.tar.gz", LENGTH);
    let (http, grant) = (
        reqwest::Client::new(),
        target(&store, "artifacts/sha256/30/66"),
    );
    let puts = (0..5).map(|_| put(&http, &file.path, &file.local, &grant));
    let sent: Vec<u64> = futures_util::future::try_join_all(puts).await.unwrap();
    assert_eq!(
        sent.iter().sum::<u64>(),
        LENGTH as u64,
        "one landed, the rest found it: {sent:?}"
    );
    assert!(store.held.lock().unwrap()["/artifacts/sha256/30/66"] == bytes);
}

#[tokio::test]
async fn a_5xx_goes_again_and_another_4xx_is_final() {
    let slow = store(|arrival| match arrival {
        0 => Turn::Answer(503),
        _ => Turn::Keep,
    })
    .await;
    let dir = tempfile::tempdir().unwrap();
    let (file, _) = file(dir.path(), "a.whl", 1 << 20);
    let sent = put(
        &reqwest::Client::new(),
        &file.path,
        &file.local,
        &target(&slow, "a"),
    )
    .await;
    assert_eq!(sent.unwrap(), 1 << 20);

    let forbidding = store(|_| Turn::Answer(403)).await;
    let refused = put(
        &reqwest::Client::new(),
        &file.path,
        &file.local,
        &target(&forbidding, "a"),
    )
    .await
    .unwrap_err();
    assert_eq!(refused.code, "release_upload_failed");
    assert!(
        refused.message.ends_with("the store answered HTTP 403"),
        "{}",
        refused.message
    );
    assert_eq!(
        forbidding.connections.load(Ordering::SeqCst),
        1,
        "a 4xx is never sent again"
    );
}

/// A store that takes the body and never answers makes no progress: the upload ends, naming the
/// whole cause chain, instead of sending forever.
#[tokio::test]
async fn sends_that_carry_the_body_no_further_end_naming_the_cause() {
    let store = store(|_| Turn::Drop(usize::MAX)).await;
    let dir = tempfile::tempdir().unwrap();
    let (file, _) = file(dir.path(), "b.tar.gz", 1 << 20);
    let refused = put(
        &reqwest::Client::new(),
        &file.path,
        &file.local,
        &target(&store, "b"),
    )
    .await
    .unwrap_err();
    assert_eq!(refused.code, "release_upload_failed");
    assert!(
        refused.message.contains("no further than the send before"),
        "{}",
        refused.message
    );
    assert!(
        refused.message.matches(": ").count() >= 3,
        "a cause chain: {}",
        refused.message
    );
    assert!(store.connections.load(Ordering::SeqCst) >= 2);
}

/// Two release paths with one digest are sent once.
#[tokio::test]
async fn one_digest_is_sent_once() {
    let store = store(|_| Turn::Keep).await;
    let dir = tempfile::tempdir().unwrap();
    let (file, _) = file(dir.path(), "c.whl", 1 << 20);
    let twin = File {
        path: "d.whl".into(),
        local: file.local.clone(),
        digest: file.digest.clone(),
        length: file.length,
    };
    let grant = |file: &File| -> types::PackageFileGrant {
        serde_json::from_value(
            json!({"digest": file.digest, "length": file.length, "path": file.path,
            "present": false, "upload": target(&store, "c")}),
        )
        .unwrap()
    };
    let grants = [grant(&file), grant(&twin)];
    let sent = upload(&reqwest::Client::new(), &[file, twin], &grants)
        .await
        .unwrap();
    assert_eq!(
        (sent, store.connections.load(Ordering::SeqCst)),
        (1 << 20, 1)
    );
}
