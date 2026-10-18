//! `cozy package publish` against a real Tensorhub and its real object store (MinIO), with an HTTP
//! forward proxy in front of the store that cuts the first PUT it carries mid-body: the publish
//! sends that file again and completes (owner-day run 39068104998). The fixture carries bytes
//! unique to this run, so the store holds none of them and the cut PUT is a real upload, not a 412.
#![cfg(unix)]
mod common;

use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use common::Home;
use serde_json::Value;

/// A forward proxy for plain-HTTP store URLs: its first PUT is cut halfway through its body,
/// everything else passes through to the URL's host, one request per connection.
fn proxy() -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let puts = Arc::new(AtomicUsize::new(0));
    let counted = puts.clone();
    std::thread::spawn(move || {
        for client in listener.incoming().flatten() {
            let puts = counted.clone();
            std::thread::spawn(move || carry(client, &puts));
        }
    });
    (address, puts)
}

fn carry(mut client: TcpStream, puts: &AtomicUsize) {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        if client.read(&mut byte).unwrap_or(0) == 0 {
            return;
        }
        head.push(byte[0]);
    }
    let text = String::from_utf8_lossy(&head).into_owned();
    let lines: Vec<&str> = text.split("\r\n").filter(|l| !l.is_empty()).collect();
    let mut words = lines
        .first()
        .copied()
        .unwrap_or_default()
        .split_whitespace();
    let (method, url) = (
        words.next().unwrap_or_default(),
        words.next().unwrap_or_default(),
    );
    let header = |name: &str| {
        lines.iter().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name).then(|| value.trim())
        })
    };
    let length: usize = header("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if method == "PUT" && puts.fetch_add(1, Ordering::SeqCst) == 0 {
        let mut half = vec![0; length / 2];
        let _ = client.read_exact(&mut half);
        return;
    }
    let authority = url
        .strip_prefix("http://")
        .and_then(|rest| rest.split('/').next())
        .unwrap_or_default();
    let Ok(mut store) = TcpStream::connect(authority) else {
        return;
    };
    // One request per connection: the store answers and closes, and so does the proxy.
    let kept = lines.iter().filter(|line| {
        let line = line.to_ascii_lowercase();
        !line.starts_with("connection:") && !line.starts_with("proxy-connection:")
    });
    let head: Vec<&str> = kept.copied().chain(["Connection: close"]).collect();
    if store
        .write_all(format!("{}\r\n\r\n", head.join("\r\n")).as_bytes())
        .is_err()
    {
        return;
    }
    let (from_client, mut to_store) = (client.try_clone().unwrap(), store.try_clone().unwrap());
    std::thread::spawn(move || {
        let mut body = from_client.take(length as u64);
        let _ = std::io::copy(&mut body, &mut to_store);
    });
    let _ = std::io::copy(&mut store, &mut client);
}

#[test]
fn a_publish_whose_upload_is_cut_sends_it_again_and_completes() {
    let Some(home) = Home::with_hub() else { return };
    let account = home.sign_in("faults@cozy.test");
    let tiny = common::package::project(&home, "tiny");
    let salt = format!(
        "{:?} {}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap(),
        std::process::id()
    );
    // Bytes no store holds and no compressor shrinks: the sdist's PUT is a real upload.
    let mut state = salt.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    });
    let noise: Vec<u8> = (0..256 << 10)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect();
    std::fs::write(tiny.join("tiny/salt.bin"), noise).unwrap();
    let cozy = |args: &[&str], proxy: Option<&str>| {
        let mut command = home.command();
        command
            .current_dir(&tiny)
            .args(args)
            .env_remove("HTTPS_PROXY")
            .env_remove("NO_PROXY");
        if let Some(proxy) = proxy {
            command.env("HTTP_PROXY", proxy);
        }
        common::out(command.output().unwrap())
    };
    let locked = cozy(&["--json", "package", "lock"], None);
    assert_eq!(locked.code, 0, "{locked:?}");
    let (address, puts) = proxy();
    let published = cozy(&["--json", "package", "publish"], Some(&address));
    assert!(
        puts.load(Ordering::SeqCst) > 0,
        "the store's URLs did not pass the HTTP proxy (not plain HTTP?): {published:?}"
    );
    assert_eq!(published.code, 0, "{published:?}");
    let record: Value = serde_json::from_str(&published.stdout).unwrap();
    assert_eq!(record["package"], format!("{account}/tiny"), "{record}");
    assert!(
        puts.load(Ordering::SeqCst) >= 2,
        "the cut PUT was not sent again"
    );
    assert!(
        published.stderr.contains("sending it again"),
        "each re-send says why: {published:?}"
    );
}
