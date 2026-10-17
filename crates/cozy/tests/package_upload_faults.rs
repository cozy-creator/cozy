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

/// A forward proxy for plain-HTTP store URLs: its first PUT is cut after `cut` bytes of body,
/// everything else passes through to the URL's host, absolute-form request lines and all.
fn proxy(cut: usize) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let puts = Arc::new(AtomicUsize::new(0));
    let counted = puts.clone();
    std::thread::spawn(move || {
        for client in listener.incoming().flatten() {
            let puts = counted.clone();
            std::thread::spawn(move || carry(client, cut, &puts));
        }
    });
    (address, puts)
}

fn carry(mut client: TcpStream, cut: usize, puts: &AtomicUsize) {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        if client.read(&mut byte).unwrap_or(0) == 0 {
            return;
        }
        head.push(byte[0]);
    }
    let line = String::from_utf8_lossy(&head)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    let mut words = line.split_whitespace();
    let (method, url) = (
        words.next().unwrap_or_default(),
        words.next().unwrap_or_default(),
    );
    if method == "PUT" && puts.fetch_add(1, Ordering::SeqCst) == 0 {
        let mut body = vec![0; cut];
        let _ = client.read_exact(&mut body);
        return;
    }
    let authority = url
        .strip_prefix("http://")
        .and_then(|rest| rest.split('/').next())
        .unwrap_or_default();
    let Ok(mut store) = TcpStream::connect(authority) else {
        return;
    };
    if store.write_all(&head).is_err() {
        return;
    }
    let (mut from_client, mut to_store) = (client.try_clone().unwrap(), store.try_clone().unwrap());
    std::thread::spawn(move || {
        let _ = std::io::copy(&mut from_client, &mut to_store);
        let _ = to_store.shutdown(std::net::Shutdown::Write);
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
    std::fs::write(tiny.join("tiny/salt.txt"), salt.repeat(4096)).unwrap();
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
    let (address, puts) = proxy(64 << 10);
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
