//! A hub whose certificate this machine does not trust is a verdict, not weather: `tfs fetch`
//! refuses it typed on the first handshake, and `--ca-file` names the private CA that makes
//! the same hub reachable.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::Command;
use std::sync::{mpsc, Arc};

const ENVELOPE: &str =
    r#"{"error":{"code":"checkpoint_not_found","message":"no such checkpoint"}}"#;

/// A loopback hub under a self-signed authority that answers every call it can read with a
/// typed 404. Every accepted peer is reported before its handshake. The first `alerts`
/// connections are answered with a fatal `internal_error` alert instead of a handshake.
fn private_hub(alerts: usize) -> (u16, String, mpsc::Receiver<SocketAddr>) {
    let key = rcgen::generate_simple_self_signed(vec!["127.0.0.1".into()]).unwrap();
    let cert = rustls::pki_types::CertificateDer::from(key.cert.der().to_vec());
    let private = rustls::pki_types::PrivateKeyDer::try_from(key.key_pair.serialize_der())
        .unwrap()
        .clone_key();
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(vec![cert], private)
    .unwrap();
    let config = Arc::new(config);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (peers, accepted) = mpsc::channel();
    std::thread::spawn(move || {
        for (index, socket) in listener.incoming().enumerate() {
            let Ok(mut socket) = socket else { return };
            let _ = peers.send(socket.peer_addr().unwrap());
            if index < alerts {
                let _ = socket.read(&mut [0u8; 4096]);
                let _ = socket.write_all(&[0x15, 0x03, 0x03, 0x00, 0x02, 0x02, 0x50]);
                continue;
            }
            let conn = rustls::ServerConnection::new(Arc::clone(&config)).unwrap();
            let mut stream = rustls::StreamOwned::new(conn, socket);
            let mut head = Vec::new();
            let mut chunk = [0u8; 4096];
            while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                match stream.read(&mut chunk) {
                    Ok(n) if n > 0 => head.extend_from_slice(&chunk[..n]),
                    _ => break,
                }
            }
            if head.windows(4).any(|w| w == b"\r\n\r\n") {
                let answer = format!(
                    "HTTP/1.1 404 Not Found\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{ENVELOPE}",
                    ENVELOPE.len()
                );
                let _ = stream.write_all(answer.as_bytes());
                stream.conn.send_close_notify();
                let _ = stream.flush();
            }
        }
    });
    (port, key.cert.pem(), accepted)
}

/// How many connections reached the hub before this call: a sentinel connection is accepted
/// after every one queued ahead of it.
fn connections(port: u16, accepted: &mpsc::Receiver<SocketAddr>) -> usize {
    let sentinel = TcpStream::connect(("127.0.0.1", port)).unwrap();
    let mine = sentinel.local_addr().unwrap();
    accepted.iter().take_while(|peer| *peer != mine).count()
}

fn fetch(root: &std::path::Path, port: u16, extra: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["fetch", root.to_str().unwrap(), "proof/model"])
        .args(["--hub", &format!("https://127.0.0.1:{port}")])
        .args(extra)
        .env("TFS_CREDENTIAL", "")
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn an_untrusted_hub_is_refused_once_and_a_named_ca_reaches_it() {
    let dir = std::env::temp_dir().join(format!("tfs-hub-tls-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let root = dir.join("store");
    let init = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["store", "init", root.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stderr)
    );
    let (port, pem, accepted) = private_hub(0);

    let (ok, stderr) = fetch(&root, port, &[]);
    assert!(
        !ok && stderr.contains("REFUSED TLS_UNTRUSTED") && stderr.contains("127.0.0.1"),
        "{stderr}"
    );
    assert_eq!(
        connections(port, &accepted),
        1,
        "an untrusted hub was asked again: {stderr}"
    );

    let ca = dir.join("hub-ca.pem");
    std::fs::write(&ca, pem).unwrap();
    let (ok, stderr) = fetch(&root, port, &["--ca-file", ca.to_str().unwrap()]);
    assert!(
        !ok && stderr.contains("REFUSED REF_NOT_FOUND") && stderr.contains("checkpoint_not_found"),
        "{stderr}"
    );

    let garbage = dir.join("garbage.pem");
    std::fs::write(&garbage, "not a certificate").unwrap();
    let (ok, stderr) = fetch(&root, port, &["--ca-file", garbage.to_str().unwrap()]);
    assert!(!ok && stderr.contains("REFUSED UNKNOWN_FORMAT"), "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A peer's alert is the link's weather, not a verdict on its certificate: the ask is made
/// again and reaches the hub's own answer.
#[test]
fn a_tls_alert_from_the_peer_is_weather_and_asked_again() {
    let dir = std::env::temp_dir().join(format!("tfs-hub-tls-alert-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let root = dir.join("store");
    let init = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["store", "init", root.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(init.status.success());
    let (port, pem, _accepted) = private_hub(1);
    let ca = dir.join("hub-ca.pem");
    std::fs::write(&ca, pem).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["ensure", root.to_str().unwrap(), "proof/model"])
        .args(["--hub", &format!("https://127.0.0.1:{port}")])
        .args([
            "--ca-file",
            ca.to_str().unwrap(),
            "--sample-seconds",
            "0.05",
        ])
        .env("TFS_CREDENTIAL", "")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("\"code\":\"REF_NOT_FOUND\"") && stdout.contains("checkpoint_not_found"),
        "{stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A hub that accepts the connection and never speaks TLS is weather, judged by the ledger's
/// patience in the handshake (a live peer answers it at once), and never a hang.
#[test]
fn a_hub_that_never_speaks_tls_is_weather_and_never_a_hang() {
    let dir = std::env::temp_dir().join(format!("tfs-hub-tls-mute-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let root = dir.join("store");
    let init = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["store", "init", root.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(init.status.success());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let mut held = Vec::new();
        for socket in listener.incoming() {
            held.extend(socket.ok());
        }
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["ensure", root.to_str().unwrap(), "proof/model"])
        .args(["--hub", &format!("https://127.0.0.1:{port}")])
        .args(["--sample-seconds", "0.05"])
        .env("TFS_CREDENTIAL", "")
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    // A hang guard on output, not a verdict: tfs ensure prints a line every second.
    let stdout = child.stdout.take().unwrap();
    let (lines, read) = mpsc::channel();
    std::thread::spawn(move || {
        use std::io::BufRead;
        for line in std::io::BufReader::new(stdout)
            .lines()
            .map_while(|l| l.ok())
        {
            let _ = lines.send(line);
        }
    });
    let mut last = String::new();
    loop {
        match read.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(line) => last = line,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = child.kill();
                panic!("tfs ensure printed nothing for 30 s; last: {last}");
            }
        }
    }
    assert!(!child.wait().unwrap().success());
    assert!(
        last.contains("\"code\":\"HUB_UNREACHABLE\"") && last.contains("\"resumable\":true"),
        "{last}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
