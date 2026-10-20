//! `cozy update` installs only what it can prove: a release whose archive matches the sha256 its
//! manifest names and carries an attestation of this repo's release workflow. Anything else is
//! refused before the binary is touched. The happy path runs in release-smoke.yml on every OS.
mod common;

use common::Home;
use sha2::{Digest, Sha256};
use std::{
    io::{BufRead, BufReader, Write},
    net::TcpListener,
    thread,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const ARCHIVE: &[u8] = b"not the release's archive";
/// A genuine bundle, but of another archive (a dev build of release.yml).
const ANOTHER: &str = include_str!("testdata/another-archive.sigstore.jsonl");

/// A release directory over plain HTTP: `dist-manifest.json`, the archive and its bundles.
fn release(sha256: &str, bundles: &str) -> String {
    let manifest = serde_json::json!({
        "announcement_tag": format!("v{VERSION}"),
        "releases": [{"app_name": "cozy", "app_version": VERSION}],
        "artifacts": {"cozy.tar.gz": {
            "kind": "executable-zip",
            "target_triples": ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu", "aarch64-apple-darwin",
                "x86_64-apple-darwin", "x86_64-pc-windows-msvc", "aarch64-pc-windows-msvc"],
            "checksums": {"sha256": sha256},
        }},
    });
    let files = [
        ("/dist-manifest.json", manifest.to_string().into_bytes()),
        ("/cozy.tar.gz", ARCHIVE.to_vec()),
        ("/cozy.tar.gz.sigstore.jsonl", bundles.as_bytes().to_vec()),
    ];
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = stream.unwrap();
            let mut reader = BufReader::new(&stream);
            let mut request = String::new();
            reader.read_line(&mut request).unwrap();
            let mut header = String::new();
            while reader.read_line(&mut header).unwrap() > 2 {
                header.clear();
            }
            let path = request.split_whitespace().nth(1).unwrap_or_default();
            let (status, body) = match files.iter().find(|(p, _)| *p == path) {
                Some((_, body)) => ("200 OK", body.as_slice()),
                None => ("404 Not Found", &[][..]),
            };
            let head = format!(
                "HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream
                .write_all(head.as_bytes())
                .and_then(|()| stream.write_all(body));
        }
    });
    url
}

/// Runs `cozy update` against that release on a hosted runner; `None` elsewhere.
fn refused(sha256: &str, bundles: &str) -> Option<serde_json::Value> {
    let home = Home::new()?;
    let binary = std::fs::read(env!("CARGO_BIN_EXE_cozy")).unwrap();
    let out = common::out(
        home.command()
            .env("COZY_DOWNLOAD_URL", release(sha256, bundles))
            .args(["--json", "update", VERSION])
            .output()
            .unwrap(),
    );
    assert_eq!(out.code, 1, "{out:?}");
    assert_eq!(
        std::fs::read(env!("CARGO_BIN_EXE_cozy")).unwrap(),
        binary,
        "the binary changed"
    );
    Some(out.refusal())
}

#[test]
fn an_archive_its_manifest_does_not_name_is_refused() {
    let Some(refusal) = refused(&"0".repeat(64), ANOTHER) else {
        return;
    };
    assert_eq!(refusal["code"], "checksum_mismatch", "{refusal}");
}

#[test]
fn an_archive_without_its_own_attestation_is_refused() {
    let sha256 = hex::encode(Sha256::digest(ARCHIVE));
    let Some(refusal) = refused(&sha256, &format!("not a bundle\n{ANOTHER}")) else {
        return;
    };
    assert_eq!(refusal["code"], "unattested", "{refusal}");
}
