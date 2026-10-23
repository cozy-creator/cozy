//! `cozy update` installs only what it can prove: a release whose archive matches the sha256 its
//! manifest names and carries an attestation of this repo's release workflow. Anything else is
//! refused before the binary is touched. The happy path runs in release-smoke.yml on every OS.
//! It keeps the shell completion in place too.
mod common;

use common::Home;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    net::TcpListener,
    process::Command,
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

/// Homebrew's cozy (its keg in `<prefix>/Cellar`, linked from `<prefix>/bin`) is Homebrew's to
/// update: `cozy update` names `brew upgrade cozy` and touches nothing.
#[cfg(unix)]
#[test]
fn homebrew_s_cozy_is_left_to_brew_upgrade() {
    let Some(home) = Home::new() else { return };
    let keg = home.path().join("homebrew/Cellar/cozy").join(VERSION).join("bin/cozy");
    let link = home.path().join("homebrew/bin/cozy");
    for dir in [keg.parent().unwrap(), link.parent().unwrap()] {
        fs::create_dir_all(dir).unwrap();
    }
    fs::rename(home.installed(), &keg).unwrap();
    std::os::unix::fs::symlink(&keg, &link).unwrap();
    home.use_binary(&link);
    let before = fs::read_dir(keg.parent().unwrap()).unwrap().count();

    let out = home.cozy(["--json", "update"]);
    assert_eq!(out.code, 1, "{out:?}");
    let refusal = out.refusal();
    assert_eq!(refusal["code"], "homebrew_managed", "{refusal}");
    assert_eq!(refusal["next"], serde_json::json!(["brew upgrade cozy"]), "{refusal}");
    assert!(refusal["message"].as_str().unwrap().contains("homebrew/bin/cozy"), "{refusal}");
    let out = home.cozy(["update"]);
    assert!(out.stdout.contains("next: brew upgrade cozy"), "{out:?}");
    assert_eq!(fs::read_dir(keg.parent().unwrap()).unwrap().count(), before);
}

/// After `cozy update`, a new bash completes a path wherever cozy takes one, as bash completes
/// any path: `~` and relative, after `name=`, a directory with `/` and no space, `--out` only
/// directories. Values still come from the grammar.
#[cfg(unix)]
#[test]
fn after_an_update_tab_completes_paths_as_bash_does() {
    let Some(home) = Home::new() else { return };
    let user = home.path();
    let data = user.join(".local/share");
    for dir in ["r/images", "my dir"] {
        fs::create_dir_all(user.join(dir)).unwrap();
    }
    for file in ["r/qwen-edit.yaml", "r/qwen-edit2.yaml", "my dir/a b.yaml"] {
        fs::write(user.join(file), "").unwrap();
    }
    let stub = data.join("bash-completion/completions/cozy");
    // What the Go CLI left there: a protocol this binary never answered.
    fs::create_dir_all(stub.parent().unwrap()).unwrap();
    fs::write(&stub, "complete -F _cozy cozy\n").unwrap();
    let out = common::out(
        home.command()
            .env("XDG_DATA_HOME", &data)
            .env("COZY_DOWNLOAD_URL", release(&"0".repeat(64), ANOTHER))
            .arg("update")
            .output()
            .unwrap(),
    );
    assert_eq!(out.code, 0, "{out:?}");
    assert!(out.stdout.contains("is the newest release"), "{out:?}");

    let mut bash = Command::new("bash");
    bash.args(["--norc", "--noprofile", "-i"])
        .current_dir(user)
        .env("HOME", user)
        .env("XDG_DATA_HOME", &data)
        .env(
            "PATH",
            format!(
                "{}:{}",
                home.root().join("bin").display(),
                std::env::var("PATH").unwrap()
            ),
        )
        .env("PS1", "$ ")
        .env("INPUTRC", "/dev/null")
        .env_remove("COMPLETE");
    let mut shell = common::pty::Pty::spawn(&bash);
    // As bash-completion loads it on the first <TAB>.
    shell.key(&format!("source {}\r", stub.display()));
    // `#` after the <TAB> shows what it left: a space, or none after a directory.
    for (typed, completed) in [
        (
            "cozy run o/p/f --input ~/r/qwen-ed",
            "cozy run o/p/f --input ~/r/qwen-edit#",
        ),
        (
            "cozy run o/p/f --input ~/r/im",
            "cozy run o/p/f --input ~/r/images/#",
        ),
        (
            "cozy run o/p/f --asset p=~/r/im",
            "cozy run o/p/f --asset p=~/r/images/#",
        ),
        (
            "cozy run o/p/f --input=./r/qwen-edit2",
            "cozy run o/p/f --input=./r/qwen-edit2.yaml #",
        ),
        (
            "cozy run o/p/f image=r/qwen-edit2",
            "cozy run o/p/f image=r/qwen-edit2.yaml #",
        ),
        ("cozy run o/p/f ./my", r"cozy run o/p/f ./my\ dir/#"),
        (
            "cozy run o/p/f --out ~/r/",
            "cozy run o/p/f --out ~/r/images/#",
        ),
        ("cozy run o/p/f --la", "cozy run o/p/f --lane #"),
        ("cozy run ./r/im", "cozy run ./r/images/#"),
        ("cozy model upload ~/my", r"cozy model upload ~/my\ dir/#"),
        ("cozy --tensorhub tens", "cozy --tensorhub tensorhub #"),
        ("cozy hub us", "cozy hub use #"),
    ] {
        shell.key(&format!("{typed}\t#"));
        let want = format!("$ {completed}");
        shell.until(&want, |screen| screen.lines().any(|l| l.trim_end() == want));
        // Ctrl-U: the next line starts empty.
        shell.key("\x15");
    }
}
