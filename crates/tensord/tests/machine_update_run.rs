//! Runs of `kind: update` on the real binary in machine mode: wheels the machine holds as objects
//! are staged, verified and activated in place (same boot, same receipt, the stable parent runs
//! the new service); the client follows the events again across the restart and reads the outcome. A
//! candidate whose service never proves readiness ends FAILED, rolled back.
//! Linux: it reads `/proc` and signals the machine.
#![cfg(target_os = "linux")]
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
mod common;

use ed25519_dalek::SigningKey;
use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tensord::api::{
    capability::{self, Grant},
    v1::{
        self,
        client::{Auth, Machine as Client},
    },
};
use tokio_stream::StreamExt;

const WORKER: &str = "update-run-test";
const OWNER: [u8; 32] = [5; 32];

struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wheel(distribution: &str, version: &str, script: Option<&[u8]>) -> (String, Vec<u8>) {
    requiring(distribution, version, script, "")
}

/// `wheel`, its METADATA requiring `requires` (`Requires-Dist:` lines).
fn requiring(distribution: &str, version: &str, script: Option<&[u8]>, requires: &str) -> (String, Vec<u8>) {
    let name = format!("{distribution}-{version}-py3-none-any.whl");
    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    zip.start_file(
        format!("{distribution}-{version}.dist-info/METADATA"),
        options,
    )
    .unwrap();
    let metadata = format!(
        "Name: {}\nVersion: {version}\n{requires}",
        distribution.replace('_', "-")
    );
    zip.write_all(metadata.as_bytes()).unwrap();
    if let Some(script) = script {
        zip.start_file(
            format!("{distribution}-{version}.data/scripts/tensord"),
            options,
        )
        .unwrap();
        zip.write_all(script).unwrap();
    }
    (name, zip.finish().unwrap().into_inner())
}

/// Puts `bytes` into the machine's store and records them as the owner's, as an upload leaves an
/// object; answers its digest.
fn hold(root: &Path, bytes: &[u8]) -> String {
    let engine = root.join("var/lib/cozy/tensord");
    let store = tensord::storage::open(&root.join("var/lib/tensorfs")).unwrap();
    let object = store.put_bytes(bytes).unwrap();
    tensord::storage::hold(&store, &tensord::storage::pin::object(&object.sha256), vec![object.clone()], None).unwrap();
    let sha256 = object.sha256.clone();
    let owner = SigningKey::from_bytes(&OWNER).verifying_key();
    tensord::journal::Journal::open(&engine.join("execution"))
        .unwrap()
        .bind_object(&tensorfs_core::sha256::hex(owner.as_bytes()), &object)
        .unwrap();
    format!("sha256:{sha256}")
}

fn cap(action: &str) -> String {
    let expires = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        + 3600;
    let grant = Grant {
        machine: WORKER.into(),
        run: if action.is_empty() {
            "up-1".into()
        } else {
            String::new()
        },
        action: action.into(),
        expires,
        ..Default::default()
    };
    capability::mint(&SigningKey::from_bytes(&OWNER), grant)
}

/// A client pinned to the machine's leaf, authorized by `cap`, once the service answers its
/// owner's Status ready.
async fn client(root: &Path, port: u16, cap: &str) -> Client {
    let start = Instant::now();
    loop {
        let owner = Auth::Token(self::cap(capability::MACHINE));
        if let Some(client) = common::pin(root)
            .and_then(|pin| Client::new(&format!("127.0.0.1:{port}"), pin, owner).ok())
        {
            if client.status().await.is_ok_and(|frame| frame.phase == "ready") {
                return client.with_auth(Auth::Token(cap.into()));
            }
        }
        assert!(
            start.elapsed() < Duration::from_secs(300),
            "the machine never answered"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

async fn status(root: &Path, port: u16) -> v1::StatusFrame {
    client(root, port, &cap(capability::MACHINE)).await.status().await.unwrap()
}

/// Submits (or attaches to) update `id` and follows its log across restarts to the outcome.
async fn update(root: &Path, port: u16, id: &str, spec: Option<v1::RunSpec>) -> Vec<v1::RunEvent> {
    let (mut log, mut spec) = (Vec::<v1::RunEvent>::new(), spec);
    loop {
        let after = log.iter().map(|e| e.sequence).max().unwrap_or(0);
        let client = client(root, port, &cap(capability::MACHINE)).await;
        if let Some(spec) = spec.take() {
            client.submit(&v1::RunRequest { id: id.into(), spec }).await.unwrap();
        }
        let Ok(events) = client.events(id, after).await else { continue };
        let mut events = Box::pin(events);
        // The stream breaks when the service restarts onto the candidate; attach again.
        while let Some(Ok(event)) = events.next().await {
            let outcome = matches!(event.event, v1::Event::Outcome(_));
            log.push(event);
            if outcome {
                return log;
            }
        }
    }
}

fn outcome(log: &[v1::RunEvent]) -> v1::Outcome {
    match log.last().map(|e| e.event.clone()) {
        Some(v1::Event::Outcome(outcome)) => outcome,
        other => panic!("no outcome: {other:?} in {log:?}"),
    }
}

fn spec(runtime: (&str, &str), tensorfs: (&str, &str)) -> v1::RunSpec {
    let payload = serde_json::json!({
        "runtime": runtime.0, "tensorfs": tensorfs.0, "note": "ignored",
    });
    let input = |field: &str, digest: &str| v1::InputFile {
        field: field.into(),
        digest: digest.into(),
        ..Default::default()
    };
    v1::RunSpec {
        kind: v1::RunKind::Update,
        payload: v1::Json::parse(payload.to_string()).ok(),
        inputs: vec![input("runtime", runtime.1), input("tensorfs", tensorfs.1)],
        ..Default::default()
    }
}

fn service_cmdline(parent: u32) -> String {
    let children = std::fs::read_to_string(format!("/proc/{parent}/task/{parent}/children"))
        .unwrap_or_default();
    children
        .split_whitespace()
        .filter_map(|pid| std::fs::read(format!("/proc/{pid}/cmdline")).ok())
        .map(|raw| String::from_utf8_lossy(&raw).replace('\0', " "))
        .find(|line| line.contains("tensord") || line.contains("agent/current"))
        .unwrap_or_default()
}

/// An update installs a Runtime/TensorFS pair and restarts the service on the machine's own
/// binary: a `tensord` a Runtime wheel still bundles is never run (the binary moves only with the
/// `cozy` release). A pair that does not resolve ends the update failed and changes nothing.
#[tokio::test]
async fn an_update_run_installs_its_pair_in_place_and_an_unresolvable_one_ends_failed() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/machine-update-run")
        .join(std::process::id().to_string());
    let _ = std::fs::remove_dir_all(&root);
    let image = root.join("opt/cozy/wheels");
    std::fs::create_dir_all(&image).unwrap();
    for (name, bytes) in [
        wheel("cozy_runtime", "0.18.102", None),
        wheel("tensorfs", "0.3.93", None),
    ] {
        std::fs::write(image.join(name), bytes).unwrap();
    }
    // A tensord this machine must never become: run, it would end the service.
    let bundled = b"#!/bin/sh\n[ \"$1\" = version ] && echo '{\"name\":\"tensord\"}' && exit 0\nexit 3\n";
    let (runtime, runtime_bytes) = wheel("cozy_runtime", "0.18.103", Some(bundled));
    let (tensorfs, tensorfs_bytes) = wheel("tensorfs", "0.3.94", None);
    let (failing, failing_bytes) =
        requiring("cozy_runtime", "0.18.104", None, "Requires-Dist: cozy-no-such-distribution==0\n");
    // The running version rebuilt: other bytes, the same label.
    let (rebuilt, rebuilt_bytes) = wheel("cozy_runtime", "0.18.103", Some(b"rebuilt"));
    let (runtime_digest, tensorfs_digest, failing_digest, rebuilt_digest) = (
        hold(&root, &runtime_bytes),
        hold(&root, &tensorfs_bytes),
        hold(&root, &failing_bytes),
        hold(&root, &rebuilt_bytes),
    );
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut machine = launch(&root, port, Some([7; 32]), false);
    let before = status(&root, port).await;
    assert_eq!(
        (before.runtime.as_str(), before.tensorfs.as_str()),
        ("0.18.102", "0.3.93")
    );
    assert!(before.capabilities.contains(&"update/1".to_string()));

    // A run-scope cap cannot update; a machine cap can.
    let refused = client(&root, port, &cap("")).await;
    let request = v1::RunRequest {
        id: "up-1".into(),
        spec: spec(
            (&runtime, &runtime_digest),
            (&tensorfs, &tensorfs_digest),
        ),
    };
    let denied = refused.submit(&request).await.unwrap_err();
    assert_eq!(denied.status(), Some(403));

    let log = update(
        &root,
        port,
        "up-1",
        Some(spec(
            (&runtime, &runtime_digest),
            (&tensorfs, &tensorfs_digest),
        )),
    )
    .await;
    let done = outcome(&log);
    assert_eq!(done.status, "succeeded", "{log:?}");
    let result: serde_json::Value = serde_json::from_str(done.result.as_ref().unwrap().get()).unwrap();
    assert_eq!((result["to"]["runtime"].as_str(), result["to"]["tensorfs"].as_str()), (Some("0.18.103"), Some("0.3.94")));
    assert_eq!(result["to"]["runtime_sha256"], runtime_digest.trim_start_matches("sha256:"));
    let stages: Vec<_> = log
        .iter()
        .filter_map(|e| match &e.event {
            v1::Event::Progress(p) => Some(p.stage.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        stages.contains(&"installing") && stages.contains(&"starting"),
        "{stages:?}"
    );
    let after = status(&root, port).await;
    assert_eq!(
        (after.runtime.as_str(), after.tensorfs.as_str()),
        ("0.18.103", "0.3.94")
    );
    assert_eq!(
        (after.boot_id.as_str(), after.receipt.as_slice()),
        (before.boot_id.as_str(), before.receipt.as_slice())
    );
    assert!(
        !service_cmdline(machine.0.id()).contains("agent/current"),
        "the service runs the machine's own binary, never the wheel's"
    );

    // The same id again attaches to the finished update; nothing runs twice.
    let again = update(
        &root,
        port,
        "up-1",
        Some(spec(
            (&runtime, &runtime_digest),
            (&tensorfs, &tensorfs_digest),
        )),
    )
    .await;
    assert_eq!(outcome(&again), done);

    // A pair that does not resolve together: FAILED before anything changes, still serving.
    let log = update(
        &root,
        port,
        "up-2",
        Some(spec(
            (&failing, &failing_digest),
            (&tensorfs, &tensorfs_digest),
        )),
    )
    .await;
    let failed = outcome(&log);
    assert_eq!(failed.status, "failed", "{log:?}");
    assert_eq!(failed.reason.unwrap().code, "update_failed");
    assert_eq!(status(&root, port).await.runtime, "0.18.103");
    assert!(
        machine.0.try_wait().unwrap().is_none(),
        "the machine keeps running after a failed update"
    );

    // A wheel rebuilt at the running version is installed: compared by bytes, never by label.
    // The very wheel the machine runs then installs nothing.
    let stages = |log: &[v1::RunEvent]| -> Vec<String> {
        log.iter()
            .filter_map(|e| match &e.event {
                v1::Event::Progress(p) => Some(p.stage.clone()),
                _ => None,
            })
            .collect()
    };
    for (id, installs) in [("up-3", true), ("up-4", false)] {
        let log = update(&root, port, id, Some(spec((&rebuilt, &rebuilt_digest), (&tensorfs, &tensorfs_digest)))).await;
        let done = outcome(&log);
        assert_eq!(done.status, "succeeded", "{log:?}");
        let result: serde_json::Value = serde_json::from_str(done.result.as_ref().unwrap().get()).unwrap();
        assert_eq!((result["from"]["runtime"].as_str(), result["to"]["runtime"].as_str()), (Some("0.18.103"), Some("0.18.103")));
        assert_eq!(result["to"]["runtime_sha256"], rebuilt_digest.trim_start_matches("sha256:"));
        assert_eq!(result["from"] != result["to"], installs, "{id}: {result}");
        assert_eq!(stages(&log).contains(&"starting".to_string()), installs, "{id}: {log:?}");
    }

    // Stopped and launched again: `cozy machine disable` / `enable` on this computer (a persistent
    // machine), then a rental's pod restarted. Each launch proves itself under a new key and keeps
    // the installed pair.
    for (persistent, key) in [(true, [8; 32]), (true, [9; 32]), (false, [10; 32])] {
        stop(&mut machine);
        std::fs::remove_file(root.join("run/cozy/bootstrap/readiness-envelope.json")).unwrap();
        machine = launch(&root, port, Some(key), persistent);
        let frame = status(&root, port).await;
        assert_eq!(frame.runtime, "0.18.103");
        assert!(!service_cmdline(machine.0.id()).contains("agent/current"), "the launch runs its own binary");
    }

    // A rental's launch that cannot prove its boot to its Hub ends, with a status, instead of
    // serving unready. (This computer's machine seals its readiness unsigned.)
    stop(&mut machine);
    std::fs::remove_file(root.join("run/cozy/bootstrap/readiness-envelope.json")).unwrap();
    let mut unprovable = launch(&root, port, None, false);
    let ended = unprovable.0.wait().unwrap();
    assert!(!ended.success(), "an unprovable launch ended {ended}");
    std::fs::remove_dir_all(&root).unwrap();
}

/// Starts the machine on `root` as a pod (or, persistent, as this computer's CLI) does, with this
/// launch's readiness key (none: it cannot prove readiness).
fn launch(root: &Path, port: u16, key: Option<[u8; 32]>, persistent: bool) -> Machine {
    let owner = SigningKey::from_bytes(&OWNER).verifying_key();
    // The root's own uv, as an image provides it: an update resolves its pair with it.
    let uv = std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|dir| dir.join("uv"))
        .find(|path| path.is_file())
        .expect("uv is required");
    std::fs::create_dir_all(root.join("usr/local/bin")).unwrap();
    let _ = std::os::unix::fs::symlink(&uv, root.join("usr/local/bin/uv"));
    let mut command = Command::new(env!("CARGO_BIN_EXE_tensord"));
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("COZY_MACHINE_ROOT", root)
        .env("HOME", root.join("home"))
        .env("TENSORFS_HOME", root.join("var/lib/tensorfs"))
        .env("COZY_WORKER_ID", WORKER)
        .env("COZY_WORKER_AUTH_TOKEN", URL_SAFE_NO_PAD.encode([1; 32]))
        .env("COZY_WORKER_INTERNAL_PORT", port.to_string())
        .env("COZY_LISTEN_HOST", "127.0.0.1")
        .env(
            "COZY_AUTHORIZED_KEYS",
            URL_SAFE_NO_PAD.encode(owner.as_bytes()),
        )
        .env("TENSORHUB_ORIGIN", "https://hub.invalid")
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if persistent {
        command.arg("--persistent");
    }
    if let Some(key) = key {
        command.env(
            "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL",
            URL_SAFE_NO_PAD.encode(key),
        );
    }
    Machine(command.spawn().unwrap())
}

/// Stops the machine as `cozy machine disable` does: SIGTERM to its parent, which ends cleanly.
fn stop(machine: &mut Machine) {
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(machine.0.id() as i32),
        nix::sys::signal::Signal::SIGTERM,
    )
    .unwrap();
    machine.0.wait().unwrap();
}


/// The persisted boundaries are interrupted deliberately, then a real supervisor and service
/// start over them. A partially published SDK must never be selected or committed by readiness.
#[tokio::test]
async fn startup_recovers_interrupted_publication_before_selecting_the_service() {
    use std::os::unix::fs::symlink;
    for (state, changed_links, runtime, outcome) in [
        ("installing", 0, "0.18.102", "rolled_back"),
        ("installing", 1, "0.18.102", "rolled_back"),
        ("succeeded", 1, "0.18.103", "succeeded"),
        ("rolled_back", 1, "0.18.102", "rolled_back"),
        // A published pair whose service restarted before readiness proves itself now.
        ("starting", 1, "0.18.103", "succeeded"),
        ("starting", 3, "0.18.102", "rolled_back"),
        ("starting", 4, "0.18.102", "rolled_back"),
    ] {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/machine-update-recovery")
            .join(uuid::Uuid::new_v4().to_string());
        let image = root.join("opt/cozy/wheels");
        let engine = root.join("var/lib/cozy/tensord");
        let candidate = engine.join("sdk/candidate");
        let updates = engine.join("update");
        for dir in [&image, &candidate, &updates] {
            std::fs::create_dir_all(dir).unwrap();
        }
        for (dir, version) in [(&image, "0.18.102"), (&candidate, "0.18.103")] {
            for (name, bytes) in [wheel("cozy_runtime", version, None), wheel("tensorfs", "0.3.94", None)] {
                std::fs::write(dir.join(name), bytes).unwrap();
            }
        }
        let status = serde_json::json!({
            "operation": "interrupted", "state": state,
            "from": {"runtime": "0.18.102", "tensorfs": "0.3.94"},
            "to": {"runtime": "0.18.103", "tensorfs": "0.3.94"},
        });
        let mut pending_status = status.clone();
        pending_status["state"] = "installing".into();
        std::fs::write(updates.join("pending.json"), serde_json::to_vec(&serde_json::json!({
            "status": pending_status, "sdk_before": null,
        })).unwrap()).unwrap();
        std::fs::write(updates.join("status.json"), serde_json::to_vec(&status).unwrap()).unwrap();
        if changed_links >= 1 && changed_links != 4 {
            let selected = if changed_links == 3 { engine.join("sdk/missing") } else { candidate.clone() };
            symlink(selected, engine.join("sdk/current")).unwrap();
        }
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let mut machine = launch(&root, port, Some([7; 32]), true);
        let frame = self::status(&root, port).await;
        assert_eq!(frame.runtime, runtime, "{state} after {changed_links} links");
        assert!(!updates.join("pending.json").exists());
        let settled: serde_json::Value = serde_json::from_slice(&std::fs::read(updates.join("status.json")).unwrap()).unwrap();
        assert_eq!(settled["state"], outcome);
        if state == "installing" {
            assert_eq!(settled["error"], "machine_restarted: software publication did not finish");
        } else if changed_links >= 3 {
            assert_eq!(settled["error"], "activated SDK unavailable: no installed cozy_runtime wheel");
        }
        stop(&mut machine);
        std::fs::remove_dir_all(root).unwrap();
    }
}
