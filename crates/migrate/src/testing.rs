//! A synthetic Go CLI home for tests, built from the Go schema with fake keys and certificates:
//! one of each kind of run, rental and file the importer sorts. Never real data.
use std::path::{Path, PathBuf};

use base64::{
    Engine as _,
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
};
use cozy_proto::machine as v1;
use ed25519_dalek::{SigningKey, pkcs8::EncodePrivateKey as _};
use prost::Message as _;
use rusqlite::{Connection, params};
use sha2::{Digest as _, Sha256};

pub const LOCAL_HUB: &str = "http://127.0.0.1:8819";
/// The ready rental's id, the Hub's name for it, and the run it still owes.
pub const LIVE: &str = "pr-live";
pub const LIVE_NAME: &str = "grimm";
pub const OWED: &str = "req-owed";
pub const OWED_CURSOR: u64 = 42;
pub const OWED_MACHINE_NUMBER: u64 = 7;
pub const BOOTING_NAME: &str = "mercy";
pub const DEV_ENDPOINT: &str = "endpoint-0123456789abcdef0123456789";

/// What the importer finds in the home `build` writes.
pub struct Expected;

impl Expected {
    pub const RUNS: usize = 10;
    pub const HISTORY: usize = 8;
    pub const COMPLETED: usize = 4;
    pub const FAILED: usize = 3;
    pub const CANCELED: usize = 1;
    pub const OUTPUTS: usize = 6;
    pub const HISTORY_MACHINES: usize = 4;
}

pub fn signing_key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn pem(seed: u8) -> String {
    signing_key(seed)
        .to_pkcs8_pem(Default::default())
        .unwrap()
        .to_string()
}

/// A PEM "certificate" whose DER is `body`: the importer pins its sha256 and reads nothing else.
pub fn certificate(body: &[u8]) -> String {
    format!(
        "-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n",
        STANDARD.encode(body)
    )
}

pub fn pin(body: &[u8]) -> [u8; 32] {
    Sha256::digest(body).into()
}

fn write(path: &Path, bytes: impl AsRef<[u8]>) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

/// Writes the Go home at `root` and answers its output files, by the runs that wrote them.
pub fn build(root: &Path) -> Vec<PathBuf> {
    write(
        &root.join("config.yaml"),
        format!(
            "tensorhub_url: local\nhubs:\n  local: {LOCAL_HUB}/\nhuggingface_token: hf_fake\nrentals:\n  development: true\n"
        ),
    );
    let device = |hub: &str, seed: u8, id: &str| {
        serde_json::json!({"version": 1, "hub": hub, "email": "dev@example.test", "device_key_id": id,
            "private_key": URL_SAFE_NO_PAD.encode([seed; 32])})
    };
    let auth = root.join("auth");
    let named = |origin: &str| hex::encode(Sha256::digest(origin.as_bytes()));
    write(
        &auth.join(format!("{}.json", named(LOCAL_HUB))),
        device(LOCAL_HUB, 1, "dk-local").to_string(),
    );
    write(
        &auth.join(format!("{}.session.json", named(LOCAL_HUB))),
        serde_json::json!({"access_token": "fake-bearer", "expires_at": "2999-01-01T00:00:00Z",
            "device_key_id": "dk-local", "email": "dev@example.test"})
        .to_string(),
    );
    let tensorhub = "https://tensorhub.com";
    write(
        &auth.join(format!("{}.json", named(tensorhub))),
        device(tensorhub, 2, "dk-th").to_string(),
    );
    write(
        &auth.join(format!("{}.session.json", named(tensorhub))),
        serde_json::json!({"access_token": "stale", "expires_at": "2020-01-01T00:00:00Z", "device_key_id": "dk-th",
            "email": "dev@example.test"})
        .to_string(),
    );
    write(&auth.join("accounts/abc.json"), r#"{"name":"dev"}"#);

    let machine = root.join("machine");
    write(&machine.join("owner.pem"), pem(3));
    write(&machine.join("leaf.pem"), certificate(b"local leaf"));
    write(
        &machine.join("agent.json"),
        r#"{"pid":1,"worker_id":"machine-local","worker_port":47001,"media_port":0,"receipt_key":"x"}"#,
    );
    write(&machine.join("machine-id"), "machine-local\n");
    write(
        &machine.join("root/state/journal"),
        "the running TensorD's own state",
    );

    let rentals = root.join("rentals");
    write(
        &rentals.join(format!("{LIVE}.pem")),
        certificate(b"live leaf"),
    );
    write(&rentals.join(format!("{LIVE}.creator.pem")), pem(4));
    write(&rentals.join(format!("{LIVE}.media-token")), "fake-media");
    let operation = "op-boot";
    let pending = hex::encode(Sha256::digest(operation.as_bytes()));
    write(
        &rentals.join(format!("pending-{pending}.creator.pem")),
        pem(5),
    );
    write(&rentals.join("pr-gone/ssh-known-hosts"), "");

    let outputs = root.join("outputs");
    let files: Vec<PathBuf> = [
        "fidika/sdxl/1/image.png",
        "fidika/sdxl/1/frames-1.png",
        "old/1/a.png",
        "old/1/b.png",
        "fidika/sdxl/6/image.png",
    ]
    .iter()
    .map(|f| outputs.join(f))
    .collect();
    for file in &files {
        write(file, b"fake pixels");
    }
    write(&root.join("scripts/s/main.py"), "print('hi')");
    write(&root.join("backups/old.tar"), "an owner's backup");
    write(&root.join("installs/x/lib.py"), "cache");
    write(&root.join("publications/_job-req-1/p.bin"), "staging");
    write(&root.join("dev/comfy/state"), "dev");
    write(&root.join("calcifer.log"), "log");
    write(&root.join("records.db"), "");
    // Go's link to its binary, where this layout keeps Calcifer's directory.
    #[cfg(unix)]
    std::os::unix::fs::symlink("/nonexistent/go/cozy", root.join("calcifer")).unwrap();
    #[cfg(not(unix))]
    write(&root.join("calcifer"), "the Go binary");

    let db = Connection::open(root.join("creator.sqlite")).unwrap();
    // Go's tables reference others the importer never reads (`installs`, `attempts`).
    db.pragma_update(None, "foreign_keys", "OFF").unwrap();
    db.execute_batch(include_str!("testing/go_schema.sql"))
        .unwrap();
    let rental = |id: &str,
                  asked: &str,
                  name: &str,
                  state: &str,
                  hub: &str,
                  address: &str,
                  worker: &str| {
        db.execute(
            "INSERT INTO rentals (id, machine_name, stable_machine_name, accelerator_model, hourly_rate_usd_micros,
                 address, cert_path, state, hub, rented_at, expected_worker_id)
             VALUES (?1, ?2, ?3, 'L4', 1, ?4, '', ?5, ?6, ?7, ?8)",
            params![id, asked, name, address, state, hub, format!("2026-10-16T0{}:00:00Z", id.len() % 10), worker],
        )
        .unwrap();
    };
    rental(
        LIVE,
        "lease-1",
        LIVE_NAME,
        "ready",
        tensorhub,
        "127.0.0.1:1",
        "worker-live",
    );
    rental(
        "pr-boot",
        "lease-2",
        BOOTING_NAME,
        "booting",
        LOCAL_HUB,
        "",
        "",
    );
    rental("pr-done", "lease-3", "done", "released", tensorhub, "", "");
    db.execute(
        "INSERT INTO rental_operations (operation_key, request_digest, request_body, hub, reason, hourly_rate_usd_micros,
             rental_id, state, created_at, updated_at) VALUES (?1, '', x'', ?2, '', 1, 'pr-boot', 'acquiring', '', '')",
        params![operation, LOCAL_HUB],
    )
    .unwrap();

    let path = |i: usize| files[i].to_string_lossy().into_owned();
    let mut minute = 0;
    let mut request = |id: &str,
                       kind: &str,
                       state: &str,
                       machine: &str,
                       worker: &str,
                       payload: &str| {
        minute += 1;
        db.execute(
            "INSERT INTO requests (id, idem_key, body_digest, package, entrypoint, plan_id, package_release, payload,
                 state, created_at, kind, hub, machine, worker)
             VALUES (?1, ?1, '', 'fidika/sdxl', ?2, '', '1.2.0', ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                id,
                if kind == "install" { "" } else { "generate" },
                payload.as_bytes(),
                state,
                format!("2026-10-16T10:{minute:02}:00Z"),
                kind,
                if worker == LIVE { tensorhub } else { LOCAL_HUB },
                machine,
                worker
            ],
        )
        .unwrap();
    };
    request("req-done", "serving", "succeeded", "", "", "{}");
    request("req-failed", "serving", "failed", "oldbox", "pr-gone", "{}");
    request("req-canceled", "serving", "canceled", "", "", "{}");
    request("req-legacy", "job", "succeeded", "", "pr-old", "{}");
    request(
        "req-install",
        "install",
        "succeeded",
        "",
        "",
        r#"{"package":"fidika/sdxl","release":"1.2.0"}"#,
    );
    request(OWED, "serving", "dispatching", LIVE_NAME, LIVE, "{}");
    request("req-sent", "serving", "queued", LIVE_NAME, LIVE, "{}");
    request("req-lost", "serving", "dispatching", "", "pr-gone", "{}");
    request("req-unsent", "serving", "queued", "", "", "{}");
    request("req-endpoint", "serving", "succeeded", "", "", "{}");

    let execution = |id: &str,
                     machine: &str,
                     receipt: Option<v1::RunState>,
                     outcome: Option<v1::Outcome>,
                     cursor: u64| {
        db.execute(
            "INSERT INTO machine_executions (request_id, machine_id, receipt, remote_cursor, outcome) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                id,
                machine,
                receipt.map(|r| r.encode_to_vec()).unwrap_or_default(),
                cursor as i64,
                outcome.map(|o| o.encode_to_vec()).unwrap_or_default()
            ],
        )
        .unwrap();
    };
    let event = |id: &str, kind: &str, payload: serde_json::Value| {
        db.execute(
            "INSERT INTO request_events (request_id, type, payload, at) VALUES (?1, ?2, ?3, '2026-10-16T11:00:00.5Z')",
            params![id, kind, payload.to_string()],
        )
        .unwrap();
    };
    let state = |number: u64, state: &str| {
        Some(v1::RunState {
            number,
            state: state.into(),
            ..Default::default()
        })
    };
    let outcome = |status: &str| {
        Some(v1::Outcome {
            status: status.into(),
            execution_ms: 1234,
            ..Default::default()
        })
    };
    let accepted = |id: &str, machine: &str| {
        event(
            id,
            "machine.api_v1_sent",
            serde_json::json!({"machine": machine}),
        );
        event(
            id,
            "machine.api_v1",
            serde_json::json!({"number": 1, "state": "queued"}),
        );
    };
    let delta = |id: &str, output: &str, index: u32, rev: u64, path: Option<String>| {
        event(
            id,
            "output_item.delta",
            serde_json::json!({"output": output, "index": index, "rev": rev, "length": 11,
            "digest": format!("sha256:{}", hex::encode([rev as u8; 32])), "media_type": "image/png", "path": path}),
        );
    };

    execution(
        "req-done",
        "local",
        state(1, "succeeded"),
        outcome("succeeded"),
        9,
    );
    accepted("req-done", "local");
    delta("req-done", "image", 0, 1, Some(path(0)));
    delta("req-done", "image", 0, 2, Some(path(0)));
    delta("req-done", "frames", 0, 1, Some(path(1)));
    event(
        "req-done",
        "run.completed",
        serde_json::json!({"status": "succeeded"}),
    );

    execution(
        "req-failed",
        "pr-gone",
        state(2, "failed"),
        outcome("failed"),
        3,
    );
    accepted("req-failed", "pr-gone");
    event(
        "req-failed",
        "run.failed",
        serde_json::json!({"error_code": "oom", "error": "out of memory"}),
    );

    execution("req-canceled", "local", None, None, 0);
    event(
        "req-canceled",
        "run.canceled",
        serde_json::json!({"error": "canceled by you"}),
    );

    db.execute(
        "INSERT INTO request_output_exports (request_id, directory, outputs, state, published_paths, updated_at)
         VALUES ('req-legacy', ?1, '[]', 'published', ?2, '')",
        params![outputs.join("old/1").to_string_lossy(), serde_json::json!([path(2), path(3), null]).to_string()],
    )
    .unwrap();

    execution(
        "req-install",
        "local",
        state(3, "succeeded"),
        outcome("succeeded"),
        2,
    );
    accepted("req-install", "local");

    execution(
        OWED,
        LIVE,
        state(OWED_MACHINE_NUMBER, "running"),
        None,
        OWED_CURSOR,
    );
    accepted(OWED, LIVE);
    delta(OWED, "image", 0, 1, Some(path(4)));

    execution("req-sent", LIVE, None, None, 0);
    event(
        "req-sent",
        "machine.api_v1_sent",
        serde_json::json!({"machine": LIVE}),
    );

    execution("req-lost", "pr-gone", state(4, "running"), None, 5);
    accepted("req-lost", "pr-gone");

    execution("req-unsent", "", None, None, 0);

    execution(
        "req-endpoint",
        DEV_ENDPOINT,
        state(5, "succeeded"),
        outcome("succeeded"),
        1,
    );
    accepted("req-endpoint", DEV_ENDPOINT);
    delta("req-endpoint", "log", 0, 1, None);
    files
}
