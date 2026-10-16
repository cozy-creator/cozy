//! The importer on a synthetic Go home: what lands in Calcifer's real store, the keys and the
//! config, what is archived, and what stays.
use std::{path::Path, time::Duration};

use calcifer::store::{State, Store};
use cozy_platform::{lock::LockFile, paths::Home};
use cozy_proto::{
    calcifer::{Holder, MachineKind, MachineState, RunList, Status},
    config::{Config, Hub, Loaded},
};
use migrate::{
    Args, Fate,
    testing::{self, Expected},
};

fn args(from: &Path, into: &Path) -> Args {
    Args {
        from: from.into(),
        into: Some(into.into()),
        dry_run: false,
        delete_caches: false,
        json: false,
    }
}

#[cfg(unix)]
fn mode(path: &Path) -> u32 {
    std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(path).unwrap().permissions()) & 0o777
}

#[test]
fn an_import_in_place_records_the_go_home_and_archives_its_state() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join(".cozy");
    let outputs = testing::build(&root);
    let home = Home::at(&root);

    let dry = migrate::run(&Args {
        dry_run: true,
        ..args(&root, &root)
    })
    .unwrap();
    let c = &dry.counts;
    assert_eq!(
        (c.runs, c.history, c.unsettled_accepted, c.unsettled_sent),
        (Expected::RUNS, Expected::HISTORY, 1, 1)
    );
    assert_eq!(
        (c.completed, c.failed, c.canceled),
        (Expected::COMPLETED, Expected::FAILED, Expected::CANCELED)
    );
    assert_eq!((c.outputs, c.outputs_without_path), (Expected::OUTPUTS, 1));
    assert_eq!((c.hubs, c.current_hub.as_deref()), (1, Some("local")));
    assert_eq!(
        (c.device_keys, c.sessions, c.owner_key, c.rental_keys),
        (2, 1, true, 2)
    );
    assert_eq!(
        (c.local_machine, c.rentals_ready, c.rentals_booting),
        (true, 1, 1)
    );
    assert_eq!(c.history_machines, Expected::HISTORY_MACHINES);
    let skipped: Vec<&str> = dry.skipped.iter().map(|s| s.what.as_str()).collect();
    for what in ["config.yaml rentals", "auth/accounts/", "rental pr-done"] {
        assert!(
            skipped.iter().any(|s| s.starts_with(what)),
            "{what} in {skipped:?}"
        );
    }
    assert_eq!(
        skipped.iter().filter(|s| s.starts_with("run ")).count(),
        2,
        "{skipped:?}"
    );
    let fate = |name: &str| dry.entries.iter().find(|e| e.name == name).map(|e| e.fate);
    assert_eq!(fate("outputs"), Some(Fate::Kept));
    assert_eq!(fate("installs"), Some(Fate::Cache));
    assert_eq!(fate("machine"), Some(Fate::Machine));
    for archived in [
        "creator.sqlite",
        "auth",
        "rentals",
        "config.yaml",
        "calcifer",
        "dev",
        "records.db",
    ] {
        assert_eq!(fate(archived), Some(Fate::Archived), "{archived}");
    }
    assert!(
        !home.config().exists() && root.join("creator.sqlite").exists(),
        "a dry run changes nothing"
    );

    let report = migrate::run(&args(&root, &root)).unwrap();
    let archive = report.archive.unwrap();
    assert!(archive.starts_with(home.backups()));
    #[cfg(unix)]
    assert_eq!(mode(&archive), 0o600);
    let mut names = vec![];
    for entry in tar::Archive::new(std::fs::File::open(&archive).unwrap())
        .entries()
        .unwrap()
    {
        names.push(
            entry
                .unwrap()
                .path()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
        );
    }
    for held in [
        "creator.sqlite",
        "config.yaml",
        "calcifer",
        "dev/comfy/state",
        "rentals/pr-live.media-token",
    ] {
        assert!(names.iter().any(|n| n == held), "{held} in the archive");
    }
    for gone in [
        "creator.sqlite",
        "config.yaml",
        "auth",
        "rentals",
        "dev",
        "records.db",
        "calcifer.log",
    ] {
        assert!(!root.join(gone).exists(), "{gone} is archived and removed");
    }
    for kept in outputs.iter().chain([
        &root.join("scripts/s/main.py"),
        &root.join("backups/old.tar"),
    ]) {
        assert!(kept.is_file(), "{} is kept", kept.display());
    }
    assert!(
        root.join("machine/root/state/journal").is_file()
            && root.join("machine/owner.pem").is_file()
    );
    assert!(
        root.join("installs").is_dir(),
        "caches wait for --delete-caches"
    );

    let store = Store::open(&home.calcifer_db()).unwrap();
    let owed = store.run(testing::OWED).unwrap().unwrap();
    assert_eq!(
        (owed.state, owed.cursor, owed.machine.as_deref()),
        (
            State::Accepted,
            testing::OWED_CURSOR,
            Some(testing::LIVE_NAME)
        )
    );
    let sent = store.run("req-sent").unwrap().unwrap();
    assert_eq!(
        (sent.state, sent.spec.len(), sent.pinned),
        (State::Sent, 0, true)
    );
    let done = store.run("req-done").unwrap().unwrap();
    assert_eq!(
        (done.number, done.state, done.machine.as_deref()),
        (1, State::Settled, Some("local"))
    );
    let record = store.record(&done).unwrap();
    assert_eq!(
        (record.status, record.execution_ms),
        (Status::Completed, Some(1234))
    );
    let paths: Vec<_> = record
        .output
        .iter()
        .map(|o| o.path.clone().unwrap())
        .collect();
    assert_eq!(paths, vec![outputs[0].clone(), outputs[1].clone()]);
    assert_eq!(record.output[0].rev, 2, "the newest revision");
    let failed = store
        .record(&store.run("req-failed").unwrap().unwrap())
        .unwrap();
    assert_eq!(failed.reason.unwrap().code, "oom");
    assert_eq!(failed.machine.as_deref(), Some("oldbox"));
    let lost = store
        .record(&store.run("req-lost").unwrap().unwrap())
        .unwrap();
    assert_eq!(
        (lost.status, lost.reason.unwrap().code.as_str()),
        (Status::Failed, "not_migrated")
    );
    let install = store.run("req-install").unwrap().unwrap();
    assert_eq!(install.target, "fidika/sdxl@1.2.0");

    let machines = store.machines().unwrap();
    let machine = |name: &str| machines.iter().find(|m| m.name == name).unwrap();
    let local = machine("local");
    let endpoint = local.endpoint.as_ref().unwrap();
    assert_eq!(
        (local.kind, local.state),
        (MachineKind::Local, MachineState::Ready)
    );
    assert_eq!(
        (endpoint.address.as_str(), endpoint.pin),
        ("127.0.0.1:47001", testing::pin(b"local leaf"))
    );
    let live = machine(testing::LIVE_NAME);
    assert_eq!(
        (live.state, live.rental_id.as_deref()),
        (MachineState::Ready, Some(testing::LIVE))
    );
    assert_eq!(
        live.endpoint.as_ref().unwrap().pin,
        testing::pin(b"live leaf")
    );
    assert!(live.key.is_file());
    let booting = machine(testing::BOOTING_NAME);
    assert_eq!(
        (booting.state, booting.endpoint.is_none()),
        (MachineState::Booting, true)
    );
    assert_eq!(
        machines
            .iter()
            .filter(|m| m.state == MachineState::Ended)
            .count(),
        Expected::HISTORY_MACHINES
    );
    assert_eq!(
        machine(&testing::DEV_ENDPOINT[..32]).kind,
        MachineKind::Added
    );

    let (config, warnings) = Config::parse(
        &std::fs::read_to_string(home.config()).unwrap(),
        "config.toml",
    )
    .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(config.current().unwrap().origin, testing::LOCAL_HUB);
    assert_eq!(config.huggingface_token.unwrap().expose(), "hf_fake");
    let local_hub = Hub {
        name: None,
        origin: testing::LOCAL_HUB.into(),
    };
    let tensorhub = Hub {
        name: None,
        origin: "https://tensorhub.com".into(),
    };
    let key = hub_client::DeviceKey::load(&local_hub.device_key(&home)).unwrap();
    assert_eq!(key.id, "dk-local");
    assert!(
        local_hub.bearer(&home).is_file() && !tensorhub.bearer(&home).exists(),
        "only an unexpired session"
    );
    #[cfg(unix)]
    for secret in [
        home.config(),
        local_hub.device_key(&home),
        local_hub.bearer(&home),
        live.key.clone(),
        home.keys().join("owner.pem"),
    ] {
        assert_eq!(mode(&secret), 0o600, "{}", secret.display());
    }

    let again = migrate::run(&args(&root, &root)).err().unwrap();
    assert_eq!(again.code, "already_migrated");
    let cleaned = migrate::run(&Args {
        delete_caches: true,
        ..args(&root, &root)
    })
    .unwrap();
    assert_eq!(
        cleaned.deleted,
        vec!["installs".to_owned(), "publications".to_owned()]
    );
    assert!(!root.join("installs").exists() && outputs.iter().all(|o| o.is_file()));
}

/// An import elsewhere (a copy's) records the same, and leaves user content and its paths alone.
#[test]
fn an_import_into_another_home_leaves_user_content_in_place() {
    let dir = tempfile::tempdir().unwrap();
    let (from, into) = (dir.path().join("go"), dir.path().join("new"));
    let outputs = testing::build(&from);
    let report = migrate::run(&args(&from, &into)).unwrap();
    assert!(
        report
            .archive
            .unwrap()
            .starts_with(Home::at(&into).backups())
    );
    let store = Store::open(&Home::at(&into).calcifer_db()).unwrap();
    let done = store
        .record(&store.run("req-done").unwrap().unwrap())
        .unwrap();
    assert_eq!(done.output[0].path.as_deref(), Some(outputs[0].as_path()));
    assert!(outputs.iter().all(|o| o.is_file()));
    assert!(from.join("scripts/s/main.py").is_file() && from.join("installs").is_dir());
    assert!(
        !from.join("creator.sqlite").exists() && from.join("calcifer").symlink_metadata().is_err()
    );
}

/// The real Calcifer serves the imported store: the history under Go's numbers, and the runs still
/// owed, waiting on their machines.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn calcifer_serves_the_imported_runs() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join(".cozy");
    testing::build(&root);
    migrate::run(&args(&root, &root)).unwrap();
    let loaded = Loaded {
        config: Config::default(),
        home: Home::at(&root),
        user_home: dir.path().to_owned(),
        warnings: vec![],
    };
    let lock = LockFile::new(loaded.home.calcifer_lock());
    let task = tokio::spawn(async move { calcifer::serve::serve(&loaded).await });
    // A hang bound for a Calcifer that never starts; the product waits without one.
    let holder: Holder = tokio::time::timeout(Duration::from_secs(120), async {
        loop {
            if let Some(holder) = lock
                .holder()
                .unwrap()
                .and_then(|b| serde_json::from_slice(&b).ok())
            {
                break holder;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    let client = reqwest::Client::new();
    let url = |path: &str| format!("http://{}{path}", holder.address);
    let list: RunList = client
        .get(url("/v1/runs?limit=100"))
        .bearer_auth(&holder.token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let numbers: Vec<u64> = list.runs.iter().map(|r| r.number).collect();
    assert_eq!(
        numbers,
        (1..=Expected::RUNS as u64).rev().collect::<Vec<_>>()
    );
    let owed = list
        .runs
        .iter()
        .find(|r| r.id.as_str() == testing::OWED)
        .unwrap();
    assert_eq!(
        (owed.status, owed.machine.as_deref()),
        (Status::InProgress, Some(testing::LIVE_NAME))
    );
    let done = list.runs.iter().find(|r| r.number == 1).unwrap();
    assert_eq!(done.status, Status::Completed);
    let down = client
        .post(url("/v1/down"))
        .bearer_auth(&holder.token)
        .send()
        .await
        .unwrap();
    assert!(down.status().is_success());
    drop(client);
    task.await.unwrap().unwrap();
}
