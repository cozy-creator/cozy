//! `cozy package install|list|update|remove --on` through the real binary: Calcifer, a real TensorD
//! (COZY_TEST_TENSORD, the cozy-runtime wheel's) added as a machine, and a real Tensorhub (its
//! dev-build `cozy_ref` job) holding a torch-free package this test publishes.
// TensorD runs on Unix only for now, and so does its fixture (common::tensord).
#![cfg(unix)]
mod common;

use std::path::Path;

use common::{Home, Out, package, tensord};
use protocol::{
    calcifer::{Kind, On, SubmitRun},
    machine as v1,
};
use serde_json::Value;

fn cozy(home: &Home, dir: &Path, args: &[&str]) -> Out {
    common::out(home.command().current_dir(dir).args(args).output().unwrap())
}

fn json(home: &Home, dir: &Path, args: &[&str]) -> Value {
    let out = cozy(home, dir, &[&["--json"], args].concat());
    assert_eq!(out.code, 0, "{args:?}: {out:?}");
    serde_json::from_str(&out.stdout).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}

/// Runs `args` and answers what it printed. One still running after `minutes` is killed and fails
/// the test with its output so far: a hang bound for the test, never the product's.
fn bounded(home: &Home, dir: &Path, args: &[&str], minutes: u64) -> Out {
    let file = |name: &str| home.path().join(format!("bounded.{name}"));
    let mut child = home
        .command()
        .current_dir(dir)
        .args(args)
        .stdout(std::fs::File::create(file("out")).unwrap())
        .stderr(std::fs::File::create(file("err")).unwrap())
        .spawn()
        .unwrap();
    let until = std::time::Instant::now() + std::time::Duration::from_secs(minutes * 60);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if std::time::Instant::now() > until {
            child.kill().unwrap();
            child.wait().unwrap();
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    let read = |name: &str| std::fs::read_to_string(file(name)).unwrap_or_default();
    let out = Out {
        code: status.and_then(|s| s.code()).unwrap_or(-1),
        stdout: read("out"),
        stderr: read("err"),
    };
    assert!(
        status.is_some(),
        "{args:?} still ran after {minutes} min: {out:?}"
    );
    out
}

/// What a person debugging this test wants: the runs and the machine as Calcifer holds them.
fn state(home: &Home, dir: &Path) -> String {
    let runs = cozy(home, dir, &["--json", "run", "list"]);
    let machine = cozy(home, dir, &["--json", "machine", "show", "t1"]);
    format!("runs: {}\nmachine: {}", runs.stdout, machine.stdout)
}

/// Follows an install with `--json` (NDJSON) and answers its exit code and last event.
fn install(home: &Home, dir: &Path, package: &str) -> (i32, Value) {
    let out = cozy(
        home,
        dir,
        &["--json", "package", "install", package, "--on", "t1"],
    );
    let last = out
        .stdout
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str(line).ok());
    (
        out.code,
        last.unwrap_or_else(|| panic!("{out:?}\n{}", state(home, dir))),
    )
}

/// The releases of `package` machine `t1` holds, from `cozy package list --on t1`, which reads the
/// machine now.
fn held(home: &Home, dir: &Path, package: &str) -> Vec<String> {
    let listed = json(home, dir, &["package", "list", "--on", "t1"]);
    let rows = listed["packages"].as_array().unwrap();
    let held = rows.iter().filter(|r| r["package"] == package);
    held.map(|r| r["release"].as_str().unwrap().to_owned())
        .collect()
}

/// A published package installs on a machine as a run the machine resolves, shows in `list` with
/// its Hub, moves to its Hub's newest release with `update` (the older one leaving), and leaves
/// with `remove`; a package the Hub lacks is refused before any run.
#[test]
fn packages_install_update_and_remove_on_a_machine() {
    let Some(home) = Home::with_hub() else { return };
    let account = home.sign_in("installer@cozy.test");
    let _machine = tensord::added(&home, "t1");
    let package = package::publish(&home, "tiny", "0.1.0");
    assert_eq!(package, format!("{account}/tiny"));
    let dir = package::project(&home, "tiny");

    for _ in 0..2 {
        let (code, last) = install(&home, &dir, &package);
        let state = state(&home, &dir);
        assert_eq!(
            (code, last["type"].as_str()),
            (0, Some("run.completed")),
            "{last}\n{state}"
        );
        assert_eq!(held(&home, &dir, &package), ["0.1.0"], "{state}");
    }

    package::publish(&home, "tiny", "0.2.0");
    let updated = json(&home, &dir, &["package", "update", "--on", "t1"]);
    let row = &updated["packages"][0];
    let moved = (row["to"].as_str(), row["status"].as_str());
    assert_eq!(moved, (Some("0.2.0"), Some("updated")), "{updated}");
    assert_eq!(
        held(&home, &dir, &package),
        ["0.2.0"],
        "{updated}\n{}",
        state(&home, &dir)
    );
    let current = json(&home, &dir, &["package", "update", &package, "--on", "t1"]);
    assert_eq!(current["packages"][0]["status"], "current", "{current}");

    // A name the Hub lacks is refused before any run, as `cozy run` refuses it.
    let nope = format!("{account}/nope");
    let missing = cozy(
        &home,
        &dir,
        &["--json", "package", "install", &nope, "--on", "t1"],
    );
    assert_eq!(missing.code, 1, "{missing:?}");
    assert_eq!(
        missing.refusal()["code"],
        "package_not_found",
        "{missing:?}"
    );

    let removed = json(&home, &dir, &["package", "remove", &package, "--on", "t1"]);
    assert_eq!(removed["removed"][0]["release"], "0.2.0", "{removed}");
    assert!(
        held(&home, &dir, &package).is_empty(),
        "{removed}\n{}",
        state(&home, &dir)
    );
    let gone = cozy(
        &home,
        &dir,
        &["--json", "package", "remove", &package, "--on", "t1"],
    );
    assert_eq!(gone.code, 1, "{gone:?}");
    assert_eq!(gone.refusal()["code"], "package_not_installed", "{gone:?}");
}

/// An unversioned install takes the Hub's newest release, as `cozy run` does, and says the older
/// one is still held. The same ask as a v0.10.1 CLI sent it (release and catalog revision empty)
/// reads the Hub's newest each time, never the machine's first answer.
#[test]
fn an_unversioned_install_takes_the_hubs_newest_release() {
    let Some(home) = Home::with_hub() else { return };
    home.sign_in("newest@cozy.test");
    let _machine = tensord::added(&home, "t1");
    let package = package::publish(&home, "tiny", "0.1.0");
    let dir = package::project(&home, "tiny");
    let (code, first) = install(&home, &dir, &package);
    assert_eq!(
        (code, first["run"]["result"]["release"].as_str()),
        (0, Some("0.1.0")),
        "{first}"
    );

    package::publish(&home, "tiny", "0.1.1");
    let out = cozy(&home, &dir, &["package", "install", &package, "--on", "t1"]);
    assert_eq!(out.code, 0, "{out:?}");
    for said in [
        format!("{package}@0.1.1 is installed on t1"),
        format!("t1 also holds {package}@0.1.0"),
        format!("cozy package update {package} --on t1"),
    ] {
        let shown = format!("{}{}", out.stdout, out.stderr);
        assert!(shown.contains(&said), "{said:?} not in {out:?}");
    }
    let mut releases = held(&home, &dir, &package);
    releases.sort();
    assert_eq!(releases, ["0.1.0", "0.1.1"]);

    // The ask as it reached the machine before: no release, no catalog revision.
    package::publish(&home, "tiny", "0.1.2");
    let lock = platform::lock::LockFile::new(
        platform::paths::Home::at(home.root()).calcifer_lock(),
    );
    let holder: protocol::calcifer::Holder =
        serde_json::from_slice(&lock.holder().unwrap().expect("Calcifer runs")).unwrap();
    let ask = SubmitRun {
        on: On::Machine("t1".into()),
        hub: first["run"]["hub"].as_str().unwrap().into(),
        kind: Kind::Warm,
        target: package.clone(),
        spec: v1::RunSpec {
            kind: v1::RunKind::Warm,
            source: Some(v1::Source::Release(v1::Release {
                package: package.clone(),
                release: String::new(),
            })),
            ..Default::default()
        },
        inputs: vec![],
        out: None,
        warnings: vec![],
    };
    let submitted: Value = tokio::runtime::Runtime::new().unwrap().block_on(async {
        let answer = reqwest::Client::new()
            .post(format!("http://{}/v1/runs", holder.address))
            .bearer_auth(&holder.token)
            .header("Idempotency-Key", "unversioned-raw")
            .json(&ask)
            .send()
            .await
            .unwrap();
        answer.json().await.unwrap()
    });
    let number = submitted["run"]["number"].to_string();
    let watched = cozy(&home, &dir, &["--json", "run", "watch", &number]);
    let last: Value = serde_json::from_str(watched.stdout.lines().last().unwrap_or_default())
        .unwrap_or_else(|e| panic!("{e}: {watched:?}"));
    assert_eq!(
        (
            last["type"].as_str(),
            last["run"]["result"]["release"].as_str()
        ),
        (Some("run.completed"), Some("0.1.2")),
        "{submitted}\n{watched:?}"
    );
}

/// R's sequence (tensorhub 39107019882): `package update --on` right after `machine rent --detach`
/// was refused while the rental booted. Each verb sent the moment its rental is rented (the CI
/// Hub's docker provider) now waits for it: update, list and remove say so and answer once it
/// starts, and an install queues and lands.
#[test]
fn package_verbs_sent_to_a_booting_rental_wait_for_it() {
    let Some(home) = Home::with_hub() else { return };
    home.sign_in("booting@cozy.test");
    let package = package::publish(&home, "tiny", "0.1.0");
    let dir = package::project(&home, "tiny");
    // R's sequence for each verb: rent, then the verb at once, while the rental boots. The CI
    // Hub's docker provider boots one container at a time, so each rental ends before the next.
    let sent = |verb: &[&str], after: &dyn Fn(&str)| {
        let rented = json(&home, &dir, &["machine", "rent", "cpu", "--detach"]);
        assert_eq!(rented["machine"]["state"], "booting", "{rented}");
        let rental = rented["machine"]["rental_id"].as_str().unwrap().to_owned();
        eprintln!("{verb:?} --on {rental}, booting");
        let out = bounded(&home, &dir, &[verb, &["--on", &rental]].concat(), 10);
        eprintln!("{out:?}");
        after(&rental);
        let ended = json(&home, &dir, &["machine", "end", &rental]);
        assert_eq!(ended["machine"]["state"], "ended", "{ended}");
        out
    };
    let nothing = |_: &str| {};
    let update = sent(&["package", "update"], &nothing);
    let list = sent(&["package", "list", "--once"], &nothing);
    let remove = sent(&["package", "remove", &package], &nothing);
    let holds = |rental: &str| {
        let listed = json(&home, &dir, &["package", "list", "--on", rental]);
        assert_eq!(listed["packages"][0]["release"], "0.1.0", "{listed}");
    };
    let install = sent(&["--json", "package", "install", &package], &holds);
    for out in [&update, &list, &remove] {
        assert!(out.stderr.contains(" to start"), "it did not wait: {out:?}");
    }
    assert_eq!(update.code, 0, "{update:?}");
    assert!(
        update.stdout.contains("Model downloads: skipped"),
        "{update:?}"
    );
    assert_eq!(list.code, 0, "{list:?}");
    assert!(list.stdout.contains("No packages installed"), "{list:?}");
    assert_eq!(remove.code, 1, "{remove:?}");
    assert!(
        remove.stdout.contains("(package_not_installed)"),
        "{remove:?}"
    );
    let last: Option<Value> =
        (install.stdout.lines().rev()).find_map(|l| serde_json::from_str(l).ok());
    assert_eq!(
        (install.code, last.as_ref().and_then(|l| l["type"].as_str())),
        (0, Some("run.completed")),
        "{install:?}"
    );
}

/// Run 11762: a local package's empty `__init__.py` files (seedvr2 ships ten) never reached a
/// rental, and its install failed `local_source_incomplete`. Zero-byte files travel like any other:
/// a package whose subpackage has an empty `__init__.py` installs on a rental (the CI Hub's docker
/// provider) and imports it when it runs.
#[test]
fn a_local_package_with_empty_files_installs_on_a_rental_and_imports_them() {
    let Some(home) = Home::with_hub() else { return };
    home.sign_in("empty@cozy.test");
    let dir = package::project(&home, "tiny");
    with_empty_subpackage(&dir.join("tiny"));
    let src = dir.parent().unwrap();
    let rented = json(&home, src, &["machine", "rent", "cpu", "--detach"]);
    let rental = rented["machine"]["rental_id"].as_str().unwrap().to_owned();

    let install = cozy(
        &home,
        src,
        &["--json", "package", "install", "./tiny", "--on", &rental],
    );
    let last: Option<Value> =
        (install.stdout.lines().rev()).find_map(|l| serde_json::from_str(l).ok());
    assert_eq!(
        (install.code, last.as_ref().and_then(|l| l["type"].as_str())),
        (0, Some("run.completed")),
        "{install:?}"
    );
    let ran = cozy(
        &home,
        src,
        &["--json", "run", "./tiny/deep", "--on", &rental],
    );
    let last: Option<Value> = (ran.stdout.lines().rev()).find_map(|l| serde_json::from_str(l).ok());
    let last = last.unwrap_or_else(|| panic!("{ran:?}"));
    assert_eq!(last["run"]["result"]["text"], "deep", "{ran:?}");

    let ended = json(&home, src, &["machine", "end", &rental]);
    assert_eq!(ended["machine"]["state"], "ended", "{ended}");
}

/// Gives the package at `code` a subpackage two levels deep whose `__init__.py` files are empty,
/// imported when the package loads, and a function `deep` that answers from it.
fn with_empty_subpackage(code: &Path) {
    let inner = code.join("upstream/inner");
    std::fs::create_dir_all(&inner).unwrap();
    std::fs::write(code.join("upstream/__init__.py"), "").unwrap();
    std::fs::write(inner.join("__init__.py"), "").unwrap();
    std::fs::write(inner.join("value.py"), "VALUE = \"deep\"\n").unwrap();
    let init = code.join("__init__.py");
    let mut text = std::fs::read_to_string(&init).unwrap();
    text.push_str(
        "\n\nfrom .upstream.inner.value import VALUE  # noqa: E402\n\n\n\
         @app.entrypoint(accelerator=False)\n\
         def deep(payload: EchoInput) -> EchoOutput:\n    \
         \"\"\"Answers from a subpackage whose __init__.py files are empty.\"\"\"\n    \
         return EchoOutput(VALUE)\n",
    );
    std::fs::write(init, text).unwrap();
}

/// A laptop names its local Hub where it listens, while machines read that Hub at its public
/// origin and record their installations there (8819 recorded `https://cozy-e2e-6.ngrok.app:443`
/// for the Hub this computer calls `http://127.0.0.1:8819`). The install is still the configured
/// Hub's: listed under its name, and `update` reaches its object store with no sign-in at the
/// public origin.
#[test]
fn an_install_recorded_at_the_hubs_public_origin_is_the_configured_hubs() {
    let Some(home) = Home::with_hub() else { return };
    home.sign_in("aliased@cozy.test");
    let public = std::env::var("TENSORHUB_ORIGIN").unwrap();
    home.at_listen_address();
    std::fs::remove_file(home.device_key(&public)).unwrap();
    let package = package::publish(&home, "tiny", "0.1.0");
    let _machine = tensord::added(&home, "t1");
    let dir = package::project(&home, "tiny");

    let (code, last) = install(&home, &dir, &package);
    assert_eq!(
        (code, last["type"].as_str()),
        (0, Some("run.completed")),
        "{last}\n{}",
        state(&home, &dir)
    );
    let scoped = ["package", "list", "--on", "t1", "--tensorhub", "ci"];
    let listed = json(&home, &dir, &scoped);
    let rows = listed["packages"].as_array().unwrap();
    let row = rows.iter().find(|r| r["package"] == package.as_str());
    assert_eq!(
        row.map(|r| (r["release"].as_str(), r["hub"].as_str())),
        Some((Some("0.1.0"), Some("ci"))),
        "{listed}"
    );

    package::publish(&home, "tiny", "0.2.0");
    let update = [
        "package",
        "update",
        &package,
        "--on",
        "t1",
        "--tensorhub",
        "ci",
    ];
    let updated = json(&home, &dir, &update);
    let row = &updated["packages"][0];
    let moved = (row["to"].as_str(), row["status"].as_str());
    assert_eq!(
        moved,
        (Some("0.2.0"), Some("updated")),
        "{updated}\n{}",
        state(&home, &dir)
    );
    assert_eq!(held(&home, &dir, &package), ["0.2.0"], "{updated}");
}
