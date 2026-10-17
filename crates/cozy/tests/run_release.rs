//! Which release a run binds and runs, on the job's Tensorhub (Tensorhub's dev-build `cozy_ref`
//! job) and a real TensorD (COZY_TEST_TENSORD). Elsewhere it skips.
#![cfg(unix)]
mod common;

use common::Home;
use serde_json::Value;

/// An unnamed release is resolved once, before binding: the release the machine holds, else the
/// Hub's latest, never the card this computer read last. The run is pinned to it.
#[test]
fn a_run_binds_and_runs_the_release_it_resolved() {
    let Some(home) = Home::with_hub() else { return };
    if std::env::var_os("COZY_TEST_TENSORD").is_none() {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return;
    }
    home.sign_in("releases@cozy.test");
    let package = common::package::publish(&home, "tiny-release", "1.0.0");
    let echo = format!("{package}/echo");
    let describe = |on: &str| home.cozy(["run", &echo, "--describe", "--on", on]);

    // 1.0.0's card is read and kept.
    let out = describe("any");
    assert!(out.stdout.contains("@1.0.0 (the Hub's latest)"), "{out:?}");

    // 1.1.0 adds a kind: the Hub's latest now, though 1.0.0's card is the one kept.
    let source = common::package::project(&home, "tiny-release").join("tiny_release/__init__.py");
    let text = std::fs::read_to_string(&source).unwrap();
    let widened = text.replace(
        r#"Literal["audio", "scene"]"#,
        r#"Literal["audio", "scene", "video"]"#,
    );
    std::fs::write(&source, widened).unwrap();
    common::package::publish(&home, "tiny-release", "1.1.0");
    let json = home.json(["run", &echo, "--describe", "--json", "--on", "any"]);
    assert_eq!(
        (json["release"].as_str(), json["release_from"].as_str()),
        (Some("1.1.0"), Some("the Hub's latest")),
        "{json}"
    );

    // A machine holding 1.0.0 runs 1.0.0, and the description says so.
    let _machine = common::tensord::added(&home, "m1");
    let pinned = format!("{package}@1.0.0");
    let out = home.cozy(["package", "install", &pinned, "--on", "m1", "--json"]);
    assert_eq!(out.code, 0, "{out:?}");
    let out = describe("m1");
    assert!(out.stdout.contains("@1.0.0 (installed on m1)"), "{out:?}");
    let out = home.cozy(["--json", "run", &echo, "kind=video", "--on", "m1"]);
    assert_ne!(out.code, 0, "1.0.0 knows no video: {out:?}");

    // Unnamed and not held: the Hub's latest, bound and run as 1.1.0.
    let out = home.cozy(["--json", "run", &echo, "kind=video", "--on", "any"]);
    assert_eq!(out.code, 0, "{out:?}");
    let end: Value = out
        .stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|event| event["type"] == "run.completed")
        .unwrap_or_else(|| panic!("no run.completed: {out:?}"));
    let run = &end["run"];
    assert_eq!(run["result"]["kind"], "video", "{run}");
    assert!(
        run["target"]
            .as_str()
            .unwrap_or_default()
            .ends_with("@1.1.0"),
        "the run was not pinned to 1.1.0: {run}"
    );
}
