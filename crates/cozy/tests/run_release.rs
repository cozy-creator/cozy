//! Which release a run binds and runs, on the job's Tensorhub (Tensorhub's dev-build `cozy_ref`
//! job) and a real TensorD (COZY_TEST_TENSORD). Elsewhere it skips.
#![cfg(unix)]
mod common;

use common::Home;
use serde_json::Value;

/// An unnamed release is the Hub's newest published one, resolved once before binding, wherever
/// the run goes: a machine holding an older release installs the newer one first. The run is
/// pinned to it.
#[test]
fn an_unnamed_release_is_the_hubs_newest_even_on_a_machine_holding_an_older_one() {
    let Some(home) = Home::with_hub() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    home.sign_in("releases@cozy.test");
    let package = common::package::publish(&home, "tiny-release", "1.0.0");
    let echo = format!("{package}/echo");

    // N is the Hub's latest; its card is read and kept, and the machine installs it.
    let out = home.cozy(["run", &echo, "--describe", "--on", "m1"]);
    assert!(
        out.stdout.contains("release: 1.0.0 (the Hub's latest)"),
        "{out:?}"
    );
    let _machine = common::tensord::added(&home, "m1");
    let pinned = format!("{package}@1.0.0");
    let out = home.cozy(["package", "install", &pinned, "--on", "m1", "--json"]);
    assert_eq!(out.code, 0, "{out:?}");

    // N+1 adds a kind. It is the Hub's latest, though the machine holds N and N's card is kept.
    let source = common::package::project(&home, "tiny-release").join("tiny_release/__init__.py");
    let text = std::fs::read_to_string(&source).unwrap();
    let widened = text.replace(
        r#"Literal["audio", "scene"]"#,
        r#"Literal["audio", "scene", "video"]"#,
    );
    std::fs::write(&source, widened).unwrap();
    common::package::publish(&home, "tiny-release", "1.1.0");
    let out = home.cozy(["run", &echo, "--describe", "--on", "m1"]);
    assert!(
        out.stdout.contains("release: 1.1.0 (the Hub's latest)"),
        "{out:?}"
    );
    let json = home.json(["run", &echo, "--describe", "--json", "--on", "m1"]);
    assert_eq!(
        (json["release"].as_str(), json["release_from"].as_str()),
        (Some("1.1.0"), Some("the Hub's latest")),
        "{json}"
    );

    // The unnamed run binds N+1's value, and the machine installs and runs N+1.
    let out = home.cozy(["--json", "run", &echo, "kind=video", "--on", "m1"]);
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
    let held = home.json(["package", "list", "--on", "m1", "--json"]);
    assert!(
        held["packages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["package"] == package.as_str() && p["release"] == "1.1.0"),
        "{held}"
    );

    // A named release still pins: N refuses N+1's value.
    let named = format!("{echo}@1.0.0");
    let out = home.cozy(["--json", "run", &named, "kind=video", "--on", "m1"]);
    assert_ne!(out.code, 0, "1.0.0 knows no video: {out:?}");
}
