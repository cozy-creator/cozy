//! Product promises of `cozy run`, through the real binary: the target grammar, a package's
//! functions, `--describe` and binding against real release cards from tensorhub.com.
mod common;

use common::Home;
use serde_json::Value;

const CARDS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../cozy-cli/tests/fixtures/cards"
);

/// Points this home at a Hub that never answers and keeps a real release card for it, as a Hub
/// read would: every command after reads no Hub.
fn kept(home: &Home, card: &str, package: &str) {
    home.json(["hub", "add", "dead", "http://127.0.0.1:1", "--json"]);
    home.json(["hub", "use", "dead", "--json"]);
    let dir = home.root().join("packages/127.0.0.1_1").join(package);
    std::fs::create_dir_all(&dir).unwrap();
    let text = std::fs::read_to_string(format!("{CARDS}/{card}.json")).unwrap();
    let doc: Value = serde_json::from_str(&text).unwrap();
    let release = doc["release"]["release"].as_str().unwrap();
    std::fs::write(dir.join(format!("{release}.json")), &text).unwrap();
    std::fs::write(dir.join("newest"), release).unwrap();
}

#[test]
fn a_package_lists_its_functions_and_describes_each() {
    let Some(home) = Home::new() else { return };
    kept(&home, "sdxl-2.10.0", "fidika/sdxl");

    let listed = home.json(["run", "Fidika/SDXL", "--json"]);
    assert_eq!(listed["release"], "2.10.0");
    let names: Vec<&str> = listed["functions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["function"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["fp8", "generate", "int8"]);
    let human = home.cozy(["run", "fidika/sdxl"]);
    assert_eq!(human.code, 0, "{human:?}");
    assert!(
        human.stdout.contains("generate")
            && human
                .stdout
                .contains("next: cozy run fidika/sdxl/fp8 --describe"),
        "{human:?}"
    );

    let contract = home.cozy(["run", "fidika/sdxl/generate", "--describe"]);
    assert_eq!(contract.code, 0, "{contract:?}");
    assert!(
        contract
            .stdout
            .contains("steps: int (>=1, <=50) (default = 20)"),
        "{contract:?}"
    );
    assert!(
        contract
            .stdout
            .contains("usage: cozy run fidika/sdxl/generate prompt=<str>"),
        "{contract:?}"
    );
    let request = home.json(["run", "fidika/sdxl/fp8", "--describe", "--json"]);
    assert_eq!(request["accelerator"], false);
    assert_eq!(request["fields"][0]["name"], "max_relative_frobenius");

    let unknown = home.cozy(["run", "fidika/sdxl/upscale", "--json"]);
    assert_eq!(
        (unknown.code, unknown.refusal()["code"].as_str()),
        (1, Some("callable_unknown")),
        "{unknown:?}"
    );
    assert_eq!(unknown.refusal()["next"][0], "cozy run fidika/sdxl/fp8");
}

#[test]
fn terms_bind_strictly_and_unknown_fields_only_warn() {
    let Some(home) = Home::new() else { return };
    kept(&home, "sdxl-2.10.0", "fidika/sdxl");

    let invalid = home.cozy(["run", "fidika/sdxl/generate", "steps=99", "--json"]);
    let refusal = invalid.refusal();
    assert_eq!(
        (invalid.code, refusal["code"].as_str()),
        (1, Some("request_payload_invalid")),
        "{invalid:?}"
    );
    let message = refusal["message"].as_str().unwrap();
    assert!(
        message.contains("provide required arguments: [prompt: str]")
            && message.contains("steps is 99"),
        "{message}"
    );
    let human = home.cozy(["run", "fidika/sdxl/generate", "steps=99"]);
    assert!(
        human.stdout.contains("Arguments:\n  prompt: str"),
        "{human:?}"
    );

    let typed = home.cozy([
        "run",
        "fidika/sdxl/generate",
        "a cat",
        "steps=lots",
        "--json",
    ]);
    assert_eq!(typed.refusal()["code"], "value_type", "{typed:?}");

    // Binding succeeds with a warning; submitting arrives with the Calcifer client.
    let warned = home.cozy([
        "run",
        "fidika/sdxl/generate",
        "a cat",
        "style=noir",
        "--json",
    ]);
    assert!(
        warned.stderr.contains("ignored unknown field style"),
        "{warned:?}"
    );
    assert_eq!(warned.refusal()["code"], "not_ported", "{warned:?}");
}

#[test]
fn the_grammar_refuses_what_it_cannot_run() {
    let Some(home) = Home::new() else { return };
    for (args, code, exit) in [
        (&["run", "sdxl", "--json"][..], "target_malformed", 2),
        (
            &["run", "fidika/sdxl/2.10.0/generate", "--json"],
            "target_malformed",
            2,
        ),
        (&["run", "./experiment.py", "--json"], "not_ported", 2),
        (
            &[
                "run",
                "fidika/unknown/generate",
                "--tensorhub",
                "http://127.0.0.1:1",
                "--json",
            ],
            "hub_unreachable",
            1,
        ),
        (
            &["run", "fidika/sdxl/generate", "--bits", "3", "--json"],
            "usage",
            2,
        ),
        (&["run", "list", "--json"], "not_ported", 2),
    ] {
        let out = home.cozy(args);
        assert_eq!(
            (out.code, out.refusal()["code"].as_str()),
            (exit, Some(code)),
            "{args:?}: {out:?}"
        );
    }
}

#[test]
fn a_release_card_is_read_from_the_hub_once_and_kept() {
    let Some(home) = Home::new() else { return };
    let listed = home.json(["run", "fidika/sdxl", "--json"]);
    let release = listed["release"].as_str().unwrap();
    let names: Vec<&str> = listed["functions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["function"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"generate"), "{listed}");
    let kept = home.root().join("packages/tensorhub.com/fidika/sdxl");
    assert!(kept.join(format!("{release}.json")).is_file());
    assert_eq!(
        std::fs::read_to_string(kept.join("newest")).unwrap(),
        release
    );

    let pinned = home.json(["run", "fidika/sdxl/generate@2.10.0", "--describe", "--json"]);
    assert_eq!(pinned["fields"][0]["name"], "prompt");
    assert!(kept.join("2.10.0.json").is_file());

    let missing = home.cozy(["run", "fidika/no-such-package/generate", "--json"]);
    assert_eq!(
        (missing.code, missing.refusal()["code"].as_str()),
        (1, Some("package_not_found")),
        "{missing:?}"
    );
}
