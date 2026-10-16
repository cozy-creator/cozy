//! `cozy model search|info` read tensorhub.com's public catalog anonymously through the real
//! binary, and the model verbs' usage refusals send nothing to any Hub.
mod common;

use common::Home;
use serde_json::Value;

fn json(out: &std::process::Output) -> Value {
    serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "not JSON ({e}), exit {:?}:\n{}{}",
            out.status.code(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    })
}

fn ok(out: std::process::Output) -> Value {
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stdout));
    json(&out)
}

#[test]
fn search_and_info_read_the_public_catalog() {
    let home = Home::new();

    let found = ok(home.cozy(["--json", "model", "search", "sdxl"]));
    let sdxl = found["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["model"] == "fidika/sdxl")
        .unwrap_or_else(|| panic!("no fidika/sdxl in {found}"));
    let release = sdxl["release"].as_str().expect("an available release").to_owned();
    assert!(sdxl["lanes"].as_array().is_some_and(|l| !l.is_empty()), "{sdxl}");

    let exact = ok(home.cozy(["--json", "model", "search", "fidika/sdxl"]));
    assert_eq!(exact["models"].as_array().unwrap().len(), 1, "{exact}");

    let one = ok(home.cozy(["--json", "model", "search", "--limit", "1"]));
    assert_eq!(one["models"].as_array().unwrap().len(), 1, "{one}");
    assert!(one["total"].as_i64().unwrap() > 1, "{one}");

    let none = ok(home.cozy(["--json", "model", "search", "--family", "no-such-family-xs043"]));
    assert!(none["models"].as_array().unwrap().is_empty(), "{none}");

    let info = ok(home.cozy(["--json", "model", "info", "fidika/sdxl"]));
    let lanes = info["releases"][0]["lanes"].as_array().unwrap();
    assert!(!lanes.is_empty(), "{info}");
    for lane in lanes {
        assert!(lane["bytes"].as_i64().unwrap() > 0, "{lane}");
        assert!(lane["checkpoint"].as_str().unwrap().starts_with("fidika/sdxl#sha256:"), "{lane}");
    }
    let first = lanes[0]["checkpoint"].as_str().unwrap().to_owned();

    let pinned = ok(home.cozy(["--json", "model", "info", &format!("fidika/sdxl@{release}")]));
    assert_eq!(pinned["releases"].as_array().unwrap().len(), 1, "{pinned}");

    let missing = home.cozy(["--json", "model", "info", "fidika/sdxl@no-such-release"]);
    assert_eq!(missing.status.code(), Some(1));
    assert_eq!(json(&missing)["error"]["code"], "model_release_not_found");

    // Piped human output is never shortened.
    let human = home.cozy(["model", "info", "fidika/sdxl"]);
    assert!(human.status.success());
    assert!(String::from_utf8_lossy(&human.stdout).contains(&first));
}

#[test]
fn usage_refusals_reach_no_hub() {
    let home = Home::new();
    let hash = format!("sha256:{}", "a".repeat(64));
    let lane = format!("int8={hash}");
    let cases: [&[&str]; 8] = [
        &["model", "delete", "fidika/x"],
        &["model", "delete", "fidika/x#latest", "--yes"],
        &["model", "publish", "local/x", "--release", "1.0.0", "--lane", &lane],
        &["model", "publish", "fidika/x", "--release", "1.0.0"],
        &["model", "publish", "fidika/x", "--release", "1.0.0", "--lane", &lane, "--remove-lane", "int8"],
        &["model", "retarget", "fidika/x", "--release", "1.0.0", "--lane", "int8", "--to", "abc"],
        &["model", "info", "sdxl"],
        &["model", "family", "fidika/x", "anime", "--clear"],
    ];
    for args in cases {
        // A Hub call would fail `unavailable` (exit 1) at this dead port before any usage check.
        let out = home.cozy(["--json", "--tensorhub", "http://127.0.0.1:9"].iter().chain(args));
        assert_eq!(out.status.code(), Some(2), "{args:?}: {}", String::from_utf8_lossy(&out.stdout));
    }
}
