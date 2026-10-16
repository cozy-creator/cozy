//! `cozy model` product promises through the real binary. The catalog reads tensorhub.com
//! anonymously; nothing here writes to it. Usage refusals reach no Hub at all.
mod common;

use common::Home;

#[test]
fn search_and_info_read_the_public_catalog() {
    let Some(home) = Home::new() else { return };

    let found = home.json(["--json", "model", "search", "sdxl"]);
    let sdxl = found["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["model"] == "fidika/sdxl")
        .unwrap_or_else(|| panic!("no fidika/sdxl in {found}"))
        .clone();
    let release = sdxl["release"].as_str().expect("an available release").to_owned();
    assert!(sdxl["lanes"].as_array().is_some_and(|l| !l.is_empty()), "{sdxl}");

    let exact = home.json(["--json", "model", "search", "fidika/sdxl"]);
    assert_eq!(exact["models"].as_array().unwrap().len(), 1, "{exact}");

    let one = home.json(["--json", "model", "search", "--limit", "1"]);
    assert_eq!(one["models"].as_array().unwrap().len(), 1, "{one}");
    assert!(one["total"].as_i64().unwrap() > 1, "{one}");

    let none = home.json(["--json", "model", "search", "--family", "no-such-family-xs043"]);
    assert!(none["models"].as_array().unwrap().is_empty(), "{none}");

    let info = home.json(["--json", "model", "info", "fidika/sdxl"]);
    let lanes = info["releases"][0]["lanes"].as_array().unwrap();
    assert!(!lanes.is_empty(), "{info}");
    for lane in lanes {
        assert!(lane["bytes"].as_i64().unwrap() > 0, "{lane}");
        assert!(lane["checkpoint"].as_str().unwrap().starts_with("fidika/sdxl#sha256:"), "{lane}");
    }
    let first = lanes[0]["checkpoint"].as_str().unwrap().to_owned();

    let pinned = home.json(["--json", "model", "info", &format!("fidika/sdxl@{release}")]);
    assert_eq!(pinned["releases"].as_array().unwrap().len(), 1, "{pinned}");

    let missing = home.cozy(["--json", "model", "info", "fidika/sdxl@no-such-release"]);
    assert_eq!((missing.code, missing.refusal()["code"].as_str()), (1, Some("model_release_not_found")), "{missing:?}");

    // Piped human output is never shortened.
    let human = home.cozy(["model", "info", "fidika/sdxl"]);
    assert_eq!(human.code, 0, "{human:?}");
    assert!(human.stdout.contains(&first), "{human:?}");
    let table = home.cozy(["model", "search", "sdxl"]);
    assert!(table.stdout.contains("fidika/sdxl"), "{table:?}");
}

#[test]
fn usage_refusals_reach_no_hub() {
    let Some(home) = Home::new() else { return };
    let lane = format!("int8=sha256:{}", "a".repeat(64));
    let cases: [&[&str]; 9] = [
        &["model", "delete", "fidika/x"],
        &["model", "delete", "fidika/x#latest", "--yes"],
        &["model", "publish", "local/x", "--release", "1.0.0", "--lane", &lane],
        &["model", "publish", "fidika/x", "--release", "1.0.0"],
        &["model", "publish", "fidika/x", "--release", "1.0.0", "--lane", &lane, "--remove-lane", "int8"],
        &["model", "retarget", "fidika/x", "--release", "1.0.0", "--lane", "int8", "--to", "abc"],
        &["model", "info", "sdxl"],
        &["model", "info", "fidika/sdxl#fp8"],
        &["model", "family", "fidika/x", "anime", "--clear"],
    ];
    for args in cases {
        // Any Hub call would end `unavailable` (exit 1) at this dead port before a usage check.
        let out = home.cozy(["--json", "--tensorhub", "http://127.0.0.1:9"].iter().chain(args));
        assert_eq!(out.code, 2, "{args:?}: {out:?}");
    }
}
