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

#[test]
fn download_and_upload_refuse_before_any_run() {
    let Some(home) = Home::new() else { return };
    std::fs::write(home.path().join("weights.bin"), b"x").unwrap();
    std::fs::write(home.path().join("w.safetensors"), b"x").unwrap();
    let file = home.path().join("w.safetensors");
    let file = file.to_str().unwrap();
    let not_safetensors = home.path().join("weights.bin");
    let cases: [(&[&str], i32, &str); 5] = [
        (&["model", "download", file], 2, "model_alias_missing"),
        (&["model", "download", "fidika/sdxl", "fidika/other"], 2, "model_alias_invalid"),
        (&["model", "upload", not_safetensors.to_str().unwrap(), "fidika/x"], 1, "model_file_refused"),
        (&["model", "upload", "fidika/sdxl", "local/x"], 2, "model_local_destination"),
        (&["model", "upload", "local/tiny", "fidika/x"], 1, "not_logged_in"),
    ];
    for (args, code, refusal) in cases {
        let out = home.cozy(["--json"].iter().chain(args));
        assert_eq!((out.code, out.refusal()["code"].as_str()), (code, Some(refusal)), "{args:?}: {out:?}");
    }
    assert!(!home.root().join("calcifer").exists(), "a refusal started Calcifer");
}

#[test]
fn quantize_names_one_checkpoint_and_one_quantizer_before_any_run() {
    let Some(home) = Home::new() else { return };
    // Grammar and sources, before any Hub read.
    let cases: [(&[&str], &str); 4] = [
        (&["model", "quantize", "fidika/sdxl@1.0.0/bf16"], ""),
        (&["model", "quantize", "fidika/sdxl@1.0.0/bf16", "--int8", "--fp8"], ""),
        (&["model", "quantize", "local/tiny", "--int8"], "model_quantize_source"),
        (&["model", "quantize", "fidika/sdxl@1.0.0/bf16", "local/x", "--int8"], "model_local_destination"),
    ];
    for (args, code) in cases {
        let out = home.cozy(["--json", "--tensorhub", "http://127.0.0.1:9"].iter().chain(args));
        assert_eq!(out.code, 2, "{args:?}: {out:?}");
        if !code.is_empty() {
            assert_eq!(out.refusal()["code"], code, "{args:?}: {out:?}");
        }
    }
    // tensorhub.com, read only: a release with several lanes needs one named, and a machine to
    // be found by `any` holds no packages to choose a quantizer from.
    let unnamed = home.cozy(["--json", "model", "quantize", "fidika/sdxl@1.0.0", "--int8", "--on", "any"]);
    assert_eq!((unnamed.code, unnamed.refusal()["code"].as_str()), (2, Some("model_lane_unnamed")), "{unnamed:?}");
    let any = home.cozy(["--json", "model", "quantize", "fidika/sdxl@1.0.0/bf16", "--int8", "--on", "any"]);
    assert_eq!((any.code, any.refusal()["code"].as_str()), (2, Some("model_quantizer_unnamed")), "{any:?}");
    assert!(!home.root().join("calcifer").exists(), "a refusal started Calcifer");
}
