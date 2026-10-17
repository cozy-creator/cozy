//! The model ingest cycle on a real Tensorhub with a real machine: Tensorhub's dev-build
//! `cozy_ref` job (TENSORHUB_URL, TENSORHUB_CA, TENSORHUB_LOG; MinIO storage) and the workspace's
//! TensorD (`cozy_tensord=workspace`, COZY_TEST_TENSORD) trusting the Hub's CA. The machine pins, makes,
//! quantizes and uploads every byte; this computer only asks.
//!
//! upload → publish → info → family → quantize (a package's quantizer) → publish its lane →
//! retarget → yank → delete.
#![cfg(unix)]
mod common;

use std::path::{Path, PathBuf};

use common::{Home, Out};
use serde_json::Value;

/// Unpinned: the machine pins it to the commit `main` names now.
const TINY: &str = "hf://hf-internal-testing/tiny-random-gpt2";

/// The last NDJSON event a followed `--json` run printed: its end.
fn end(out: &Out) -> Value {
    let line = out.stdout.lines().last().unwrap_or_else(|| panic!("no events: {out:?}"));
    serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}

fn completed(out: &Out) -> Value {
    let ended = end(out);
    assert_eq!(ended["type"], "run.completed", "{out:?}");
    ended["run"].clone()
}

fn copy(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        match entry.file_type().unwrap().is_dir() {
            true => copy(&entry.path(), &to.join(entry.file_name())),
            false => drop(std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap()),
        }
    }
}

/// `tests/fixtures/model/tiny-quantizer`, locked and published by this home's account.
fn publish_quantizer(home: &Home) {
    let project: PathBuf = home.path().join("src/tiny-quantizer");
    copy(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/model/tiny-quantizer"), &project);
    for verb in ["lock", "publish"] {
        let out = common::out(home.command().current_dir(&project).args(["--json", "package", verb]).output().unwrap());
        assert_eq!(out.code, 0, "package {verb}: {out:?}");
    }
}

#[test]
fn a_model_is_uploaded_quantized_and_published_by_its_machine() {
    let Some(home) = Home::with_hub() else { return };
    let ca = std::env::var("TENSORHUB_CA").expect("the CI Hub's CA");
    let account = home.sign_in("modelcycle@cozy.test");
    let _calcifer = home.want_calcifer();
    let _machine = common::tensord::added_with(&home, "m1", &[("SSL_CERT_FILE", &ca)]);
    let model = format!("{account}/tiny");

    // The machine pins the source, makes it and uploads a private checkpoint.
    let run = completed(&home.cozy(["--json", "model", "upload", TINY, &model, "--on", "m1"]));
    let made = &run["result"]["models"][0];
    let pinned = made["resolved"].as_str().and_then(|r| r.strip_prefix(&format!("{TINY}@"))).unwrap_or_default();
    assert!(pinned.len() >= 40 && pinned[..40].bytes().all(|b| b.is_ascii_hexdigit()), "{run}");
    assert_eq!(made["published"]["destination"], model.as_str(), "{run}");
    let source = made["published"]["checkpoint"].as_str().unwrap().to_owned();

    // A release names it; info shows the lane and its checkpoint.
    let lane = format!("f32={source}");
    let release = home.json(["--json", "model", "publish", &model, "--release", "1.0.0", "--lane", &lane]);
    assert_eq!((release["lanes"]["f32"].as_str(), &release["changed"]), (Some(source.as_str()), &Value::Bool(true)), "{release}");
    let info = home.json(["--json", "model", "info", &model]);
    assert_eq!(info["releases"][0]["lanes"][0]["checkpoint"], format!("{model}#{source}"), "{info}");

    home.json(["--json", "model", "family", &model, "cycle-test"]);
    let found = home.json(["--json", "model", "search", "--family", "cycle-test"]);
    assert_eq!(found["models"][0]["model"], model.as_str(), "{found}");

    // The package's quantizer runs on the machine and uploads the fp8 lane to the model.
    publish_quantizer(&home);
    let package = format!("{account}/tiny-quantizer");
    let pinned = format!("{model}@1.0.0/f32");
    let run = completed(&home.cozy(["--json", "model", "quantize", &pinned, "--fp8", "--package", &package, "--on", "m1"]));
    println!("quantize run: {run}");
    let fp8 = run["result"]["manifest"]["digest"].as_str().unwrap_or_else(|| panic!("no checkpoint in {run}")).to_owned();

    // The machine holds the package now: quantize finds its quantizer there, and none for int8.
    let none = home.cozy(["--json", "model", "quantize", &pinned, "--int8", "--on", "m1"]);
    assert_eq!((none.code, none.refusal()["code"].as_str()), (1, Some("model_quantizer_not_found")), "{none:?}");

    let lane = format!("fp8={fp8}");
    let release = home.json(["--json", "model", "publish", &model, "--release", "1.0.0", "--lane", &lane]);
    assert_eq!(release["lanes"]["fp8"].as_str(), Some(fp8.as_str()), "{release}");
    let moved = home.json(["--json", "model", "retarget", &model, "--release", "1.0.0", "--lane", "fp8", "--to", &source]);
    assert_eq!(moved["lanes"]["fp8"].as_str(), Some(source.as_str()), "{moved}");
    let emptied = home.cozy(["--json", "model", "publish", &model, "--release", "1.0.0", "--remove-lane", "f32", "--remove-lane", "fp8"]);
    assert_eq!((emptied.code, emptied.refusal()["code"].as_str()), (2, Some("model_release_empty")), "{emptied:?}");

    let yanked = home.json(["--json", "model", "yank", &model, "--release", "1.0.0"]);
    assert_eq!(yanked["yanked"], true, "{yanked}");
    let deleted = home.json(["--json", "model", "delete", &model, "--yes"]);
    assert_eq!(deleted["model"], model.as_str(), "{deleted}");
}
