//! The model ingest cycle on a real Tensorhub with real machines: Tensorhub's dev-build `cozy_ref`
//! job (TENSORHUB_URL, TENSORHUB_LOG; MinIO storage; its public origin behind a private CA) and the
//! TensorD COZY_TEST_TENSORD names (the released one by default). The machines trust that CA
//! only as each run's Hub access names it, never through their environment, so every Hub call a
//! run makes proves it: the capability's trade, the upload's presign and publication, a private
//! checkpoint's closure, a package's card and install plan. The machine pins, makes, quantizes
//! and uploads every byte; this computer only asks.
//!
//! upload → private download elsewhere → publish → info → family → quantize (a package's
//! quantizer) → publish its lane → retarget → yank → delete.
#![cfg(unix)]
mod common;

use std::path::{Path, PathBuf};

use common::{Home, Out};
use serde_json::Value;

/// `cozy args` in `dir`. Each step is named, timestamped and flushed before it starts, and its
/// output streams into the log as it comes, so the job's own timeout shows the step it stopped in.
fn cozy(home: &Home, dir: Option<&Path>, args: &[&str]) -> Out {
    use std::io::{BufRead as _, Write as _};
    let at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
    println!("step {at}: cozy {}", args.join(" "));
    let _ = std::io::stdout().flush();
    let mut command = home.command();
    if let Some(dir) = dir {
        command.current_dir(dir);
    }
    let mut child = command
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .unwrap();
    let mut stdout = String::new();
    for line in std::io::BufReader::new(child.stdout.take().unwrap()).lines() {
        let line = line.unwrap();
        println!("  | {line}");
        stdout.push_str(&line);
        stdout.push('\n');
    }
    let code = child.wait().unwrap().code().unwrap_or(-1);
    Out { code, stdout, stderr: String::new() }
}

/// What Calcifer and its machines say now: printed when a step fails.
fn diagnose(home: &Home) {
    for args in [&["run", "list", "--limit", "5"][..], &["machine", "list"]] {
        let out = common::out(home.command().args(args).output().unwrap());
        println!("cozy {}:\n{}{}", args.join(" "), out.stdout, out.stderr);
    }
    let log = std::fs::read_to_string(home.root().join("calcifer/calcifer.log")).unwrap_or_default();
    let tail: Vec<&str> = log.lines().rev().take(50).collect();
    println!("calcifer.log (last 50 lines):\n{}", tail.into_iter().rev().collect::<Vec<_>>().join("\n"));
}

/// A step that succeeds, and its JSON document.
fn json(home: &Home, args: &[&str]) -> Value {
    let out = cozy(home, None, args);
    if out.code != 0 {
        diagnose(home);
    }
    assert_eq!(out.code, 0, "{args:?}: {out:?}");
    serde_json::from_str(&out.stdout).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}

/// Pinned, so a released machine takes it; `model_machine` covers a machine pinning it.
const TINY: &str = "hf://hf-internal-testing/tiny-random-gpt2@71034c5d8bde858ff824298bdedc65515b97d2b9";

/// The last NDJSON event a followed `--json` run printed: its end.
fn end(out: &Out) -> Value {
    let line = out.stdout.lines().last().unwrap_or_else(|| panic!("no events: {out:?}"));
    serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}

fn completed(home: &Home, out: &Out) -> Value {
    let ended = end(out);
    if ended["type"] != "run.completed" {
        diagnose(home);
    }
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
        let out = cozy(home, Some(&project), &["--json", "package", verb]);
        if out.code != 0 {
            diagnose(home);
        }
        assert_eq!(out.code, 0, "package {verb}: {out:?}");
    }
}

/// Whether COZY_TEST_TENSORD is this workspace's TensorD (`cozy_tensord=workspace` sets
/// COZY_TENSORD), not a released one.
fn workspace() -> bool {
    std::env::var("COZY_TENSORD").as_deref() == Ok("workspace")
}

/// A machine added to this home. TensorD v0.7.3 doesn't apply the run's CA to every Hub call
/// (fixed in #1384, ships with the cohort), so a released one also trusts the Hub's CA through
/// SSL_CERT_FILE; the workspace's (`cozy_tensord=workspace`, which sets COZY_TENSORD) gets none,
/// and every Hub call it makes proves the CA arrives through the run's access alone.
fn machine(home: &Home, name: &str) -> common::tensord::Tensord {
    match workspace() {
        true => common::tensord::added(home, name),
        false => {
            let ca = std::env::var("TENSORHUB_CA").expect("the CI Hub's CA");
            common::tensord::added_with(home, name, &[("SSL_CERT_FILE", &ca)])
        }
    }
}

#[test]
fn a_model_is_uploaded_quantized_and_published_by_its_machine() {
    let Some(home) = Home::with_hub() else { return };
    let account = home.sign_in("modelcycle@cozy.test");
    let _machine = machine(&home, "m1");
    let model = format!("{account}/tiny");

    // The machine makes the source and uploads a private checkpoint.
    let run = completed(&home, &cozy(&home, None, &["--json", "model", "upload", TINY, &model, "--on", "m1"]));
    let made = &run["result"]["models"][0];
    assert_eq!(made["resolved"], TINY, "{run}");
    assert_eq!(made["published"]["destination"], model.as_str(), "{run}");
    let source = made["published"]["checkpoint"].as_str().unwrap().to_owned();

    // Another machine reads the private checkpoint by its digest, under the run's capability.
    // TensorD v0.7.3 cannot read a by-hash checkpoint without its length (NOT_CONTAINED; fixed in
    // #1384, ships with the cohort), so this runs on the workspace's TensorD.
    if workspace() {
        let _other = machine(&home, "m2");
        let checkpoint = format!("{model}#{source}");
        let run = completed(&home, &cozy(&home, None, &["--json", "model", "download", &checkpoint, "--on", "m2"]));
        assert_eq!(run["result"]["models"][0]["manifest"], source.as_str(), "{run}");
    }

    // A release names it; info shows the lane and its checkpoint.
    let lane = format!("f32={source}");
    let release = json(&home, &["--json", "model", "publish", &model, "--release", "1.0.0", "--lane", &lane]);
    assert_eq!((release["lanes"]["f32"].as_str(), &release["changed"]), (Some(source.as_str()), &Value::Bool(true)), "{release}");
    let info = json(&home, &["--json", "model", "info", &model]);
    assert_eq!(info["releases"][0]["lanes"][0]["checkpoint"], format!("{model}#{source}"), "{info}");

    json(&home, &["--json", "model", "family", &model, "cycle-test"]);
    let found = json(&home, &["--json", "model", "search", "--family", "cycle-test"]);
    assert_eq!(found["models"][0]["model"], model.as_str(), "{found}");

    // The package's quantizer runs on the machine and uploads the fp8 lane to the model.
    publish_quantizer(&home);
    let package = format!("{account}/tiny-quantizer");
    let pinned = format!("{model}@1.0.0/f32");
    let run = completed(&home, &cozy(&home, None, &["--json", "model", "quantize", &pinned, "--fp8", "--package", &package, "--on", "m1"]));
    println!("quantize run: {run}");
    let fp8 = run["result"]["manifest"]["digest"].as_str().unwrap_or_else(|| panic!("no checkpoint in {run}")).to_owned();

    // A person sees the checkpoint the machine wrote and how to publish it.
    let human = cozy(&home, None, &["model", "quantize", &pinned, "--fp8", "--package", &package, "--on", "m1"]);
    assert_eq!(human.code, 0, "{human:?}");
    assert!(human.stdout.contains(&format!("{model}#{fp8}")), "{human:?}");
    let next = format!("cozy model publish {model} --release 1.0.0 --lane fp8={fp8}");
    assert!(human.stdout.contains(&next), "{human:?}");

    // The machine holds the package now: quantize finds its quantizer there, and none for int8.
    let none = cozy(&home, None, &["--json", "model", "quantize", &pinned, "--int8", "--on", "m1"]);
    assert_eq!((none.code, none.refusal()["code"].as_str()), (1, Some("model_quantizer_not_found")), "{none:?}");

    let lane = format!("fp8={fp8}");
    let release = json(&home, &["--json", "model", "publish", &model, "--release", "1.0.0", "--lane", &lane]);
    assert_eq!(release["lanes"]["fp8"].as_str(), Some(fp8.as_str()), "{release}");
    let moved = json(&home, &["--json", "model", "retarget", &model, "--release", "1.0.0", "--lane", "fp8", "--to", &source]);
    assert_eq!(moved["lanes"]["fp8"].as_str(), Some(source.as_str()), "{moved}");
    let emptied = cozy(&home, None, &["--json", "model", "publish", &model, "--release", "1.0.0", "--remove-lane", "f32", "--remove-lane", "fp8"]);
    assert_eq!((emptied.code, emptied.refusal()["code"].as_str()), (2, Some("model_release_empty")), "{emptied:?}");

    let yanked = json(&home, &["--json", "model", "yank", &model, "--release", "1.0.0"]);
    assert_eq!(yanked["yanked"], true, "{yanked}");
    let deleted = json(&home, &["--json", "model", "delete", &model, "--yes"]);
    assert_eq!(deleted["model"], model.as_str(), "{deleted}");
}
