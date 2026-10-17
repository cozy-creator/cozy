//! `cozy model download|list` on a real machine: a TensorD (COZY_TEST_TENSORD) added to this
//! home's Calcifer makes a tiny public Hugging Face repository into its own store; no weight
//! byte lands in the controller's home.
#![cfg(unix)]
mod common;

use common::Home;
use serde_json::Value;

const TINY: &str = "hf://hf-internal-testing/tiny-random-gpt2@71034c5d8bde858ff824298bdedc65515b97d2b9";

/// The last NDJSON event a followed `--json` run printed: its end.
fn end(out: &common::Out) -> Value {
    let line = out.stdout.lines().last().unwrap_or_else(|| panic!("no events: {out:?}"));
    serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}

#[test]
fn download_makes_a_provider_source_on_the_machine() {
    let Some(home) = Home::new() else { return };
    let _machine = common::tensord::added(&home, "m1");

    let out = home.cozy(["--json", "model", "download", TINY, "local/tiny", "--on", "m1"]);
    assert_eq!(out.code, 0, "{out:?}");
    let ended = end(&out);
    assert_eq!(ended["type"], "run.completed", "{ended}");
    let model = &ended["run"]["result"]["models"][0];
    assert_eq!(model["resolved"], TINY, "{model}");
    assert_eq!(model["published"]["destination"], "local/tiny", "{model}");

    // The machine holds it by name now; the human record names the checkpoint.
    let held = home.cozy(["model", "download", "local/tiny", "--on", "m1"]);
    assert_eq!(held.code, 0, "{held:?}");
    assert!(held.stdout.contains("local/tiny"), "{held:?}");

    // `model list` reads the machine's Status, which follows the store within a frame or two:
    // every machine by default, one with --on.
    let holds = |args: &[&str]| {
        let listed = home.json(args);
        listed["models"].as_array().unwrap().iter().any(|r| r["machine"] == "m1" && r["model"] == "local/tiny")
    };
    // A hang bound for the test; the product never waits on it.
    let until = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !holds(&["--json", "model", "list"]) {
        let machines = home.cozy(["--json", "machine", "list"]);
        assert!(std::time::Instant::now() < until, "local/tiny never listed: {machines:?}");
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    assert!(holds(&["--json", "model", "list", "--on", "m1"]));
    let missing = home.cozy(["--json", "model", "list", "--on", "nowhere"]);
    assert_eq!((missing.code, missing.refusal()["code"].as_str()), (1, Some("machine_not_found")), "{missing:?}");

    // The 454 KB model never passed through this computer's home.
    let state = |path: &std::path::Path| {
        let name = path.file_name().unwrap().to_string_lossy();
        [".db", ".db-wal", ".db-shm", ".log"].iter().any(|s| name.ends_with(s))
    };
    let big: Vec<_> = walk(&home.root()).into_iter().filter(|(path, len)| *len >= 400 << 10 && !state(path)).collect();
    assert!(big.is_empty(), "weights in the controller's home: {big:?}");
}

fn walk(dir: &std::path::Path) -> Vec<(std::path::PathBuf, u64)> {
    let mut files = vec![];
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let meta = entry.metadata().unwrap();
        if meta.is_dir() {
            files.extend(walk(&entry.path()));
        } else {
            files.push((entry.path(), meta.len()));
        }
    }
    files
}

/// A moving revision and a pasted URL are pinned by the machine itself: only TensorD from this
/// workspace does that, so this runs where the job builds it (`cozy_tensord=workspace`, which
/// sets COZY_TENSORD); a released TensorD refuses unpinned spellings.
#[test]
fn the_machine_pins_a_moving_source() {
    let Some(home) = Home::new() else { return };
    if std::env::var("COZY_TENSORD").as_deref() != Ok("workspace") {
        eprintln!("skipped: COZY_TEST_TENSORD is not this workspace's TensorD (COZY_TENSORD=workspace)");
        return;
    }
    let _machine = common::tensord::added(&home, "m1");
    // It pins a moving revision and reads a pasted URL as its source; the result
    // names the commit. Anything ambiguous is the machine's refusal, naming the spellings.
    let pinned = |resolved: &Value| {
        let commit = resolved.as_str().and_then(|r| r.strip_prefix("hf://hf-internal-testing/tiny-random-gpt2@"));
        commit.is_some_and(|c| c.len() >= 40 && c[..40].bytes().all(|b| b.is_ascii_hexdigit()))
    };
    for (source, alias) in [
        ("hf://hf-internal-testing/tiny-random-gpt2", "local/main"),
        ("https://huggingface.co/hf-internal-testing/tiny-random-gpt2/blob/main/model.safetensors", "local/pasted"),
    ] {
        let out = home.cozy(["--json", "model", "download", source, alias, "--on", "m1"]);
        let ended = end(&out);
        assert_eq!(ended["type"], "run.completed", "{source}: {out:?}");
        let resolved = &ended["run"]["result"]["models"][0]["resolved"];
        assert!(pinned(resolved), "{source} → {resolved}");
    }
    let vague = home.cozy(["--json", "model", "download", "https://huggingface.co/hf-internal-testing", "local/x", "--on", "m1"]);
    assert_eq!(vague.code, 1, "{vague:?}");
    let failed = end(&vague);
    assert_eq!(failed["run"]["reason"]["code"], "model_source_failed", "{failed}");
    assert!(failed["run"]["reason"]["message"].as_str().unwrap().contains("hf://"), "{failed}");

}
