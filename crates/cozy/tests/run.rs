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
    // Bound and warned, the run reaches Calcifer, which has no local machine to send it to.
    assert_eq!(warned.refusal()["code"], "machine_not_found", "{warned:?}");
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
        (&["run", "play", "1", "--json"], "not_ported", 2),
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

/// Reads `child`'s stdout lines until one satisfies `want`, answering every line read.
fn lines_until(reader: &mut impl std::io::BufRead, want: impl Fn(&Value) -> bool) -> Vec<Value> {
    let mut seen = Vec::new();
    loop {
        let mut line = String::new();
        assert!(
            reader.read_line(&mut line).unwrap() > 0,
            "stdout closed after {seen:?}"
        );
        let event: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line}"));
        let done = want(&event);
        seen.push(event);
        if done {
            return seen;
        }
    }
}

/// A run nothing can place stays queued in Calcifer's outbox: submitting, detaching, replaying a
/// key, following it as NDJSON, cancelling it and reattaching all go through the real Calcifer.
#[test]
fn a_run_is_submitted_followed_detached_and_canceled() {
    let Some(home) = Home::new() else { return };
    kept(&home, "sdxl-2.10.0", "fidika/sdxl");

    let local = home.cozy(["run", "fidika/sdxl/generate", "a cat", "--json"]);
    assert_eq!(
        (local.code, local.refusal()["code"].as_str()),
        (1, Some("machine_not_found")),
        "{local:?}"
    );

    let args = [
        "run",
        "fidika/sdxl/generate",
        "a cat",
        "style=noir",
        "--on",
        "any",
        "--idempotency-key",
        "k1",
    ];
    let detached = home.json([&args[..], &["--detach", "--json"]].concat());
    assert_eq!(detached["status"], "queued", "{detached}");
    assert_eq!(detached["target"], "fidika/sdxl/generate");
    assert_eq!(detached["warnings"][0]["code"], "request_fields_ignored");
    let replay = home.json([&args[..], &["--detach", "--json"]].concat());
    assert_eq!(replay["id"], detached["id"]);
    let human = home.cozy([&args[..], &["--detach"]].concat());
    assert!(
        human.stdout.contains("queued ⚠") && human.stdout.contains("next: cozy run watch"),
        "{human:?}"
    );
    let number = detached["number"].as_u64().unwrap().to_string();

    // Following with --json: NDJSON of Calcifer's events until a cancel ends the run.
    let mut follower = home
        .command()
        .args(["run", "watch", &number, "--json"])
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = std::io::BufReader::new(follower.stdout.take().unwrap());
    let first = lines_until(&mut stdout, |e| {
        e["type"].as_str().is_some_and(|t| t.starts_with("run."))
    });
    assert_eq!(
        first.last().unwrap()["run"]["status"],
        "queued",
        "{first:?}"
    );
    // Calcifer stopping under a follower: the follower starts it again and reattaches.
    let down = home.cozy(["down"]);
    assert_eq!(down.code, 0, "{down:?}");
    let canceled = home.json(["run", "cancel", &number, "--json"]);
    assert_eq!(canceled["id"], detached["id"]);
    let rest = lines_until(&mut stdout, |e| e["type"] == "run.canceled");
    assert_eq!(rest.last().unwrap()["run"]["status"], "canceled");
    assert_eq!(follower.wait().unwrap().code(), Some(1));

    let again = home.cozy(["run", "watch", &number]);
    assert_eq!(again.code, 1, "{again:?}");
    assert!(
        again.stdout.contains("canceled") && again.stdout.contains("next: cozy run show"),
        "{again:?}"
    );
    let settled = home.json(["run", "cancel", &number, "--json"]);
    assert_eq!(settled["status"], "canceled");
}

/// Ctrl-C stops following and never cancels: the run is still queued, and watch follows it again.
#[cfg(unix)]
#[test]
fn ctrl_c_detaches_and_never_cancels() {
    let Some(home) = Home::new() else { return };
    kept(&home, "sdxl-2.10.0", "fidika/sdxl");
    let mut follower = home
        .command()
        .args([
            "run",
            "fidika/sdxl/generate",
            "a dog",
            "--on",
            "any",
            "--json",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = std::io::BufReader::new(follower.stdout.take().unwrap());
    let created = lines_until(&mut stdout, |e| {
        e["type"] == "run.created" || e["type"] == "run.queued"
    });
    let number = created.last().unwrap()["run"]["number"]
        .as_u64()
        .unwrap()
        .to_string();
    let signal = std::process::Command::new("kill")
        .args(["-INT", &follower.id().to_string()])
        .status()
        .unwrap();
    assert!(signal.success());
    let out = follower.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{out:?}");
    let mut watcher = home
        .command()
        .args(["run", "watch", &number, "--json"])
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = std::io::BufReader::new(watcher.stdout.take().unwrap());
    let still = lines_until(&mut stdout, |e| {
        e["type"].as_str().is_some_and(|t| t.starts_with("run."))
    });
    let run = &still.last().unwrap()["run"];
    assert_eq!(
        (run["status"].as_str(), run["cancel_requested"].as_bool()),
        (Some("queued"), Some(false)),
        "{run}"
    );
    home.json(["run", "cancel", &number, "--json"]);
    assert_eq!(watcher.wait().unwrap().code(), Some(1));
}

/// Runs list newest first with their state, filter by state, say how many older runs a limit
/// left out, and show one run whole; a live board needs a terminal.
#[test]
fn runs_list_filter_and_show() {
    let Some(home) = Home::new() else { return };
    kept(&home, "sdxl-2.10.0", "fidika/sdxl");
    let submit = |prompt: &str| {
        let run = home.json([
            "run",
            "fidika/sdxl/generate",
            prompt,
            "mood=calm",
            "--on",
            "any",
            "--detach",
            "--json",
        ]);
        run["number"].as_u64().unwrap()
    };
    let first = submit("a cat");
    let second = submit("a dog");
    home.json(["run", "cancel", &first.to_string(), "--json"]);

    let listed = home.json(["run", "list", "--json"]);
    let numbers: Vec<u64> = listed["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["number"].as_u64().unwrap())
        .collect();
    assert_eq!(numbers, [second, first]);
    let canceled = home.json(["run", "list", "--state", "canceled", "--json"]);
    assert_eq!(canceled["runs"].as_array().unwrap().len(), 1, "{canceled}");
    assert_eq!(canceled["runs"][0]["number"], first);
    let newest = home.json(["run", "list", "--limit", "1", "--json"]);
    assert_eq!(
        (
            newest["runs"].as_array().unwrap().len(),
            newest["older"].as_u64()
        ),
        (1, Some(1)),
        "{newest}"
    );

    let table = home.cozy(["run", "list", "--limit", "1"]);
    assert!(
        table.stdout.contains("queued ⚠") && table.stdout.contains("1 older runs"),
        "{table:?}"
    );

    let shown = home.json(["run", "show", &first.to_string(), "--json"]);
    assert_eq!(
        (
            shown["status"].as_str(),
            shown["warnings"][0]["code"].as_str()
        ),
        (Some("canceled"), Some("request_fields_ignored"))
    );
    let human = home.cozy(["run", "show", &first.to_string()]);
    assert!(
        human.stdout.contains("warning: ignored unknown field mood"),
        "{human:?}"
    );
    assert!(!human.stdout.contains("next:"), "{human:?}");

    let board = home.cozy(["run", "list", "--watch"]);
    assert!(
        board.code == 2 && board.stdout.contains("watch_needs_terminal"),
        "{board:?}"
    );
    home.json(["run", "cancel", &second.to_string(), "--json"]);
}

/// An upload names a weights output the run wrote, on the run's machine; one that wrote none,
/// or a local destination, is refused before anything is sent.
#[test]
fn an_upload_needs_a_weights_output_and_a_hub_destination() {
    let Some(home) = Home::new() else { return };
    kept(&home, "sdxl-2.10.0", "fidika/sdxl");
    let run = home.json([
        "run",
        "fidika/sdxl/generate",
        "a cat",
        "--on",
        "any",
        "--detach",
        "--json",
    ]);
    let number = run["number"].as_u64().unwrap().to_string();
    for (args, code, exit) in [
        (
            vec!["run", "upload", &number, "paul/sdxl-int8", "--json"],
            "run_output_absent",
            1,
        ),
        (
            vec![
                "run",
                "upload",
                &format!("{number}#int8"),
                "paul/sdxl-int8",
                "--json",
            ],
            "run_output_absent",
            1,
        ),
        (
            vec!["run", "upload", &number, "local/mine", "--json"],
            "model_local_destination",
            2,
        ),
    ] {
        let out = home.cozy(&args);
        assert_eq!(
            (out.code, out.refusal()["code"].as_str()),
            (exit, Some(code)),
            "{args:?}: {out:?}"
        );
    }
    home.json(["run", "cancel", &number, "--json"]);
}
