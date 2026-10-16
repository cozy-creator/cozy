//! `cozy machine` and `cozy doctor` through the real binary, a real Calcifer and a real TensorD added by
//! address. No Hub: an added machine needs none. TensorD 0.6.0 runs on Linux only.
#![cfg(unix)]
mod common;

use common::{
    Home,
    tensord::{self, Tensord},
};
use serde_json::Value;

fn listed(home: &Home, name: &str) -> Option<Value> {
    home.json(["machine", "list", "--json"])["machines"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == name)
        .cloned()
}

#[test]
fn an_added_machine_is_checked_before_it_is_recorded_then_listed_shown_checked_and_removed() {
    let Some(home) = Home::new() else { return };
    let _calcifer = home.want_calcifer();
    // A machine that admits some other controller.
    let mut machine = Tensord::start(&[tensord::stranger()]);
    let add = |fingerprint: &str| {
        home.cozy([
            "machine",
            "add",
            "box",
            machine.address.as_str(),
            "--fingerprint",
            fingerprint,
            "--json",
        ])
    };

    // A wrong pin is refused and nothing is recorded.
    let out = add(&format!("sha256:{}", "0".repeat(64)));
    assert_eq!(
        (out.code, out.refusal()["code"].clone()),
        (1, "fingerprint_mismatch".into()),
        "{out:?}"
    );
    assert!(listed(&home, "box").is_none());

    // The right pin, but the machine does not admit this controller: the refusal names the command to
    // run on that computer, and nothing is recorded.
    let out = add(&machine.fingerprint);
    assert_eq!(
        (out.code, out.refusal()["code"].clone()),
        (1, "machine_unauthorized".into()),
        "{out:?}"
    );
    let key = tensord::key_to_authorize(&out);
    assert!(listed(&home, "box").is_none());

    // Once it admits the key (as `cozy machine authorize` will), the add succeeds.
    machine.restart(&[key]);
    let add = |fingerprint: &str| {
        home.cozy([
            "machine",
            "add",
            "box",
            machine.address.as_str(),
            "--fingerprint",
            fingerprint,
            "--json",
        ])
    };
    let out = add(&machine.fingerprint);
    assert_eq!(out.code, 0, "{out:?}");
    let row = listed(&home, "box").expect("the added machine is listed");
    assert_eq!(
        (row["kind"].as_str(), row["state"].as_str()),
        (Some("added"), Some("ready")),
        "{row}"
    );
    let human = home.cozy(["machine", "list"]);
    assert!(
        human.stdout.contains("box") && human.stdout.contains("added"),
        "{human:?}"
    );

    // show and doctor render what the machine reports of itself.
    let shown = home.json(["machine", "show", "box", "--json"]);
    assert!(
        !shown["status"]["version"]
            .as_str()
            .unwrap_or_default()
            .is_empty(),
        "{shown}"
    );
    let out = home.cozy(["machine", "show", "box"]);
    assert!(out.stdout.contains("platform"), "{out:?}");
    let out = home.cozy(["doctor", "--on", "box"]);
    assert_eq!(out.code, 0, "{out:?}");
    assert!(
        out.stdout.contains("GPU") || out.stdout.contains("none: runs use the CPU"),
        "{out:?}"
    );

    // A log the machine keeps, possibly empty, reads without failing.
    for log in ["memory", "tensorfs"] {
        let out = home.cozy(["machine", "logs", "box", log]);
        assert_eq!(out.code, 0, "{log}: {out:?}");
    }

    // End removes it here; a second end is an honest no-op, and a name never added claims nothing.
    let out = home.cozy(["machine", "end", "box"]);
    assert_eq!(out.code, 0, "{out:?}");
    assert!(listed(&home, "box").is_none());
    let again = home.json(["machine", "end", "box", "--json"]);
    assert_eq!(again["changed"], false, "{again}");
    let out = home.cozy(["machine", "end", "never-added", "--json"]);
    assert_eq!(
        (out.code, out.refusal()["kind"].clone()),
        (1, "not_found".into()),
        "{out:?}"
    );
    // The fixture other groups' tests use adds a machine the same way.
    let _again = tensord::added(&home, "again");
    assert_eq!(listed(&home, "again").expect("added")["state"], "ready");
}

#[test]
fn machine_refusals_say_what_to_do_instead() {
    let Some(home) = Home::new() else { return };
    let stranger = tensord::stranger();
    for (args, exit, code) in [
        // This computer's machine role comes with TensorD inside the binary.
        (vec!["machine", "enable"], 2, "not_ported"),
        (vec!["machine", "disable"], 2, "not_ported"),
        (
            vec!["machine", "authorize", stranger.as_str()],
            2,
            "not_ported",
        ),
        // `local` is ended by disabling it, and there is no local machine to check yet.
        (vec!["machine", "end", "local"], 1, "machine_local"),
        (vec!["doctor"], 1, "no_local_machine"),
        (
            vec!["machine", "list", "--watch"],
            2,
            "watch_needs_terminal",
        ),
        (vec!["machine", "show", "../etc"], 2, "machine_name_invalid"),
    ] {
        let out = home.cozy(args.iter().chain(&["--json"]));
        assert_eq!(
            (out.code, out.refusal()["code"].clone()),
            (exit, code.into()),
            "{args:?}: {out:?}"
        );
    }
    // Rental options without a product, and a malformed fingerprint, are usage errors.
    for args in [
        vec!["machine", "rent", "--gpus", "2"],
        vec![
            "machine",
            "add",
            "box",
            "127.0.0.1:1",
            "--fingerprint",
            "abc",
        ],
    ] {
        assert_eq!(home.cozy(&args).code, 2, "{args:?}");
    }
}
