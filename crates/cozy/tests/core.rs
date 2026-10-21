//! Product promises of the CLI's foundation: config.toml, Hub contexts, typed refusals and
//! shell completion, through the real binary.
mod common;

use common::Home;
use serde_json::Value;

fn hubs(home: &Home, args: &[&str]) -> Vec<Value> {
    let doc = home.json([&["hub", "list", "--json"], args].concat());
    doc["hubs"].as_array().unwrap().clone()
}

fn current(hubs: &[Value]) -> Vec<String> {
    hubs.iter()
        .filter(|h| h["current"] == true)
        .map(|h| {
            h["name"]
                .as_str()
                .unwrap_or(h["url"].as_str().unwrap())
                .to_owned()
        })
        .collect()
}

/// Hub names work like kubectl contexts: one identity in any case, URLs in one spelling, the
/// current one kept in config.toml, and `--tensorhub` picks one for a single command.
#[test]
fn hub_contexts_select_the_current_hub() {
    let Some(home) = Home::new() else { return };
    let fresh = hubs(&home, &[]);
    assert_eq!(fresh.len(), 1);
    assert_eq!(fresh[0]["url"], "https://tensorhub.com");
    assert_eq!(current(&fresh), ["tensorhub"]);
    assert_eq!(std::fs::read_dir(home.root()).unwrap().count(), 1, "a read created the home");

    home.json(["hub", "add", "Local", "http://127.0.0.1:1/", "--json"]);
    home.json([
        "hub",
        "add",
        "spelled",
        "HTTPS://Example.TEST:443/",
        "--json",
    ]);
    let listed = hubs(&home, &[]);
    assert!(
        listed
            .iter()
            .any(|h| h["name"] == "spelled" && h["url"] == "https://example.test"),
        "{listed:?}"
    );
    home.json(["hub", "remove", "SPELLED", "--json"]);

    let bad = home.cozy(["hub", "add", "Bad Name", "http://127.0.0.1:1", "--json"]);
    assert_eq!(
        (bad.code, bad.refusal()["code"].as_str()),
        (2, Some("hub_name_invalid")),
        "{bad:?}"
    );

    home.json(["hub", "use", "LOCAL", "--json"]);
    let written = home.config();
    assert!(
        written.contains("hub = \"local\"") && written.contains("local = \"http://127.0.0.1:1\""),
        "{written}"
    );
    assert_eq!(current(&hubs(&home, &[])), ["local"]);

    let refused = home.cozy(["hub", "remove", "local", "--json"]);
    assert_eq!(
        (refused.code, refused.refusal()["code"].as_str()),
        (1, Some("hub_current")),
        "{refused:?}"
    );

    let unknown = home.cozy(["--tensorhub", "nosuch", "hub", "list", "--json"]);
    assert_eq!(
        (unknown.code, unknown.refusal()["code"].as_str()),
        (1, Some("hub_not_found")),
        "{unknown:?}"
    );

    // --tensorhub narrows to one Hub, by name or URL, for this command only.
    let narrowed = hubs(&home, &["--tensorhub", "TensorHub"]);
    assert_eq!(
        (narrowed.len(), narrowed[0]["current"].as_bool()),
        (1, Some(false)),
        "{narrowed:?}"
    );
    let by_url = hubs(&home, &["--tensorhub", "http://127.0.0.1:1"]);
    assert_eq!(by_url[0]["name"], "local");
    assert_eq!(current(&hubs(&home, &[])), ["local"]);

    home.json(["hub", "use", "tensorhub", "--json"]);
    home.json(["hub", "remove", "Local", "--json"]);
    assert_eq!(hubs(&home, &[]).len(), 1);

    // A capital in a hand-written name is the same Hub; the file keeps its comment.
    home.write_config("# mine\n[hubs]\nStaging = \"http://127.0.0.1:2\"\n");
    assert_eq!(
        hubs(&home, &["--tensorhub", "STAGING"])[0]["name"],
        "staging"
    );
    home.json(["hub", "use", "staging", "--json"]);
    let written = home.config();
    assert!(
        written.contains("hub = \"staging\"") && written.contains("# mine\n[hubs]\nStaging = "),
        "{written}"
    );
}

/// A file another version wrote never disables a command: unknown keys are named once and
/// ignored. A known key with the wrong type is a refusal that names it.
#[test]
fn a_config_from_another_version_keeps_commands_working() {
    let Some(home) = Home::new() else { return };
    home.write_config(
        "future_setting = true\n[calcifer]\nport = 7433\nidle_s = 3\n[rentals]\nmax = 1\n",
    );
    let out = home.cozy(["hub", "list", "--json"]);
    assert_eq!(out.code, 0, "{out:?}");
    for key in ["`future_setting`", "`calcifer.idle_s`", "`rentals`"] {
        assert_eq!(out.stderr.matches(key).count(), 1, "{key}: {}", out.stderr);
    }
    home.write_config("[calcifer]\nport = \"soon\"\n");
    let out = home.cozy(["hub", "list", "--json"]);
    let error = out.refusal();
    assert_eq!(
        (out.code, error["code"].as_str()),
        (2, Some("config_invalid")),
        "{out:?}"
    );
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("`calcifer.port`"),
        "{out:?}"
    );
    // An edit never writes a file the load would refuse.
    let out = home.cozy(["hub", "add", "x", "http://127.0.0.1:3", "--json"]);
    assert_ne!(out.code, 0, "{out:?}");
    assert_eq!(home.config(), "[calcifer]\nport = \"soon\"\n");
}

/// With `--json` a usage error is the typed envelope on stdout, exit 2; for a person it is
/// clap's message.
#[test]
fn usage_errors_are_typed() {
    let Some(home) = Home::new() else { return };
    let out = home.cozy(["hub", "add", "--json"]);
    assert_eq!(
        (out.code, out.refusal()["kind"].as_str()),
        (2, Some("usage")),
        "{out:?}"
    );
    let out = home.cozy(["hub", "add"]);
    assert_eq!(out.code, 2);
    assert!(
        out.stdout.is_empty() && out.stderr.contains("Usage"),
        "{out:?}"
    );
}

/// The shell asks the binary itself; completion reads the grammar and config.toml and writes
/// nothing.
#[test]
fn completion_answers_from_the_grammar_and_the_hubs() {
    let Some(home) = Home::new() else { return };
    for shell in ["bash", "zsh", "fish", "powershell"] {
        let out = home.cozy(["completion", shell]);
        assert!(
            out.code == 0 && out.stdout.contains("COMPLETE"),
            "{shell}: {out:?}"
        );
    }
    let complete = |words: &[&str]| {
        let out = common::out(
            home.command()
                .env("COMPLETE", "fish")
                .arg("--")
                .args(words)
                .output()
                .unwrap(),
        );
        assert_eq!(out.code, 0, "{out:?}");
        out.stdout
    };
    let verbs = complete(&["cozy", "hu"]);
    assert!(verbs.lines().any(|l| l.starts_with("hub")), "{verbs}");
    assert_eq!(std::fs::read_dir(home.root()).unwrap().count(), 1, "completion created the home");
    home.json(["hub", "add", "local", "http://127.0.0.1:1", "--json"]);
    let before = home.config();
    let names = complete(&["cozy", "hub", "use", "lo"]);
    assert!(names.lines().any(|l| l.starts_with("local")), "{names}");
    let flag = complete(&["cozy", "--tensorhub", "t"]);
    assert!(flag.lines().any(|l| l.starts_with("tensorhub")), "{flag}");
    assert_eq!(home.config(), before);
}
