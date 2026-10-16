//! Signing in at a real Tensorhub (Tensorhub's dev-build `cozy_ref` job), through the binary.
mod common;

use common::Home;

/// `whoami` reads the account from the Hub with this computer's device key, and says so when
/// there is none.
#[test]
fn whoami_says_who_this_computer_signs_in_as() {
    let Some(home) = Home::with_hub() else { return };
    let out = home.json(["whoami", "--json"]);
    assert_eq!(
        (out["logged_in"].as_bool(), out["hub"]["name"].as_str()),
        (Some(false), Some("ci")),
        "{out}"
    );

    let account = home.sign_in("whoami@example.test");
    let out = home.json(["whoami", "--json"]);
    assert_eq!(out["logged_in"], true, "{out}");
    assert_eq!(out["email"], "whoami@example.test", "{out}");
    assert_eq!(out["account"], account.as_str(), "{out}");
    let hubs = home.json(["hub", "list", "--json"]);
    assert!(
        hubs["hubs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|h| h["name"] == "ci" && h["logged_in"] == true),
        "{hubs}"
    );

    // Another Hub this computer has no key for is not signed in.
    let out = home.json(["--tensorhub", "tensorhub", "whoami", "--json"]);
    assert_eq!(out["logged_in"], false, "{out}");
}

/// `cozy login <email>`, answering its prompts as a person would: the emailed code once the
/// Hub has sent it, then the account name if asked.
fn login(home: &Home, email: &str, name: &str, args: &[&str]) -> common::Out {
    use std::io::{Read as _, Write as _};
    let log = std::env::var("TENSORHUB_LOG").unwrap();
    let mut child = home
        .command()
        .args(args)
        .arg("--json")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let mut said = vec![];
    let mut byte = [0];
    while !String::from_utf8_lossy(&said).contains("Code: ") {
        if stderr.read(&mut byte).unwrap() == 0 {
            break;
        }
        said.push(byte[0]);
    }
    if String::from_utf8_lossy(&said).contains("Code: ") {
        let code = hub_client::testing::emailed_code(log.as_ref(), email);
        let mut stdin = child.stdin.take().unwrap();
        writeln!(stdin, "{code}\n{name}").unwrap();
    }
    let mut out = common::out(child.wait_with_output().unwrap());
    out.stderr = format!("{}{}", String::from_utf8_lossy(&said), out.stderr);
    out
}

fn doc(out: &common::Out) -> serde_json::Value {
    assert_eq!(out.code, 0, "{out:?}");
    serde_json::from_str(&out.stdout).unwrap()
}

/// Login enrolls a key by the emailed code and names the account; a second login with a key the
/// Hub accepts asks nothing; logout erases the key.
#[test]
fn login_names_the_account_and_logout_forgets_the_key() {
    let Some(home) = Home::with_hub() else { return };
    let email = "login@example.test";
    let first = doc(&login(&home, email, "LoginName", &["login", email]));
    assert_eq!(
        (first["status"].as_str(), first["account"].as_str()),
        (Some("signed in"), Some("loginname")),
        "{first}"
    );
    assert_eq!(home.json(["whoami", "--json"])["account"], "loginname");
    let again = doc(&login(&home, email, "", &["login", email]));
    assert_eq!(again["status"], "already signed in", "{again}");

    let out = home.json(["logout", "--json"]);
    assert_eq!(out["status"], "signed out", "{out}");
    assert!(
        out["notes"].as_array().unwrap().is_empty(),
        "the Hub did not revoke the key: {out}"
    );
    assert_eq!(home.json(["whoami", "--json"])["logged_in"], false);
    assert!(
        !home
            .root()
            .join("keys")
            .read_dir()
            .unwrap()
            .any(|e| e.is_ok()),
        "a key file is left"
    );
    let back = doc(&login(&home, email, "", &["login", email]));
    assert_eq!(back["account"], "loginname", "{back}");
}

/// `logout --everywhere` proves the mailbox, then signs every other computer out too.
#[test]
fn logout_everywhere_signs_out_the_other_computers() {
    let (Some(here), Some(there)) = (Home::with_hub(), Home::with_hub()) else {
        return;
    };
    let email = "everywhere@example.test";
    doc(&login(&here, email, "everywhere", &["login", email]));
    doc(&login(&there, email, "", &["login", email]));
    assert_eq!(there.json(["whoami", "--json"])["logged_in"], true);
    let out = doc(&login(&here, email, "", &["logout", "--everywhere"]));
    assert_eq!(
        (out["status"].as_str(), out["everywhere"].as_bool()),
        (Some("signed out"), Some(true)),
        "{out}"
    );
    assert_eq!(there.json(["whoami", "--json"])["logged_in"], false);
}

/// Credit reads answer metered or unmetered; a purchase gets a payment page or a typed refusal;
/// money is never parsed loosely.
#[test]
fn credits_read_and_buy() {
    let Some(home) = Home::with_hub() else { return };
    let out = home.cozy(["credits", "--json"]);
    assert_eq!(
        (out.code, out.refusal()["code"].as_str()),
        (1, Some("not_logged_in")),
        "{out:?}"
    );
    home.sign_in("credits@example.test");
    let credit = home.json(["credits", "--json"]);
    assert!(
        credit["billing"] == "unmetered" || credit["balance_micros"].is_i64(),
        "{credit}"
    );
    let out = home.cozy(["credits", "buy", "ten", "--json"]);
    assert_eq!(
        (out.code, out.refusal()["code"].as_str()),
        (2, Some("usd_invalid")),
        "{out:?}"
    );
    for args in [
        &["credits", "history", "--json"][..],
        &[
            "credits",
            "buy",
            "10",
            "--no-browser",
            "--no-wait",
            "--json",
        ],
    ] {
        let out = home.cozy(args);
        match out.code {
            0 => assert!(!out.stdout.is_empty(), "{out:?}"),
            _ => assert!(
                !out.refusal()["code"]
                    .as_str()
                    .unwrap_or_default()
                    .is_empty(),
                "{out:?}"
            ),
        }
    }
}
