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
