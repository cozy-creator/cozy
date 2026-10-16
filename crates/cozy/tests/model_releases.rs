//! `cozy model family|publish|retarget|yank|delete` against a real Tensorhub (Tensorhub's
//! dev-build `cozy_ref` job sets TENSORHUB_URL): signed out they refuse before any write, signed
//! in each reaches the Hub as the account and its refusal comes back verbatim. The success path
//! (upload, then publish, retarget, yank, delete) runs with `cozy model upload`.
mod common;

use common::Home;

const CHECKPOINT: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

#[test]
fn release_verbs_speak_as_the_signed_in_account() {
    let Some(home) = Home::with_hub() else { return };
    let lane = format!("int8={CHECKPOINT}");

    let out = home.cozy(["--json", "model", "publish", "lmodel/none", "--release", "1.0.0", "--lane", &lane]);
    assert_eq!((out.code, out.refusal()["code"].as_str()), (1, Some("not_logged_in")), "{out:?}");

    let account = home.sign_in("lmodel@example.test");
    let model = format!("{account}/none");
    let asks: [Vec<&str>; 6] = [
        vec!["model", "family", &model, "anime"],
        vec!["model", "publish", &model, "--release", "1.0.0", "--lane", &lane],
        vec!["model", "retarget", &model, "--release", "1.0.0", "--lane", "int8", "--to", CHECKPOINT],
        vec!["model", "yank", &model, "--release", "1.0.0"],
        vec!["model", "delete", &model, "--yes"],
        vec!["model", "publish", "fidika/elsewhere", "--release", "1.0.0", "--lane", &lane],
    ];
    for ask in asks {
        let out = home.cozy(["--json"].into_iter().chain(ask.iter().copied()));
        let refusal = out.refusal();
        println!("{ask:?} → {refusal}");
        assert_eq!(out.code, 1, "{ask:?}: {out:?}");
        // The Hub's own code, verbatim, and never a transport or credential failure.
        assert_eq!(refusal["cause"], refusal["code"], "{ask:?}: {out:?}");
        assert!(!matches!(refusal["code"].as_str(), Some("not_logged_in" | "hub_unreachable" | "hub_credential")), "{ask:?}: {out:?}");
    }

    let found = home.json(["--json", "model", "search"]);
    assert_eq!(found["models"].as_array().map(Vec::len), Some(0), "{found}");
}
