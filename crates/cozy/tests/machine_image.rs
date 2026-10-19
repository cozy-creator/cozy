//! A gated image bake boots a rental on a worker image by digest: only a Hub admin may, through
//! the hidden `cozy machine rent --image-digest`, and `machine show` names the image a rental
//! boots. Tensorhub's dev-build `cozy_ref` job names its Hub binary (TENSORHUB_ADMIN); elsewhere
//! it skips.
#![cfg(unix)]
mod common;

use std::process::Command;

use common::Home;
use serde_json::Value;

/// A rental of `cpu` with `extra`, ready: its name, its id, and how `machine list` lists it.
fn rent(home: &Home, extra: &[&str]) -> (String, String, Value) {
    let rented = home.cozy(["machine", "rent", "cpu"].iter().chain(extra));
    assert_eq!(rented.code, 0, "{rented:?}");
    let name = rented
        .stdout
        .lines()
        .find_map(|l| l.strip_prefix("machine"))
        .map(|rest| rest.trim().to_owned())
        .unwrap_or_else(|| panic!("no machine named: {rented:?}"));
    let list = home.json(["machine", "list", "--json"]);
    let machine = list["machines"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == name.as_str())
        .unwrap_or_else(|| panic!("{name} is not listed: {list}"))
        .clone();
    let id = machine["rental_id"].as_str().unwrap().to_owned();
    (name, id, machine)
}

fn rentals(home: &Home) -> usize {
    let list = home.json(["machine", "list", "--json"]);
    let machines = list["machines"].as_array().unwrap();
    machines
        .iter()
        .filter(|m| m["rental_id"].is_string())
        .count()
}

#[test]
fn only_an_admin_pins_a_rentals_image_and_show_names_it() {
    let Some(home) = Home::with_hub() else { return };
    let hub = "the Hub's binary: tensorhub's cozy_ref job names it";
    let Some(admin) = common::needs("TENSORHUB_ADMIN", hub) else {
        return;
    };
    let account = home.sign_in("baker@cozy.test");

    // A malformed digest never reaches the Hub.
    let malformed = home.cozy(["machine", "rent", "cpu", "--image-digest", "sha256:abc"]);
    assert_eq!(malformed.code, 2, "{malformed:?}");
    // A renter chooses no image: the Hub refuses the quote, and nothing is rented.
    let pinned = format!("sha256:{}", "c".repeat(64));
    let refused = home.cozy([
        "--json",
        "machine",
        "rent",
        "cpu",
        "--image-digest",
        &pinned,
    ]);
    assert_eq!(refused.code, 1, "{refused:?}");
    let refusal = refused.refusal();
    assert!(
        refusal
            .to_string()
            .contains("rental.image_choice_operator_only"),
        "{refused:?}"
    );
    assert_eq!(rentals(&home), 0);

    // `machine show` names the image a rental boots: the current one when none is named.
    let granted = Command::new(&admin)
        .args(["admin", "grant", &account])
        .output()
        .unwrap();
    assert!(granted.status.success(), "{granted:?}");
    let (_, id, machine) = rent(&home, &[]);
    let current = machine["rental"]["base_worker_image_digest"]
        .as_str()
        .unwrap_or_else(|| panic!("no image digest: {machine}"))
        .to_owned();
    assert!(
        current.starts_with("sha256:") && current.len() == 71,
        "{machine}"
    );
    let shown = home.cozy(["machine", "show", &id]);
    assert!(
        shown
            .stdout
            .lines()
            .any(|l| l.starts_with("image") && l.ends_with(&current)),
        "{shown:?}"
    );
    let ended = home.json(["machine", "end", &id, "--json"]);
    assert_eq!(ended["changed"], true, "{ended}");

    // An admin pins a rental to a digest: it boots exactly that image.
    let (_, id, machine) = rent(&home, &["--image-digest", &current]);
    assert_eq!(
        machine["rental"]["base_worker_image_digest"].as_str(),
        Some(current.as_str()),
        "{machine}"
    );
    assert_eq!(machine["state"], "ready", "{machine}");
    let shown = home.cozy(["machine", "show", &id]);
    assert!(shown.stdout.contains(&current), "{shown:?}");
    let ended = home.json(["machine", "end", &id, "--json"]);
    assert_eq!(ended["changed"], true, "{ended}");
}
