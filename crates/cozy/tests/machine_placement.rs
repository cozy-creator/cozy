//! A rental the Hub placed off its requester's continent says why, wherever a person meets it:
//! renting it, a run that rents it, `machine show` and `machine list`. Tensorhub's dev-build
//! `cozy_ref` job with `cozy_client_ip` makes the requester North American (COZY_TEST_CLIENT_IP);
//! its docker provider states no location, so every offer is remote. Elsewhere it skips.
mod common;

use common::Home;
use serde_json::Value;

const FAR: &str = "no CPU in North America fits now";

fn rental(home: &Home, name: &str) -> Value {
    let list = home.json(["machine", "list", "--json"]);
    let machines = list["machines"].as_array().unwrap();
    let found = machines.iter().find(|m| m["name"] == name);
    found
        .unwrap_or_else(|| panic!("{name} is not listed: {list}"))
        .clone()
}

#[test]
fn a_rental_placed_off_its_requesters_continent_says_why() {
    let Some(home) = Home::with_hub() else { return };
    // The job names it empty when no client address was asked for.
    let client = "a client continent: tensorhub's cozy_ref job with cozy_client_ip";
    let Some(_) = common::needs("COZY_TEST_CLIENT_IP", client) else {
        return;
    };
    home.sign_in("placed@cozy.test");

    // The Hub places a rental after it accepts the order: the boot's follower says why.
    let rented = home.cozy(["machine", "rent", "cpu", "--yes"]);
    assert_eq!(rented.code, 0, "{rented:?}");
    assert!(rented.stderr.contains(FAR), "{rented:?}");
    let name = rented
        .stdout
        .lines()
        .find_map(|l| l.strip_prefix("machine"))
        .map(|rest| rest.trim().to_owned())
        .unwrap_or_else(|| panic!("no machine named: {rented:?}"));
    let machine = rental(&home, &name);
    let note = machine["rental"]["placement_note"]
        .as_str()
        .unwrap_or_default();
    assert!(note.starts_with(FAR), "{machine}");
    let id = machine["rental_id"].as_str().unwrap().to_owned();
    let shown = home.cozy(["machine", "show", &id]);
    assert!(
        shown.stdout.contains("placement") && shown.stdout.contains(FAR),
        "{shown:?}"
    );
    let listed = home.cozy(["machine", "list"]);
    assert!(
        listed.stdout.contains(&format!("{name}: {FAR}")),
        "{listed:?}"
    );
    let ended = home.json(["machine", "end", &id, "--json"]);
    assert_eq!(ended["changed"], true, "{ended}");

    // A run that rents its machine says it too, once it is placed.
    let package = common::package::publish(&home, "tiny", "0.1.0");
    let echo = format!("{package}/echo");
    // Renting for a run is agreed to like any rental: refused with its quote without a yes.
    let unagreed = home.cozy(["run", &echo, "text=far", "--on", "new:cpu", "--json"]);
    let refusal = unagreed.refusal();
    assert!(refusal["code"] == "quote_unagreed" && refusal["message"].to_string().contains("cpu ×1: $"), "{unagreed:?}");
    let out = home.cozy(["run", &echo, "text=far", "--on", "new:cpu", "--yes"]);
    assert_eq!(out.code, 0, "{out:?}");
    assert!(out.stderr.contains(FAR), "{out:?}");
    let list = home.json(["machine", "list", "--json"]);
    for machine in list["machines"].as_array().unwrap() {
        if let Some(id) = machine["rental_id"].as_str() {
            home.json(["machine", "end", id, "--json"]);
        }
    }
}
