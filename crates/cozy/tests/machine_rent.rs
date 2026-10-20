//! `cozy machine rent` against a real Tensorhub: Tensorhub's dev-build `cozy_ref` job starts one and
//! names it in TENSORHUB_URL. Nothing is bought: the catalog and a product's standing are reads.
mod common;

use common::Home;
use serde_json::Value;

/// The catalog lists every product, cheapest expected total first, and refuses nothing.
#[test]
fn the_catalog_lists_products_by_expected_total() {
    let Some(home) = Home::with_hub() else { return };
    let skus = home.json(["machine", "rent", "--json"]);
    let skus = skus.as_array().expect("the catalog is a list");
    let totals: Vec<i64> = skus
        .iter()
        .map(|s| {
            s["widths"][0]["expected_price"]["total_usd_micros"]
                .as_i64()
                .unwrap()
        })
        .collect();
    assert!(totals.is_sorted(), "{totals:?}");
    eprintln!("the CI Hub lists {} products", skus.len());
    let out = home.cozy(["machine", "rent"]);
    assert_eq!(out.code, 0, "{out:?}");
    assert!(
        out.stdout.contains("NAME") && out.stdout.contains("EXPECTED"),
        "{out:?}"
    );
    let download = (skus.iter())
        .map(|s| s["widths"][0]["expected_price"]["download_gb"].as_i64().unwrap())
        .max();
    match download {
        Some(gb) => assert!(out.stderr.contains(&format!("and a {gb} GB model download")), "{out:?}"),
        None => assert!(out.stderr.contains("No machine is for rent"), "{out:?}"),
    }
}

/// An ask the Hub cannot fill is refused, naming the absence, and never widened to another card or
/// GPU count.
#[test]
fn an_ask_the_hub_cannot_fill_is_refused_and_never_widened() {
    let Some(home) = Home::with_hub() else { return };
    let out = home.cozy(["machine", "rent", "no-such-gpu", "--json"]);
    let refusal = out.refusal();
    assert_eq!(
        (out.code, refusal["code"].as_str()),
        (1, Some("sku_unknown")),
        "{out:?}"
    );
    assert!(
        refusal["message"]
            .as_str()
            .unwrap()
            .contains("nothing was rented"),
        "{out:?}"
    );
    assert_eq!(refusal["next"][0], "cozy machine rent");

    let skus = home.json(["machine", "rent", "--json"]);
    let Some(sku) = skus.as_array().unwrap().first().cloned() else {
        eprintln!("the CI Hub lists no product: the out-of-stock refusal is not exercised here");
        return;
    };
    let name = sku["name"].as_str().unwrap();
    let widest = sku["widths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["accelerator_count"].as_i64().unwrap());
    let unsold = (widest.max().unwrap() + 1).to_string();
    let out = home.cozy(["machine", "rent", name, "--gpus", unsold.as_str(), "--json"]);
    assert_eq!(
        (out.code, out.refusal()["code"].as_str()),
        (1, Some("sku_out_of_stock")),
        "{out:?}"
    );
    assert!(
        out.refusal()["message"]
            .as_str()
            .unwrap()
            .contains("is offered now with"),
        "{out:?}"
    );
}

/// Refusals that need no Hub: rental options without a product are usage errors.
#[test]
fn machine_usage_refusals() {
    let Some(home) = Home::new() else { return };
    for args in [
        vec!["machine", "rent", "--gpus", "2"],
        vec!["machine", "rent", "--disk-gb", "500"],
        vec![
            "machine",
            "add",
            "box",
            "127.0.0.1:1",
            "--fingerprint",
            "abc",
        ],
        vec!["machine", "logs", "box", "nonsense"],
    ] {
        let out = home.cozy(&args);
        assert_eq!(out.code, 2, "{args:?}: {out:?}");
    }
    let help = home.cozy(["machine", "--help"]);
    for verb in [
        "list",
        "show",
        "logs",
        "rent",
        "end",
        "keepalive",
        "update",
        "add",
        "enable",
        "disable",
        "authorize",
    ] {
        assert!(help.stdout.contains(verb), "{verb}: {help:?}");
    }
}

/// The Hub's quote for exactly the order comes before it is placed, in the catalog's terms: on a
/// terminal it asks, and a no rents nothing; without one the command is the yes, and the rental
/// costs what was quoted. On the Hub's docker provider: nothing is bought from a provider.
#[cfg(unix)]
#[test]
fn a_rent_says_its_quote_before_it_is_placed() {
    let Some(home) = Home::with_hub() else { return };
    home.sign_in("quote@example.com");
    let mut command = home.command();
    command.args(["machine", "rent", "cpu"]);
    let mut asked = common::pty::Pty::spawn(&command);
    let screen = asked.until("the quote and its question", |s| s.contains("Rent it? [Y/n]"));
    assert!(
        screen.contains("cpu ×1: $") && screen.contains("; expected $"),
        "{screen}"
    );
    asked.key("n\r");
    assert!(asked.wait().success());
    assert!(asked.screen().contains("nothing was rented"), "{}", asked.screen());
    let rentals = |home: &Home| -> Vec<Value> {
        let listed = home.json(["machine", "list", "--json"]);
        let machines = listed["machines"].as_array().unwrap().iter();
        machines.filter(|m| m["kind"] == "rental").cloned().collect()
    };
    assert!(rentals(&home).is_empty(), "a no rented nothing");

    let out = home.cozy(["machine", "rent", "cpu", "--detach"]);
    assert_eq!(out.code, 0, "{out:?}");
    let quoted = out.stderr.lines().find(|l| l.starts_with("cpu ×1: $"));
    let quoted = quoted.unwrap_or_else(|| panic!("no quote: {out:?}"));
    let rented = rentals(&home);
    let [rental] = rented.as_slice() else {
        panic!("one rental: {rented:?}")
    };
    let hourly = rental["rental"]["usd_micros_per_hour"].as_i64().unwrap();
    assert!(
        quoted.contains(&format!(": ${:.2}/h (", hourly as f64 / 1e6)),
        "rented at {hourly} µ$/h, quoted {quoted}"
    );
    let id = rental["rental_id"].as_str().unwrap();
    let ended = home.json(["machine", "end", id, "--json"]);
    assert_eq!(ended["machine"]["state"], "ended", "{ended}");
}

/// With the Hub's docker provider (Tensorhub's dev-build `hub_set`), a CPU rental is quoted and asked
/// once under its key, seen booting, and ended only when the Hub affirms its container gone. Nothing
/// is bought from a provider.
#[test]
fn a_rental_is_asked_once_per_key_and_ends_with_the_hubs_proof() {
    let Some(home) = Home::with_hub() else { return };
    let skus = home.json(["machine", "rent", "--json"]);
    assert!(
        skus.as_array().unwrap().iter().any(|s| s["name"] == "cpu"),
        "the Hub offers no cpu product: the job's hub_set must turn its docker provider on"
    );
    home.sign_in("renter@example.com");
    let ask = [
        "machine",
        "rent",
        "cpu",
        "--detach",
        "--idempotency-key",
        "rent-test-1",
        "--json",
    ];
    let first = home.json(ask);
    let machine = &first["machine"];
    let id = machine["rental_id"]
        .as_str()
        .expect("a rental id")
        .to_string();
    assert!(
        matches!(machine["state"].as_str(), Some("booting" | "ready")),
        "{first}"
    );
    assert!(
        first["quote"]["expected_total_usd_micros"].is_i64(),
        "{first}"
    );
    assert_eq!(first["replay"], false, "{first}");
    // The same key is the same order: no second machine.
    let again = home.json(ask);
    assert_eq!(
        (
            again["machine"]["rental_id"].as_str(),
            again["replay"].as_bool()
        ),
        (Some(id.as_str()), Some(true)),
        "{again}"
    );
    let listed: Vec<Value> = home.json(["machine", "list", "--json"])["machines"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["rental_id"] == id.as_str())
        .cloned()
        .collect();
    assert_eq!(listed.len(), 1, "{listed:?}");

    // End returns once the Hub affirms the container gone.
    let ended = home.json(["machine", "end", id.as_str(), "--json"]);
    assert_eq!(
        (
            ended["changed"].as_bool(),
            ended["machine"]["state"].as_str()
        ),
        (Some(true), Some("ended")),
        "{ended}"
    );
    let again = home.json(["machine", "end", id.as_str(), "--json"]);
    assert_eq!(again["changed"], false, "{again}");
}

/// The rental path end to end on the Hub's docker provider: a CPU machine is rented; a run sent to it
/// while it boots waits, then completes on it once it is ready; the rental ends only on the Hub's
/// affirmed absence and leaves the list.
#[test]
fn a_run_sent_to_a_booting_rental_completes_on_it_and_the_rental_ends() {
    let Some(home) = Home::with_hub() else { return };
    home.sign_in("cycle@cozy.test");
    let package = common::package::publish(&home, "tiny", "0.1.0");
    let rented = home.json(["machine", "rent", "cpu", "--detach", "--json"]);
    let machine = rented["machine"]["name"].as_str().unwrap().to_string();
    let id = rented["machine"]["rental_id"].as_str().unwrap().to_string();
    assert_eq!(rented["machine"]["state"], "booting", "{rented}");

    // Sent while the machine boots: the run waits for it, unsent.
    let target = format!("{package}/echo");
    let queued = home.json([
        "run",
        &target,
        "text=cycle",
        "--on",
        &machine,
        "--detach",
        "--json",
    ]);
    assert_eq!(queued["status"], "queued", "{queued}");
    let number = queued["number"].to_string();

    // Followed to its end: the machine boots, takes the run and answers it. The Hub may name the
    // machine once it boots; the run moves with it, and the rental id names it throughout.
    let out = home.cozy(["run", "watch", &number, "--json"]);
    let last: Value = serde_json::from_str(out.stdout.lines().last().unwrap_or_default())
        .unwrap_or_else(|e| panic!("{e}: {out:?}"));
    let shown = home.json(["machine", "show", &id, "--json"]);
    assert_eq!(
        (
            out.code,
            last["type"].as_str(),
            last["run"]["machine"].as_str()
        ),
        (0, Some("run.completed"), shown["name"].as_str()),
        "{out:?}\n{shown}"
    );
    assert_eq!(last["run"]["result"]["text"], "cycle", "{last}");
    assert_eq!(shown["state"], "ready", "{shown}");

    // End returns once the Hub affirms the pod gone; it leaves the list.
    let ended = home.json(["machine", "end", &id, "--json"]);
    assert_eq!(
        (
            ended["changed"].as_bool(),
            ended["machine"]["state"].as_str()
        ),
        (Some(true), Some("ended")),
        "{ended}"
    );
    let list = home.json(["machine", "list", "--json"]);
    assert!(
        !list["machines"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["rental_id"] == id.as_str() && m["state"] != "ended"),
        "{list}"
    );
}
