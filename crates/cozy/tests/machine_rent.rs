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
    match skus.first() {
        Some(_) => assert!(out.stderr.contains("EXPECTED is"), "{out:?}"),
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

/// Refusals that need no Hub: this computer's machine role waits for TensorD in the binary, and rental
/// options without a product are usage errors.
#[test]
fn machine_usage_refusals() {
    let Some(home) = Home::new() else { return };
    for verb in ["enable", "disable"] {
        let out = home.cozy(["machine", verb, "--json"]);
        assert_eq!(
            (out.code, out.refusal()["code"].as_str()),
            (2, Some("not_ported")),
            "{out:?}"
        );
    }
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

/// With the Hub's docker provider (Tensorhub's dev-build `hub_set`), a CPU rental is quoted and asked
/// once under its key, seen booting, and ended only when the Hub affirms its container gone. Nothing
/// is bought from a provider.
#[test]
fn a_rental_is_asked_once_per_key_and_ends_with_the_hubs_proof() {
    let Some(home) = Home::with_hub() else { return };
    let skus = home.json(["machine", "rent", "--json"]);
    if !skus.as_array().unwrap().iter().any(|s| s["name"] == "cpu") {
        eprintln!("skipped: the Hub offers no cpu product (its docker provider is off)");
        return;
    }
    home.sign_in("renter@example.com");
    let _calcifer = home.want_calcifer();
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

/// A home whose current Hub is the job's public origin (its TLS front, trusted through
/// SSL_CERT_FILE), signed in as `email`: the production shape, in which the CLI, Calcifer and a
/// rented pod all name the Hub by one origin. Answers the account.
fn public_home(email: &str) -> Option<(Home, String)> {
    let Some(origin) = std::env::var("TENSORHUB_ORIGIN")
        .ok()
        .filter(|o| !o.is_empty())
    else {
        eprintln!("skipped: TENSORHUB_ORIGIN is unset; Tensorhub's dev-build cozy_ref job sets it");
        return None;
    };
    let home = Home::new()?;
    home.write_config(&format!("hub = \"ci\"\n\n[hubs]\nci = \"{origin}\"\n"));
    let hub = cozy_proto::config::Hub {
        name: Some("ci".into()),
        origin: cozy_proto::config::origin(&origin).unwrap(),
    };
    let paths = cozy_platform::paths::Home::at(home.root());
    let log = std::env::var("TENSORHUB_LOG").expect("the Tensorhub's log");
    let account = email.split('@').next().unwrap().to_owned();
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let http = reqwest::Client::new();
        let anonymous =
            hub_client::Client::connect(&origin, http.clone(), hub_client::Credential::Anonymous);
        let (key, bearer) = hub_client::testing::sign_up(&anonymous, email, log.as_ref())
            .await
            .expect("sign up");
        std::fs::create_dir_all(paths.keys()).unwrap();
        key.save(&hub.device_key(&paths)).unwrap();
        hub_client::Client::connect(&origin, http, hub_client::Credential::Bearer(bearer.token))
            .register_account(&account, &hub_client::types::Empty(Default::default()))
            .await
            .expect("an account name");
    });
    Some((home, account))
}

/// `tests/fixtures/package/<name>` published to the home's Hub at 0.1.0 by `cozy package lock` and
/// `cozy package publish`.
fn publish_fixture(home: &Home, name: &str) {
    fn copy(from: &std::path::Path, to: &std::path::Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            match entry.file_type().unwrap().is_dir() {
                true => copy(&entry.path(), &to.join(entry.file_name())),
                false => drop(std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap()),
            }
        }
    }
    let dir = home.path().join("src").join(name);
    copy(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/package")
            .join(name),
        &dir,
    );
    for verb in ["lock", "publish"] {
        let out = common::out(
            home.command()
                .current_dir(&dir)
                .args(["package", verb, "--json"])
                .output()
                .unwrap(),
        );
        assert_eq!(out.code, 0, "package {verb}: {out:?}");
    }
}

/// The rental path end to end on the Hub's docker provider: a CPU machine is rented; a run sent to it
/// while it boots waits, then completes on it once it is ready; the rental ends only on the Hub's
/// affirmed absence and leaves the list.
#[test]
fn a_run_sent_to_a_booting_rental_completes_on_it_and_the_rental_ends() {
    let Some((home, account)) = public_home("cycle@cozy.test") else {
        return;
    };
    let _calcifer = home.want_calcifer();
    publish_fixture(&home, "tiny");
    let rented = home.json(["machine", "rent", "cpu", "--detach", "--json"]);
    let machine = rented["machine"]["name"].as_str().unwrap().to_string();
    let id = rented["machine"]["rental_id"].as_str().unwrap().to_string();
    assert_eq!(rented["machine"]["state"], "booting", "{rented}");

    // Sent while the machine boots: the run waits for it, unsent.
    let target = format!("{account}/tiny/echo");
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
