//! `cozy machine rent` against a real Tensorhub: Tensorhub's dev-build `cozy_ref` job starts one and
//! names it in TENSORHUB_URL. Nothing is bought: the catalog and a product's standing are reads.
mod common;

use common::Home;

/// A home whose current Hub is the job's; `None` (skipped, said aloud) outside that job.
fn hub_home() -> Option<Home> {
    let Some(url) = std::env::var("TENSORHUB_URL").ok().filter(|u| !u.is_empty()) else {
        eprintln!("skipped: TENSORHUB_URL is unset; Tensorhub's dev-build cozy_ref job sets it");
        return None;
    };
    let home = Home::new()?;
    home.write_config(&format!("hub = \"ci\"\n\n[hubs]\nci = \"{url}\"\n"));
    Some(home)
}

/// The catalog lists every product, cheapest expected total first, and refuses nothing.
#[test]
fn the_catalog_lists_products_by_expected_total() {
    let Some(home) = hub_home() else { return };
    let skus = home.json(["machine", "rent", "--json"]);
    let skus = skus.as_array().expect("the catalog is a list");
    let totals: Vec<i64> = skus
        .iter()
        .map(|s| s["widths"][0]["expected_price"]["total_usd_micros"].as_i64().unwrap())
        .collect();
    assert!(totals.is_sorted(), "{totals:?}");
    eprintln!("the CI Hub lists {} products", skus.len());
    let out = home.cozy(["machine", "rent"]);
    assert_eq!(out.code, 0, "{out:?}");
    assert!(out.stdout.contains("NAME") && out.stdout.contains("EXPECTED"), "{out:?}");
    match skus.first() {
        Some(_) => assert!(out.stderr.contains("EXPECTED is"), "{out:?}"),
        None => assert!(out.stderr.contains("No machine is for rent"), "{out:?}"),
    }
}

/// An ask the Hub cannot fill is refused, naming the absence, and never widened to another card or
/// GPU count.
#[test]
fn an_ask_the_hub_cannot_fill_is_refused_and_never_widened() {
    let Some(home) = hub_home() else { return };
    let out = home.cozy(["machine", "rent", "no-such-gpu", "--json"]);
    let refusal = out.refusal();
    assert_eq!((out.code, refusal["code"].as_str()), (1, Some("sku_unknown")), "{out:?}");
    assert!(refusal["message"].as_str().unwrap().contains("nothing was rented"), "{out:?}");
    assert_eq!(refusal["next"][0], "cozy machine rent");

    let skus = home.json(["machine", "rent", "--json"]);
    let Some(sku) = skus.as_array().unwrap().first().cloned() else {
        eprintln!("the CI Hub lists no product: the out-of-stock refusal is not exercised here");
        return;
    };
    let name = sku["name"].as_str().unwrap();
    let widest = sku["widths"].as_array().unwrap().iter().map(|w| w["accelerator_count"].as_i64().unwrap());
    let unsold = (widest.max().unwrap() + 1).to_string();
    let out = home.cozy(["machine", "rent", name, "--gpus", unsold.as_str(), "--json"]);
    assert_eq!((out.code, out.refusal()["code"].as_str()), (1, Some("sku_out_of_stock")), "{out:?}");
    assert!(out.refusal()["message"].as_str().unwrap().contains("is offered now with"), "{out:?}");
}

/// Refusals that need no Hub: this computer's machine role waits for TensorD in the binary, and rental
/// options without a product are usage errors.
#[test]
fn machine_usage_refusals() {
    let Some(home) = Home::new() else { return };
    for verb in ["enable", "disable"] {
        let out = home.cozy(["machine", verb, "--json"]);
        assert_eq!((out.code, out.refusal()["code"].as_str()), (2, Some("not_ported")), "{out:?}");
    }
    for args in [
        vec!["machine", "rent", "--gpus", "2"],
        vec!["machine", "rent", "--disk-gb", "500"],
        vec!["machine", "add", "box", "127.0.0.1:1", "--fingerprint", "abc"],
        vec!["machine", "logs", "box", "nonsense"],
    ] {
        let out = home.cozy(&args);
        assert_eq!(out.code, 2, "{args:?}: {out:?}");
    }
    let help = home.cozy(["machine", "--help"]);
    for verb in ["list", "show", "logs", "rent", "end", "keepalive", "update", "add", "enable", "disable", "authorize"] {
        assert!(help.stdout.contains(verb), "{verb}: {help:?}");
    }
}
