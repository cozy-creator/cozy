//! `cozy run … --on new` asks the Hub where to run and says why, with two other machines it would
//! rent instead (xs-068, tracker design/placement.md): the real CLI, Calcifer and Hub, nothing
//! rented. Tensorhub's dev-build `cozy_ref` job with `hub_set` naming vast.ai
//! (`provider.vast.api_key=ci provider.vast.base_url=http://127.0.0.1:9`) sells vast cards beside
//! its docker CPU machines; this test seeds their offers in the Hub's database
//! (TENSORHUB_DATABASE_URL), as a vast observation would publish them.
#![cfg(unix)]
mod common;

use std::process::Command;

use common::Home;

/// One vast offer per (card, GPUs, $/h): every card holds a function that loads no model.
const OFFERS: [(&str, u32, u64); 9] = [
    ("NVIDIA GeForce RTX 3070", 1, 200_000),
    ("NVIDIA GeForce RTX 5080", 1, 500_000),
    ("NVIDIA GeForce RTX 4090", 1, 700_000),
    ("NVIDIA GeForce RTX 5090", 1, 1_000_000),
    ("NVIDIA H100 PCIe", 2, 4_000_000),
    ("NVIDIA H100 PCIe", 4, 8_400_000),
    ("NVIDIA H100 80GB HBM3", 1, 3_000_000),
    ("NVIDIA B200", 1, 7_000_000),
    ("NVIDIA B200", 2, 14_000_000),
];

/// The offers as vast's head generation, newer than any it observed.
fn seed() -> String {
    let generation = 1_000_000_000;
    let rows: Vec<String> = OFFERS
        .iter()
        .map(|(card, gpus, rate)| {
            format!(
                "('vast',{generation},'ci:{card}:{gpus}','{card}','ci-vast','{card}',{gpus},true,3,\
                 {rate},500,8,32,0,0,2000,'US')"
            )
        })
        .collect();
    format!(
        "BEGIN; \
         INSERT INTO hub_v2.provider_inventory_snapshots(provider,generation,snapshot_digest,observed_at) \
         VALUES ('vast',{generation},'sha256:{digest}',now()); \
         INSERT INTO hub_v2.provider_inventory_offers(provider,generation,offer_key,provider_sku,datacenter,\
         accelerator_model,accelerator_count,available,stock_grade,list_price_usd_micros_per_hour,\
         max_container_disk_gb,vcpu_count,memory_gb,storage_usd_micros_per_gb_hour,\
         bandwidth_usd_micros_per_gb,download_mbps,country) VALUES {rows}; \
         INSERT INTO hub_v2.provider_inventory_heads(provider,generation) VALUES ('vast',{generation}) \
         ON CONFLICT (provider) DO UPDATE SET generation=EXCLUDED.generation; COMMIT;",
        digest = "a".repeat(64),
        rows = rows.join(","),
    )
}

#[test]
fn a_new_run_says_where_the_hub_placed_it_and_what_else_it_would_rent() {
    let Some(home) = Home::with_hub() else { return };
    const HUB: &str = "the Hub's database: tensorhub's cozy_ref job names it";
    let Some(database) = common::needs("TENSORHUB_DATABASE_URL", HUB) else {
        return;
    };
    let database = database.to_string_lossy().into_owned();
    let seeded = Command::new("psql")
        .args([&database, "-v", "ON_ERROR_STOP=1", "-c", &seed()])
        .output()
        .unwrap();
    assert!(seeded.status.success(), "seeding the offers: {seeded:?}");
    home.sign_in("placed@cozy.test");
    let catalog = home.json(["machine", "rent", "--json"]);
    assert!(
        catalog.as_array().unwrap().iter().any(|p| p["name"] == "rtx-4090"),
        "the Hub sells no vast card: run with hub_set \
         'provider.vast.api_key=ci provider.vast.base_url=http://127.0.0.1:9': {catalog}"
    );
    let package = common::package::publish(&home, "placed", "0.1.0");

    // No --yes and no terminal: the quote is said and refused, and nothing is rented. Of the
    // cards fast enough (4090, 5090, H100, B200), the 4090 is the cheapest per run; the 3070 is
    // cheaper per run but slow; the B200 is the fastest. The H100 PCIe sells no single GPU.
    let quoted = home.cozy(["run", &format!("{package}/draw"), "--on", "new"]);
    assert_ne!(quoted.code, 0, "{quoted:?}");
    let said = &quoted.stderr;
    for line in [
        "rtx-4090 ×1: $",
        "the Hub placed it on rtx-4090 (1× NVIDIA GeForce RTX 4090, 24 GB)",
        "the cheapest per run of 4 machines fast enough for 0 GB of 8-bit weights",
        "or rtx-3070 ×1 (cheaper per run): 3.30× the time, 0.94× the cost per run",
        "or b200 ×1 (faster): 0.36× the time, 3.59× the cost per run",
        "quote_unagreed",
    ] {
        assert!(said.contains(line), "{line:?} is not said: {quoted:?}");
    }
    let listed = home.json(["machine", "list", "--json"]);
    assert!(
        listed["machines"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["rental_id"].is_null()),
        "nothing is rented: {listed}"
    );

    // `--on new:<sku>`: the Hub ranks that product alone; `--gpus` fixes its count.
    let b200 = home.cozy(["run", &format!("{package}/draw"), "--on", "new:b200"]);
    assert!(
        b200.stderr.contains("b200 ×1: $") && b200.stderr.contains("the Hub placed it on b200"),
        "{b200:?}"
    );
    let pair = home.cozy(["run", &format!("{package}/draw"), "--on", "new:b200", "--gpus", "2"]);
    assert!(pair.stderr.contains("b200 ×2: $"), "{pair:?}");
}
