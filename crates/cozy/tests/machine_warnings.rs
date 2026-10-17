//! What a rental's Hub warns of, wherever a person meets the rental: an admin hears that its
//! prepaid provider account runs low (renting it, `machine show`, `machine list`), and a ready
//! rental the Hub cannot reach, though its provider reports it running, reads so with the way to
//! end it. Tensorhub's dev-build `cozy_ref` job names its Hub binary (TENSORHUB_ADMIN) and
//! database (TENSORHUB_DATABASE_URL); elsewhere it skips.
#![cfg(unix)]
mod common;

use std::{
    process::Command,
    time::{Duration, Instant},
};

use common::Home;
use serde_json::Value;

const LOW: &str = "docker balance $0.20 covers about 2 h";

/// The docker provider's account, as a prepaid adapter would record it: $0.20 left at $0.10/h.
const FUNDS: &str = "INSERT INTO hub_v2.provider_funds(provider,balance_usd_micros,burn_usd_micros_per_hour) \
     VALUES('docker',200000,100000) ON CONFLICT (provider) DO UPDATE SET \
     balance_usd_micros=EXCLUDED.balance_usd_micros, \
     burn_usd_micros_per_hour=EXCLUDED.burn_usd_micros_per_hour, observed_at=now()";

fn run(program: &std::ffi::OsStr, args: &[&str]) -> String {
    let out = Command::new(program).args(args).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(out.status.success(), "{program:?} {args:?}: {out:?}");
    text
}

fn rental(home: &Home, name: &str) -> Value {
    let list = home.json(["machine", "list", "--json"]);
    let machines = list["machines"].as_array().unwrap();
    let found = machines.iter().find(|m| m["name"] == name);
    found
        .unwrap_or_else(|| panic!("{name} is not listed: {list}"))
        .clone()
}

/// The docker container this rental runs in: the one publishing its machine's port, else the one
/// container whose TensorD answers `pidof`.
fn container(machine: &Value) -> String {
    let address = machine["address"].as_str().unwrap_or_default();
    let port = address.rsplit(':').next().unwrap_or_default();
    let filter = format!("publish={port}");
    let published = run("docker".as_ref(), &["ps", "-q", "--filter", &filter]);
    let all = run("docker".as_ref(), &["ps", "-q"]);
    let holds = |id: &&str| {
        let probe = Command::new("docker")
            .args(["exec", id, "pidof", "tensord"])
            .output();
        probe.is_ok_and(|out| out.status.success())
    };
    let found: Vec<&str> = match published.lines().count() {
        1 => published.lines().collect(),
        _ => all.lines().filter(holds).collect(),
    };
    match found[..] {
        [id] => id.to_owned(),
        _ => panic!("no one container holds {machine}: {found:?}"),
    }
}

#[test]
fn a_rentals_hub_warnings_reach_whoever_meets_it() {
    let Some(home) = Home::with_hub() else { return };
    let (Some(admin), Some(database)) = (
        std::env::var_os("TENSORHUB_ADMIN"),
        std::env::var_os("TENSORHUB_DATABASE_URL"),
    ) else {
        eprintln!("skipped: no Hub to administer (TENSORHUB_ADMIN is unset)");
        return;
    };
    let account = home.sign_in("warned@cozy.test");
    run(&admin, &["admin", "grant", &account]);
    let database = database.to_string_lossy().into_owned();
    run(
        "psql".as_ref(),
        &[&database, "-v", "ON_ERROR_STOP=1", "-c", FUNDS],
    );

    // The Hub places it after it accepts the order: the boot's follower says what it warns of.
    let rented = home.cozy(["machine", "rent", "cpu"]);
    assert_eq!(rented.code, 0, "{rented:?}");
    assert!(rented.stderr.contains(LOW), "{rented:?}");
    let name = rented
        .stdout
        .lines()
        .find_map(|l| l.strip_prefix("machine"))
        .map(|rest| rest.trim().to_owned())
        .unwrap_or_else(|| panic!("no machine named: {rented:?}"));
    let machine = rental(&home, &name);
    let warning = machine["rental"]["provider_warning"].as_str();
    assert!(warning.is_some_and(|w| w.starts_with(LOW)), "{machine}");
    let id = machine["rental_id"].as_str().unwrap().to_owned();
    let shown = home.cozy(["machine", "show", &id]);
    assert!(
        shown.stdout.contains("provider") && shown.stdout.contains(LOW),
        "{shown:?}"
    );
    let listed = home.cozy(["machine", "list"]);
    assert!(
        listed.stdout.contains(&format!("{name}: {LOW}")),
        "{listed:?}"
    );

    // TensorD frozen in its container: docker reports it running, and the Hub cannot reach it.
    let container = container(&machine);
    let signal = |sig: &str| {
        let line = format!("kill -{sig} $(pidof tensord)");
        run("docker".as_ref(), &["exec", &container, "sh", "-c", &line]);
    };
    signal("STOP");
    // A hang bound for a Hub that never notices; the product waits on its own passes.
    let until = Instant::now() + Duration::from_secs(600);
    let machine = loop {
        let machine = rental(&home, &name);
        if machine["rental"]["unreachable_since_ms"].is_i64() {
            break machine;
        }
        assert!(Instant::now() < until, "the Hub never missed it: {machine}");
        std::thread::sleep(Duration::from_secs(1));
    };
    let listed = home.cozy(["machine", "list"]);
    // Read while it is frozen: show asks the machine itself, which answers no one now.
    let shown = home.cozy(["machine", "show", &id]);
    signal("CONT");
    assert!(
        listed.stdout.contains("unreachable from the Hub since")
            && listed
                .stdout
                .contains(&format!("`cozy machine end {name}`")),
        "{machine}\n{listed:?}"
    );
    assert!(
        shown.stdout.contains("the Hub cannot reach it"),
        "{shown:?}"
    );
    let ended = home.json(["machine", "end", &id, "--json"]);
    assert_eq!(ended["changed"], true, "{ended}");
}
