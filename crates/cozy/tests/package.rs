//! `cozy package search|info` through the real binary, against tensorhub.com, read anonymously.
mod common;

use common::Home;

/// A search finds a published package, says how many a `--limit` left out and how to see them.
#[test]
fn search_finds_packages_and_counts_what_it_left_out() {
    let Some(home) = Home::new() else { return };
    let found = home.json(["--json", "package", "search", "sdxl"]);
    let names: Vec<&str> = found["packages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["package"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"fidika/sdxl"), "{found}");

    let every = home.json(["--json", "package", "search"]);
    let total = every["total"].as_i64().unwrap();
    assert!(total >= 2, "{every}");
    let one = home.json(["--json", "package", "search", "--limit", "1"]);
    assert_eq!(one["packages"].as_array().unwrap().len(), 1, "{one}");
    assert_eq!(one["omitted"].as_i64(), Some(total - 1), "{one}");
    let human = home.cozy(["package", "search", "--limit", "1"]);
    assert_eq!(human.code, 0, "{human:?}");
    assert!(
        human.stdout.contains(&format!(
            "{} more not shown. Use --limit {total}",
            total - 1
        )),
        "{human:?}"
    );
}

/// Info lists a package's releases newest first by version, whatever case its name is typed in;
/// a package the Hub lacks is a typed refusal that names where to look.
#[test]
fn info_lists_releases_newest_first() {
    let Some(home) = Home::new() else { return };
    let info = home.json(["--json", "package", "info", "Fidika/SDXL"]);
    assert_eq!(info["package"], "fidika/sdxl");
    let releases: Vec<Vec<u64>> = info["releases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            assert!(r["published"].as_str().unwrap().ends_with('Z'), "{r}");
            r["release"]
                .as_str()
                .unwrap()
                .split('.')
                .map(|n| n.parse().unwrap())
                .collect()
        })
        .collect();
    assert!(releases.len() > 1, "{info}");
    assert!(releases.windows(2).all(|w| w[0] > w[1]), "{info}");

    let missing = home.cozy(["--json", "package", "info", "fidika/no-such-package-xs043"]);
    assert_eq!(missing.code, 1, "{missing:?}");
    let refusal = missing.refusal();
    assert_eq!(refusal["code"], "package_not_found", "{refusal}");
    assert!(
        refusal["message"].as_str().unwrap().contains("tensorhub"),
        "{refusal}"
    );
    assert_eq!(
        refusal["next"][0], "cozy package search no-such-package-xs043",
        "{refusal}"
    );

    let usage = home.cozy(["package", "info", "sdxl"]);
    assert_eq!(usage.code, 2, "{usage:?}");
}
