//! `cozy migrate` through the real binary, in place in its own `~/.cozy`: a synthetic Go home
//! becomes this layout once, and the CLI reads it.
mod common;

use common::Home;
use migrate::testing::{self, Expected};

#[test]
fn the_go_home_is_imported_once_in_place() {
    let Some(home) = Home::new() else { return };
    let root = home.root();
    let outputs = testing::build(&root);
    let from = root.to_str().unwrap();

    let dry = home.json(["migrate", "--from", from, "--dry-run", "--json"]);
    assert_eq!(dry["runs"], Expected::RUNS);
    assert_eq!(dry["history"], Expected::HISTORY);
    assert_eq!(dry["unsettled_accepted"], 1);
    assert_eq!(dry["outputs"], Expected::OUTPUTS);
    assert!(root.join("creator.sqlite").is_file() && !home.config_path().exists());

    let done = home.json(["migrate", "--from", from, "--json"]);
    assert_eq!(done["runs"], Expected::RUNS);
    assert!(std::path::Path::new(done["archive"].as_str().unwrap()).is_file());
    assert!(!root.join("creator.sqlite").exists() && outputs.iter().all(|o| o.is_file()));

    let hubs = home.json(["hub", "list", "--json"]);
    let current: Vec<&str> = hubs["hubs"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|h| h["current"] == true)
        .map(|h| h["url"].as_str().unwrap())
        .collect();
    assert_eq!(current, [testing::LOCAL_HUB]);

    let again = home.cozy(["migrate", "--from", from, "--json"]);
    assert_eq!(again.code, 1, "{again:?}");
    assert!(again.stdout.contains("already_migrated"), "{again:?}");
}
