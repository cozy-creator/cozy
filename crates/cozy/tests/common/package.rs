//! A test package published to the job's Tensorhub the way an author publishes one: the fixture
//! copied into the home, then `cozy package lock` and `cozy package publish` in its directory.
use std::path::{Path, PathBuf};

use super::Home;

/// Publishes `tests/fixtures/package/<fixture>` at `version` as `home`'s signed-in account and
/// answers `<account>/<name>`. Each account publishes its own, so tests sharing a Hub never collide.
pub fn publish(home: &Home, fixture: &str, version: &str) -> String {
    let dir = project(home, fixture);
    let pyproject = dir.join("pyproject.toml");
    let text: Vec<String> = std::fs::read_to_string(&pyproject)
        .unwrap()
        .lines()
        .map(|line| match line.starts_with("version = ") {
            true => format!("version = \"{version}\""),
            false => line.to_owned(),
        })
        .collect();
    std::fs::write(&pyproject, text.join("\n") + "\n").unwrap();
    for verb in ["lock", "publish"] {
        let out = super::out(
            home.command()
                .current_dir(&dir)
                .args(["--json", "package", verb])
                .output()
                .unwrap(),
        );
        assert_eq!(out.code, 0, "cozy package {verb}: {out:?}");
        if verb == "publish" {
            let published: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
            return published["package"].as_str().unwrap().to_owned();
        }
    }
    unreachable!()
}

/// The fixture's project directory in `home`, copied there on first use.
pub fn project(home: &Home, fixture: &str) -> PathBuf {
    let to = home.path().join("src").join(fixture);
    if !to.exists() {
        let from = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/package")
            .join(fixture);
        copy(&from, &to);
    }
    to
}

fn copy(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        match entry.file_type().unwrap().is_dir() {
            true => copy(&entry.path(), &to.join(entry.file_name())),
            false => drop(std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap()),
        }
    }
}
