//! `cozy tools` and `cozy doctor` on this computer, through the real binary. With no ffmpeg on
//! PATH, doctor says what its absence costs and what installs it; `cozy tools install ffmpeg`
//! fetches the pinned build into `~/.cozy/tools`, held to its sha256, and doctor then shows its
//! version. uv installs the same way. Every OS cozy ships for.
mod common;

use common::{Home, Out};
use serde_json::Value;

fn json(out: &Out) -> Value {
    assert_eq!(out.code, 0, "{out:?}");
    serde_json::from_str(&out.stdout).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}

fn ffmpeg(doctor: &Value) -> Value {
    let tools = doctor["tools"]
        .as_array()
        .unwrap_or_else(|| panic!("{doctor}"));
    tools
        .iter()
        .find(|t| t["name"] == "ffmpeg")
        .cloned()
        .unwrap_or_else(|| panic!("{doctor}"))
}

#[test]
fn a_missing_ffmpeg_is_installed_pinned_and_doctor_shows_it() {
    let Some(home) = Home::new() else { return };
    // A PATH with no ffmpeg on it.
    let empty = tempfile::tempdir().unwrap();
    let cozy = |args: &[&str]| {
        common::out(
            home.command()
                .env("PATH", empty.path())
                .args(args)
                .output()
                .unwrap(),
        )
    };
    let before = json(&cozy(&["doctor", "--json"]));
    // xdg-open and systemctl are Linux's: macOS and Windows have their own opener and login start.
    let names: Vec<&str> = before["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    for linux in ["xdg-open", "systemctl"] {
        assert_eq!(names.contains(&linux), cfg!(target_os = "linux"), "{names:?}");
    }
    let missing = ffmpeg(&before);
    assert_eq!(
        (
            missing["found"].as_bool(),
            missing["cost"].as_str(),
            missing["install"].as_str()
        ),
        (
            Some(false),
            Some("uploads go untrimmed (larger, slower)"),
            Some("cozy tools install ffmpeg")
        ),
        "{before}"
    );
    assert!(before["machine"].is_null(), "{before}");

    let installed = json(&cozy(&["tools", "install", "ffmpeg", "--json"]));
    let tools = home.root().join("tools");
    assert_eq!(installed["installed"].as_bool(), Some(true), "{installed}");
    assert!(
        installed["path"]
            .as_str()
            .is_some_and(|p| p.starts_with(tools.to_str().unwrap())),
        "{installed}"
    );
    assert!(
        installed["version"]
            .as_str()
            .is_some_and(|v| v.starts_with("ffmpeg version 9.0.2")),
        "{installed}"
    );
    let after = ffmpeg(&json(&cozy(&["doctor", "--json"])));
    assert_eq!(after["found"].as_bool(), Some(true), "{after}");
    assert_eq!(after["version"], installed["version"], "{after}");
    let again = json(&cozy(&[
        "tools",
        "install",
        "ffmpeg",
        "--if-missing",
        "--json",
    ]));
    assert_eq!(again["installed"].as_bool(), Some(false), "{again}");

    let uv = json(&cozy(&["tools", "install", "uv", "--json"]));
    assert!(
        uv["version"]
            .as_str()
            .is_some_and(|v| v.starts_with("uv 0.12.24")),
        "{uv}"
    );
    let listed = cozy(&["tools", "list"]);
    assert!(
        listed.code == 0
            && listed.stdout.contains("ffmpeg")
            && listed.stdout.contains("uv 0.12.24"),
        "{listed:?}"
    );
}
