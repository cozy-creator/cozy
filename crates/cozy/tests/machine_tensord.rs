//! `cozy machine update <machine> --tensord <FILE>`: a dev `cozy` binary runs as the machine. The
//! machine checks the binary runs there before its accepted runs drain, restarts on it, and reports
//! its version and commit as dev. A binary for another platform is refused with the mismatch, and
//! nothing drains.
mod common;

use std::{
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use common::{Home, Out};
use serde_json::Value;

/// `cozy` installed in this home, as the installer puts it (the machine's launch path), and its
/// machine enabled.
fn enabled(home: &Home) -> PathBuf {
    let bin = home.root().join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let launch = bin.join(format!("cozy{}", std::env::consts::EXE_SUFFIX));
    std::fs::copy(env!("CARGO_BIN_EXE_cozy"), &launch).unwrap();
    home.use_binary(&launch);
    let out = home.cozy(["machine", "enable", "--json"]);
    assert_eq!(out.code, 0, "{out:?}");
    launch
}

/// Disables the machine and stops Calcifer, however the test ended.
struct Disabled<'a>(&'a Home);

impl Drop for Disabled<'_> {
    fn drop(&mut self) {
        let _ = self.0.cozy(["machine", "disable", "--json"]);
        let _ = self.0.cozy(["down", "--json"]);
    }
}

/// `cozy machine update local --tensord <binary>`: its output and its last event.
fn tensord(home: &Home, binary: &Path) -> (Out, Value) {
    let binary = binary.to_str().unwrap();
    let out = home.cozy(["machine", "update", "local", "--tensord", binary, "--json"]);
    let last = serde_json::from_str(out.stdout.lines().last().unwrap_or_default())
        .unwrap_or_else(|e| panic!("{e}: {out:?}"));
    (out, last)
}

/// A build of this checkout that the machines here run (`COZY_TEST_DEV_BINARY`): dev-build's
/// `build=cozy` (`<version>+dev.g<sha8>`), or the worker image's own cozy in Tensorhub's live job.
fn dev_binary() -> (PathBuf, String) {
    let provides =
        "a cozy the machines here run (dev-build build=cozy, tensorhub cozy_worker=branch)";
    let dev = PathBuf::from(common::needs("COZY_TEST_DEV_BINARY", provides).unwrap());
    let said = Command::new(&dev).arg("--version").output().unwrap().stdout;
    let said = String::from_utf8(said).unwrap();
    let version = said.trim().strip_prefix("cozy ").unwrap_or_default();
    assert!(
        !version.is_empty(),
        "{} --version said {said:?}",
        dev.display()
    );
    (dev, version.to_owned())
}

/// `machine show local`, once its status says `done`: Calcifer hears a restarted machine when
/// its link reconnects.
fn shown(home: &Home, done: impl Fn(&Value) -> bool) -> Value {
    loop {
        let shown = home.json(["machine", "show", "local", "--json"]);
        if done(&shown["status"]) {
            return shown;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

/// A dev build replaces the machine's binary. The machine restarts on it and reports its version
/// and commit as dev; the release it replaced is kept beside it as `.cozy-release`; this
/// computer's cozy is the dev build too.
#[cfg(target_os = "linux")]
#[test]
fn a_dev_binary_runs_as_the_machine_and_reports_its_version_and_commit() {
    let Some(home) = Home::new() else { return };
    let (dev, version) = dev_binary();
    let launch = enabled(&home);
    let _disabled = Disabled(&home);
    let release = std::fs::read(&launch).unwrap();

    let (out, last) = tensord(&home, &dev);
    assert_eq!(
        (
            out.code,
            last["type"].as_str(),
            last["run"]["result"]["to"]["cozy"].as_str()
        ),
        (0, Some("run.completed"), Some(version.as_str())),
        "{out:?}"
    );
    assert!(
        std::fs::read(&launch).unwrap() == std::fs::read(&dev).unwrap(),
        "the launch path is not the dev binary"
    );
    assert!(
        std::fs::read(launch.with_file_name(".cozy-release")).unwrap() == release,
        "the release is not kept beside it"
    );
    assert!(out.stderr.contains("a dev build"), "{out:?}");
    let shown = shown(&home, |s| s["version"] == version.as_str());
    assert_eq!(
        (
            shown["status"]["commit"].as_str(),
            shown["status"]["dev"].as_bool()
        ),
        (Some(protocol::COMMIT), Some(true)),
        "{shown}"
    );
    assert_eq!(
        home.cozy(["--version"]).stdout.trim(),
        format!("cozy {version}"),
        "cozy --version and machine show local disagree"
    );
}

/// A binary built for another platform is refused with the mismatch before anything drains: the
/// machine keeps its binary and serves on, not dev.
#[test]
fn a_binary_for_another_platform_is_refused() {
    let Some(home) = Home::new() else { return };
    let launch = enabled(&home);
    let _disabled = Disabled(&home);
    let before = std::fs::read(&launch).unwrap();
    // An ELF header for the other arch on Linux x86_64; a Linux x86_64 one everywhere else.
    let (arch, machine): (&str, u16) = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => ("aarch64", 0xb7),
        _ => ("x86_64", 0x3e),
    };
    let mut elf = vec![0u8; 64];
    elf[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
    elf[18..20].copy_from_slice(&machine.to_le_bytes());
    let foreign = home.path().join("cozy-foreign");
    std::fs::write(&foreign, &elf).unwrap();

    let (out, last) = tensord(&home, &foreign);
    assert_eq!(
        (
            out.code,
            last["type"].as_str(),
            last["run"]["reason"]["code"].as_str()
        ),
        (1, Some("run.failed"), Some("binary_incompatible")),
        "{out:?}"
    );
    let here = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
    let message = last["run"]["reason"]["message"]
        .as_str()
        .unwrap_or_default();
    assert!(
        message.contains(&format!("built for linux {arch}, this machine is {here}")),
        "{message}"
    );
    assert!(
        std::fs::read(&launch).unwrap() == before,
        "the launch path changed"
    );
    let shown = shown(&home, |s| s["dev"].is_boolean());
    assert_eq!(shown["status"]["dev"].as_bool(), Some(false), "{shown}");
}

/// On a rental (the CI Hub's docker provider booting this branch's worker image, Debian's glibc):
/// a `cozy` linked on this newer runner is refused with both glibc versions named, and nothing
/// drains; the image's own build (`COZY_TEST_DEV_BINARY`) swaps in, and the rental reports it as
/// dev at this commit. The rental ends and leaves the list.
#[test]
fn a_rental_runs_a_dev_binary_and_refuses_one_its_glibc_cannot_load() {
    let Some(home) = Home::with_hub() else { return };
    let (worker, version) = dev_binary();
    home.sign_in("tensord@cozy.test");
    let rented = home.json(["machine", "rent", "cpu", "--yes", "--detach", "--json"]);
    let id = rented["machine"]["rental_id"].as_str().unwrap().to_string();
    let update = |binary: &Path| {
        let binary = binary.to_str().unwrap();
        let out = home.cozy(["machine", "update", &id, "--tensord", binary, "--json"]);
        let last: Value = serde_json::from_str(out.stdout.lines().last().unwrap_or_default())
            .unwrap_or_else(|e| panic!("{e}: {out:?}"));
        (out, last)
    };

    // Sent while it boots: Calcifer holds the update until the rental serves.
    let (out, last) = update(Path::new(env!("CARGO_BIN_EXE_cozy")));
    assert_eq!(
        (out.code, last["run"]["reason"]["code"].as_str()),
        (1, Some("binary_incompatible")),
        "{out:?}"
    );
    let message = last["run"]["reason"]["message"]
        .as_str()
        .unwrap_or_default();
    assert!(
        message.contains("needs glibc 2.") && message.contains("this machine has 2."),
        "{message}"
    );

    let (out, last) = update(&worker);
    assert_eq!(
        (
            out.code,
            last["type"].as_str(),
            last["run"]["result"]["to"]["cozy"].as_str()
        ),
        (0, Some("run.completed"), Some(version.as_str())),
        "{out:?}"
    );
    let shown = loop {
        let shown = home.json(["machine", "show", &id, "--json"]);
        if shown["status"]["dev"] == true {
            break shown;
        }
        std::thread::sleep(Duration::from_secs(1));
    };
    assert_eq!(
        shown["status"]["commit"].as_str(),
        Some(protocol::COMMIT),
        "{shown}"
    );

    let ended = home.json(["machine", "end", &id, "--json"]);
    assert_eq!(ended["machine"]["state"].as_str(), Some("ended"), "{ended}");
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
