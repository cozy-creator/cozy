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

/// The dev build (dev-build `build=cozy`: `<version>+dev.g<sha8>`) replaces the machine's binary.
/// The machine restarts on it and reports its version and commit as dev; the release it replaced
/// is kept beside it as `.cozy-release`; this computer's cozy is the dev build too.
#[cfg(target_os = "linux")]
#[test]
fn a_dev_binary_runs_as_the_machine_and_reports_its_version_and_commit() {
    let Some(home) = Home::new() else { return };
    let dev = PathBuf::from(
        std::env::var_os("COZY_TEST_DEV_BINARY")
            .expect("COZY_TEST_DEV_BINARY: the dev cozy (dev-build build=cozy)"),
    );
    let said = Command::new(&dev).arg("--version").output().unwrap().stdout;
    let said = String::from_utf8(said).unwrap();
    let version = said
        .trim()
        .strip_prefix("cozy ")
        .unwrap_or_default()
        .to_owned();
    assert!(version.contains("+dev.g"), "{said}");
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
        (Some(cozy_proto::COMMIT), Some(true)),
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
