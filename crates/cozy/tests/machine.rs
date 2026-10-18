//! `cozy machine` and `cozy doctor` through the real binary, a real Calcifer and a real TensorD added by
//! address. No Hub: an added machine needs none. TensorD 0.6.0 runs on Linux only.
#![cfg(unix)]
mod common;

use std::{
    path::{Path, PathBuf},
    sync::atomic::Ordering,
};

use common::{
    Home,
    https::Https,
    tensord::{self, Tensord},
};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

fn listed(home: &Home, name: &str) -> Option<Value> {
    home.json(["machine", "list", "--json"])["machines"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == name)
        .cloned()
}

#[test]
fn an_added_machine_is_checked_before_it_is_recorded_then_listed_shown_checked_and_removed() {
    let Some(home) = Home::new() else { return };
    // A machine that admits some other controller.
    let mut machine = Tensord::start(&[tensord::stranger()]);
    let add = |fingerprint: &str| {
        home.cozy([
            "machine",
            "add",
            "box",
            machine.address.as_str(),
            "--fingerprint",
            fingerprint,
            "--json",
        ])
    };

    // A wrong pin is refused and nothing is recorded.
    let out = add(&format!("sha256:{}", "0".repeat(64)));
    assert_eq!(
        (out.code, out.refusal()["code"].clone()),
        (1, "fingerprint_mismatch".into()),
        "{out:?}"
    );
    assert!(listed(&home, "box").is_none());

    // The right pin, but the machine does not admit this controller: the refusal names the command to
    // run on that computer, and nothing is recorded.
    let out = add(&machine.fingerprint);
    assert_eq!(
        (out.code, out.refusal()["code"].clone()),
        (1, "machine_unauthorized".into()),
        "{out:?}"
    );
    let key = tensord::key_to_authorize(&out);
    assert!(listed(&home, "box").is_none());

    // Once it admits the key (as `cozy machine authorize` will), the add succeeds.
    machine.restart(&[key]);
    let add = |fingerprint: &str| {
        home.cozy([
            "machine",
            "add",
            "box",
            machine.address.as_str(),
            "--fingerprint",
            fingerprint,
            "--json",
        ])
    };
    let out = add(&machine.fingerprint);
    assert_eq!(out.code, 0, "{out:?}");
    let row = listed(&home, "box").expect("the added machine is listed");
    assert_eq!(
        (row["kind"].as_str(), row["state"].as_str()),
        (Some("added"), Some("ready")),
        "{row}"
    );
    let human = home.cozy(["machine", "list"]);
    assert!(
        human.stdout.contains("box") && human.stdout.contains("added"),
        "{human:?}"
    );

    // show and doctor render what the machine reports of itself.
    let shown = home.json(["machine", "show", "box", "--json"]);
    // It names the leaf another computer's `machine add` pins.
    assert_eq!(
        shown["fingerprint"],
        machine.fingerprint.as_str(),
        "{shown}"
    );
    assert!(
        !shown["status"]["version"]
            .as_str()
            .unwrap_or_default()
            .is_empty(),
        "{shown}"
    );
    // A persistent machine reports its store's disk.
    assert!(
        shown["status"]["disk_total_bytes"].as_u64() > Some(0),
        "{shown}"
    );
    let out = home.cozy(["machine", "show", "box"]);
    assert!(out.stdout.contains("platform"), "{out:?}");
    let out = home.cozy(["doctor", "--on", "box"]);
    assert_eq!(out.code, 0, "{out:?}");
    assert!(
        out.stdout.contains("GPU") || out.stdout.contains("none: runs use the CPU"),
        "{out:?}"
    );

    // A log the machine keeps, possibly empty, reads without failing.
    for log in ["memory", "tensorfs"] {
        let out = home.cozy(["machine", "logs", "box", log]);
        assert_eq!(out.code, 0, "{log}: {out:?}");
    }

    // End removes it here; a second end is an honest no-op, and a name never added claims nothing.
    let out = home.cozy(["machine", "end", "box"]);
    assert_eq!(out.code, 0, "{out:?}");
    assert!(listed(&home, "box").is_none());
    let again = home.json(["machine", "end", "box", "--json"]);
    assert_eq!(again["changed"], false, "{again}");
    let out = home.cozy(["machine", "end", "never-added", "--json"]);
    assert_eq!(
        (out.code, out.refusal()["kind"].clone()),
        (1, "not_found".into()),
        "{out:?}"
    );
    // The fixture other groups' tests use adds a machine the same way.
    let _again = tensord::added(&home, "again");
    assert_eq!(listed(&home, "again").expect("added")["state"], "ready");
}

/// `machine authorize` adds a controller's key to `keys/authorized` once, at 0600; TensorD reads it
/// again when a signer is unknown.
#[test]
fn authorize_adds_a_controllers_key_once() {
    let Some(home) = Home::new() else { return };
    let key = tensord::stranger();
    let first = home.json(["machine", "authorize", key.as_str(), "--json"]);
    let again = home.json(["machine", "authorize", key.as_str(), "--json"]);
    assert_eq!(
        (first["changed"].as_bool(), again["changed"].as_bool()),
        (Some(true), Some(false))
    );
    let path = home.root().join("keys/authorized");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), format!("{key}\n"));
    use std::os::unix::fs::PermissionsExt as _;
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
}

#[test]
fn machine_refusals_say_what_to_do_instead() {
    let Some(home) = Home::new() else { return };
    for (args, exit, code) in [
        // A controller's key is 32 bytes of unpadded base64url.
        (
            vec!["machine", "authorize", "not-a-key"],
            1,
            "public_key_invalid",
        ),
        // `local` is ended by disabling it.
        (vec!["machine", "end", "local"], 1, "machine_local"),
        // The live view is the terminal's default: `--watch` is gone.
        (vec!["machine", "list", "--watch"], 2, "usage"),
        (vec!["machine", "show", "../etc"], 2, "machine_name_invalid"),
    ] {
        let out = home.cozy(args.iter().chain(&["--json"]));
        assert_eq!(
            (out.code, out.refusal()["code"].clone()),
            (exit, code.into()),
            "{args:?}: {out:?}"
        );
    }
    // A bad key's next step starts on the controller's computer: installing cozy there.
    let next = home
        .cozy(["machine", "authorize", "not-a-key", "--json"])
        .refusal()["next"][0]
        .clone();
    assert!(
        next.as_str()
            .is_some_and(|n| n.contains("https://tensorhub.com/install.sh")),
        "{next}"
    );
    // Rental options without a product, and a malformed fingerprint, are usage errors.
    for args in [
        vec!["machine", "rent", "--gpus", "2"],
        vec![
            "machine",
            "add",
            "box",
            "127.0.0.1:1",
            "--fingerprint",
            "abc",
        ],
    ] {
        assert_eq!(home.cozy(&args).code, 2, "{args:?}");
    }
}

/// `machine update` is a numbered run of kind update, sent to the machine and followed like any run
/// across the machine's restart. The target is the default Hub's public `/v1/software`; once the
/// machine runs it, another update sends nothing.
#[test]
fn an_update_installs_the_hubs_target_and_then_has_nothing_to_do() {
    let Some(home) = Home::new() else { return };
    let _machine = tensord::added(&home, "upd");
    let out = home.cozy(["machine", "update", "upd", "--json"]);
    let last: serde_json::Value =
        serde_json::from_str(out.stdout.lines().last().unwrap_or_default())
            .unwrap_or_else(|e| panic!("{e}: {out:?}"));
    let run = &last["run"];
    assert_eq!(
        (out.code, last["type"].as_str(), run["kind"].as_str()),
        (0, Some("run.completed"), Some("update")),
        "{out:?}"
    );
    let to = run["result"]["to"]["runtime"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(
        run["target"].as_str().unwrap().contains(&to) && !to.is_empty(),
        "{run}"
    );
    let again = home.json(["machine", "update", "upd", "--json"]);
    assert_eq!(
        (again["changed"].as_bool(), again["runtime"].as_str()),
        (Some(false), Some(to.as_str())),
        "{again}"
    );
}

/// PyPI's newest Runtime and the TensorFS it accepts (a newer TensorFS may be out first), wheels
/// for this platform, downloaded into `dir`.
fn downloaded(dir: &Path) -> (PathBuf, PathBuf) {
    let download = |requirement: &str| {
        let fetched = std::process::Command::new("python3")
            .args([
                "-m",
                "pip",
                "download",
                "-q",
                "--no-deps",
                "--only-binary=:all:",
            ])
            .args([
                "--python-version",
                "3.12",
                "--platform",
                "manylinux_2_28_x86_64",
                "--platform",
                "manylinux_2_17_x86_64",
                "-d",
            ])
            .arg(dir)
            .arg(requirement)
            .status()
            .unwrap();
        assert!(fetched.success(), "pip download {requirement}");
    };
    let wheel = |prefix: &str| {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.file_name().unwrap().to_str().unwrap().starts_with(prefix))
            .unwrap_or_else(|| panic!("no {prefix} wheel"))
    };
    download("cozy-runtime");
    let runtime = wheel("cozy_runtime-");
    download(&requirement(&runtime, "tensorfs"));
    (runtime, wheel("tensorfs-"))
}

/// The requirement a wheel's metadata names for `name`, without extras (`tensorfs<0.11,>=0.10`).
fn requirement(wheel: &Path, name: &str) -> String {
    const READ: &str = r#"
import re, sys, zipfile
wheel, name = sys.argv[1:]
archive = zipfile.ZipFile(wheel)
metadata = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
for line in archive.read(metadata).decode().splitlines():
    if line.startswith("Requires-Dist: ") and "extra ==" not in line:
        requirement = line.removeprefix("Requires-Dist: ").replace(" ", "")
        if re.match(re.escape(name) + r"[^A-Za-z0-9._-]", requirement):
            print(requirement)
            break
"#;
    let out = std::process::Command::new("python3")
        .args(["-I", "-c", READ])
        .arg(wheel)
        .arg(name)
        .output()
        .unwrap();
    let requirement = String::from_utf8(out.stdout).unwrap().trim().to_owned();
    assert!(
        !requirement.is_empty(),
        "{} names no {name}",
        wheel.display()
    );
    requirement
}

/// A wheel's version, from its file name.
fn version(wheel: &Path) -> String {
    wheel.file_name().unwrap().to_str().unwrap().split('-').nth(1).unwrap().to_owned()
}

/// `machine update --runtime-wheel/--tensorfs-wheel` installs local wheels, each written to the machine
/// with the run, in place of the Hub's target: how a release gate runs an unpublished Runtime. The
/// machine's Status then reports the wheels' versions.
#[test]
fn an_update_installs_local_wheels() {
    let Some(home) = Home::new() else { return };
    let _machine = tensord::added(&home, "wheels");
    let dir = tempfile::tempdir().unwrap();
    let (runtime, tensorfs) = downloaded(dir.path());
    let out = home.cozy([
        "machine".as_ref(),
        "update".as_ref(),
        "wheels".as_ref(),
        "--runtime-wheel".as_ref(),
        runtime.as_os_str(),
        "--tensorfs-wheel".as_ref(),
        tensorfs.as_os_str(),
        "--json".as_ref(),
    ]);
    let last: serde_json::Value =
        serde_json::from_str(out.stdout.lines().last().unwrap_or_default())
            .unwrap_or_else(|e| panic!("{e}: {out:?}"));
    assert_eq!(
        (
            out.code,
            last["type"].as_str(),
            last["run"]["result"]["to"]["runtime"].as_str()
        ),
        (0, Some("run.completed"), Some(version(&runtime).as_str())),
        "{out:?}"
    );
    let shown = home.json(["machine", "show", "wheels", "--json"]);
    assert_eq!(shown["status"]["runtime"], version(&runtime).as_str(), "{shown}");
    // A file that is not the member's wheel is refused before anything is sent.
    let out = home.cozy([
        "machine".as_ref(),
        "update".as_ref(),
        "wheels".as_ref(),
        "--runtime-wheel".as_ref(),
        tensorfs.as_os_str(),
        "--json".as_ref(),
    ]);
    assert_eq!(
        (out.code, out.refusal()["code"].clone()),
        (1, "wheel_refused".into()),
        "{out:?}"
    );
}

/// `machine update --runtime-wheel/--tensorfs-wheel https://…#sha256=<hex>`: the machine fetches each
/// wheel itself, through a redirect to another host, and nothing is written to it from here. The
/// bytes are held to the digest: under a wrong one the update fails `wheel_digest_mismatch` naming
/// both digests, and the machine keeps its software.
#[test]
fn an_update_fetches_wheels_by_url_held_to_their_digests() {
    let Some(home) = Home::new() else { return };
    let (dir, trust) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let (runtime, tensorfs) = downloaded(dir.path());
    let server = Https::serve(dir.path());
    // The machine's bundle, which uv reads too: the runner's roots and the server's.
    let ca = trust.path().join("ca.pem");
    let system = std::fs::read_to_string("/etc/ssl/certs/ca-certificates.crt").unwrap_or_default();
    std::fs::write(&ca, system + &server.ca).unwrap();
    let env = [("SSL_CERT_FILE", ca.to_str().unwrap())];
    let _machine = tensord::added_with(&home, "fetching", &env);
    let url = |wheel: &Path, sha256: &str| {
        let name = wheel.file_name().unwrap().to_str().unwrap();
        format!("{}/to/{name}#sha256={sha256}", server.base)
    };
    let digest = |wheel: &Path| hex::encode(Sha256::digest(std::fs::read(wheel).unwrap()));
    let update = |runtime: &str, tensorfs: &str| {
        let out = home.cozy([
            "machine",
            "update",
            "fetching",
            "--runtime-wheel",
            runtime,
            "--tensorfs-wheel",
            tensorfs,
            "--json",
        ]);
        let last: Value = serde_json::from_str(out.stdout.lines().last().unwrap_or_default())
            .unwrap_or_else(|e| panic!("{e}: {out:?}"));
        (out, last)
    };
    let running =
        || home.json(["machine", "show", "fetching", "--json"])["status"]["runtime"].clone();
    let before = running();
    // Plain http, or a URL without its digest, is refused before anything is sent.
    let http = url(&runtime, &digest(&runtime)).replace("https://", "http://");
    let bare = format!(
        "{}/to/{}",
        server.base,
        runtime.file_name().unwrap().to_str().unwrap()
    );
    for refused in [http, bare] {
        let out = home.cozy([
            "machine",
            "update",
            "fetching",
            "--runtime-wheel",
            refused.as_str(),
            "--json",
        ]);
        assert_eq!(
            (out.code, out.refusal()["code"].clone()),
            (1, "wheel_refused".into()),
            "{out:?}"
        );
    }
    let wrong = "ab".repeat(32);
    let (out, last) = update(
        url(&runtime, &wrong).as_str(),
        url(&tensorfs, &digest(&tensorfs)).as_str(),
    );
    let reason = &last["run"]["reason"];
    assert_eq!(
        (out.code, last["type"].as_str(), reason["code"].as_str()),
        (1, Some("run.failed"), Some("wheel_digest_mismatch")),
        "{out:?}"
    );
    let message = reason["message"].as_str().unwrap();
    assert!(
        message.contains(&wrong) && message.contains(&digest(&runtime)),
        "{message}"
    );
    assert_eq!(running(), before);
    let (out, last) = update(
        url(&runtime, &digest(&runtime)).as_str(),
        url(&tensorfs, &digest(&tensorfs)).as_str(),
    );
    assert_eq!(
        (
            out.code,
            last["type"].as_str(),
            last["run"]["result"]["to"]["runtime"].as_str()
        ),
        (0, Some("run.completed"), Some(version(&runtime).as_str())),
        "{out:?}"
    );
    assert_eq!(running(), version(&runtime).as_str());
    assert!(
        server.served.load(Ordering::Relaxed) > 0,
        "the machine fetched the wheels"
    );
}

/// `machine update --runtime-wheel` alone: the machine keeps its TensorFS when the Runtime resolves
/// with it, else installs the one uv resolves for the Runtime over the index (PyPI's newest it
/// accepts), held to PyPI's digest. A Runtime no published TensorFS satisfies fails naming
/// `--tensorfs-wheel`, and the machine keeps its software.
#[test]
fn a_runtime_wheel_alone_brings_the_tensorfs_it_accepts() {
    let Some(home) = Home::new() else { return };
    let (dir, trust) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let (runtime, tensorfs) = downloaded(dir.path());
    let server = Https::serve(dir.path());
    // The machine's bundle: the runner's roots and the server's.
    let ca = trust.path().join("ca.pem");
    let system = std::fs::read_to_string("/etc/ssl/certs/ca-certificates.crt").unwrap_or_default();
    std::fs::write(&ca, system + &server.ca).unwrap();
    let _machine = tensord::added_with(&home, "paired", &[("SSL_CERT_FILE", ca.to_str().unwrap())]);
    let update = |wheel: &str| {
        let out = home.cozy([
            "machine",
            "update",
            "paired",
            "--runtime-wheel",
            wheel,
            "--json",
        ]);
        let last: Value = serde_json::from_str(out.stdout.lines().last().unwrap_or_default())
            .unwrap_or_else(|e| panic!("{e}: {out:?}"));
        (out, last)
    };
    let running = || home.json(["machine", "show", "paired", "--json"])["status"].clone();
    let before = running();

    // A Runtime that needs a TensorFS PyPI has never published.
    let unpaired = trust.path().join("cozy_runtime-0.0.1-py3-none-any.whl");
    let made = std::process::Command::new("python3")
        .args([
            "-I",
            "-c",
            "import sys, zipfile\n\
             with zipfile.ZipFile(sys.argv[1], 'w') as w:\n\
             \x20   w.writestr('cozy_runtime-0.0.1.dist-info/METADATA', \
             'Metadata-Version: 2.1\\nName: cozy-runtime\\nVersion: 0.0.1\\nRequires-Dist: tensorfs>=999\\n')",
        ])
        .arg(&unpaired)
        .status()
        .unwrap();
    assert!(made.success());
    let (out, last) = update(unpaired.to_str().unwrap());
    let reason = &last["run"]["reason"];
    assert_eq!(
        (out.code, last["type"].as_str(), reason["code"].as_str()),
        (1, Some("run.failed"), Some("tensorfs_unresolved")),
        "{out:?}"
    );
    assert!(
        reason["message"]
            .as_str()
            .is_some_and(|m| m.contains("--tensorfs-wheel")),
        "{out:?}"
    );
    let after = running();
    assert_eq!(
        (&after["runtime"], &after["tensorfs"]),
        (&before["runtime"], &before["tensorfs"])
    );

    // The machine runs no TensorFS the Runtime accepts: it installs PyPI's newest that it does.
    let name = runtime.file_name().unwrap().to_str().unwrap();
    let sha256 = hex::encode(Sha256::digest(std::fs::read(&runtime).unwrap()));
    let (out, last) = update(&format!("{}/to/{name}#sha256={sha256}", server.base));
    assert_eq!(
        (
            out.code,
            last["type"].as_str(),
            last["run"]["result"]["to"]["tensorfs"].as_str()
        ),
        (0, Some("run.completed"), Some(version(&tensorfs).as_str())),
        "{out:?}"
    );
    let status = running();
    assert_eq!(
        (&status["runtime"], &status["tensorfs"]),
        (&version(&runtime).into(), &version(&tensorfs).into()),
        "{status}"
    );
}
