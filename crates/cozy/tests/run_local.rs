//! Local code on a real machine: `cozy run ./script.py` and `cozy run ./dir/fn`. TensorD
//! (COZY_TEST_TENSORD) installs the code it is sent, answers its interface, and runs it; an
//! edited script runs as edited. Nothing Python runs on this computer.
#![cfg(unix)]
mod common;

use std::path::Path;

use common::{Home, Out};
use serde_json::Value;

const SCRIPT: &str = r#"# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.25.1"]
# [tool.cozy]
# accelerator = false
# ///
def main() -> str:
    return "hello from a script"
"#;

fn cozy(home: &Home, dir: &Path, args: &[&str]) -> Out {
    common::out(home.command().current_dir(dir).args(args).output().unwrap())
}

/// A followed `--json` run's end, which must be a completion.
fn completed(out: &Out) -> Value {
    assert_eq!(out.code, 0, "{out:?}");
    let end = out
        .stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .rfind(|e| e["type"] == "run.completed");
    end.unwrap_or_else(|| panic!("no completion: {out:?}"))["run"].clone()
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

#[test]
fn local_code_runs_on_a_real_machine_and_runs_again_as_edited() {
    let Some(home) = Home::new() else { return };
    if std::env::var_os("COZY_TEST_TENSORD").is_none() {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return;
    }
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    std::fs::create_dir_all(&work).unwrap();

    // A script: the machine installs its generated project, answers `main`'s interface, runs it.
    std::fs::write(work.join("greet.py"), SCRIPT).unwrap();
    let described = cozy(
        &home,
        &work,
        &["run", "./greet.py", "--describe", "--on", "m1"],
    );
    assert_eq!(described.code, 0, "{described:?}");
    assert!(
        described.stdout.contains("request: none") && described.stdout.contains("runs on: CPU"),
        "{described:?}"
    );
    let run = completed(&cozy(
        &home,
        &work,
        &["--json", "run", "greet.py", "--on", "m1"],
    ));
    assert_eq!(run["machine"], "m1");
    assert_eq!(run["result"]["value"], "hello from a script", "{run}");
    assert!(
        run["target"].as_str().unwrap().starts_with("local/"),
        "{run}"
    );

    // Edited, it runs as edited: only the changed file travels.
    std::fs::write(
        work.join("greet.py"),
        SCRIPT.replace("hello from a script", "edited, and run again"),
    )
    .unwrap();
    let again = completed(&cozy(
        &home,
        &work,
        &["--json", "run", "greet.py", "--on", "m1"],
    ));
    assert_eq!(again["result"]["value"], "edited, and run again", "{again}");

    // A project directory: its functions, then one of them.
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny"),
        &work.join("tiny"),
    );
    let listed = cozy(&home, &work, &["--json", "run", "./tiny", "--on", "m1"]);
    assert_eq!(listed.code, 0, "{listed:?}");
    let functions: Value = serde_json::from_str(&listed.stdout).unwrap();
    assert_eq!(functions["functions"][0]["function"], "echo", "{functions}");
    let echoed = completed(&cozy(
        &home,
        &work,
        &[
            "--json",
            "run",
            "./tiny/echo",
            "from a directory",
            "--on",
            "m1",
        ],
    ));
    assert_eq!(echoed["result"]["text"], "from a directory", "{echoed}");
}

/// A wheel no index serves (an unpublished build, as cozy-kernels before its release), named by a
/// path source: the machine installs it as a carried wheel, for a script (resolved on the machine)
/// and for a locked project (`uv sync` leaves it to the carried file), and the code imports it.
#[test]
fn a_local_wheel_a_path_source_names_is_carried_to_the_machine() {
    let Some(home) = Home::new() else { return };
    if std::env::var_os("COZY_TEST_TENSORD").is_none() {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return;
    }
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    let write = |path: &str, text: &str| {
        let path = work.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    };
    let uv = |dir: &Path, args: &[&str]| {
        let out = std::process::Command::new("uv")
            .current_dir(dir)
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "uv {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    let backend = "[build-system]\nrequires = [\"uv_build>=0.9,<1\"]\nbuild-backend = \"uv_build\"\n\n\
                   [tool.uv.build-backend]\nmodule-root = \"\"\n";
    write(
        "probe/pyproject.toml",
        &format!(
            "[project]\nname = \"probe-kernels\"\nversion = \"0.1.0\"\nrequires-python = \">=3.12\"\n\n{backend}"
        ),
    );
    write(
        "probe/probe_kernels/__init__.py",
        "VALUE = \"from a carried wheel\"\n",
    );
    uv(&work, &["build", "--wheel", "--out-dir", "dist", "probe"]);
    let wheel = "dist/probe_kernels-0.1.0-py3-none-any.whl";
    assert!(work.join(wheel).is_file());

    // A script, which nothing locks.
    write(
        "carried.py",
        &format!(
            "# /// script\n# requires-python = \">=3.12\"\n# dependencies = [\"cozy-runtime>=0.25.1\", \"probe-kernels\"]\n\
         # [tool.uv.sources]\n# probe-kernels = {{ path = \"{wheel}\" }}\n# [tool.cozy]\n# accelerator = false\n# ///\n\
         import probe_kernels\n\n\ndef main() -> str:\n    return probe_kernels.VALUE\n"
        ),
    );
    let run = completed(&cozy(
        &home,
        &work,
        &["--json", "run", "carried.py", "--on", "m1"],
    ));
    assert_eq!(run["result"]["value"], "from a carried wheel", "{run}");

    // A locked project.
    write(
        "locked/pyproject.toml",
        &format!(
            "[project]\nname = \"locked-probe\"\nversion = \"0.1.0\"\nrequires-python = \">=3.12\"\n\
         dependencies = [\"cozy-runtime>=0.25.1\", \"msgspec>=0.19\", \"probe-kernels\"]\n\n\
         [project.entry-points.\"cozy.application\"]\ndefault = \"locked_probe:app\"\n\n\
         [tool.uv.sources]\nprobe-kernels = {{ path = \"../{wheel}\" }}\n\n{backend}"
        ),
    );
    write(
        "locked/package.toml",
        "[application]\nobject = \"locked_probe:app\"\n",
    );
    write(
        "locked/locked_probe/__init__.py",
        "import msgspec\nimport probe_kernels\nfrom cozy_runtime.author import App\n\n\
        app = App()\n\n\nclass ProbeInput(msgspec.Struct):\n    text: str = \"\"\n\n\nclass ProbeOutput(msgspec.Struct):\n    text: str\n\n\n\
        @app.entrypoint(accelerator=False)\ndef probe(payload: ProbeInput) -> ProbeOutput:\n    \
        \"\"\"Answers the carried wheel's value.\"\"\"\n    return ProbeOutput(probe_kernels.VALUE)\n",
    );
    uv(&work.join("locked"), &["lock", "--quiet"]);
    let run = completed(&cozy(
        &home,
        &work,
        &["--json", "run", "./locked/probe", "x", "--on", "m1"],
    ));
    assert_eq!(run["result"]["text"], "from a carried wheel", "{run}");
}

/// Run 10297: `cozy run --json` exited at `run.completed` while its 11 MB video was still being
/// saved, and its caller ended the machine. A followed run returns only once each output's last
/// revision landed here, checked against its digest: a 128 MiB output still exporting when the
/// run completed is whole at its path when `cozy run` exits, its landing before the final record.
#[test]
fn a_followed_run_returns_only_once_its_outputs_landed() {
    use sha2::{Digest as _, Sha256};
    let Some(home) = Home::new() else { return };
    if std::env::var_os("COZY_TEST_TENSORD").is_none() {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return;
    }
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    let script = SCRIPT
        .replace(
            "def main() -> str:\n    return \"hello from a script\"",
            "from cozy_runtime.author import FileAsset, Outputs\n\n\
             def main(out: Outputs) -> FileAsset:\n    \
             return out.save_bytes(bytes(range(256)) * (512 << 10), media_type=\"application/octet-stream\")",
        );
    std::fs::write(work.join("big.py"), script).unwrap();
    let out = cozy(
        &home,
        &work,
        &["--json", "run", "big.py", "--on", "m1", "--out", "saved"],
    );
    assert_eq!(out.code, 0, "{}", out.stderr);
    let events: Vec<Value> = out
        .stdout
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let last = events.last().expect("events");
    assert_eq!(last["type"], "run.completed", "{last}");
    let item = &last["run"]["output"][0];
    let path = Path::new(
        item["path"]
            .as_str()
            .expect("the record names the landed file"),
    );
    let expected: Vec<u8> = (0..=255u8).cycle().take(128 << 20).collect();
    let digest = format!("sha256:{}", hex::encode(Sha256::digest(&expected)));
    assert_eq!(item["sha256"], digest.as_str(), "{item}");
    let held = std::fs::read(path).unwrap();
    assert!(
        held.len() == expected.len() && held == expected,
        "{} holds other bytes",
        path.display()
    );
    assert!(!Path::new(&format!("{}.part", path.display())).exists());
    let landed = events
        .iter()
        .find(|e| e["type"] == "output_item.landed" && e["item"]["rev"] == item["rev"])
        .expect("its landing comes before the final record");
    // It was still being saved when the run completed: the follower waited for it.
    assert!(
        landed["sequence_number"].as_u64() > last["sequence_number"].as_u64(),
        "landed {landed} before completing {last}"
    );
}
