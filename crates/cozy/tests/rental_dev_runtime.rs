//! A rental updated to a development Runtime (`cozy machine update --runtime-wheel`, a version PyPI
//! never had) runs a script on it, though the script pins PyPI's Runtime: scripts follow a
//! release's rule, and the run says so. The CI Hub of Tensorhub's dev-build `cozy_ref` job rents a
//! CPU machine on its docker provider (`cozy_worker=branch`: this checkout's cozy in the image);
//! elsewhere it skips.
#![cfg(unix)]
mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::Home;
use serde_json::Value;

/// Rewrites a wheel at another version with one more module, `cozy_runtime/_xs043_marker.py`:
/// what a developer's Runtime build carries that no published one has.
const MARKED: &str = r#"
import base64, hashlib, pathlib, sys, zipfile
source, out, version = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
name, old = source.name.split("-")[:2]
target = out / source.name.replace(f"-{old}-", f"-{version}-", 1)
added = {"cozy_runtime/_xs043_marker.py": f"MARKER = {version!r}\n".encode()}
rows = []
with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as dst:
    items = [(i.filename, src.read(i)) for i in src.infolist()] + list(added.items())
    for path, data in items:
        path = path.replace(f"{name}-{old}.", f"{name}-{version}.", 1)
        if path.endswith(".dist-info/RECORD"):
            continue
        if path.endswith(".dist-info/METADATA"):
            data = "\n".join(f"Version: {version}" if l.startswith("Version: ") else l
                             for l in data.decode().split("\n")).encode()
        dst.writestr(path, data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append(f"{path},sha256={digest},{len(data)}")
    record = f"{name}-{version}.dist-info/RECORD"
    dst.writestr(record, "\n".join(rows + [f"{record},,"]) + "\n")
print(target)
"#;

fn python(args: &[&str]) -> String {
    let out = Command::new("python3").args(args).output().unwrap();
    assert!(out.status.success(), "python3 {args:?}: {out:?}");
    String::from_utf8(out.stdout).unwrap().trim().to_owned()
}

/// PyPI's newest Runtime and the TensorFS it accepts, for the rental's platform.
fn pypi_pair(dir: &Path) -> (PathBuf, PathBuf) {
    let download = |requirement: &str| {
        python(&[
            "-m",
            "pip",
            "download",
            "-q",
            "--no-deps",
            "--only-binary=:all:",
            "--python-version",
            "3.12",
            "--platform",
            "manylinux_2_28_x86_64",
            "--platform",
            "manylinux_2_17_x86_64",
            "-d",
            dir.to_str().unwrap(),
            requirement,
        ]);
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
    let requires = python(&[
        "-I",
        "-c",
        "import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); \
        m=next(n for n in z.namelist() if n.endswith('.dist-info/METADATA')); \
        print(next(l[15:].split(';')[0].replace(' ','') for l in z.read(m).decode().splitlines() \
        if l.startswith('Requires-Dist: tensorfs')))",
        runtime.to_str().unwrap(),
    ]);
    download(&requires);
    (runtime, wheel("tensorfs-"))
}

/// A rental, ended by its id however the test goes; `end` says how it ended.
struct Rental<'a>(&'a Home, String);
impl Rental<'_> {
    fn end(&self) -> Value {
        self.0.json(["machine", "end", &self.1, "--json"])
    }
}
impl Drop for Rental<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let out = self.0.cozy(["machine", "end", &self.1, "--json"]);
            eprintln!("ended {} after a failure: {out:?}", self.1);
        }
    }
}

fn followed(out: &common::Out, kind: &str) -> Value {
    assert_eq!(out.code, 0, "{out:?}");
    let end = out
        .stdout
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .rfind(|e| e["type"] == kind);
    end.unwrap_or_else(|| panic!("no {kind}: {out:?}"))["run"].clone()
}

#[test]
fn a_rentals_dev_runtime_runs_a_script_pinned_to_pypis() {
    let Some(home) = Home::with_hub() else { return };
    home.sign_in("dev-runtime@cozy.test");
    let work = tempfile::tempdir().unwrap();
    let (runtime, tensorfs) = pypi_pair(work.path());
    let published = runtime
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .split('-')
        .nth(1)
        .unwrap()
        .to_owned();
    let dev = format!("{published}.1+xs043.marker");
    let marked = python(&[
        "-I",
        "-c",
        MARKED,
        runtime.to_str().unwrap(),
        work.path().to_str().unwrap(),
        &dev,
    ]);

    let rented = home.json(["machine", "rent", "cpu", "--detach", "--json"]);
    let name = rented["machine"]["name"].as_str().unwrap().to_owned();
    let rental = Rental(
        &home,
        rented["machine"]["rental_id"].as_str().unwrap().to_owned(),
    );
    let update = home.cozy([
        "--json",
        "machine",
        "update",
        &name,
        "--runtime-wheel",
        &marked,
        "--tensorfs-wheel",
        tensorfs.to_str().unwrap(),
    ]);
    let updated = followed(&update, "run.completed");

    std::fs::write(work.path().join("marker.py"), format!(
        "# /// script\n# requires-python = \">=3.12\"\n# dependencies = [\"cozy-runtime=={published}\"]\n\
         # [tool.cozy]\n# accelerator = false\n# ///\nimport importlib.metadata\n\n\
         import cozy_runtime._xs043_marker as marker\n\n\n\
         def main() -> str:\n    return f\"{{marker.MARKER}} {{importlib.metadata.version('cozy-runtime')}}\"\n")).unwrap();
    let out = common::out(
        home.command()
            .current_dir(work.path())
            .args(["--json", "run", "marker.py", "--on", &name])
            .output()
            .unwrap(),
    );
    let number = followed(&out, "run.completed")["number"].to_string();
    let shown = home.cozy(["run", "show", number.as_str()]);
    let ended = rental.end();
    assert_eq!(ended["machine"]["state"].as_str(), Some("ended"), "{ended}");
    assert!(
        shown
            .stdout
            .contains("(`--runtime-wheel`) runs it regardless"),
        "{shown:?}"
    );

    assert!(updated.to_string().contains(&dev), "{updated}");
    let run = followed(&out, "run.completed");
    assert_eq!(
        run["result"]["value"].as_str(),
        Some(format!("{dev} {dev}").as_str()),
        "{run}"
    );
    // The machine's warning reaches whoever runs it: a `run.log` warning line, what `cozy run`
    // prints as `warning: …` and `run show` lists.
    let warned = out
        .stdout
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|e| e["type"] == "run.log" && e["level"] == "warning")
        .filter_map(|e| e["text"].as_str().map(str::to_owned))
        .find(|text| text.contains("(`--runtime-wheel`) runs it regardless"));
    let warned = warned
        .unwrap_or_else(|| panic!("no warning that the dev Runtime runs regardless: {out:?}"));
    assert!(
        warned.contains(&format!(
            "requires `cozy-runtime=={published}`, but `{dev}` is installed"
        )),
        "{warned}"
    );
}
