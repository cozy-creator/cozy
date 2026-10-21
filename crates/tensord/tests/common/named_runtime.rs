//! The Runtime wheel a hosted job names (`COZY_TEST_RUNTIME_WHEEL`, with `COZY_TEST_TENSORFS_WHEEL`
//! when it built one): the pair every environment a test builds runs, and the commit that proves
//! it did. With none named the pair is empty and nothing is asserted.
#![allow(dead_code)]
use std::{
    io::Write as _,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

/// The TensorFS wheel a Runtime wheel accepts, from PyPI, into `dir`.
const ACCEPTED_TENSORFS: &str = "import re, subprocess, sys, zipfile, pathlib
wheel, out = sys.argv[1], pathlib.Path(sys.argv[2])
archive = zipfile.ZipFile(wheel)
meta = next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
req = next(l[15:].split(';')[0].replace(' ', '') for l in archive.read(meta).decode().splitlines()
           if re.match(r'Requires-Dist: tensorfs[^A-Za-z0-9._-]', l))
subprocess.run([sys.executable, '-m', 'pip', 'download', '-q', '--no-deps', '--only-binary=:all:', '--python-version', '3.12',
                '--platform', 'manylinux_2_28_x86_64', '--platform', 'manylinux_2_17_x86_64', '-d', str(out), req], check=True)
print(next(out.glob('tensorfs-*.whl')))
";

/// The named Runtime with the TensorFS it accepts, as an update naming it brings, fetched once.
pub fn pair() -> &'static [PathBuf] {
    static PAIR: OnceLock<Vec<PathBuf>> = OnceLock::new();
    PAIR.get_or_init(|| {
        let Some(runtime) = std::env::var_os("COZY_TEST_RUNTIME_WHEEL").map(PathBuf::from) else {
            return vec![];
        };
        let tensorfs = std::env::var_os("COZY_TEST_TENSORFS_WHEEL").map(PathBuf::from).unwrap_or_else(|| {
            let dir = std::env::temp_dir().join(format!("tensorfs-wheel-{}", std::process::id()));
            let out = Command::new("python3").args(["-I", "-c", ACCEPTED_TENSORFS]).arg(&runtime).arg(&dir).output().unwrap();
            assert!(out.status.success(), "the TensorFS {runtime:?} accepts: {out:?}");
            PathBuf::from(String::from_utf8(out.stdout).unwrap().trim())
        });
        vec![runtime, tensorfs]
    })
}

/// The named pair as the image's (`<root>/opt/cozy/wheels`), so a machine started on `root` runs
/// it rather than PyPI's newest.
pub fn image(root: &Path) {
    let wheels = root.join("opt/cozy/wheels");
    for wheel in pair() {
        std::fs::create_dir_all(&wheels).unwrap();
        std::fs::copy(wheel, wheels.join(wheel.file_name().unwrap())).unwrap();
    }
}

/// The named Runtime's version and the commit stamped in it; a wheel without one proves nothing.
pub fn named() -> Option<&'static (String, String)> {
    static NAMED: OnceLock<Option<(String, String)>> = OnceLock::new();
    NAMED
        .get_or_init(|| {
            let wheel = PathBuf::from(std::env::var_os("COZY_TEST_RUNTIME_WHEEL")?);
            let commit = stamped(&wheel);
            assert!(
                !commit.is_empty() && commit != "unknown",
                "{wheel:?} names no commit (COMMIT = {commit:?}): a wheel without provenance proves nothing"
            );
            Some((wheel.file_name()?.to_str()?.split('-').nth(1)?.to_owned(), commit))
        })
        .as_ref()
}

/// The commit a Runtime wheel was built from: its `cozy_runtime._build_provenance.COMMIT`.
pub fn stamped(wheel: &Path) -> String {
    let read = "import re, sys, zipfile\n\
        text = zipfile.ZipFile(sys.argv[1]).read('cozy_runtime/_build_provenance.py').decode()\n\
        print(re.search(r'^COMMIT = \"([^\"]*)\"', text, re.M).group(1))\n";
    let out = Command::new("python3").args(["-I", "-c", read]).arg(wheel).output().unwrap();
    assert!(out.status.success(), "reading {wheel:?}: {out:?}");
    String::from_utf8(out.stdout).unwrap().trim().to_owned()
}

/// Every environment `pythons` names that holds a Runtime loads the named one, by its commit (one
/// without any, a package that runs no code, holds none to check); the log says so, since a
/// silent pass reads as missing. Nothing named, nothing asserted.
pub fn loaded(pythons: &[PathBuf], what: &str) {
    const COMMIT: &str = "import importlib.util\n\
        if importlib.util.find_spec('cozy_runtime'):\n\
        \x20   from cozy_runtime._build_provenance import COMMIT\n\
        \x20   print(COMMIT)\n";
    let Some((version, commit)) = named() else { return };
    let mut checked = 0;
    for python in pythons {
        let out = Command::new(python).args(["-I", "-c", COMMIT]).output().unwrap();
        assert!(out.status.success(), "{python:?}: {out:?}");
        let found = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        if found.is_empty() {
            continue;
        }
        assert_eq!(&found, commit, "{python:?} loads another Runtime than the named {version}: {out:?}");
        checked += 1;
    }
    if checked > 0 {
        let _ = writeln!(
            std::io::stderr(),
            "named Runtime {version} (commit {commit}) loads: asserted in {checked} environments of {what}"
        );
    }
}

/// The environments under a generations directory.
pub fn environments(generations: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(generations)
        .into_iter()
        .flatten()
        .flatten()
        .map(|generation| generation.path().join("env/bin/python"))
        .filter(|python| python.exists())
        .collect()
}

/// The development Runtime the named pair is, as an update's `--runtime-wheel` names it
/// (`runtime_build`): the operator's choice runs whatever a fixture's bounds. "" with none named.
pub fn build() -> String {
    named().map(|(version, _)| version.clone()).unwrap_or_default()
}

/// What a test's local installs run with: the repository's helper interpreter (the client and
/// the test extras, `uv run --locked --extra test`) and the machine client wheel, built once.
pub fn installer() -> &'static (PathBuf, PathBuf) {
    static INSTALLER: OnceLock<(PathBuf, PathBuf)> = OnceLock::new();
    INSTALLER.get_or_init(|| {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let run = |command: &mut Command| {
            let out = command.current_dir(repo).output().unwrap();
            assert!(out.status.success(), "{command:?}: {out:?}");
            String::from_utf8(out.stdout).unwrap().trim().to_owned()
        };
        let helper = run(Command::new("uv").args(["run", "--locked", "--extra", "test", "python", "-c", "import sys; print(sys.executable)"]));
        let out = std::env::temp_dir().join(format!("tensord-client-{}", std::process::id()));
        run(Command::new("uv").args(["build", "--wheel", "--out-dir"]).arg(&out));
        let client = std::fs::read_dir(&out).unwrap().flatten().map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "whl")).unwrap();
        (PathBuf::from(helper.lines().last().unwrap()), client)
    })
}
