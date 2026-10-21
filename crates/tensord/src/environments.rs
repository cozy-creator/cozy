//! What every package environment's installation does once its own closure is in place, a
//! release's (`published`, from its lock) or a local package's (`api::install`, what the helper
//! captured): this machine's Runtime/TensorFS pair over it, the machine client, uv's check, the
//! description, and the one typed record (`catalog::Generation`). One path for both.

use crate::{
    catalog::{normalized, Callee, Dependency, Generation},
    process::Supervised,
    published::CANCELED,
    stop::Stop,
};
use serde_json::{json, value::RawValue};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    path::Path,
    process::{Command, Stdio},
};

/// A refusal: its code and its words.
pub(crate) type Failure = (&'static str, String);

/// Cozy's SDK: the pair a machine installs over each environment's own.
pub(crate) const SDK: [&str; 2] = ["cozy-runtime", "tensorfs"];

/// uv as one installation runs it: its environment and its supervision are the caller's.
pub(crate) type Uv<'a> = &'a dyn Fn(&[&str]) -> Result<(), Failure>;

/// One prepared uv command to its exit, `read` told about once a second how many bytes it has
/// read. Its process group ends with `stop`, or when its CPU and bytes stop moving past its
/// measured patience (`process::supervise`); uv's failure carries its stderr's tail.
pub(crate) fn run_uv(mut command: Command, step: &str, stop: Option<&Stop>, read: &dyn Fn(u64)) -> Result<(), Failure> {
    command.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped());
    platform::process::own_group(&mut command);
    let mut child = command
        .spawn()
        .map_err(|e| ("package_installation_uv_absent", format!("cannot run uv: {e}")))?;
    let mut stderr = child.stderr.take().expect("piped uv stderr");
    let (ended, stderr) = std::thread::scope(|scope| {
        let drained = scope.spawn(move || {
            let mut kept = Vec::new();
            let _ = stderr.read_to_end(&mut kept);
            kept
        });
        let ended = crate::process::supervise(&mut child, stop, "uv", read);
        (ended, drained.join().unwrap_or_default())
    });
    match ended.map_err(|e| ("package_installation_uv_failed", format!("uv: {e}")))? {
        Supervised::Exited(status) if status.success() => Ok(()),
        Supervised::Exited(_) => Err((
            "package_installation_uv_failed",
            format!("uv {step}:\n{}", crate::api::install::stderr_tail(&stderr)),
        )),
        Supervised::Canceled => Err((CANCELED, "the run was canceled while it prepared".into())),
        Supervised::Wedged(verdict) => Err(("no_progress", format!("uv {step}: {verdict}"))),
    }
}

/// This machine's Runtime/TensorFS pair: the wheels each environment takes over its own.
pub(crate) struct Pair<'a> {
    pub requirements: &'a [String],
    pub find_links: Option<&'a Path>,
    /// The development Runtime an update named (`--runtime-wheel`): the operator's choice.
    pub runtime_build: &'a str,
}

/// This machine's own pair in an environment holding a package's closure, constrained to leave
/// every other selected version (`constraints`), where every installed requirement admits it
/// (uv's check); else the package's own (`restore`; None: there is none to restore), and why.
/// A development Runtime an update named (`--runtime-wheel`) is the operator's choice: it runs
/// whatever the package's bounds on the pair, which the warning names, and one that cannot
/// install refuses the package. Returns whether the machine's pair stands, and the run's
/// warning: never silent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn install_pair(
    uv: Uv,
    pair: &Pair,
    py: &str,
    env: &Path,
    constraints: &Path,
    restore: Option<&dyn Fn() -> Result<(), Failure>>,
    compile: &str,
) -> Result<(bool, String), Failure> {
    let constraints = constraints.to_string_lossy().to_string();
    let mut args = vec!["pip", "install", "--no-config", "--python", py, compile, "--constraints", &constraints];
    let links = pair.find_links.map(|p| p.to_string_lossy().to_string());
    if let Some(links) = &links {
        args.extend(["--find-links", links]);
    }
    args.extend(pair.requirements.iter().map(String::as_str));
    // What the package's own closure reports already stands: the pair is refused only for a
    // finding it adds. uv's failure names its step and its output.
    let before = check(uv, py)?;
    let own = uv(&args).and_then(|()| {
        let added: Vec<String> = check(uv, py)?.difference(&before).cloned().collect();
        match added.is_empty() {
            true => Ok(()),
            false => Err(("package_installation_uv_failed", format!("uv pip check:\n{}", added.join("\n")))),
        }
    });
    let Err((code, detail)) = own else {
        return Ok((true, String::new()));
    };
    let build = pair.runtime_build;
    if code == CANCELED {
        return Err((code, detail));
    } else if !build.is_empty() {
        // uv's check names each requirement the installed pair misses; any other failure (the
        // pair's own dependencies against the closure) leaves nothing it could run.
        let missed: Vec<&str> = detail.lines().map(str::trim).filter(|l| l.starts_with("The package `")).collect();
        if missed.is_empty() || !missed.iter().all(|l| pair_bound(l)) {
            return Err((code, format!("this machine's development Runtime {build} cannot run this package: {detail}")));
        }
        return Ok((true, format!("{}; this machine's development Runtime {build} (`--runtime-wheel`) runs it regardless", missed.join("; "))));
    }
    let Some(restore) = restore else {
        return Err((code, detail));
    };
    let lines: Vec<&str> = detail.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let detail = lines[lines.len().saturating_sub(12)..].join(" | ");
    // The package's own pair failing too (TensorFS's sdist on aarch64) still names why the
    // machine's did not.
    restore().map_err(|(own_code, why)| {
        (own_code, format!("{why}; this machine's own Runtime and TensorFS did not install first ({code}: {})", crate::triage::ends(&detail, 1000)))
    })?;
    let own: Vec<String> = installed(env)
        .into_iter()
        .filter(|d| SDK.contains(&d.name.as_str()))
        .map(|d| format!("{}=={}", d.name, d.version))
        .collect();
    Ok((false, format!(
        "this machine's own Runtime and TensorFS did not install ({code}: {}); it runs the package's own {}",
        crate::triage::ends(&detail, 2000),
        own.join(", ")
    )))
}

/// The machine's description client beside an environment's selected closure: it may add its
/// own dependencies, never move a selected version (`constraints`).
pub(crate) fn add_client(uv: Uv, py: &str, client: &Path, constraints: &Path, compile: &str) -> Result<(), Failure> {
    let (client, constraints) = (client.to_string_lossy(), constraints.to_string_lossy());
    uv(&["pip", "install", "--no-config", "--python", py, compile, "--constraints", &constraints, &client])
}

/// uv's check of the finished environment: each installed requirement a version misses, but a
/// bound on the pair the machine's development Runtime runs regardless (`runtime_build`, said
/// in `sdk_fallback`). A finding never refuses an install (version skew is allowed): the install
/// log and every run say it, and a real import failure surfaces as itself.
pub(crate) fn findings(uv: Uv, py: &str, runtime_build: &str) -> Result<Vec<String>, Failure> {
    Ok(check(uv, py)?
        .into_iter()
        .filter(|finding| runtime_build.is_empty() || !pair_bound(finding))
        .collect())
}

/// uv's check of an environment: one line per installed requirement a version misses (`The
/// package …`). A wheel tagged for another platform still runs and stands (NVIDIA's aarch64
/// cusparseLt wheel says sbsa, not aarch64): no finding.
pub(crate) fn check(uv: Uv, py: &str) -> Result<BTreeSet<String>, Failure> {
    match uv(&["pip", "check", "--no-config", "--python", py]) {
        Ok(()) => Ok(BTreeSet::new()),
        Err((code, detail)) => {
            let lines: Vec<&str> = detail.lines().map(str::trim).filter(|l| l.starts_with("The package ")).collect();
            if lines.is_empty() {
                return Err((code, detail));
            }
            Ok(lines
                .into_iter()
                .filter(|l| !l.ends_with("was built for a different platform"))
                .map(String::from)
                .collect())
        }
    }
}

/// uv's check line for a bound on the machine's own pair that the installed pair misses:
/// "The package `anima` requires `cozy-runtime[media]>=0.24.7`, but `0.24.6` is installed".
pub(crate) fn pair_bound(line: &str) -> bool {
    let required = line.split_once("` requires `").and_then(|(_, rest)| rest.split_once("`, but `")).map(|(r, _)| r);
    required.is_some_and(|r| {
        let end = r.find(|c: char| !(c.is_alphanumeric() || matches!(c, '-' | '_' | '.'))).unwrap_or(r.len());
        SDK.contains(&normalized(&r[..end]).as_str())
    })
}

/// Every distribution an environment holds, by its installed metadata's directory name.
pub(crate) fn installed(env: &Path) -> Vec<Dependency> {
    let mut found: Vec<Dependency> = platform::paths::venv_site_packages(env)
        .into_iter()
        .flat_map(|site| fs::read_dir(site).into_iter().flatten().flatten())
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let (distribution, version) = name.strip_suffix(".dist-info")?.split_once('-')?;
            Some(Dependency { name: normalized(distribution), version: version.into() })
        })
        .collect();
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

/// Every distribution `env` holds but `except` as constraints (`name==version`): what the
/// machine's pair and client install against, so neither moves one.
pub(crate) fn pin_installed(env: &Path, except: &[&str], out: &Path) -> Result<(), Failure> {
    let pins: String = installed(env)
        .into_iter()
        .filter(|d| !except.contains(&d.name.as_str()))
        .map(|d| format!("{}=={}\n", d.name, d.version))
        .collect();
    fs::write(out, pins).map_err(storage)
}

/// What an environment's description says of it.
#[derive(Default)]
pub struct Described {
    /// The root's own installed files, hashed.
    pub source_digest: String,
    /// The other packages' Apps it holds.
    pub callees: Vec<Callee>,
    /// The root's interface document, when asked for.
    pub interface: Option<Box<RawValue>>,
}

/// The environment's root digest and the other Apps it holds, described inside it by the
/// machine's client without importing them (`runtime_describe`), and with `interface` the
/// root's own interface. A failed description fails this preparation: it never publishes an
/// environment with silently missing callees.
pub fn describe_environment(
    python: &str,
    root: &str,
    packages: &BTreeMap<String, String>,
    interface: bool,
) -> Result<Described, Failure> {
    #[derive(serde::Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum Reply {
        DescribedEnvironment {
            source_digest: String,
            callees: Vec<Callee>,
            #[serde(default)]
            interface: String,
        },
        DescribeFailed { code: String, detail: String },
    }
    let request = json!({"kind": "describe_environment", "root": root, "packages": packages, "interface": interface});
    let (stdout, success) = in_environment(python, &request)?;
    match serde_json::from_slice::<Reply>(&stdout).map_err(undescribed)? {
        Reply::DescribedEnvironment { source_digest, callees, interface: own } if success => {
            let interface = match own.is_empty() {
                true => None,
                false => Some(RawValue::from_string(own).map_err(undescribed)?),
            };
            Ok(Described { source_digest, callees, interface })
        }
        Reply::DescribeFailed { code, detail } => Err(("package_prepare_description_failed", format!("{code}: {detail}"))),
        _ => Err(("package_prepare_description_failed", "environment description did not complete".into())),
    }
}

/// A live project's interface, described from its own directory by the environment's Runtime.
pub fn describe_project(python: &str, project: &Path) -> Result<Box<RawValue>, Failure> {
    // A flat record: serde buffers an internally tagged enum, which a RawValue cannot pass.
    #[derive(serde::Deserialize)]
    struct Reply {
        kind: String,
        interface: Option<Box<RawValue>>,
        #[serde(default)]
        code: String,
        #[serde(default)]
        detail: String,
    }
    let request = json!({"kind": "describe", "project": project, "environment_python": python});
    let (stdout, success) = in_environment(python, &request)?;
    let reply = serde_json::from_slice::<Reply>(&stdout).map_err(undescribed)?;
    match (reply.kind.as_str(), reply.interface) {
        ("described", Some(interface)) if success => Ok(interface),
        ("describe_failed", _) => Err(("package_prepare_description_failed", format!("{}: {}", reply.code, reply.detail))),
        _ => Err(("package_prepare_description_failed", "project description did not complete".into())),
    }
}

/// The client's answer to one request inside an environment. -S keeps authored .pth startup
/// code out; only the venv's own site-packages are added.
fn in_environment(python: &str, request: &serde_json::Value) -> Result<(Vec<u8>, bool), Failure> {
    const BOOTSTRAP: &str = "import pathlib,runpy,sys;\
        root=pathlib.Path(sys.executable).absolute().parent.parent;\
        sys.path[:0]=[str(p) for p in (*(root/'lib').glob('python*/site-packages'),root/'Lib'/'site-packages') if p.is_dir()];\
        runpy.run_module('tensord_client.runtime_describe',run_name='__main__')";
    let output = Command::new(python)
        .args(["-I", "-S", "-c", BOOTSTRAP])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .and_then(|mut child| {
            child.stdin.take().expect("piped").write_all(request.to_string().as_bytes())?;
            child.wait_with_output()
        })
        .map_err(storage)?;
    Ok((output.stdout, output.status.success()))
}

fn undescribed(error: serde_json::Error) -> Failure {
    ("package_prepare_description_failed", format!("environment description failed: {error}"))
}

/// The generation's record, published atomically at `dir`: only it makes the environment
/// resolvable.
pub(crate) fn publish(dir: &Path, generations: &Path, record: &Generation) -> Result<(), Failure> {
    let staged = dir.join(".generation.json.new");
    let mut file = OpenOptions::new().create_new(true).write(true).open(&staged).map_err(storage)?;
    let bytes = serde_json::to_vec(record).map_err(|e| storage(io::Error::other(e)))?;
    file.write_all(&bytes).and_then(|_| file.sync_all()).map_err(storage)?;
    fs::rename(&staged, dir.join("generation.json")).map_err(storage)?;
    platform::durable::sync_dir(dir)
        .and_then(|()| platform::durable::sync_dir(generations))
        .map_err(|error| storage(error.into()))
}

fn storage(error: io::Error) -> Failure {
    ("package_installation_failed", error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// uv as a test runs it: whatever uv this machine has, under no stop.
    fn uv(args: &[&str]) -> Result<(), Failure> {
        let mut command = Command::new("uv");
        command.args(args);
        run_uv(command, &args[..2].join(" "), None, &|_| ())
    }

    /// A venv holding installed metadata alone: (name, version, wheel tag, requirement).
    fn holding(root: &Path, rows: &[(&str, &str, &str, &str)]) -> String {
        let env = root.join("env");
        uv(&["venv", "--no-config", "--python", "3.12", &env.to_string_lossy()]).unwrap();
        let site = platform::paths::venv_site_packages(&env).unwrap();
        for (name, version, tag, requires) in rows {
            let info = site.join(format!("{name}-{version}.dist-info"));
            fs::create_dir_all(&info).unwrap();
            let requires = if requires.is_empty() { String::new() } else { format!("Requires-Dist: {requires}\n") };
            fs::write(info.join("METADATA"), format!("Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{requires}")).unwrap();
            fs::write(info.join("WHEEL"), format!("Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}\n")).unwrap();
            fs::write(info.join("RECORD"), format!("{name}-{version}.dist-info/METADATA,,\n")).unwrap();
        }
        platform::paths::venv_python(&env).to_string_lossy().into_owned()
    }

    /// uv's real check of an environment holding both (NVIDIA's aarch64 cusparseLt wheel says
    /// sbsa): the platform tag stands, the missed requirement is a finding.
    #[test]
    fn a_wheel_tagged_for_another_platform_stands_and_a_missed_requirement_is_a_finding() {
        let root = std::env::temp_dir().join(format!("cm-env-check-{}", uuid::Uuid::new_v4()));
        let py = holding(&root, &[
            ("numpy", "2.0.0", "cp312-cp312-manylinux_2_17_x86_64", ""),
            ("conflicted_app", "0.1.0", "py3-none-any", "numpy==1.0.0"),
            ("nvidia_cusparselt_cu12", "0.6.3", "py3-none-manylinux2014_sbsa", ""),
        ]);
        let missed = "The package `conflicted-app` requires `numpy==1.0.0`, but `2.0.0` is installed".to_string();
        assert_eq!(check(&uv, &py).unwrap(), BTreeSet::from([missed.clone()]));
        // The environment installs all the same, and says what uv found.
        assert_eq!(findings(&uv, &py, "").unwrap(), [missed]);
        let _ = fs::remove_dir_all(root);
    }

    /// A bound on the pair the machine's development Runtime runs regardless is its warning
    /// (`sdk_fallback`), no finding; any other miss is one.
    #[test]
    fn a_development_runtime_runs_a_package_whose_pair_bound_it_misses() {
        let root = std::env::temp_dir().join(format!("cm-env-dev-{}", uuid::Uuid::new_v4()));
        let py = holding(&root, &[
            ("cozy_runtime", "0.30.1", "py3-none-any", ""),
            ("script", "0.1.0", "py3-none-any", "cozy-runtime==0.30.0"),
        ]);
        assert!(findings(&uv, &py, "0.30.1+0123abcd").unwrap().is_empty());
        assert_eq!(findings(&uv, &py, "").unwrap().len(), 1);
        assert_eq!(
            installed(&root.join("env")).into_iter().map(|d| (d.name, d.version)).collect::<Vec<_>>(),
            [("cozy-runtime".to_string(), "0.30.1".to_string()), ("script".to_string(), "0.1.0".to_string())]
        );
        let _ = fs::remove_dir_all(root);
    }
}
