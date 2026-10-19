//! A real TensorD (`COZY_TEST_TENSORD`, this workspace's build) on a temp root, for
//! tests that add it as another computer's machine. It runs as a persistent machine from a grant in
//! its environment, as a computer's machine does, so its Status carries its store, disk, models and
//! environments. Its TLS identity outlives a restart.
use std::{
    io::Write as _,
    os::unix::{fs::PermissionsExt as _, process::CommandExt as _},
    path::Path,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

use base64::{
    Engine as _,
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
};
use sha2::{Digest as _, Sha256};

pub struct Tensord {
    child: Option<Child>,
    root: tempfile::TempDir,
    /// Extra environment for the process, such as where to find a test Hub's CA.
    env: Vec<(String, String)>,
    /// `host:port` it listens on.
    pub address: String,
    /// `sha256:<hex>` of its TLS leaf.
    pub fingerprint: String,
    /// The port it listens on across restarts, when a test names one (a shaped link's).
    port: Option<u16>,
}

/// A controller key no Calcifer holds, base64url as TensorD's keys file spells it.
pub fn stranger() -> String {
    let key = ed25519_dalek::SigningKey::from_bytes(&[9; 32]).verifying_key();
    URL_SAFE_NO_PAD.encode(key.as_bytes())
}

impl Tensord {
    /// Its store's root.
    pub fn store(&self) -> std::path::PathBuf {
        self.root.path().join("machine/var/lib/tensorfs")
    }

    /// Starts a machine that admits the controllers holding `keys` (base64url Ed25519 public keys).
    pub fn start(keys: &[String]) -> Self {
        Self::start_with(keys, &[])
    }

    /// `start`, with extra environment for the process.
    pub fn start_with(keys: &[String], env: &[(&str, &str)]) -> Self {
        Self::start_on(None, keys, env)
    }

    /// `start_with`, listening on `port` (a new free one each start when `None`).
    pub fn start_on(port: Option<u16>, keys: &[String], env: &[(&str, &str)]) -> Self {
        let root = tempfile::tempdir().unwrap();
        std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let mut machine = Self {
            child: None,
            root,
            env: env
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            address: String::new(),
            fingerprint: String::new(),
            port,
        };
        machine.restart(keys);
        machine
    }

    /// Its machine root (`COZY_MACHINE_ROOT`).
    pub fn root(&self) -> std::path::PathBuf {
        self.root.path().join("machine")
    }

    /// Restarts it admitting `keys`: same identity and leaf, a new port.
    pub fn restart(&mut self, keys: &[String]) {
        self.stop();
        let root = self.root.path().join("machine");
        let key_file = self.root.path().join("receipt-key");
        std::fs::write(&key_file, URL_SAFE_NO_PAD.encode([7u8; 32])).unwrap();
        std::fs::set_permissions(&key_file, std::fs::Permissions::from_mode(0o600)).unwrap();
        let port = self.port.unwrap_or_else(|| {
            std::net::TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .port()
        });
        let binary = std::env::var_os("COZY_TEST_TENSORD")
            .expect("COZY_TEST_TENSORD names the tensord binary these tests run against");
        let mut child = Command::new(binary)
            .args(["run", "--persistent"])
            .env_remove("COZY_TEST_TENSORD")
            // A home of its own: this computer's keys and config are not the test machine's.
            .env("HOME", self.root.path())
            .env("COZY_MACHINE_ROOT", &root)
            .env("COZY_LISTEN_HOST", "127.0.0.1")
            .env("COZY_WORKER_ID", "cli-test")
            .env("COZY_WORKER_INTERNAL_PORT", port.to_string())
            .env("COZY_AUTHORIZED_KEYS", keys.join(","))
            .env("COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_FILE", &key_file)
            .env("CUDA_VISIBLE_DEVICES", "")
            // It trusts a private Hub only as a run's Hub access names its CA.
            .env_remove("SSL_CERT_FILE")
            .env_remove("SSL_CERT_DIR")
            .envs(self.env.iter().map(|(k, v)| (k, v)))
            // Its own process group: the machine's stable parent starts children, and stop ends them all.
            .process_group(0)
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        self.address = format!("127.0.0.1:{port}");
        let leaf = root.join("run/cozy/bootstrap/tls.crt");
        wait_ready(&mut child, &leaf, &self.address);
        self.child = Some(child);
        let body: String = std::fs::read_to_string(&leaf)
            .unwrap()
            .lines()
            .filter(|line| !line.starts_with("-----"))
            .collect();
        let der = STANDARD.decode(body).unwrap();
        self.fingerprint = format!("sha256:{}", hex::encode(Sha256::digest(&der)));
    }

    /// Ends the exact processes this test started: its process group. Its service lets go of the
    /// root's owner lock only as it exits, after its parent's wait returns, so this waits for that
    /// lock too: a restart on the same root never finds it held.
    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = Command::new("kill")
                .args(["-KILL", "--", &format!("-{}", child.id())])
                .status();
            let _ = child.wait();
            let lock = self
                .root
                .path()
                .join("machine/var/lib/cozy/tensord/owner.lock");
            if let Ok(file) = std::fs::File::open(lock) {
                let _ = file.lock();
            }
        }
    }
}

impl Drop for Tensord {
    fn drop(&mut self) {
        self.stop();
    }
}

fn wait_ready(child: &mut Child, leaf: &Path, address: &str) {
    // A hang bound for a machine that never starts; the product waits without one.
    let until = Instant::now() + Duration::from_secs(120);
    while !(leaf.exists() && std::net::TcpStream::connect(address).is_ok()) {
        if let Ok(Some(status)) = child.try_wait() {
            panic!("tensord exited before it served: {status}");
        }
        assert!(
            Instant::now() < until,
            "tensord did not listen on {address}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// A TensorD added to `home`'s Calcifer as `name`, the way a person adds one: `cozy machine add`
/// answers the key this controller signs with, the machine admits it, and the add is made again.
pub fn added(home: &super::Home, name: &str) -> Tensord {
    added_with(home, name, &[])
}

/// `added`, with extra environment for the TensorD process.
pub fn added_with(home: &super::Home, name: &str, env: &[(&str, &str)]) -> Tensord {
    added_on(home, name, None, env)
}

/// `added_with`, listening on `port`.
pub fn added_on(
    home: &super::Home,
    name: &str,
    port: Option<u16>,
    env: &[(&str, &str)],
) -> Tensord {
    let machine = joined(home, name, port, env);
    if std::env::var_os("COZY_TEST_RUNTIME_WHEEL").is_some() {
        named_runtime(home, name);
        PROBED.get_or_init(probe);
    }
    machine
}

/// A TensorD added as `name` without the job's named Runtime: the software a machine has before
/// any update, for a test of what an update to the Hub's target does.
pub fn added_unupdated(home: &super::Home, name: &str) -> Tensord {
    joined(home, name, None, &[])
}

/// A TensorD that `home`'s Calcifer added as `name`.
fn joined(home: &super::Home, name: &str, port: Option<u16>, env: &[(&str, &str)]) -> Tensord {
    let mut machine = Tensord::start_on(port, &[stranger()], env);
    let add = |machine: &Tensord| {
        home.cozy([
            "machine",
            "add",
            name,
            machine.address.as_str(),
            "--fingerprint",
            machine.fingerprint.as_str(),
            "--json",
        ])
    };
    machine.restart(&[key_to_authorize(&add(&machine))]);
    let out = add(&machine);
    assert_eq!(out.code, 0, "{out:?}");
    machine
}

/// The Runtime wheel the job names (`COZY_TEST_RUNTIME_WHEEL`, with `COZY_TEST_TENSORFS_WHEEL` when
/// it built one) installed on `name` as a person installs a development Runtime: `cozy machine
/// update --runtime-wheel`. The update must end on that version.
fn named_runtime(home: &super::Home, name: &str) {
    let runtime = std::env::var_os("COZY_TEST_RUNTIME_WHEEL").expect("a named Runtime wheel");
    let mut update: Vec<std::ffi::OsString> = vec![
        "--json".into(),
        "machine".into(),
        "update".into(),
        name.into(),
        "--runtime-wheel".into(),
        runtime.clone(),
    ];
    if let Some(tensorfs) = std::env::var_os("COZY_TEST_TENSORFS_WHEEL") {
        update.extend(["--tensorfs-wheel".into(), tensorfs]);
    }
    let out = home.cozy(&update);
    let version = Path::new(&runtime)
        .file_name()
        .and_then(|n| n.to_str()?.split('-').nth(1));
    let completed = out
        .stdout
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .rfind(|e| e["type"] == "run.completed");
    let to = completed
        .as_ref()
        .and_then(|e| e["run"]["result"]["to"]["runtime"].as_str());
    assert!(
        out.code == 0 && to.is_some() && to == version,
        "installing {runtime:?} on {name} (want {version:?}, have {to:?}): {out:?}"
    );
}

/// Once per test process.
static PROBED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// A script that answers the commit of the Runtime it runs on.
const PROVENANCE: &str = "# /// script\n# requires-python = \">=3.12\"\n# dependencies = [\"cozy-runtime\"]\n\
    # [tool.cozy]\n# accelerator = false\n# ///\nfrom cozy_runtime._build_provenance import COMMIT\n\n\n\
    def main() -> str:\n    return COMMIT\n";

/// The Runtime a machine updated to the named wheel loads is that build, by the commit stamped in
/// it (`cozy_runtime._build_provenance.COMMIT`); a wheel without one proves nothing. On a machine
/// of its own, in a home of its own, so no test's machine holds the probe's script.
fn probe() {
    let runtime = std::env::var_os("COZY_TEST_RUNTIME_WHEEL").expect("a named Runtime wheel");
    let want = stamped(Path::new(&runtime));
    assert!(
        !want.is_empty() && want != "unknown",
        "{runtime:?} names no commit (COMMIT = {want:?}): a wheel without provenance proves nothing"
    );
    let home = super::Home::new().expect("a home where the named wheel is");
    let _machine = joined(&home, "provenance", None, &[]);
    named_runtime(&home, "provenance");
    std::fs::write(home.root().join("provenance.py"), PROVENANCE).unwrap();
    let probe = super::out(
        home.command()
            .current_dir(home.root())
            .args(["--json", "run", "provenance.py", "--on", "provenance"])
            .output()
            .unwrap(),
    );
    let loaded = probe
        .stdout
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .rfind(|e| e["type"] == "run.completed")
        .and_then(|e| e["run"]["result"]["value"].as_str().map(str::to_owned));
    assert_eq!(
        loaded.as_deref(),
        Some(want.as_str()),
        "a machine updated to {runtime:?} runs another Runtime: {probe:?}"
    );
    // Past libtest's capture: a silent pass reads as missing.
    let version = Path::new(&runtime)
        .file_name()
        .and_then(|n| n.to_str()?.split('-').nth(1));
    let _ = writeln!(
        std::io::stderr(),
        "named Runtime {} (commit {want}) loads: asserted",
        version.unwrap_or("?")
    );
}

/// The commit a Runtime wheel was built from: its `_build_provenance.COMMIT`.
pub fn stamped(wheel: &Path) -> String {
    const READ: &str = "import re, sys, zipfile\n\
        text = zipfile.ZipFile(sys.argv[1]).read('cozy_runtime/_build_provenance.py').decode()\n\
        print(re.search(r'^COMMIT = \"([^\"]*)\"', text, re.M).group(1))\n";
    let out = Command::new("python3")
        .args(["-I", "-c", READ])
        .arg(wheel)
        .output()
        .unwrap();
    assert!(out.status.success(), "reading {wheel:?}: {out:?}");
    String::from_utf8(out.stdout).unwrap().trim().to_owned()
}

/// The controller key a refused `cozy machine add --json` asks that computer to authorize.
pub fn key_to_authorize(refused: &super::Out) -> String {
    let remedy = refused.refusal()["remedy"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    remedy
        .split("cozy machine authorize ")
        .nth(1)
        .and_then(|rest| rest.split(|c: char| c == '`' || c.is_whitespace()).next())
        .unwrap_or_else(|| panic!("the add names no key to authorize: {refused:?}"))
        .to_string()
}
