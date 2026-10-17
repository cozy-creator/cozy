//! TensorD's configuration: the binary's one load ([`cozy_proto::config`]: the `[machine]` table
//! of `~/.cozy/config.toml`, then a pod's environment), the command line's lifetime, and the
//! grant judged from them. Nothing else in TensorD reads the environment; the processes it starts
//! inherit [`cozy_proto::config::environment`], filtered.
use crate::machine::{
    endpoint::Endpoint,
    grant::{Grant, Lifetime},
};
use cozy_proto::config::{self as shared, Loaded, ENV};
use std::{
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
};

/// `cozy machine serve` (and the `tensord` binary's run modes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::Args)]
#[group(multiple = false)]
pub struct Flags {
    /// A provider's pod: Hub lease, readiness receipt, idle release.
    #[arg(long)]
    pub rental: bool,
    /// This computer's machine, run as a user service.
    #[arg(long)]
    pub persistent: bool,
}

impl Flags {
    pub fn lifetime(&self) -> Option<Lifetime> {
        match (self.rental, self.persistent) {
            (true, _) => Some(Lifetime::Rental),
            (_, true) => Some(Lifetime::Persistent),
            _ => None,
        }
    }
}

/// The readiness key's names, taken out of the environment once read.
const RECEIPT_KEYS: [&str; 2] = ["COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL", "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_FILE"];
/// The provider's own names: where it maps this pod's ports, and its credential.
const PROVIDER: &[&str] = &[
    "COZY_WEBRTC_PUBLIC_ADDRESS",
    "COZY_PROVIDER_API_ORIGIN",
    "RUNPOD_PUBLIC_IP",
    "RUNPOD_POD_ID",
    "RUNPOD_API_KEY",
    "PUBLIC_IPADDR",
    "CONTAINER_ID",
    "CONTAINER_API_KEY",
];
const PROVIDER_PREFIXES: &[&str] = &["RUNPOD_TCP_PORT_", "VAST_TCP_PORT_"];

/// One machine's configuration.
pub struct Config {
    pub grant: Grant,
    /// CUDA_VISIBLE_DEVICES as the launcher set it: indices or UUIDs; empty is no GPU.
    pub visible_devices: Option<String>,
    /// The provider's public mapping of this pod's ports, told to the Hub at boot.
    pub endpoint: Option<Endpoint>,
    /// The provider's names, for [`crate::machine::provider::ProviderSelf::take`].
    pub provider: BTreeMap<String, String>,
    /// Where this machine's own log rolls: `<root>/var/lib/cozy/machine/logs`.
    pub logs: PathBuf,
    /// What the load ignored, said once the log is up.
    pub warnings: Vec<String>,
}

/// The one load. Call before any thread starts: the readiness key leaves the environment here, so
/// nothing this machine starts inherits it.
pub fn load(lifetime: Lifetime) -> io::Result<Config> {
    let loaded = shared::load(shared::Flags::default()).map_err(|e| invalid(&named(&e.message)))?;
    for name in RECEIPT_KEYS {
        std::env::remove_var(name);
    }
    from_loaded(lifetime, loaded, shared::environment())
}

/// The load over a given environment and user home (what the tests drive).
pub fn load_from(lifetime: Lifetime, environment: &BTreeMap<String, String>, user_home: PathBuf) -> io::Result<Config> {
    let loaded = shared::load_from(shared::Flags::default(), environment, user_home).map_err(|e| invalid(&named(&e.message)))?;
    from_loaded(lifetime, loaded, environment)
}

fn from_loaded(lifetime: Lifetime, loaded: Loaded, environment: &BTreeMap<String, String>) -> io::Result<Config> {
    let machine = &loaded.config.machine;
    let provider: BTreeMap<String, String> = environment
        .iter()
        .filter(|(n, _)| PROVIDER.contains(&n.as_str()) || PROVIDER_PREFIXES.iter().any(|p| n.starts_with(p)))
        .map(|(n, v)| (n.clone(), v.clone()))
        .collect();
    let reach = crate::machine::player::reach(machine.webrtc_port, &provider);
    let grant = Grant::new(lifetime, machine, reach, &loaded.home)?;
    let mut warnings = loaded.warnings;
    warnings.extend(
        environment
            .keys()
            .filter(|n| (n.starts_with("COZY_") || n.starts_with("TENSORHUB_")) && !known(n))
            .map(|n| format!("ignoring {n}, which this machine does not read")),
    );
    let rental = lifetime == Lifetime::Rental;
    Ok(Config {
        logs: grant.layout.state.join("logs"),
        visible_devices: machine.visible_devices.clone(),
        endpoint: rental.then(|| crate::machine::endpoint::from_env(provider.clone())).flatten(),
        provider,
        grant,
        warnings,
    })
}

fn known(name: &str) -> bool {
    ENV.iter().any(|(n, _)| *n == name) || PROVIDER.contains(&name) || name == shared::AUTHORIZED_KEYS
}

/// A load refusal naming a `machine.*` key also names the variable a pod sets it with.
fn named(message: &str) -> String {
    let mut message = message.to_owned();
    for (name, key) in ENV {
        message = message.replace(&format!("`{key}`"), &format!("`{key}` ({name})"));
    }
    message
}

/// Logs to stderr, and to `<dir>/tensord.<date>.log` when a directory is given: rolled daily,
/// the newest seven kept.
pub fn init_logs(dir: Option<&Path>) {
    use tracing_subscriber::fmt::writer::MakeWriterExt;
    let builder = tracing_subscriber::fmt().with_ansi(false);
    let file = dir.and_then(|dir| {
        std::fs::create_dir_all(dir)
            .map_err(|e| eprintln!("tensord: no log directory {}: {e}", dir.display()))
            .ok()?;
        tracing_appender::rolling::Builder::new()
            .rotation(tracing_appender::rolling::Rotation::DAILY)
            .filename_prefix("tensord")
            .filename_suffix("log")
            .max_log_files(7)
            .build(dir)
            .map_err(|e| eprintln!("tensord: no log file in {}: {e}", dir.display()))
            .ok()
    });
    let _ = match file {
        Some(file) => builder.with_writer(file.and(std::io::stderr)).try_init(),
        None => builder.with_writer(std::io::stderr).try_init(),
    };
}

fn invalid(detail: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, detail.to_owned())
}
