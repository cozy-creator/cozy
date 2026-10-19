//! `~/.cozy/config.toml`, loaded once per process through figment and shared by the CLI,
//! Calcifer and the machine (TensorD). Sources, each over the last: the file, the environment
//! (credentials and locations only: two model-source tokens, where releases come from, and a
//! pod's machine contract), the command line. The home is always `<user home>/.cozy`. A key this
//! build does not read is named once and ignored, so a file another version wrote never disables
//! a command.
use std::{
    collections::BTreeMap,
    fmt,
    path::PathBuf,
    sync::OnceLock,
};

use cozy_platform::paths::Home;
use figment::{
    Figment,
    providers::{Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use url::Url;

use crate::error::{Error, Kind};

/// The Hub every install knows by name.
pub const DEFAULT_HUB: &str = "tensorhub";
pub const DEFAULT_HUB_URL: &str = "https://tensorhub.com";

/// Every environment variable the load reads, with its key: credentials for model sources, by the
/// names their own tools use; where releases are downloaded from (dist's installers read the
/// same one); and the machine's settings a pod's launcher gives (Tensorhub `podenv`, plus the
/// devices the container may use).
pub const ENV: [(&str, &str); 18] = [
    ("HF_TOKEN", "huggingface_token"),
    ("CIVITAI_TOKEN", "civitai_token"),
    ("COZY_DOWNLOAD_URL", "download_url"),
    ("COZY_MACHINE_ROOT", "machine.root"),
    ("COZY_TENSORFS_ROOT", "machine.store"),
    ("COZY_TENSORFS_CACHE", "machine.cache"),
    ("COZY_LISTEN_HOST", "machine.listen_host"),
    ("COZY_WORKER_ID", "machine.worker_id"),
    ("COZY_WORKER_AUTH_TOKEN", "machine.worker_token"),
    ("COZY_WORKER_INTERNAL_PORT", "machine.port"),
    ("COZY_WEBRTC_INTERNAL_PORT", "machine.webrtc_port"),
    ("COZY_SSH_PUBLIC_KEY", "machine.ssh_public_key"),
    ("COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL", "machine.receipt_key"),
    ("COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_FILE", "machine.receipt_key_file"),
    ("TENSORHUB_ORIGIN", "machine.hub_origin"),
    ("TENSORHUB_CA_DER_B64URL", "machine.hub_ca_der"),
    ("TENSORHUB_OBJECT_STORAGE_HOSTS", "machine.object_storage_hosts"),
    ("CUDA_VISIBLE_DEVICES", "machine.visible_devices"),
];

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct Config {
    /// The current Hub: a name in `hubs`, or a URL. Absent: `tensorhub`.
    pub hub: Option<String>,
    /// Named Hubs, like kubectl contexts: name (lowercase) → canonical URL.
    pub hubs: BTreeMap<String, String>,
    pub calcifer: CalciferConfig,
    pub machine: MachineConfig,
    pub huggingface_token: Option<Secret>,
    pub civitai_token: Option<Secret>,
    /// The directory holding one release's files (dist-manifest.json, the archives); absent: GitHub
    /// Releases.
    pub download_url: Option<Url>,
    /// `--tensorhub`: this command's Hub. From the command line only.
    pub tensorhub: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct CalciferConfig {
    /// Calcifer's loopback port; 0 takes any free one (its lock record says which).
    pub port: u16,
}

/// `[machine]`: this computer's machine (TensorD, `cozy machine serve`), which judges these. A
/// pod's launcher gives the same settings through the environment ([`ENV`]).
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct MachineConfig {
    /// The machine API's port. On this computer, absent or 0: a free port, kept across restarts.
    pub port: Option<u16>,
    /// The WebRTC player's port, kept across restarts the same way.
    pub webrtc_port: Option<u16>,
    /// The TensorFS store it fills; absent: `var/lib/tensorfs` under its root.
    pub store: Option<PathBuf>,
    /// How long a rental stays with no job before it releases itself; absent: 15 minutes
    /// (shorter only in tests).
    pub rental_idle_seconds: Option<u32>,
    /// The image layout's root: `/` on a pod, `~/.cozy/machine` on this computer.
    pub root: Option<PathBuf>,
    /// The datacenter cache volume's mount.
    pub cache: Option<PathBuf>,
    /// `0.0.0.0` or `127.0.0.1`.
    pub listen_host: Option<String>,
    pub worker_id: Option<String>,
    pub worker_token: Option<Secret>,
    /// A pod's boot keys (`COZY_AUTHORIZED_KEYS=a,b`), from the environment only: this
    /// computer's controllers are `keys/owner.pem`'s and `keys/authorized`'s.
    #[serde(skip)]
    pub authorized_keys: Vec<String>,
    /// A development rental's SSH key.
    pub ssh_public_key: Option<String>,
    pub receipt_key: Option<Secret>,
    pub receipt_key_file: Option<PathBuf>,
    pub hub_origin: Option<String>,
    pub hub_ca_der: Option<String>,
    /// Exact hosts, or `.suffix` for subdomains (`a,b` in the environment).
    #[serde(deserialize_with = "list")]
    pub object_storage_hosts: Vec<String>,
    /// CUDA_VISIBLE_DEVICES: indices or UUIDs; empty is no GPU.
    pub visible_devices: Option<String>,
    /// The most memory each GPU offers, below its own (`8GiB` or bytes): a larger card stands in
    /// for a smaller one (`cozy machine update --gpu-memory-limit`).
    #[serde(deserialize_with = "size")]
    pub gpu_memory_limit: Option<u64>,
}

impl MachineConfig {
    /// This computer's machine root: `root`, else `~/.cozy/machine`. Its API's endpoint is
    /// `<root>/var/lib/cozy/tensord/api-ready.json`.
    pub fn root(&self, home: &Home) -> PathBuf {
        self.root.clone().unwrap_or_else(|| home.machine())
    }
}

/// The environment name of a pod's boot keys.
pub const AUTHORIZED_KEYS: &str = "COZY_AUTHORIZED_KEYS";

/// A list as a TOML array, or as one comma-separated string (how the environment spells it).
fn list<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum List {
        Spelled(String),
        Items(Vec<String>),
    }
    Ok(match List::deserialize(d)? {
        List::Spelled(text) => split(&text),
        List::Items(items) => items,
    })
}

fn size<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Size {
        Bytes(u64),
        Spelled(String),
    }
    match Option::<Size>::deserialize(d)? {
        None => Ok(None),
        Some(Size::Bytes(n)) => Ok(Some(n)),
        Some(Size::Spelled(text)) => parse_size(&text).map(Some).map_err(serde::de::Error::custom),
    }
}

/// A size as people write it: bytes, or a number of `KiB`, `MiB`, `GiB` or `TiB` (any case,
/// with or without a space): `8GiB`, `1.5 GiB`, `512MiB`.
pub fn parse_size(text: &str) -> Result<u64, String> {
    let text = text.trim();
    let digits = text.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(text.len());
    let (number, unit) = text.split_at(digits);
    let shift = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 0,
        "kib" => 10,
        "mib" => 20,
        "gib" => 30,
        "tib" => 40,
        _ => return Err(format!("{text:?} is not a size: write bytes or KiB, MiB, GiB, TiB (8GiB)")),
    };
    let value: f64 = number.parse().map_err(|_| format!("{text:?} is not a size: no number"))?;
    Ok((value * (1u64 << shift) as f64) as u64)
}

fn split(text: &str) -> Vec<String> {
    text.split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect()
}

/// A credential: never printed.
#[derive(Clone, PartialEq, Eq, Deserialize)]
#[serde(transparent)]
pub struct Secret(String);

impl Secret {
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(…)")
    }
}

/// One Tensorhub, named or by URL.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct Hub {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "url")]
    pub origin: Origin,
}

/// A Tensorhub base URL in its one spelling: lowercase scheme and host, no default port, no
/// trailing slash. Every record and credential is keyed by it, so `https://tensorhub.com:443/`
/// and `https://tensorhub.com` are one Hub.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct Origin(String);

impl Origin {
    pub fn parse(text: &str) -> Result<Self, Error> {
        let text = text.trim();
        let url = Url::parse(text).ok().filter(|u| {
            matches!(u.scheme(), "http" | "https")
                && u.host_str().is_some()
                && u.username().is_empty()
                && u.password().is_none()
                && u.query().is_none()
                && u.fragment().is_none()
        });
        match url {
            Some(url) => Ok(Self(url.as_str().trim_end_matches('/').to_owned())),
            None => Err(Error::new(
                Kind::Usage,
                "hub_invalid",
                format!(
                    "{text:?} is not an http or https URL without credentials, query or fragment"
                ),
            )),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::ops::Deref for Origin {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<Origin> for String {
    fn from(origin: Origin) -> String {
        origin.0
    }
}

impl PartialEq<str> for Origin {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for Origin {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl PartialEq<String> for Origin {
    fn eq(&self, other: &String) -> bool {
        &self.0 == other
    }
}

/// What the command line contributes to the load.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Flags {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorhub: Option<String>,
}

/// One process's configuration and where it came from.
#[derive(Clone, Debug)]
pub struct Loaded {
    pub config: Config,
    pub home: Home,
    /// The user's home directory, which services and `~` are relative to.
    pub user_home: PathBuf,
    /// What the load ignored, for the caller to show once.
    pub warnings: Vec<String>,
}

/// The one read of the environment and the config file.
pub fn load(flags: Flags) -> Result<Loaded, Error> {
    let user_home = std::env::home_dir()
        .ok_or_else(|| Error::new(Kind::Internal, "no_home", "this user has no home directory"))?;
    load_from(flags, environment(), user_home)
}

/// The load over a given environment and user home (what tests drive).
pub fn load_from(
    flags: Flags,
    environment: &BTreeMap<String, String>,
    user_home: PathBuf,
) -> Result<Loaded, Error> {
    let home = Home::under(&user_home);
    let path = home.config();
    let mut figment = Figment::new();
    if path.exists() {
        figment = figment.merge(Toml::file_exact(&path));
    }
    // As spelled: a value is a string until its setting's type reads it (`"0"` stays a device).
    for (var, key) in ENV {
        if let Some(value) = environment.get(var).filter(|v| !v.is_empty()) {
            figment = figment.merge(Serialized::default(key, value));
        }
    }
    let figment = figment.merge(Serialized::defaults(flags));
    let (mut config, warnings) = extract(figment, &path.display().to_string())?;
    config.machine.authorized_keys = split(environment.get(AUTHORIZED_KEYS).map_or("", String::as_str));
    Ok(Loaded {
        config,
        home,
        user_home,
        warnings,
    })
}

impl Config {
    /// A config file's text as the load would read it: how an edit proves it still loads.
    pub fn parse(text: &str, name: &str) -> Result<(Config, Vec<String>), Error> {
        extract(Figment::from(Toml::string(text)), name)
    }

    /// Every named Hub, `tensorhub` included, by name.
    pub fn hubs(&self) -> Vec<Hub> {
        let mut hubs: BTreeMap<&str, &str> = [(DEFAULT_HUB, DEFAULT_HUB_URL)].into();
        hubs.extend(self.hubs.iter().map(|(n, u)| (n.as_str(), u.as_str())));
        hubs.into_iter()
            .filter_map(|(name, origin)| {
                Some(Hub {
                    name: Some(name.to_owned()),
                    origin: Origin::parse(origin).ok()?,
                })
            })
            .collect()
    }

    /// A Hub's name in any case, or its URL; a URL takes the name it has here.
    pub fn resolve(&self, text: &str) -> Result<Hub, Error> {
        let text = text.trim();
        let hubs = self.hubs();
        let lower = text.to_ascii_lowercase();
        if let Some(hub) = hubs.iter().find(|h| h.name.as_deref() == Some(&lower)) {
            return Ok(hub.clone());
        }
        if !text.contains("://") {
            return Err(Error::new(
                Kind::NotFound,
                "hub_not_found",
                format!("no Hub is named {text:?}"),
            )
            .next("cozy hub list")
            .next("cozy hub add <name> <url>"));
        }
        let origin = Origin::parse(text)?;
        let name = hubs
            .into_iter()
            .find(|h| h.origin == origin)
            .and_then(|h| h.name);
        Ok(Hub { name, origin })
    }

    /// The current Hub (`hub`, set by `cozy hub use`).
    pub fn current(&self) -> Result<Hub, Error> {
        self.resolve(self.hub.as_deref().unwrap_or(DEFAULT_HUB))
            .map_err(|e| e.remedy("set `hub` with `cozy hub use <name>`"))
    }

    /// Where new work goes: `--tensorhub`, else the current Hub.
    pub fn selected(&self) -> Result<Hub, Error> {
        match self.narrowed()? {
            Some(hub) => Ok(hub),
            None => self.current(),
        }
    }

    /// `--tensorhub`'s Hub: list verbs show every Hub without it.
    pub fn narrowed(&self) -> Result<Option<Hub>, Error> {
        self.tensorhub
            .as_deref()
            .map(|t| self.resolve(t))
            .transpose()
    }

    /// Lowercases names and canonicalizes URLs, dropping (and naming) entries that are neither.
    fn normalize(&mut self, file: &str, warnings: &mut Vec<String>) {
        let mut hubs = BTreeMap::new();
        for (name, url) in std::mem::take(&mut self.hubs) {
            let lower = name.to_ascii_lowercase();
            match (valid_name(&lower), Origin::parse(&url)) {
                (true, Ok(origin)) => {
                    hubs.insert(lower, origin.into());
                }
                (false, _) => warnings.push(format!(
                    "{file}: hub name {name:?} is not letters, digits, '.', '_' or '-'; ignored"
                )),
                (true, Err(e)) => {
                    warnings.push(format!("{file}: hub {name:?}: {}; ignored", e.message))
                }
            }
        }
        self.hubs = hubs;
        for secret in [
            &mut self.huggingface_token,
            &mut self.civitai_token,
            &mut self.machine.worker_token,
            &mut self.machine.receipt_key,
        ] {
            if secret.as_ref().is_some_and(|s| s.0.trim().is_empty()) {
                *secret = None;
            }
        }
    }
}

impl Hub {
    /// How a person reads it: its name, else its URL.
    pub fn label(&self) -> &str {
        self.name.as_deref().unwrap_or(self.origin.as_str())
    }

    /// This computer's device key at this Hub (hub-client's `DeviceKey`), mode 0600.
    pub fn device_key(&self, home: &Home) -> PathBuf {
        home.keys().join(format!("device-{}.json", self.key_name()))
    }

    /// The short Bearer that device key last minted, kept for the next process.
    pub fn bearer(&self, home: &Home) -> PathBuf {
        home.keys()
            .join(format!("device-{}.bearer.json", self.key_name()))
    }

    fn key_name(&self) -> String {
        hex::encode(Sha256::digest(self.origin.as_bytes()))
    }
}

impl fmt::Display for Hub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.name {
            Some(name) => write!(f, "{name} ({})", self.origin),
            None => f.write_str(self.origin.as_str()),
        }
    }
}

/// This process's environment, read once: the load's source, and what the processes it starts
/// inherit (each filters it further).
pub fn environment() -> &'static BTreeMap<String, String> {
    static ENVIRONMENT: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    ENVIRONMENT.get_or_init(|| std::env::vars().collect())
}

/// Whether `name` (already lowercase) can name a Hub. A name is never a URL.
pub fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    name.len() <= 63
        && chars
            .next()
            .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "._-".contains(c))
}

fn extract(figment: Figment, file: &str) -> Result<(Config, Vec<String>), Error> {
    let invalid = |e: figment::Error| {
        let key = e.path.join(".");
        let at = if key.is_empty() {
            String::new()
        } else {
            format!(" `{key}`")
        };
        Error::new(
            Kind::Usage,
            "config_invalid",
            format!("{file}{at}: {}", e.kind),
        )
        .remedy(format!("correct or remove it in {file}"))
    };
    let mut config: Config = figment.extract_lossy().map_err(invalid)?;
    let mut warnings = vec![];
    let value: figment::value::Value = figment.extract().map_err(invalid)?;
    let _: Result<Config, _> = serde_ignored::deserialize(&value, |path| {
        warnings.push(format!(
            "{file}: `{path}` is not a setting this cozy reads; ignored"
        ))
    });
    config.normalize(file, &mut warnings);
    Ok((config, warnings))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> (Config, Vec<String>) {
        Config::parse(text, "config.toml").unwrap()
    }

    #[test]
    fn a_gpu_memory_limit_reads_as_bytes_or_a_size() {
        for (text, bytes) in [("8GiB", 8 << 30), ("1.5 gib", 3 << 29), ("512MiB", 512 << 20), ("4096", 4096)] {
            assert_eq!(parse_size(text), Ok(bytes), "{text}");
        }
        assert!(parse_size("8GB").is_err() && parse_size("GiB").is_err());
        let limit = |text: &str| parse(text).0.machine.gpu_memory_limit;
        assert_eq!(limit("[machine]\ngpu_memory_limit = \"8GiB\"\n"), Some(8 << 30));
        assert_eq!(limit("[machine]\ngpu_memory_limit = 1073741824\n"), Some(1 << 30));
        assert_eq!(limit(""), None);
        assert!(Config::parse("[machine]\ngpu_memory_limit = \"lots\"\n", "config.toml").is_err());
    }

    #[test]
    fn urls_have_one_spelling() {
        for (text, canonical) in [
            ("HTTPS://Example.TEST:443/", "https://example.test"),
            ("https://tensorhub.com:443", "https://tensorhub.com"),
            ("http://Hub.test:80", "http://hub.test"),
            ("https://hub.test:80", "https://hub.test:80"),
            ("http://127.0.0.1:8819/", "http://127.0.0.1:8819"),
            ("https://hub.test/api/", "https://hub.test/api"),
        ] {
            assert_eq!(Origin::parse(text).unwrap(), canonical);
        }
        assert_eq!(
            Origin::parse("https://tensorhub.com:443/").unwrap(),
            Origin::parse("https://tensorhub.com").unwrap()
        );
        for bad in [
            "ftp://x",
            "https://u:p@x",
            "https://x/?q=1",
            "x",
            "https://x#f",
        ] {
            assert_eq!(Origin::parse(bad).unwrap_err().code, "hub_invalid", "{bad}");
        }
    }

    #[test]
    fn hubs_resolve_by_name_in_any_case_or_by_url() {
        let (mut config, warnings) = parse(
            "hub = \"LOCAL\"\nfuture = 1\n[hubs]\nLocal = \"http://127.0.0.1:8819/\"\n\"Bad Name\" = \"http://x\"\n[calcifer]\nport = 7\nidle = 3\n",
        );
        assert_eq!(config.calcifer.port, 7);
        assert!(
            warnings.iter().any(|w| w.contains("`future`")),
            "{warnings:?}"
        );
        assert!(
            warnings.iter().any(|w| w.contains("`calcifer.idle`")),
            "{warnings:?}"
        );
        assert!(
            warnings.iter().any(|w| w.contains("Bad Name")),
            "{warnings:?}"
        );
        let local = config.current().unwrap();
        assert_eq!(local.name.as_deref(), Some("local"));
        assert_eq!(local.origin, "http://127.0.0.1:8819");
        assert_eq!(config.resolve("HTTP://127.0.0.1:8819").unwrap(), local);
        assert_eq!(config.resolve("tensorhub").unwrap().origin, DEFAULT_HUB_URL);
        assert_eq!(config.resolve("https://other.test").unwrap().name, None);
        assert_eq!(config.resolve("nosuch").unwrap_err().code, "hub_not_found");
        config.tensorhub = Some("tensorhub".into());
        assert_eq!(config.selected().unwrap().origin, DEFAULT_HUB_URL);
        assert_ne!(
            local.device_key(&Home::at("/h")),
            config.selected().unwrap().device_key(&Home::at("/h"))
        );
    }

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn the_machine_reads_its_table_and_a_pods_environment_as_spelled() {
        let user = tempfile::tempdir().unwrap();
        let home = Home::under(user.path());
        std::fs::create_dir_all(home.root()).unwrap();
        std::fs::write(home.config(), "[machine]\nport = 7443\nstore = \"/s\"\n").unwrap();
        let environment = env(&[
            ("COZY_WEBRTC_INTERNAL_PORT", "8445"),
            ("CUDA_VISIBLE_DEVICES", "0"),
            ("COZY_WORKER_AUTH_TOKEN", "secret"),
            ("COZY_AUTHORIZED_KEYS", "a, b,"),
            ("HF_TOKEN", "12345"),
        ]);
        let loaded = load_from(Flags::default(), &environment, user.path().into()).unwrap();
        let machine = &loaded.config.machine;
        assert_eq!((machine.port, machine.webrtc_port), (Some(7443), Some(8445)));
        assert_eq!(machine.store.as_deref(), Some(std::path::Path::new("/s")));
        assert_eq!(machine.visible_devices.as_deref(), Some("0"));
        assert_eq!(machine.worker_token.as_ref().map(Secret::expose), Some("secret"));
        assert_eq!(machine.authorized_keys, ["a", "b"]);
        assert_eq!(loaded.config.huggingface_token.as_ref().map(Secret::expose), Some("12345"));
        assert!(loaded.warnings.is_empty(), "{:?}", loaded.warnings);

        // A pod's variable left in a shell refuses every command, naming its key.
        let stray = env(&[("COZY_WORKER_INTERNAL_PORT", "abc")]);
        let error = load_from(Flags::default(), &stray, user.path().into()).unwrap_err();
        assert_eq!(error.code, "config_invalid");
        assert!(error.message.contains("`machine.port`"), "{}", error.message);
    }

    #[test]
    fn a_wrong_type_names_its_key() {
        let error = Config::parse("[calcifer]\nport = \"soon\"\n", "config.toml").unwrap_err();
        assert_eq!(error.code, "config_invalid");
        assert!(
            error.message.contains("`calcifer.port`"),
            "{}",
            error.message
        );
    }
}
