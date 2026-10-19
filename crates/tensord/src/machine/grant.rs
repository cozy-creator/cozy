//! The machine's launch grant: what a Hub (rental) or this computer's home (persistent) gives
//! it, judged once from [`crate::config`]'s load.
use protocol::config::MachineConfig;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use platform::paths::Home;
use ed25519_dalek::VerifyingKey;
use std::{
    fs::{self, File},
    io::{self, Read},
    net::IpAddr,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifetime {
    /// A provider allocation: Hub lease of keys, readiness receipt, idle release.
    Rental,
    /// This computer's machine: no Hub lifecycle.
    Persistent,
}

/// One Hub's worker registration of this machine.
#[derive(Clone, Debug)]
pub struct HubGrant {
    pub origin: String,
    pub worker_id: String,
    pub worker_token: String,
    pub ca_der: Option<Vec<u8>>,
    pub object_hosts: Vec<String>,
}

pub struct Grant {
    pub lifetime: Lifetime,
    pub layout: Layout,
    pub listen_host: IpAddr,
    pub worker_id: String,
    /// The API's port; None on this computer: any free port, kept across restarts.
    pub worker_port: Option<u16>,
    pub webrtc_port: Option<u16>,
    /// Where browsers reach the WebRTC listener ([`super::player::reach`]).
    pub webrtc_reach: super::player::Reach,
    pub hub: Option<HubGrant>,
    pub authorized: Vec<VerifyingKey>,
    /// The one-shot readiness key; taken by the receipt and never kept elsewhere.
    pub receipt_key: Option<Vec<u8>>,
    /// A development rental's SSH key (`COZY_SSH_PUBLIC_KEY`, which the Hub sends only then):
    /// serve SSH maintenance.
    pub developer_key: Option<String>,
    /// The pod's own provider credential, which ends it when the Hub cannot. Never read from the
    /// environment here: [`super::provider::ProviderSelf::take`] keeps it from everything else.
    pub provider: Option<super::provider::ProviderSelf>,
    /// How long a rental stays with no job before it releases itself (15 minutes).
    pub idle_ms: i64,
    /// The user's cozy home, whose `keys/` this machine admits too.
    pub home: Home,
}

/// A machine root's own TensorFS store.
const STORE: &str = "var/lib/tensorfs";

/// The image's filesystem, rooted: "/" on a pod, a directory on an owned machine. The paths
/// preserve the machine's identity across software updates.
#[derive(Clone, Debug)]
pub struct Layout {
    pub root: PathBuf,
    pub bootstrap: PathBuf,
    pub state: PathBuf,
    /// The TensorFS store this machine reads and fills: the one `COZY_TENSORFS_ROOT` names (an
    /// owned computer's box store; a worker image sets it), else the root's standard
    /// `var/lib/tensorfs`.
    pub store: PathBuf,
    /// The datacenter volume the Hub mounted (`COZY_TENSORFS_CACHE`): the machine's one Fetcher
    /// reads it before the Hub when it is a mount point.
    pub cache: Option<PathBuf>,
}
impl Layout {
    pub fn new(root: &Path, store: Option<PathBuf>) -> Self {
        Self {
            root: root.into(),
            bootstrap: root.join("run/cozy/bootstrap"),
            state: root.join("var/lib/cozy/machine"),
            store: store.unwrap_or_else(|| root.join(STORE)),
            cache: None,
        }
    }
    /// Whether the store is this root's own, not one an operator named elsewhere (and may share).
    pub fn own_store(&self) -> bool {
        self.store == self.root.join(STORE)
    }
    /// This service's execution journal and package generations.
    pub fn engine(&self) -> PathBuf {
        self.root.join("var/lib/cozy/tensord")
    }
}

impl Grant {
    pub fn new(lifetime: Lifetime, raw: &MachineConfig, webrtc_reach: super::player::Reach, home: &Home) -> io::Result<Self> {
        fn get(text: &Option<String>) -> Option<&str> {
            text.as_deref().filter(|v| !v.is_empty())
        }
        let rental = lifetime == Lifetime::Rental;
        let root = match (&raw.root, rental) {
            (None, true) => PathBuf::from("/"),
            _ => raw.root(home),
        };
        clean_absolute(&root, "COZY_MACHINE_ROOT")?;
        for (path, name) in [(&raw.store, "COZY_TENSORFS_ROOT"), (&raw.cache, "COZY_TENSORFS_CACHE")] {
            if let Some(path) = path {
                clean_absolute(path, name)?;
            }
        }
        let listen_host: IpAddr = match get(&raw.listen_host).unwrap_or(if rental { "0.0.0.0" } else { "127.0.0.1" }) {
            host @ ("0.0.0.0" | "127.0.0.1") => host.parse().expect("literal address"),
            _ => return Err(invalid("COZY_LISTEN_HOST must be 0.0.0.0 or 127.0.0.1")),
        };
        let worker_id = get(&raw.worker_id).unwrap_or(if rental { "" } else { "local" }).to_owned();
        if worker_id.is_empty() || worker_id.len() > 256 || worker_id.trim() != worker_id {
            return Err(invalid("COZY_WORKER_ID must be one identifier of at most 256 bytes"));
        }
        let worker_port = raw.port.filter(|p| *p > 0);
        if rental && worker_port.is_none() {
            return Err(invalid("COZY_WORKER_INTERNAL_PORT is required"));
        }
        let webrtc_port = raw.webrtc_port.filter(|p| *p > 0);
        let hub = match lifetime {
            Lifetime::Persistent => None,
            Lifetime::Rental => {
                let token = raw.worker_token.as_ref().map(|t| t.expose()).unwrap_or_default();
                if URL_SAFE_NO_PAD.decode(token).map(|t| t.len()) != Ok(32) {
                    return Err(invalid("COZY_WORKER_AUTH_TOKEN must be 32 bytes of unpadded base64url"));
                }
                let ca_der = match get(&raw.hub_ca_der) {
                    None => None,
                    Some(text) => Some(
                        URL_SAFE_NO_PAD
                            .decode(text)
                            .map_err(|_| invalid("TENSORHUB_CA_DER_B64URL must be unpadded base64url DER"))?,
                    ),
                };
                Some(HubGrant {
                    origin: origin(get(&raw.hub_origin), "TENSORHUB_ORIGIN")?
                        .ok_or_else(|| invalid("TENSORHUB_ORIGIN is required for a rental"))?,
                    worker_id: worker_id.clone(),
                    worker_token: token.into(),
                    ca_der,
                    object_hosts: raw.object_storage_hosts.clone(),
                })
            }
        };
        let authorized = raw.authorized_keys.iter().map(|k| public_key(k)).collect::<io::Result<Vec<_>>>()?;
        if rental && authorized.is_empty() && home_keys(home)?.is_empty() {
            return Err(invalid("the grant authorizes no key: COZY_AUTHORIZED_KEYS is required"));
        }
        let receipt_key = match (raw.receipt_key.as_ref().map(|k| k.expose()), &raw.receipt_key_file) {
            (Some(_), Some(_)) => return Err(invalid("specify only one bootstrap receipt key source")),
            (Some(text), None) => Some(text.trim().to_owned()),
            (None, Some(path)) => Some(private_file(path)?),
            (None, None) => None,
        };
        let receipt_key = match receipt_key {
            None => None,
            Some(text) => match URL_SAFE_NO_PAD.decode(&text) {
                Ok(key) if key.len() == 32 => Some(key),
                _ => return Err(invalid("bootstrap receipt key must be 32 bytes of unpadded base64url")),
            },
        };
        let idle_ms = match raw.rental_idle_seconds {
            None => super::lifecycle::IDLE_GRACE_MS,
            Some(seconds) if seconds > 0 => i64::from(seconds) * 1000,
            Some(_) => return Err(invalid("rental_idle_seconds must be a positive number of seconds")),
        };
        Ok(Self {
            lifetime,
            layout: Layout { cache: raw.cache.clone(), ..Layout::new(&root, raw.store.clone()) },
            listen_host,
            worker_id,
            worker_port,
            webrtc_port,
            webrtc_reach,
            hub,
            authorized,
            receipt_key,
            developer_key: rental.then(|| get(&raw.ssh_public_key).map(str::to_owned)).flatten(),
            provider: None,
            idle_ms,
            home: home.clone(),
        })
    }

    /// The keys this machine admits: the grant's and its home's. On a rental the Hub lease
    /// replaces them once it is read; this computer's are read again when a signer is unknown.
    pub fn keys(&self) -> io::Result<crate::api::auth::Keys> {
        let (named, home) = (self.authorized.clone(), self.home.clone());
        let admitted = move || Ok([named.clone(), home_keys(&home)?].concat());
        match self.lifetime {
            Lifetime::Rental => admitted().map(crate::api::auth::Keys::fixed),
            Lifetime::Persistent => crate::api::auth::Keys::read(admitted),
        }
    }
}

/// The controllers the home admits, besides those the grant names: the public half of its owner
/// key (`keys/owner.pem`, this computer's Calcifer's signer) and one unpadded base64url Ed25519
/// key per line of `keys/authorized` (`cozy machine authorize`; `#` starts a comment). Either
/// may be absent.
pub fn home_keys(home: &Home) -> io::Result<Vec<VerifyingKey>> {
    use ed25519_dalek::{pkcs8::DecodePrivateKey, SigningKey};
    let owner = home.keys().join("owner.pem");
    let mut keys = match fs::read_to_string(&owner) {
        Ok(pem) => vec![SigningKey::from_pkcs8_pem(&pem)
            .map_err(|e| invalid(&format!("{}: {e}", owner.display())))?
            .verifying_key()],
        Err(e) if e.kind() == io::ErrorKind::NotFound => vec![],
        Err(e) => return Err(e),
    };
    match fs::read_to_string(home.keys().join("authorized")) {
        Ok(text) => {
            for line in text.lines().map(|l| l.split('#').next().unwrap_or_default().trim()) {
                if !line.is_empty() {
                    keys.push(public_key(line)?);
                }
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => (),
        Err(e) => return Err(e),
    }
    Ok(keys)
}

fn public_key(spelled: &str) -> io::Result<VerifyingKey> {
    let raw: [u8; 32] = URL_SAFE_NO_PAD
        .decode(spelled)
        .ok()
        .and_then(|raw| raw.try_into().ok())
        .ok_or_else(|| {
            invalid(&format!(
                "{spelled:?} is not a 32-byte unpadded base64url Ed25519 key"
            ))
        })?;
    VerifyingKey::from_bytes(&raw).map_err(|e| invalid(&e.to_string()))
}

fn origin(text: Option<&str>, name: &str) -> io::Result<Option<String>> {
    let Some(text) = text else { return Ok(None) };
    match text.strip_prefix("https://") {
        Some(rest) if !rest.is_empty() && !rest.contains(['/', '?', '#', '@']) => {
            Ok(Some(text.to_owned()))
        }
        _ => Err(invalid(&format!("{name} must be https://host[:port]"))),
    }
}

fn clean_absolute(path: &Path, name: &str) -> io::Result<()> {
    let clean: PathBuf = path.components().collect();
    if !path.is_absolute()
        || clean != path
        || path
            .to_str()
            .is_some_and(|p| p.len() > 1 && p.ends_with('/'))
    {
        return Err(invalid(&format!(
            "{name} must be one clean absolute directory"
        )));
    }
    Ok(())
}

fn private_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let meta = file.metadata()?;
    if !path.is_absolute()
        || !meta.is_file()
        || platform::durable::mode(&meta).is_some_and(|mode| mode & 0o077 != 0)
        || fs::symlink_metadata(path)?.file_type().is_symlink()
    {
        return Err(invalid(
            "the receipt key file must be one private regular file",
        ));
    }
    let mut text = String::new();
    file.by_ref().take(256).read_to_string(&mut text)?;
    Ok(text.trim().to_owned())
}

fn invalid(detail: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, detail.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{load_from, Config};
    use std::collections::BTreeMap;

    #[cfg(unix)]
    const TOKEN: &str = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    const KEY: &str = "O2onvM62pC1io6jQKm8Nc2UyFXcd4kOmOsBIoYtZ2ik";

    /// A user home of its own, and the load over `pairs` as this process's environment.
    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!("tensord-grant-{}", uuid::Uuid::new_v4())))
        }
        fn home(&self) -> Home {
            Home::under(&self.0)
        }
        fn load(&self, lifetime: Lifetime, pairs: &[(&str, &str)]) -> io::Result<Config> {
            let env: BTreeMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            load_from(lifetime, &env, self.0.clone())
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// A rental is a pod: its root and cache are Unix paths.
    #[cfg(unix)]
    #[test]
    fn rental_grant_matches_the_hub_environment() {
        let config = Scratch::new()
            .load(Lifetime::Rental, &[
                ("COZY_WORKER_ID", "ra-1"),
                ("COZY_WORKER_AUTH_TOKEN", TOKEN),
                ("COZY_WORKER_INTERNAL_PORT", "8443"),
                ("COZY_WEBRTC_INTERNAL_PORT", "8445"),
                ("COZY_AUTHORIZED_KEYS", KEY),
                ("COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL", TOKEN),
                ("TENSORHUB_ORIGIN", "https://hub.example"),
                ("COZY_FUTURE_SETTING", "x"),
                ("COZY_MACHINE_LIFETIME", "persistent"),
                ("WORKER_MODE", "development"),
                ("COZY_SSH_PUBLIC_KEY", "ssh-ed25519 AAAA"),
                ("COZY_TENSORFS_CACHE", "/cache"),
                ("CUDA_VISIBLE_DEVICES", "0"),
                ("RUNPOD_PUBLIC_IP", "198.51.100.2"),
                ("RUNPOD_TCP_PORT_8443", "40001"),
            ])
            .unwrap();
        let grant = &config.grant;
        assert_eq!(grant.lifetime, Lifetime::Rental);
        assert_eq!(grant.layout.cache.as_deref(), Some(Path::new("/cache")));
        assert_eq!(grant.layout.state, Path::new("/var/lib/cozy/machine"));
        assert_eq!((grant.worker_port, grant.webrtc_port), (Some(8443), Some(8445)));
        assert_eq!(grant.receipt_key.as_deref(), Some(&[0u8; 32][..]));
        assert_eq!(grant.hub.as_ref().unwrap().origin, "https://hub.example");
        assert_eq!(grant.developer_key.as_deref(), Some("ssh-ed25519 AAAA"));
        assert_eq!(config.visible_devices.as_deref(), Some("0"));
        assert_eq!(config.endpoint.unwrap().ports["8443"], 40001);
        assert_eq!(config.logs, Path::new("/var/lib/cozy/machine/logs"));
        // The lifetime is the command line's: the old variable is named and ignored.
        assert_eq!(
            config.warnings,
            [
                "ignoring COZY_FUTURE_SETTING, which this machine does not read",
                "ignoring COZY_MACHINE_LIFETIME, which this machine does not read"
            ]
        );
    }

    #[test]
    fn this_computers_machine_lives_in_the_home_and_admits_its_keys() {
        use ed25519_dalek::{pkcs8::EncodePrivateKey, SigningKey};
        let scratch = Scratch::new();
        let home = scratch.home();
        fs::create_dir_all(home.keys()).unwrap();
        let owner = SigningKey::from_bytes(&[7; 32]);
        fs::write(home.keys().join("owner.pem"), owner.to_pkcs8_pem(Default::default()).unwrap().as_bytes()).unwrap();
        fs::write(home.keys().join("authorized"), format!("# another controller\n{KEY} # laptop\n")).unwrap();
        fs::write(home.config(), "hub = \"tensorhub\"\n[machine]\nport = 7443\n").unwrap();
        let config = scratch.load(Lifetime::Persistent, &[]).unwrap();
        let grant = &config.grant;
        assert_eq!(grant.layout.root, home.machine());
        assert_eq!((grant.worker_id.as_str(), grant.worker_port), ("local", Some(7443)));
        assert_eq!(grant.listen_host.to_string(), "127.0.0.1");
        assert!(grant.authorized.is_empty());
        let keys = grant.keys().unwrap();
        assert_eq!(keys.admitted(), [owner.verifying_key(), public_key(KEY).unwrap()]);
        // `cozy machine authorize` appends a line: the next unknown signer reads it.
        let added = SigningKey::from_bytes(&[9; 32]).verifying_key();
        let line = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(added.as_bytes());
        fs::write(home.keys().join("authorized"), format!("{KEY}\n{line}\n")).unwrap();
        let known = |k: &[VerifyingKey]| if k.contains(&added) { Ok(()) } else { Err(()) };
        assert_eq!(keys.check(known), Ok(()));
        assert!(keys.admitted().contains(&added));
        assert!(grant.hub.is_none() && grant.developer_key.is_none() && grant.receipt_key.is_none());
        assert!(config.warnings.is_empty(), "{:?}", config.warnings);
    }

    #[cfg(unix)]
    #[test]
    fn malformed_grants_refuse_by_name() {
        let base = [
            ("COZY_WORKER_ID", "ra-1"),
            ("COZY_WORKER_AUTH_TOKEN", TOKEN),
            ("COZY_WORKER_INTERNAL_PORT", "8443"),
            ("COZY_AUTHORIZED_KEYS", KEY),
            ("TENSORHUB_ORIGIN", "https://hub.example"),
        ];
        let scratch = Scratch::new();
        assert!(scratch.load(Lifetime::Rental, &base).is_ok());
        for (name, value) in [
            ("COZY_WORKER_INTERNAL_PORT", "8443x"),
            ("COZY_MACHINE_ROOT", "relative"),
            ("COZY_TENSORFS_CACHE", "cache"),
            ("COZY_LISTEN_HOST", "::"),
            ("COZY_AUTHORIZED_KEYS", "short"),
            ("TENSORHUB_ORIGIN", "http://hub.example"),
        ] {
            let mut pairs = base.to_vec();
            pairs.retain(|(n, _)| *n != name);
            pairs.push((name, value));
            let error = scratch.load(Lifetime::Rental, &pairs).err().unwrap_or_else(|| panic!("{name}={value}"));
            assert!(error.to_string().contains(name) || name == "COZY_AUTHORIZED_KEYS", "{name}: {error}");
        }
        let without_token: Vec<_> = base.iter().filter(|(n, _)| *n != "COZY_WORKER_AUTH_TOKEN").copied().collect();
        assert!(scratch.load(Lifetime::Rental, &without_token).is_err(), "a rental needs its worker token");
    }
}
