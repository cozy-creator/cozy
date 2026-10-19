//! Retained TLS identity and explicit typed configuration. No executor/business state.
use super::{auth::Authority, server::MachineIdentity};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::VerifyingKey;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use cozy_platform::durable::Mode;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufReader, Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Deserialize, Serialize)]
pub struct MachineConfig {
    pub worker_id: String,
    pub identity_directory: PathBuf,
    pub authorized_keys_file: PathBuf,
    pub readiness_hmac_key_file: PathBuf,
}
#[derive(Deserialize, Serialize)]
pub struct AuthorizedKeys {
    pub keys: Vec<String>,
}
#[derive(Deserialize, Serialize)]
pub struct ReadinessSecret {
    pub key_b64url: String,
}
#[derive(Deserialize, Serialize)]
struct Retained {
    worker_id: String,
    certificate_pem: String,
    private_key_pem: String,
}

impl MachineConfig {
    pub fn load(path: &Path) -> io::Result<Self> {
        let mut config: Self = read_json(path, false)?;
        let parent = path.parent().unwrap_or(Path::new("."));
        for selected in [
            &mut config.identity_directory,
            &mut config.authorized_keys_file,
            &mut config.readiness_hmac_key_file,
        ] {
            if selected.is_relative() {
                *selected = parent.join(&*selected);
            }
        }
        config.validate()?;
        Ok(config)
    }
    fn validate(&self) -> io::Result<()> {
        if self.worker_id.is_empty()
            || self.worker_id.len() > 256
            || !self.worker_id.bytes().all(|b| (0x21..=0x7e).contains(&b))
        {
            return Err(invalid("worker_id must be one printable ASCII identifier"));
        }
        Ok(())
    }
}

impl MachineIdentity {
    /// One stable worker, leaf and machine lifetime (boot id), and current configured keys.
    /// Failure never replaces an existing identity or touches the execution journal.
    pub fn retained(config: &MachineConfig) -> io::Result<Self> {
        config.validate()?;
        config.validate()?;
        private_directory(&config.identity_directory)?;
        let lock = cozy_platform::durable::open_nofollow(&config.identity_directory.join("identity.lock"), OpenOptions::new().create(true).truncate(false).read(true).write(true).mode(0o600)).map_err(std::io::Error::from)?;
        check_file(&lock, true)?;
        lock.lock_exclusive()?;
        let path = config.identity_directory.join("identity.json");
        let pending = config.identity_directory.join("identity.pending");
        let retained = if path.exists() {
            read_json::<Retained>(&path, true)?
        } else {
            let retained = if pending.exists() {
                read_json::<Retained>(&pending, true)?
            } else {
                let key =
                    rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).map_err(other)?;
                let cert =
                    rcgen::CertificateParams::new(vec!["cozy-worker".into(), "localhost".into()])
                        .map_err(other)?
                        .self_signed(&key)
                        .map_err(other)?;
                Retained {
                    worker_id: config.worker_id.clone(),
                    certificate_pem: cert.pem(),
                    private_key_pem: key.serialize_pem(),
                }
            };
            if retained.worker_id != config.worker_id {
                return Err(invalid(
                    "retained worker identity differs from configured worker_id",
                ));
            }
            write_private(&pending, &retained)?;
            fs::rename(&pending, &path)?;
            cozy_platform::durable::sync_dir(&config.identity_directory)?;
            retained
        };
        if retained.worker_id != config.worker_id {
            return Err(invalid(
                "retained worker identity differs from configured worker_id",
            ));
        }
        let key = rcgen::KeyPair::from_pem(&retained.private_key_pem).map_err(other)?;
        if key.algorithm() != &rcgen::PKCS_ECDSA_P256_SHA256 {
            return Err(invalid("retained TLS key must be P256"));
        }
        let certificates: Vec<_> =
            rustls_pemfile::certs(&mut BufReader::new(retained.certificate_pem.as_bytes()))
                .collect::<Result<_, _>>()?;
        if certificates.len() != 1 {
            return Err(invalid("retained identity must contain one exact TLS leaf"));
        }
        let private_key =
            rustls_pemfile::private_key(&mut BufReader::new(retained.private_key_pem.as_bytes()))?
                .ok_or_else(|| invalid("retained identity has no TLS key"))?;
        rustls::sign::CertifiedKey::from_der(
            certificates.clone(),
            private_key,
            &rustls::crypto::aws_lc_rs::default_provider(),
        )
        .map_err(other)?;
        let cert_der = certificates[0].as_ref().to_vec();
        let configured: AuthorizedKeys = read_json(&config.authorized_keys_file, false)?;
        if configured.keys.is_empty() || configured.keys.len() > 256 {
            return Err(invalid("at least one bounded authorized key is required"));
        }
        let mut keys = Vec::new();
        for spelling in configured.keys {
            let raw: [u8; 32] = URL_SAFE_NO_PAD
                .decode(spelling)
                .map_err(other)?
                .try_into()
                .map_err(|_| invalid("authorized key must encode 32 Ed25519 bytes"))?;
            keys.push(VerifyingKey::from_bytes(&raw).map_err(other)?);
        }
        let secret: ReadinessSecret = read_json(&config.readiness_hmac_key_file, true)?;
        let receipt_key = URL_SAFE_NO_PAD.decode(secret.key_b64url).map_err(other)?;
        if receipt_key.len() < 32 || receipt_key.len() > 4096 {
            return Err(invalid("readiness HMAC key must contain 32..4096 bytes"));
        }
        let boot_id = retained_boot_id(&config.identity_directory)?;
        let authority = Authority {
            worker_id: retained.worker_id,
            boot_id,
            keys: keys.into(),
        };
        Ok(Self {
            authority,
            cert_pem: retained.certificate_pem,
            key_pem: retained.private_key_pem,
            cert_der,
            readiness: crate::machine::receipt::Readiness::open(None, Some(receipt_key), false)?,
            started_at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(other)?
                .as_millis() as u64,
            lifecycle: None,
            hubs: vec![],
            updates: None,
            store: None,
            gpus: vec![],
            gpu_memory_limit: None,
            webrtc: None,
            player: Default::default(),
            webrtc_port: None,
        })
    }
}

impl MachineIdentity {
    /// The pod or computer machine: its lifetime files under the machine root, keys from its
    /// grant (a rental's Hub answer replaces them) and this boot's readiness receipt.
    pub fn machine(
        worker_id: String,
        keys: super::auth::Keys,
        lifetime: crate::machine::identity::Lifetime,
        readiness: std::sync::Arc<crate::machine::receipt::Readiness>,
    ) -> io::Result<Self> {
        let authority = Authority {
            worker_id,
            boot_id: lifetime.boot_id,
            keys,
        };
        Ok(Self {
            authority,
            cert_pem: lifetime.cert_pem,
            key_pem: lifetime.key_pem,
            cert_der: lifetime.cert_der,
            readiness,
            started_at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(other)?
                .as_millis() as u64,
            lifecycle: None,
            hubs: vec![],
            updates: None,
            store: None,
            gpus: vec![],
            gpu_memory_limit: None,
            webrtc: None,
            player: Default::default(),
            webrtc_port: None,
        })
    }
}

/// The boot id names this machine lifetime: Claims signed for it stay valid across restarts,
/// and records accepted before a restart remain addressable. Format as the Hub requires.
fn retained_boot_id(directory: &Path) -> io::Result<String> {
    let path = directory.join("boot-id");
    match cozy_platform::durable::open_nofollow(&path, OpenOptions::new().read(true)).map_err(std::io::Error::from)
    {
        Ok(file) => {
            check_file(&file, true)?;
            let mut text = String::new();
            file.take(64).read_to_string(&mut text)?;
            if URL_SAFE_NO_PAD.decode(text.trim()).map(|raw| raw.len()) != Ok(32) {
                return Err(invalid(
                    "retained boot id must be unpadded base64url for 32 bytes",
                ));
            }
            Ok(text.trim().to_owned())
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let boot_id = URL_SAFE_NO_PAD.encode(crate::machine::identity::random::<32>()?);
            crate::machine::identity::write_atomic(&path, boot_id.as_bytes(), 0o600)?;
            Ok(boot_id)
        }
        Err(error) => Err(error),
    }
}

fn private_directory(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            fs::create_dir_all(path)?;
            cozy_platform::durable::set_mode(path, 0o700).map_err(std::io::Error::from)?;
            if let Some(parent) = path.parent() {
                cozy_platform::durable::sync_dir(parent)?;
            }
        }
        Err(error) => return Err(error),
        Ok(metadata) if !metadata.is_dir() || metadata.file_type().is_symlink() || !owned(&metadata, Some(0o700)) => {
            return Err(invalid(
                "identity directory must be owned, real and mode0700",
            ))
        }
        Ok(_) => (),
    }
    Ok(())
}
fn check_file(file: &File, private: bool) -> io::Result<()> {
    let metadata = file.metadata()?;
    if !metadata.is_file() || !owned(&metadata, private.then_some(0o600)) {
        return Err(invalid(
            "configuration/identity file ownership or modes are unsafe",
        ));
    }
    Ok(())
}
fn read_json<T: serde::de::DeserializeOwned>(path: &Path, private: bool) -> io::Result<T> {
    let file = cozy_platform::durable::open_nofollow(path, OpenOptions::new().read(true)).map_err(std::io::Error::from)?;
    check_file(&file, private)?;
    serde_json::from_reader(file.take(64 << 10)).map_err(other)
}
fn write_private<T: Serialize>(path: &Path, record: &T) -> io::Result<()> {
    let mut file = cozy_platform::durable::open_nofollow(path, OpenOptions::new().create(true).truncate(true).write(true).mode(0o600)).map_err(std::io::Error::from)?;
    check_file(&file, true)?;
    file.write_all(&serde_json::to_vec(record).map_err(other)?)?;
    file.sync_all()
}
/// This user's, on one link, writable by no one else, and exactly `mode` when given. Windows
/// keeps an ACL instead, which the profile owns.
fn owned(metadata: &fs::Metadata, mode: Option<u32>) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let bits = metadata.mode() & 0o777;
        metadata.uid() == nix::unistd::geteuid().as_raw()
            && (metadata.is_dir() || metadata.nlink() == 1)
            && bits & 0o022 == 0
            && mode.is_none_or(|mode| bits == mode)
    }
    #[cfg(not(unix))]
    {
        let _ = (metadata, mode);
        true
    }
}
fn invalid(detail: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, detail)
}
fn other(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}
