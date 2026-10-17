//! A user's device key (Ed25519, AuthKit device keys): enrollment by emailed code, silent login
//! to a short Bearer, and the run capabilities it signs offline for a machine to trade.
use ed25519_dalek::{Signer as _, SigningKey};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::Mutex;

use crate::jws::{b64, jti, now, signing_input, unb64};
use crate::{Client, Error, types};
use progenitor_client::ClientInfo;

const ENROLL_DOMAIN: &str = "authkit.device-key-enrollment/1";
const LOGIN_DOMAIN: &str = "authkit.device-key-login/1";
/// A Bearer this close to its expiry is minted again rather than sent.
const MARGIN: u64 = 30;

/// A device key and the account it signs in, as kept in its file (mode 0600).
pub struct DeviceKey {
    pub hub: String,
    pub email: String,
    pub id: String,
    key: SigningKey,
}

#[derive(Deserialize, Serialize)]
struct Kept {
    version: u32,
    hub: String,
    email: String,
    device_key_id: String,
    /// The Ed25519 seed, base64url.
    private_key: String,
}

impl DeviceKey {
    pub fn load(path: &Path) -> Result<Self, Error> {
        let raw =
            std::fs::read(path).map_err(|e| Error::Key(format!("{}: {e}", path.display())))?;
        let kept: Kept = serde_json::from_slice(&raw)
            .map_err(|e| Error::Key(format!("{}: {e}", path.display())))?;
        let seed: [u8; 32] = unb64(&kept.private_key)
            .and_then(|seed| seed.try_into().ok())
            .filter(|_| kept.version == 1 && !kept.device_key_id.is_empty())
            .ok_or_else(|| Error::Key(format!("{}: not a device key", path.display())))?;
        Ok(Self {
            hub: kept.hub,
            email: kept.email,
            id: kept.device_key_id,
            key: SigningKey::from_bytes(&seed),
        })
    }

    /// Writes the key to `path`, owner-only, replacing any key there.
    pub fn save(&self, path: &Path) -> Result<(), Error> {
        let kept = Kept {
            version: 1,
            hub: self.hub.clone(),
            email: self.email.clone(),
            device_key_id: self.id.clone(),
            private_key: b64(self.key.to_bytes()),
        };
        write_private(path, &serde_json::to_vec(&kept).expect("a key serializes"))
    }

    /// The public key, unpadded base64url: what a rental's `creator_public_key` names.
    pub fn public_key(&self) -> String {
        b64(self.key.verifying_key().as_bytes())
    }

    fn sign_challenge(key: &SigningKey, domain: &str, challenge: &str) -> Result<String, Error> {
        let challenge = unb64(challenge)
            .filter(|bytes| bytes.len() == 32)
            .ok_or_else(|| Error::Answer("a malformed device-key challenge".into()))?;
        let message = [domain.as_bytes(), &[0], &challenge].concat();
        Ok(b64(key.sign(&message).to_bytes()))
    }

    /// Signs `capability` offline: the compact JWS a machine trades for a token (th-241).
    pub fn sign_run_capability(&self, capability: &RunCapability) -> String {
        let header = json!({"alg": "EdDSA", "typ": "authkit-capability+jwt", "kid": self.id});
        let issued = now();
        let claims = json!({
            "sub": capability.user_id, "aud": capability.hub, "cnf": {"jkt": capability.machine},
            "run": capability.run, "jti": jti(), "iat": issued,
            "exp": issued + capability.lifetime.as_secs(),
            "authorization_details": capability.operations,
        });
        let input = signing_input(&header, &claims);
        format!(
            "{input}.{}",
            b64(self.key.sign(input.as_bytes()).to_bytes())
        )
    }
}

/// What one run may do at a Hub for its owner: the operations (RFC 9396
/// `authorization_details`) the machine whose leaf thumbprint is `machine` performs for `run`.
pub struct RunCapability {
    pub user_id: String,
    /// The Hub's public origin, the token's audience.
    pub hub: String,
    pub machine: String,
    pub run: String,
    pub operations: Vec<Value>,
    pub lifetime: Duration,
}

/// A short user Bearer.
#[derive(Clone, Deserialize, Serialize)]
pub struct Bearer {
    pub token: String,
    /// Unix seconds.
    pub expires: u64,
    pub device_key_id: String,
}

impl Bearer {
    fn from_answer(answer: &types::AuthResult, started: u64) -> Result<Self, Error> {
        let set = answer
            .token_set
            .as_ref()
            .ok_or_else(|| Error::Answer("no access token".into()))?;
        if !set.token_type.eq_ignore_ascii_case("bearer")
            || set.access_token.is_empty()
            || set.expires_in <= 0
        {
            return Err(Error::Answer("an unusable access token".into()));
        }
        let device_key_id = answer
            .device_key
            .as_ref()
            .map(|key| key.id.clone())
            .unwrap_or_default();
        Ok(Self {
            token: set.access_token.clone(),
            expires: started + set.expires_in as u64,
            device_key_id,
        })
    }

    fn live(&self) -> bool {
        self.expires > now() + MARGIN
    }
}

/// One user signed in by a device key: a Bearer kept in memory and, if given a path, beside
/// the key for the next process, minted again when stale or refused.
pub struct UserSession {
    key: DeviceKey,
    cache: Option<PathBuf>,
    bearer: Mutex<Option<Bearer>>,
}

impl UserSession {
    pub fn new(key: DeviceKey, cache: Option<PathBuf>) -> Self {
        Self {
            key,
            cache,
            bearer: Mutex::new(None),
        }
    }

    pub fn key(&self) -> &DeviceKey {
        &self.key
    }

    /// A live Bearer: held, kept, or from a fresh login at `hub`.
    pub async fn bearer(&self, hub: &Client) -> Result<String, Error> {
        let mut held = self.bearer.lock().await;
        let ours = |b: &Bearer| b.live() && b.device_key_id == self.key.id;
        if let Some(bearer) = held.as_ref().filter(|b| ours(b)) {
            return Ok(bearer.token.clone());
        }
        let kept = self
            .cache
            .as_deref()
            .and_then(|path| std::fs::read(path).ok())
            .and_then(|raw| serde_json::from_slice::<Bearer>(&raw).ok())
            .filter(ours);
        let bearer = match kept {
            Some(bearer) => bearer,
            None => {
                let bearer = self.login(hub).await?;
                if let Some(path) = &self.cache
                    && let Err(error) = write_private(
                        path,
                        &serde_json::to_vec(&bearer).expect("a bearer serializes"),
                    )
                {
                    tracing::warn!(%error, "keeping the session");
                }
                bearer
            }
        };
        let token = bearer.token.clone();
        *held = Some(bearer);
        Ok(token)
    }

    /// The Hub refused `token`: the next request logs in again.
    pub async fn refused(&self, token: &str) {
        let mut held = self.bearer.lock().await;
        if held.as_ref().is_some_and(|b| b.token == token) {
            *held = None;
            if let Some(path) = &self.cache {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    async fn login(&self, hub: &Client) -> Result<Bearer, Error> {
        let begun = hub
            .begin_device_key_login(&types::AuthDeviceKeyLoginBeginRequest {
                device_key_id: self.key.id.clone(),
            })
            .await?
            .into_inner();
        let signature = DeviceKey::sign_challenge(&self.key.key, LOGIN_DOMAIN, &begun.challenge)?;
        let started = now();
        let answer = hub
            .finish_device_key_login(&types::AuthDeviceKeyLoginFinishRequest {
                challenge_id: begun.challenge_id,
                signature,
            })
            .await?
            .into_inner();
        Bearer::from_answer(&answer, started)
    }
}

/// A device key's enrollment, waiting for the code AuthKit emailed.
pub struct Enrollment {
    id: String,
    challenge: String,
    key: SigningKey,
    email: String,
    hub: String,
}

impl Enrollment {
    /// Asks `hub` to email `email` a code that enrolls a new device key, or proves the mailbox
    /// for `existing` (to revoke the account's other keys).
    pub async fn begin(
        hub: &Client,
        email: &str,
        label: &str,
        existing: Option<&DeviceKey>,
    ) -> Result<Self, Error> {
        let key = match existing {
            Some(existing) => existing.key.clone(),
            None => SigningKey::from_bytes(&crate::jws::random()),
        };
        let begun = hub
            .begin_device_key_enrollment(&types::AuthDeviceKeyEnrollBeginRequest {
                email: email.to_string(),
                public_key: b64(key.verifying_key().as_bytes()),
                label: (!label.is_empty())
                    .then(|| label.parse())
                    .transpose()
                    .map_err(|_| Error::Key("an unusable key label".into()))?,
            })
            .await?
            .into_inner();
        Ok(Self {
            id: begun.enrollment_id,
            challenge: begun.challenge,
            key,
            email: email.to_string(),
            hub: ClientInfo::baseurl(hub).to_string(),
        })
    }

    /// Completes enrollment with the emailed `code`: the key, and its first Bearer.
    pub async fn finish(self, hub: &Client, code: &str) -> Result<(DeviceKey, Bearer), Error> {
        let signature = DeviceKey::sign_challenge(&self.key, ENROLL_DOMAIN, &self.challenge)?;
        let started = now();
        let answer = hub
            .finish_device_key_enrollment(&types::AuthDeviceKeyEnrollFinishRequest {
                enrollment_id: self.id,
                code: code.trim().to_string(),
                code_2fa: None,
                signature,
            })
            .await?
            .into_inner();
        let bearer = Bearer::from_answer(&answer, started)?;
        let key = DeviceKey {
            hub: self.hub,
            email: self.email,
            id: bearer.device_key_id.clone(),
            key: self.key,
        };
        Ok((key, bearer))
    }
}

/// Writes `bytes` to `path` owner-only, through a rename so a reader never sees half.
fn write_private(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let fail = |e: std::io::Error| Error::Key(format!("{}: {e}", path.display()));
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir).map_err(fail)?;
    let staged = dir.join(format!(".{}.tmp", jti()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let written = options.open(&staged).and_then(|mut file| {
        std::io::Write::write_all(&mut file, bytes)?;
        file.sync_all()
    });
    let renamed = written.and_then(|()| std::fs::rename(&staged, path));
    if renamed.is_err() {
        let _ = std::fs::remove_file(&staged);
    }
    renamed.map_err(fail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signature, Verifier as _};

    fn key() -> DeviceKey {
        DeviceKey {
            hub: "https://hub.example".into(),
            email: "a@example.test".into(),
            id: "dk1".into(),
            key: SigningKey::from_bytes(&[7; 32]),
        }
    }

    /// A key survives its file, and the run capability it signs verifies under it.
    #[test]
    fn a_kept_key_signs_a_verifiable_capability() {
        let dir = std::env::temp_dir().join(format!("hub-client-{}", jti()));
        let path = dir.join("key.json");
        key().save(&path).unwrap();
        let loaded = DeviceKey::load(&path).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(
            (loaded.id.as_str(), loaded.key.to_bytes()),
            ("dk1", [7; 32])
        );
        let jws = loaded.sign_run_capability(&RunCapability {
            user_id: "u1".into(),
            hub: "https://hub.example".into(),
            machine: "jkt".into(),
            run: "r1".into(),
            operations: vec![json!({"type": "tensorhub_model_publish", "model": "a/b"})],
            lifetime: Duration::from_secs(60),
        });
        let (input, signature) = jws.rsplit_once('.').unwrap();
        let signature = Signature::from_slice(&unb64(signature).unwrap()).unwrap();
        loaded
            .key
            .verifying_key()
            .verify(input.as_bytes(), &signature)
            .unwrap();
        let claims = crate::jws::claims(&jws).unwrap();
        assert_eq!(
            (claims["cnf"]["jkt"].as_str(), claims["aud"].as_str()),
            (Some("jkt"), Some("https://hub.example"))
        );
    }
}
