//! Identifiers that cross process boundaries.
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

/// A run's id: the caller's id on the machine (`RunRequest.id`), the same in Calcifer, the CLI
/// and TensorD. Minted as `run-` and 24 hex characters; any 1–256 bytes are accepted, so runs
/// another client sent keep their ids.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct RunId(String);

impl RunId {
    pub fn mint(random: [u8; 12]) -> Self {
        Self(format!("run-{}", hex::encode(random)))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for RunId {
    type Error = &'static str;
    fn try_from(id: String) -> Result<Self, Self::Error> {
        if id.is_empty() || id.len() > 256 {
            return Err("a run id is 1-256 bytes");
        }
        Ok(Self(id))
    }
}

impl From<RunId> for String {
    fn from(id: RunId) -> Self {
        id.0
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Content's name on the wire: `sha256:` and 64 lowercase hex characters.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Digest([u8; 32]);

impl Digest {
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::str::FromStr for Digest {
    type Err = &'static str;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let hex = text
            .strip_prefix("sha256:")
            .ok_or("a digest is sha256:<64 hex>")?;
        let mut bytes = [0; 32];
        if hex.len() != 64 || hex.bytes().any(|b| b.is_ascii_uppercase()) {
            return Err("a digest is sha256:<64 lowercase hex>");
        }
        hex::decode_to_slice(hex, &mut bytes)
            .map_err(|_| "a digest is sha256:<64 lowercase hex>")?;
        Ok(Self(bytes))
    }
}

impl TryFrom<String> for Digest {
    type Error = &'static str;
    fn try_from(text: String) -> Result<Self, Self::Error> {
        text.parse()
    }
}

impl From<Digest> for String {
    fn from(digest: Digest) -> Self {
        digest.to_string()
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", hex::encode(self.0))
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
