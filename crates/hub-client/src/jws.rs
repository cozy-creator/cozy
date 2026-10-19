//! Compact JWS: the P-256 keys a machine proves with (DPoP, RFC 9449; JWT-bearer assertions,
//! RFC 7523) and the base64url and random helpers the Ed25519 side shares.
use aws_lc_rs::encoding::{AsBigEndian, EcPrivateKeyBin};
use aws_lc_rs::rand::{SecureRandom, SystemRandom};
use aws_lc_rs::signature::{ECDSA_P256_SHA256_FIXED_SIGNING, EcdsaKeyPair, KeyPair};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};
use x509_cert::Certificate;
use x509_cert::der::{Decode as _, oid::ObjectIdentifier};

use crate::Error;

pub(crate) fn b64(bytes: impl AsRef<[u8]>) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

pub(crate) fn unb64(text: &str) -> Option<Vec<u8>> {
    URL_SAFE_NO_PAD.decode(text).ok()
}

pub(crate) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

pub(crate) fn random<const N: usize>() -> [u8; N] {
    let mut raw = [0u8; N];
    SystemRandom::new()
        .fill(&mut raw)
        .expect("the system random source");
    raw
}

/// A fresh unguessable id (`jti`).
pub(crate) fn jti() -> String {
    b64(random::<24>())
}

/// `base64url(header).base64url(claims)`, the bytes a JWS signs.
pub(crate) fn signing_input(header: &Value, claims: &Value) -> String {
    format!("{}.{}", b64(header.to_string()), b64(claims.to_string()))
}

/// A JWS's claims, unverified: what a holder reads of a token it was handed.
#[cfg(test)]
pub(crate) fn claims(jws: &str) -> Option<Value> {
    let mut parts = jws.split('.');
    let (_, claims, _, None) = (parts.next()?, parts.next()?, parts.next()?, parts.next()) else {
        return None;
    };
    serde_json::from_slice(&unb64(claims)?).ok()
}

/// RFC 7638 thumbprint of a P-256 public key: what `cnf.jkt` names.
pub fn thumbprint(x: &str, y: &str) -> String {
    b64(Sha256::digest(format!(
        r#"{{"crv":"P-256","kty":"EC","x":"{x}","y":"{y}"}}"#
    )))
}

/// The thumbprint of a machine's TLS leaf (`der`): the `machine` a run capability binds to, the
/// leaf's P-256 key being the one the machine proves DPoP and signs assertions with.
pub fn leaf_thumbprint(der: &[u8]) -> Result<String, Error> {
    const EC: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.10045.2.1");
    const P256: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.10045.3.1.7");
    let refused = |why: &str| Error::Key(format!("the machine's leaf {why}"));
    let leaf =
        Certificate::from_der(der).map_err(|e| refused(&format!("is not a certificate: {e}")))?;
    let key = leaf.tbs_certificate.subject_public_key_info;
    let curve = key
        .algorithm
        .parameters
        .and_then(|p| p.decode_as::<ObjectIdentifier>().ok());
    match key.subject_public_key.raw_bytes() {
        [4, point @ ..] if key.algorithm.oid == EC && curve == Some(P256) && point.len() == 64 => {
            Ok(thumbprint(&b64(&point[..32]), &b64(&point[32..])))
        }
        _ => Err(refused("key is not P-256")),
    }
}

/// A P-256 key that signs ES256: a machine's TLS leaf key, or a fresh one held in memory.
pub struct P256Key {
    pair: EcdsaKeyPair,
    x: String,
    y: String,
}

impl P256Key {
    /// A private P-256 JWK, for the request that holds this key.
    pub fn private_jwk(&self) -> String {
        let scalar: EcPrivateKeyBin<'_> =
            self.pair.private_key().as_be_bytes().expect("P-256 scalar");
        let mut jwk = self.jwk();
        jwk["d"] = b64(scalar.as_ref()).into();
        jwk.to_string()
    }

    pub fn from_jwk(key: &str) -> Result<Self, Error> {
        let bad = || Error::Key("not a private P-256 JWK".into());
        let jwk: Value = serde_json::from_str(key).map_err(|_| bad())?;
        if jwk["kty"] != "EC" || jwk["crv"] != "P-256" {
            return Err(bad());
        }
        let field = |name: &str| {
            jwk[name]
                .as_str()
                .and_then(unb64)
                .filter(|v| v.len() == 32)
                .ok_or_else(bad)
        };
        let point = [&[4u8][..], &field("x")?, &field("y")?].concat();
        let pair = EcdsaKeyPair::from_private_key_and_public_key(
            &ECDSA_P256_SHA256_FIXED_SIGNING,
            &field("d")?,
            &point,
        )
        .map_err(|_| bad())?;
        Ok(Self::from_pair(pair))
    }

    /// An unencrypted PKCS#8 key, DER.
    pub fn from_pkcs8(der: &[u8]) -> Result<Self, Error> {
        let pair = EcdsaKeyPair::from_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, der)
            .map_err(|_| Error::Key("not an unencrypted PKCS#8 P-256 key".into()))?;
        Ok(Self::from_pair(pair))
    }

    /// The same key in PEM (`-----BEGIN PRIVATE KEY-----`).
    pub fn from_pem(pem: &str) -> Result<Self, Error> {
        let body = pem
            .split("-----BEGIN PRIVATE KEY-----")
            .nth(1)
            .and_then(|rest| rest.split("-----END PRIVATE KEY-----").next())
            .ok_or_else(|| Error::Key("no PKCS#8 PRIVATE KEY block".into()))?;
        let der = base64::engine::general_purpose::STANDARD
            .decode(body.split_whitespace().collect::<String>())
            .map_err(|_| Error::Key("the PRIVATE KEY block is not base64".into()))?;
        Self::from_pkcs8(&der)
    }

    /// A fresh key, held only in memory.
    pub fn generate() -> Self {
        let pair =
            EcdsaKeyPair::generate(&ECDSA_P256_SHA256_FIXED_SIGNING).expect("P-256 key generation");
        Self::from_pair(pair)
    }

    fn from_pair(pair: EcdsaKeyPair) -> Self {
        let point = pair.public_key().as_ref();
        let (x, y) = (b64(&point[1..33]), b64(&point[33..65]));
        Self { pair, x, y }
    }

    pub fn thumbprint(&self) -> String {
        thumbprint(&self.x, &self.y)
    }

    pub(crate) fn jwk(&self) -> Value {
        json!({"kty": "EC", "crv": "P-256", "x": self.x, "y": self.y})
    }

    /// A compact ES256 JWS of `claims` under `header` (its `alg` set here).
    pub(crate) fn sign(&self, mut header: Value, claims: &Value) -> String {
        header["alg"] = "ES256".into();
        let input = signing_input(&header, claims);
        let signature = self
            .pair
            .sign(&SystemRandom::new(), input.as_bytes())
            .expect("ES256 signing");
        format!("{input}.{}", b64(signature.as_ref()))
    }

    /// A DPoP proof for one request: `htu` is its URL without query or fragment; `token` the
    /// access token it presents, if any; `nonce` the server's last one.
    pub fn proof(
        &self,
        method: &str,
        htu: &str,
        token: Option<&str>,
        nonce: Option<&str>,
    ) -> String {
        let mut claims = json!({"jti": jti(), "htm": method, "htu": htu, "iat": now()});
        if let Some(token) = token {
            claims["ath"] = b64(Sha256::digest(token.as_bytes())).into();
        }
        if let Some(nonce) = nonce {
            claims["nonce"] = nonce.into();
        }
        self.sign(json!({"typ": "dpop+jwt", "jwk": self.jwk()}), &claims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_lc_rs::signature::{ECDSA_P256_SHA256_FIXED, UnparsedPublicKey};

    /// A proof verifies under the key its header carries and binds the request and token.
    #[test]
    fn a_proof_binds_its_request_and_token() {
        let generated = P256Key::generate();
        let key = P256Key::from_jwk(&generated.private_jwk()).unwrap();
        assert_eq!(key.thumbprint(), generated.thumbprint());
        assert_ne!(key.thumbprint(), P256Key::generate().thumbprint());
        let proof = key.proof(
            "POST",
            "https://hub.example/v1/tensorfs/closure",
            Some("tok"),
            Some("n1"),
        );
        let (input, signature) = proof.rsplit_once('.').unwrap();
        let (header, _) = input.split_once('.').unwrap();
        let header: Value = serde_json::from_slice(&unb64(header).unwrap()).unwrap();
        let (x, y) = (
            unb64(header["jwk"]["x"].as_str().unwrap()).unwrap(),
            unb64(header["jwk"]["y"].as_str().unwrap()).unwrap(),
        );
        let point = [&[4u8][..], &x, &y].concat();
        UnparsedPublicKey::new(&ECDSA_P256_SHA256_FIXED, point)
            .verify(input.as_bytes(), &unb64(signature).unwrap())
            .unwrap();
        let claims = claims(&proof).unwrap();
        assert_eq!(
            (header["typ"].as_str(), header["alg"].as_str()),
            (Some("dpop+jwt"), Some("ES256"))
        );
        assert_eq!(claims["htm"], "POST");
        assert_eq!(claims["htu"], "https://hub.example/v1/tensorfs/closure");
        assert_eq!(claims["ath"], b64(Sha256::digest(b"tok")));
        assert_eq!(claims["nonce"], "n1");
        assert_eq!(key.thumbprint(), thumbprint(&b64(&x), &b64(&y)));
    }
}
