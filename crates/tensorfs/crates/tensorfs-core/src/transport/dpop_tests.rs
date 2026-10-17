//! DPoP over the real transport, against a server that verifies every proof the way
//! AuthKit's resource servers and token endpoint do (`internal/dpop/proof.go`).
use super::*;
use crate::transport::{api_get, form_post, AccessToken, Client, DpopCredential, DpopKey};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use aws_lc_rs::signature::{UnparsedPublicKey, ECDSA_P256_SHA256_FIXED};
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

/// What a proof must match (AuthKit's `dpop.Check`).
struct Check<'a> {
    url: String,
    method: &'a str,
    token: Option<&'a str>,
    thumbprint: &'a str,
    nonce: Option<&'a str>,
}

#[derive(Debug, PartialEq)]
enum Verdict {
    Invalid(&'static str),
    NonceRequired,
    Replay,
}

fn object(segment: &str) -> Option<Map<String, Value>> {
    let raw = URL_SAFE_NO_PAD.decode(segment).ok()?;
    match serde_json::from_slice(&raw).ok()? {
        Value::Object(map) => Some(map),
        _ => None,
    }
}

fn members(map: &Map<String, Value>, want: &[&str]) -> bool {
    map.len() == want.len() && want.iter().all(|key| map.contains_key(*key))
}

fn canonical(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    let (authority, path) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
    let (scheme, authority) = (scheme.to_ascii_lowercase(), authority.to_ascii_lowercase());
    let host = authority
        .rsplit_once(':')
        .map_or(authority.as_str(), |(h, _)| h);
    if scheme != "https" && !(scheme == "http" && matches!(host, "localhost" | "127.0.0.1")) {
        return None;
    }
    let authority = match (scheme.as_str(), authority.rsplit_once(':')) {
        ("https", Some((host, "443"))) | ("http", Some((host, "80"))) => host.to_string(),
        _ => authority,
    };
    Some(format!(
        "{scheme}://{authority}{}",
        if path.is_empty() { "/" } else { path }
    ))
}

fn verify(
    proof: &str,
    check: &Check,
    seen: &Mutex<HashSet<String>>,
) -> std::result::Result<(), Verdict> {
    use Verdict::Invalid;
    let parts: Vec<&str> = proof.split('.').collect();
    if proof.len() > 4 << 10 || parts.len() != 3 {
        return Err(Invalid("shape"));
    }
    let header = object(parts[0]).ok_or(Invalid("header"))?;
    if !members(&header, &["typ", "alg", "jwk"])
        || header["typ"] != "dpop+jwt"
        || header["alg"] != "ES256"
    {
        return Err(Invalid("header members"));
    }
    let jwk = header["jwk"].as_object().ok_or(Invalid("jwk"))?;
    if !members(jwk, &["kty", "crv", "x", "y"]) || jwk["kty"] != "EC" || jwk["crv"] != "P-256" {
        return Err(Invalid("jwk members"));
    }
    let (x, y) = (
        jwk["x"].as_str().unwrap_or(""),
        jwk["y"].as_str().unwrap_or(""),
    );
    let (xb, yb) = (
        URL_SAFE_NO_PAD.decode(x).unwrap_or_default(),
        URL_SAFE_NO_PAD.decode(y).unwrap_or_default(),
    );
    if xb.len() != 32 || yb.len() != 32 {
        return Err(Invalid("coordinates"));
    }
    let point = [&[4u8][..], &xb, &yb].concat();
    let signature = URL_SAFE_NO_PAD
        .decode(parts[2])
        .map_err(|_| Invalid("signature encoding"))?;
    UnparsedPublicKey::new(&ECDSA_P256_SHA256_FIXED, &point)
        .verify(format!("{}.{}", parts[0], parts[1]).as_bytes(), &signature)
        .map_err(|_| Invalid("signature"))?;
    let claims = object(parts[1]).ok_or(Invalid("claims"))?;
    let jti = claims.get("jti").and_then(Value::as_str).unwrap_or("");
    if !(16..=128).contains(&jti.len()) || jti.contains(char::is_whitespace) {
        return Err(Invalid("jti"));
    }
    if claims.get("htm").and_then(Value::as_str) != Some(check.method) {
        return Err(Invalid("htm"));
    }
    let htu = claims.get("htu").and_then(Value::as_str).unwrap_or("");
    if htu.contains(['?', '#'])
        || canonical(htu).is_none()
        || canonical(htu) != canonical(&check.url)
    {
        return Err(Invalid("htu"));
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let iat = claims
        .get("iat")
        .and_then(Value::as_i64)
        .ok_or(Invalid("iat"))?;
    if (iat - now).abs() > 60 {
        return Err(Invalid("iat window"));
    }
    match (check.token, claims.get("ath")) {
        (None, None) => {}
        (Some(token), Some(Value::String(ath)))
            if *ath == URL_SAFE_NO_PAD.encode(crate::sha256::digest(token.as_bytes())) => {}
        _ => return Err(Invalid("ath")),
    }
    let jwk = format!(r#"{{"crv":"P-256","kty":"EC","x":"{x}","y":"{y}"}}"#);
    if URL_SAFE_NO_PAD.encode(crate::sha256::digest(jwk.as_bytes())) != check.thumbprint {
        return Err(Invalid("thumbprint"));
    }
    if check.nonce.is_some() && claims.get("nonce").and_then(Value::as_str) != check.nonce {
        return Err(Verdict::NonceRequired);
    }
    if !seen.lock().unwrap().insert(jti.to_string()) {
        return Err(Verdict::Replay);
    }
    Ok(())
}

fn reply(status: &str, headers: &[(&str, &str)], body: &str) -> Reply {
    let mut head = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\n", body.len());
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("\r\n");
    Reply {
        bytes: [head.as_bytes(), body.as_bytes()].concat(),
        close: false,
    }
}

/// A token whose holder swaps it for the next one when it is refused.
struct Rotating {
    tokens: Mutex<Vec<&'static str>>,
    refusals: AtomicUsize,
}
impl Rotating {
    fn new(tokens: &[&'static str]) -> Self {
        Rotating {
            tokens: Mutex::new(tokens.iter().rev().copied().collect()),
            refusals: AtomicUsize::new(0),
        }
    }
}
impl AccessToken for Rotating {
    fn current(&self) -> Option<String> {
        self.tokens.lock().unwrap().last().map(|t| t.to_string())
    }
    fn refused(&self, token: &str) {
        self.refusals.fetch_add(1, Ordering::Relaxed);
        let mut tokens = self.tokens.lock().unwrap();
        if tokens.len() > 1 && tokens.last() == Some(&token) {
            tokens.pop();
        }
    }
}

/// A resource server that accepts `accepted` bound to `thumbprint` and, once `nonce` is
/// set, only proofs carrying it (challenging the rest the way AuthKit's gate does).
fn resource(
    thumbprint: String,
    accepted: &'static str,
    nonce: Option<&'static str>,
    proving: Option<&'static str>,
) -> (Origin, Arc<Mutex<String>>) {
    let base = Arc::new(Mutex::new(String::new()));
    let seen = Arc::new(Mutex::new(HashSet::new()));
    let held = base.clone();
    let origin = Origin::new(false, Duration::ZERO, move |ask, _| {
        let refuse = || {
            reply(
                "401 Unauthorized",
                &[],
                r#"{"error":{"code":"auth.invalid_credentials","message":"no"}}"#,
            )
        };
        let (Some(authorization), Some(proof)) = (ask.header("authorization"), ask.header("dpop"))
        else {
            return refuse();
        };
        let Some(token) = authorization.strip_prefix("DPoP ") else {
            return refuse();
        };
        let path = ask.path.split('?').next().unwrap();
        let url = format!(
            "{}{path}",
            proving.map_or_else(|| held.lock().unwrap().clone(), String::from)
        );
        let check = Check {
            url,
            method: &ask.method,
            token: Some(token),
            thumbprint: &thumbprint,
            nonce,
        };
        match verify(proof, &check, &seen) {
            Err(Verdict::NonceRequired) => reply(
                "401 Unauthorized",
                &[
                    (
                        "WWW-Authenticate",
                        r#"DPoP error="use_dpop_nonce", algs="ES256""#,
                    ),
                    ("DPoP-Nonce", nonce.unwrap()),
                ],
                r#"{"error":{"code":"use_dpop_nonce","message":"nonce"}}"#,
            ),
            Err(verdict) => panic!("a proof AuthKit refuses: {verdict:?}"),
            Ok(()) if token != accepted => refuse(),
            Ok(()) => reply("200 OK", &[("Content-Type", "application/json")], "{}"),
        }
    });
    *base.lock().unwrap() = origin.url.clone();
    (origin, base)
}

#[test]
fn resource_requests_carry_proofs_authkit_accepts() {
    let key = Arc::new(DpopKey::generate().unwrap());
    let (origin, _) = resource(key.thumbprint(), "at", Some("n-1"), None);
    let policy = Origin::policy();
    let credential = DpopCredential::new(
        key.clone(),
        Rotating::new(&["at"]),
        vec!["localhost".into()],
    );

    // The first proof carries no nonce: challenged once, answered with it.
    crate::transport::hub::call_method(
        &Client::new(),
        "POST",
        &origin.url,
        "/v1/tensorfs/closure",
        b"{}",
        &credential,
        &policy,
        Deadline::none(),
        &Ledger::new(),
    )
    .unwrap();
    let asks = origin.asks.lock().unwrap().clone();
    assert_eq!(asks.len(), 2);
    assert_eq!(asks[1].header("content-type"), Some("application/json"));

    // Later requests carry the held nonce, a fresh proof each, and share one connection.
    let before = origin.count();
    for _ in 0..3 {
        api_get(
            &format!("{}/v1/models?page=2", origin.url),
            &policy,
            &credential,
            1024,
            Deadline::none(),
            &Ledger::new(),
        )
        .unwrap();
    }
    assert_eq!(origin.asks.lock().unwrap().len(), 5, "no further challenge");
    assert_eq!(
        origin.count() - before,
        1,
        "a per-request proof does not split the pool"
    );
    assert_eq!(credential.tokens().refusals.load(Ordering::Relaxed), 0);

    // A host the credential does not name sees neither the token nor a proof.
    let elsewhere = DpopCredential::new(
        key.clone(),
        Rotating::new(&["at"]),
        vec!["hub.example".into()],
    );
    assert!(api_get(
        &format!("{}/v1/models", origin.url),
        &policy,
        &elsewhere,
        1024,
        Deadline::none(),
        &Ledger::new()
    )
    .is_err());
    let last = origin.asks.lock().unwrap().last().cloned().unwrap();
    assert!(last.header("authorization").is_none() && last.header("dpop").is_none());
}

#[test]
fn a_refused_token_is_refreshed_once_and_never_looped() {
    let key = Arc::new(DpopKey::generate().unwrap());
    let (origin, _) = resource(key.thumbprint(), "fresh", None, None);
    let policy = Origin::policy();
    let url = format!("{}/v1/models", origin.url);

    let credential = DpopCredential::new(
        key.clone(),
        Rotating::new(&["stale", "fresh"]),
        vec!["localhost".into()],
    );
    api_get(
        &url,
        &policy,
        &credential,
        1024,
        Deadline::none(),
        &Ledger::new(),
    )
    .unwrap();
    assert_eq!(origin.asks.lock().unwrap().len(), 2);
    assert_eq!(credential.tokens().refusals.load(Ordering::Relaxed), 1);

    // A holder with nothing fresher: one retry, then the refusal stands.
    let dead = DpopCredential::new(
        key.clone(),
        Rotating::new(&["revoked"]),
        vec!["localhost".into()],
    );
    assert!(api_get(&url, &policy, &dead, 1024, Deadline::none(), &Ledger::new()).is_err());
    assert_eq!(origin.asks.lock().unwrap().len(), 4);
}

#[test]
fn proofs_can_name_the_resource_servers_public_origin() {
    let key = Arc::new(DpopKey::generate().unwrap());
    let (origin, _) = resource(key.thumbprint(), "at", None, Some("https://hub.example"));
    let credential = DpopCredential::new(
        key.clone(),
        Rotating::new(&["at"]),
        vec!["localhost".into()],
    )
    .proving_origin("https://hub.example/");
    crate::transport::hub::call_method(
        &Client::new(),
        "GET",
        &origin.url,
        "/v1/me",
        b"",
        &credential,
        &Origin::policy(),
        Deadline::none(),
        &Ledger::new(),
    )
    .unwrap();
    assert_eq!(origin.asks.lock().unwrap().len(), 1);
}

#[test]
fn a_token_request_proves_the_key_without_a_token() {
    let key = Arc::new(DpopKey::generate().unwrap());
    let thumbprint = key.thumbprint();
    let base = Arc::new(Mutex::new(String::new()));
    let seen = Mutex::new(HashSet::new());
    let held = base.clone();
    let origin = Origin::new(false, Duration::ZERO, move |ask, number| {
        assert!(ask.header("authorization").is_none());
        assert_eq!(
            ask.header("content-type"),
            Some("application/x-www-form-urlencoded")
        );
        assert_eq!(
            ask.body,
            b"grant_type=authorization_code&code=c%2B1%2F2&code_verifier=v~_.-"
        );
        let check = Check {
            url: format!("{}/oauth2/token", held.lock().unwrap()),
            method: "POST",
            token: None,
            thumbprint: &thumbprint,
            nonce: Some("t-1"),
        };
        match verify(ask.header("dpop").unwrap(), &check, &seen) {
            // A token endpoint asks for its nonce with a 400 (RFC 9449 §8).
            Err(Verdict::NonceRequired) if number == 1 => reply(
                "400 Bad Request",
                &[("DPoP-Nonce", "t-1"), ("Content-Type", "application/json")],
                r#"{"error":"use_dpop_nonce"}"#,
            ),
            Ok(()) => reply(
                "200 OK",
                &[],
                r#"{"access_token":"at","token_type":"DPoP"}"#,
            ),
            Err(verdict) => panic!("a proof AuthKit refuses: {verdict:?}"),
        }
    });
    *base.lock().unwrap() = origin.url.clone();
    let proof_only = DpopCredential::new(key.clone(), (), vec!["localhost".into()]);
    let (status, body) = form_post(
        &format!("{}/oauth2/token", origin.url),
        &[
            ("grant_type", "authorization_code"),
            ("code", "c+1/2"),
            ("code_verifier", "v~_.-"),
        ],
        &Origin::policy(),
        &proof_only,
        4096,
        Deadline::none(),
    )
    .unwrap();
    assert_eq!(
        (status, body.as_slice()),
        (200, &br#"{"access_token":"at","token_type":"DPoP"}"#[..])
    );
    assert_eq!(origin.asks.lock().unwrap().len(), 2);
}

#[test]
fn a_key_reads_from_pkcs8_pem_and_keeps_its_thumbprint() {
    let generated = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
    let from_pem = DpopKey::from_pem(&generated.serialize_pem()).unwrap();
    let from_der = DpopKey::from_pkcs8(&generated.serialize_der()).unwrap();
    assert_eq!(from_pem.thumbprint(), from_der.thumbprint());
    assert_eq!(from_pem.thumbprint().len(), 43);
    let ed = rcgen::KeyPair::generate_for(&rcgen::PKCS_ED25519).unwrap();
    assert!(
        DpopKey::from_pem(&ed.serialize_pem()).is_err(),
        "ES256 over P-256 only"
    );
}

#[test]
fn the_verifier_refuses_what_authkit_refuses() {
    let key = Arc::new(DpopKey::generate().unwrap());
    let other = DpopKey::generate().unwrap();
    let thumbprint = key.thumbprint();
    let seen = Mutex::new(HashSet::new());
    let url = "https://hub.example/v1/x";
    let check = |method, token, nonce| Check {
        url: url.into(),
        method,
        token,
        thumbprint: &thumbprint,
        nonce,
    };
    let proof = |htu: &str, token, nonce| key.proof("GET", htu, token, nonce).unwrap();
    assert_eq!(
        verify(
            &proof(url, Some("t"), None),
            &check("POST", Some("t"), None),
            &seen
        ),
        Err(Verdict::Invalid("htm"))
    );
    assert_eq!(
        verify(
            &proof("https://hub.example/v1/y", Some("t"), None),
            &check("GET", Some("t"), None),
            &seen
        ),
        Err(Verdict::Invalid("htu"))
    );
    assert_eq!(
        verify(
            &proof(url, Some("t"), None),
            &check("GET", Some("u"), None),
            &seen
        ),
        Err(Verdict::Invalid("ath"))
    );
    assert_eq!(
        verify(
            &proof(url, None, None),
            &check("GET", Some("t"), None),
            &seen
        ),
        Err(Verdict::Invalid("ath"))
    );
    assert_eq!(
        verify(
            &proof(url, Some("t"), None),
            &check("GET", None, None),
            &seen
        ),
        Err(Verdict::Invalid("ath"))
    );
    assert_eq!(
        verify(
            &proof(url, Some("t"), None),
            &check("GET", Some("t"), Some("n")),
            &seen
        ),
        Err(Verdict::NonceRequired)
    );
    let foreign = other.proof("GET", url, Some("t"), None).unwrap();
    assert_eq!(
        verify(&foreign, &check("GET", Some("t"), None), &seen),
        Err(Verdict::Invalid("thumbprint"))
    );
    let mut forged = proof(url, Some("t"), None);
    forged.replace_range(forged.len() - 4.., "AAAA");
    assert_eq!(
        verify(&forged, &check("GET", Some("t"), None), &seen),
        Err(Verdict::Invalid("signature"))
    );
    let once = proof("https://HUB.example:443/v1/x", Some("t"), Some("n"));
    assert_eq!(
        verify(&once, &check("GET", Some("t"), Some("n")), &seen),
        Ok(())
    );
    assert_eq!(
        verify(&once, &check("GET", Some("t"), Some("n")), &seen),
        Err(Verdict::Replay)
    );
}
