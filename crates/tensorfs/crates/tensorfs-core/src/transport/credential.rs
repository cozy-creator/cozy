//! Credential presentation. **The credential arrives as a provider trait, never a credential
//! type**, so one implementation serves whatever a caller holds. It is asked per request and
//! per redirect hop with the request's method and checked target, so a credential stays
//! scoped to the hosts its provider names (it cannot leak onto a presigned object host or a
//! redirect target) and a per-request proof (`dpop`) is minted for exactly the request it
//! signs. It sees every answer, so a provider can take a server nonce or a refused token
//! and ask for the one retry that cures it.

use crate::err::{refuse, Code, Result};
use crate::transport::http::Response;
use crate::transport::policy::CheckedUrl;

/// The request a credential is asked to present on.
pub struct Ask<'a> {
    pub method: &'a str,
    pub url: &'a CheckedUrl,
}

impl Ask<'_> {
    /// The URL without query or fragment, default port elided: RFC 9449's `htu`.
    pub fn htu(&self) -> String {
        let scheme = if self.url.https { "https" } else { "http" };
        let host = self.url.host.to_ascii_lowercase();
        let host = if host.contains(':') {
            format!("[{host}]")
        } else {
            host
        };
        let path = self.url.target.split(['?', '#']).next().unwrap_or("");
        let path = if path.is_empty() { "/" } else { path };
        match (self.url.https, self.url.port) {
            (true, 443) | (false, 80) => format!("{scheme}://{host}{path}"),
            (_, port) => format!("{scheme}://{host}:{port}{path}"),
        }
    }
}

/// What a credential is shown of an answer: the status and headers it came back with, and
/// the headers the credential presented on the request.
pub struct Answer<'a> {
    pub status: u16,
    headers: &'a [(String, String)],
    presented: &'a [(String, String)],
}

impl<'a> Answer<'a> {
    pub(crate) fn of(status: u16, headers: &'a [(String, String)], presented: &'a [(String, String)]) -> Answer<'a> {
        Answer { status, headers, presented }
    }
    pub fn header(&self, name: &str) -> Option<&'a str> {
        find(self.headers, name)
    }
    pub fn presented(&self, name: &str) -> Option<&'a str> {
        find(self.presented, name)
    }
}

fn find<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

pub trait CredentialProvider: Sync {
    /// The headers that present a credential on `ask`. An empty answer presents nothing,
    /// which is what a presigned URL wants.
    fn headers(&self, ask: &Ask) -> Vec<(String, String)>;
    /// Shown every answer to a request this provider was asked about. `true` asks for one
    /// more attempt with fresh headers; it is honoured once per request.
    fn answered(&self, _ask: &Ask, _answer: &Answer) -> bool {
        false
    }
}

/// One request under `credential`: headers asked, the answer shown, and the one retry the
/// credential may ask for. `send` makes the request with the credential's headers.
pub(crate) fn presenting<'l>(
    credential: &dyn CredentialProvider,
    ask: &Ask,
    mut send: impl FnMut(&[(String, String)]) -> Result<Response<'l>>,
) -> Result<Response<'l>> {
    let mut retried = false;
    loop {
        let presented = credential.headers(ask);
        let response = send(&presented)?;
        let again = credential.answered(
            ask,
            &Answer {
                status: response.status,
                headers: response.headers(),
                presented: &presented,
            },
        );
        if !again || retried {
            return Ok(response);
        }
        retried = true;
        response.discard();
    }
}

fn names(hosts: &[String], ask: &Ask) -> bool {
    hosts.iter().any(|h| h.eq_ignore_ascii_case(&ask.url.host))
}

/// No credential at all.
pub struct Anonymous;

impl CredentialProvider for Anonymous {
    fn headers(&self, _ask: &Ask) -> Vec<(String, String)> {
        Vec::new()
    }
}

/// Arbitrary presentation headers, scoped to named hosts — the generic impl a caller
/// builds when its credential is not a bearer.
pub struct ScopedHeaders {
    pub hosts: Vec<String>,
    pub headers: Vec<(String, String)>,
}

impl CredentialProvider for ScopedHeaders {
    fn headers(&self, ask: &Ask) -> Vec<(String, String)> {
        if names(&self.hosts, ask) {
            self.headers.clone()
        } else {
            Vec::new()
        }
    }
}

/// Two providers' headers on one request (a sender-constrained token beside a worker
/// capability); each sees every answer, and either may ask for the retry.
pub struct Both<A, B>(pub A, pub B);

impl<A: CredentialProvider, B: CredentialProvider> CredentialProvider for Both<A, B> {
    fn headers(&self, ask: &Ask) -> Vec<(String, String)> {
        let mut headers = self.0.headers(ask);
        headers.extend(self.1.headers(ask));
        headers
    }
    fn answered(&self, ask: &Ask, answer: &Answer) -> bool {
        let first = self.0.answered(ask, answer);
        self.1.answered(ask, answer) || first
    }
}

/// The ONE credential spelling every surface passes (tfs-050): exactly one scoped value,
/// handed explicitly — a file, one deliberately-set variable — never inherited.
///
/// - `bearer <token>`      → `authorization: Bearer <token>`
/// - `worker <id> <token>` → `x-cozy-worker-id` + `x-cozy-worker-token`: a machine's
///   worker capability, which the hub's closure and presign read as its owner's grant
/// - empty                 → anonymous: published checkpoints only.
///
/// A sender-constrained (DPoP) token has no spelling: its key never leaves the process
/// that holds it, which builds a `DpopCredential` instead.
pub fn credential_from_spec(spec: &str, hosts: Vec<String>) -> Result<ScopedHeaders> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Ok(ScopedHeaders {
            hosts,
            headers: Vec::new(),
        });
    }
    let parts: Vec<&str> = spec.split_whitespace().collect();
    let headers = match parts.as_slice() {
        ["bearer", token] => {
            vec![("authorization".to_string(), format!("Bearer {token}"))]
        }
        ["worker", id, token] => vec![
            ("x-cozy-worker-id".to_string(), id.to_string()),
            ("x-cozy-worker-token".to_string(), token.to_string()),
        ],
        _ => {
            return refuse(
                Code::CREDENTIAL_REQUIRED,
                "a credential is `bearer <token>`, `worker <id> <token>` or empty — one \
                 scoped value, passed explicitly",
            )
        }
    };
    Ok(ScopedHeaders { hosts, headers })
}
