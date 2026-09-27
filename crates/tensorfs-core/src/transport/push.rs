//! The upload half (tfs-050, th-132's mover) — the port of cozy-runtime
//! `egress.put_from`, minus nothing that was measured into it.
//!
//! One verified object streams to one exact presigned URL. The URL is already the control
//! plane's per-attempt capability, so a response cannot choose a replacement destination:
//! a redirect target is fully validated — making a metadata/private-address redirect a
//! precise security refusal — and then refused anyway, because the grant named exactly one
//! destination. `If-None-Match: *` makes the PUT idempotent by construction: a 412 on a
//! re-send of a body an earlier send completed is the first send having landed, a
//! successful end state for a content-addressed write, not a failure.
//!
//! The object is RE-SENT when the socket goes silent or the store answers a retryable
//! status, `PUT_ATTEMPTS` times, exactly as the pull re-asks for one object. Every wait of
//! the exchange — the answer after the last body byte included — is judged by its own
//! silence against the ledger's patience (`Judge::Silence`), and one write that moves
//! nothing by the ledger's floor; never an invented constant. A retryable answer is re-sent
//! after the wait the store states (`Retry-After`), which spends no attempt, or after the
//! ledger's patience, which does. Authority stays where it was: `podweights` (or whoever
//! minted the grant) decided WHETHER these bytes may go; this module only moves them.
//!
//! **A grant is a URL AND the headers its signature covers, together.** tensorhub signs
//! `if-none-match: *` and `x-amz-checksum-sha256: <base64 of the raw digest>` into every
//! upload grant it mints (`podweights.validateGrant` demands exactly those two), and a
//! SigV4 store answers 400 to a PUT that omits a signed header. A mover that carried the
//! URL and dropped the conditions therefore could not spend ANY minted grant — which is
//! what th-132 found. So the destination and its conditions are ONE value, `UploadGrant`,
//! and no spelling reaches this function with one and not the other. The headers cross the
//! wire verbatim, exactly once each, on every attempt, and they win over this module's own
//! defaults and over the credential's: the grant's spelling is what the signature covers.

use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{Doc, ObjectRef};
use crate::store::Store;
use crate::transport::http::{self, Client};
use crate::transport::hub::CredentialProvider;
use crate::transport::ledger::{Deadline, Ledger};
use crate::transport::policy::SourcePolicy;
use std::sync::OnceLock;

/// How many times one object is re-sent — the same shape as the pull's `FETCH_ATTEMPTS`,
/// for the same reason: a retry costs one object's bytes and never its identity.
pub const PUT_ATTEMPTS: u32 = 4;

const HTTP_PRECONDITION_FAILED: u16 = 412;

/// One upload grant: the exact destination and the exact headers its signature covers.
/// The pair is indivisible — a presigned PUT spent without its signed conditions is not a
/// degraded push, it is a 400 — so this is the only way to name a push destination.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UploadGrant {
    pub url: String,
    /// Lowercase names in the grant's own order, each sent verbatim exactly once.
    pub headers: Vec<(String, String)>,
}

/// A grant is a URL, a few signed headers and nothing else; anything larger is not one.
pub const MAX_GRANT_BYTES: usize = 64 * 1024;

impl UploadGrant {
    /// A destination that signs no conditions — a plain PUT target, not a minted grant.
    pub fn to(url: &str) -> UploadGrant {
        UploadGrant {
            url: url.to_string(),
            headers: Vec::new(),
        }
    }

    /// The grant as its minter writes it, in the line grammar the credential file already
    /// uses: the destination on the first line, then one `name: value` per line for each
    /// header the signature covers. No escaping, no canonical-bytes question — a presigned
    /// URL is full of `&` and a base64 checksum of `+/=`, and a grammar that can mangle
    /// either is a grammar that loses the signature.
    ///
    /// ```text
    /// https://bucket.s3.amazonaws.com/o/<key>?X-Amz-Algorithm=...&X-Amz-Signature=...
    /// if-none-match: *
    /// x-amz-checksum-sha256: 47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=
    /// ```
    pub fn parse(text: &str) -> Result<UploadGrant> {
        if text.len() > MAX_GRANT_BYTES {
            return refuse(
                Code::SIZE_CAP,
                format!(
                    "{} bytes is not an upload grant — one is a URL and a few headers",
                    text.len()
                ),
            );
        }
        let mut lines = text.lines().map(str::trim).filter(|line| !line.is_empty());
        let Some(url) = lines.next() else {
            return refuse(
                Code::SOURCE_NOT_ALLOWED,
                "an upload grant is a destination on the first line, then one \
                 `name: value` per signed header",
            );
        };
        let mut grant = UploadGrant::to(url);
        for line in lines {
            let Some((name, value)) = line.split_once(':') else {
                return refuse(
                    Code::SOURCE_NOT_ALLOWED,
                    format!("upload grant line {line:?} is not `name: value`"),
                );
            };
            let (name, value) = (name.trim(), value.trim());
            if name.is_empty() || name != name.to_ascii_lowercase() || value.is_empty() {
                return refuse(
                    Code::SOURCE_NOT_ALLOWED,
                    format!(
                        "upload grant header {name:?} is not one canonical lowercase name \
                         with a value — send the grant's headers exactly as its signature \
                         spells them"
                    ),
                );
            }
            if grant.headers.iter().any(|(held, _)| held == name) {
                return refuse(
                    Code::SOURCE_NOT_ALLOWED,
                    format!("upload grant repeats the signed header {name:?}"),
                );
            }
            grant.headers.push((name.to_string(), value.to_string()));
        }
        Ok(grant)
    }
}

fn shared_client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(Client::new)
}

/// What one push did. `landed_precondition` says the key already held these exact bytes —
/// this process's send or an earlier one; both are the object being durable remotely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pushed {
    pub object: ObjectRef,
    pub http_status: u16,
    pub landed_precondition: bool,
    pub bytes_sent: u64,
}

/// What one push sends: a manifest's bounded document bytes, or a blob's verified
/// descriptor, streamed in bounded chunks. Either re-sends from byte zero with no scratch.
enum Body {
    Bytes(Vec<u8>),
    Object(crate::store::VerifiedFile),
}

fn verified_body(store: &Store, sha256: &str, manifest: bool) -> Result<(Body, ObjectRef)> {
    if manifest {
        let path = store.manifest_path(sha256);
        let bytes = std::fs::read(&path).map_err(|error| Refusal {
            code: Code::OBJECT_ABSENT,
            detail: format!("manifest sha256:{sha256}: {error}"),
        })?;
        let object = ObjectRef::of(&bytes);
        if object.sha256 != sha256 {
            return refuse(
                Code::OBJECT_CORRUPT,
                format!("manifest sha256:{sha256} re-hashes to {}", object.sha256),
            );
        }
        crate::manifest::Manifest::parse(&bytes)?;
        Ok((Body::Bytes(bytes), object))
    } else {
        let file = store.open_verified(sha256)?;
        let object = ObjectRef {
            sha256: sha256.to_string(),
            length: file.len(),
        };
        Ok((Body::Object(file), object))
    }
}

/// Stream one verified store object to the one destination a grant names, under the exact
/// headers that grant's signature covers.
#[allow(clippy::too_many_arguments)]
pub fn push_object(
    store: &Store,
    sha256: &str,
    manifest: bool,
    grant: &UploadGrant,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    content_type: &str,
) -> Result<Pushed> {
    let client = shared_client();
    let (body, object) = verified_body(store, sha256, manifest)?;
    let payload_length = object.length;
    let payload = match &body {
        Body::Bytes(bytes) => http::Payload::Bytes(bytes),
        Body::Object(file) => http::Payload::Object(file),
    };
    let mut complete_send = false;
    let mut last_refusal: Option<Refusal> = None;
    let mut attempt = 0;
    while attempt < PUT_ATTEMPTS {
        attempt += 1;
        let last = attempt == PUT_ATTEMPTS;
        let checked = policy.check(&grant.url)?;
        let mut headers = vec![
            (
                "content-type".to_string(),
                if content_type.is_empty() {
                    "application/octet-stream".to_string()
                } else {
                    content_type.to_string()
                },
            ),
            ("if-none-match".to_string(), "*".to_string()),
        ];
        let credentials = credential.headers(&checked.host);
        headers.extend(credentials.iter().cloned());
        // The grant's own spelling is the one the signature covers, so it replaces a
        // default or a credential header of the same name rather than joining it.
        for (name, value) in &grant.headers {
            match headers.iter_mut().find(|(held, _)| held == name) {
                Some(slot) => slot.1 = value.clone(),
                None => headers.push((name.clone(), value.clone())),
            }
        }
        let sent = http::send(
            client,
            &checked,
            "PUT",
            &headers,
            Some(payload),
            // One origin's pushes share kept-alive sockets: a grant's URL and signed
            // conditions are per object, the connection is scoped by the credentials.
            Some(&credentials),
            deadline,
            ledger,
            // A peer that stops answering — during the handshake, a write, or after the
            // last body byte — is judged by that silence, never waited on forever.
            http::Judge::Silence,
            Some(ledger.floor()),
            Code::TRANSFER_FAILED,
        );
        let response = match sent {
            Ok(response) => response,
            Err(refusal) => match refusal.code {
                Code::TRANSFER_FAILED if !last => {
                    // The socket died somewhere in the exchange. If the whole body left
                    // this process, the send may have LANDED and only the answer was
                    // lost; a later 412 is that answer.
                    complete_send = true;
                    last_refusal = Some(refusal);
                    continue;
                }
                _ => return Err(refusal),
            },
        };
        let status = response.status;
        let location = response.header("location").map(str::to_string);
        if (300..400).contains(&status) {
            if let Some(location) = location {
                if location.contains("://") {
                    // Validate the whole redirect even though it is never followed: a
                    // private-address target becomes a precise security refusal, and a
                    // benign one still cannot replace the exact granted destination.
                    policy.check(&location)?;
                }
            }
            return refuse(
                Code::REDIRECT_REFUSED,
                format!("HTTP {status} tried to replace the exact granted URL"),
            );
        }
        let stated = response.stated_wait();
        let body = response.read_capped(64 * 1024)?;
        let _ = body;
        if crate::transport::pull::RETRYABLE_STATUS.contains(&status) {
            let (wait, counted) = ledger.retry_wait(stated);
            if last && counted {
                return refuse(
                    Code::TRANSFER_FAILED,
                    format!("the object store answered HTTP {status} to every PUT"),
                );
            }
            if !counted {
                attempt -= 1;
            }
            complete_send = true;
            ledger.pause(wait, deadline)?;
            continue;
        }
        if status == HTTP_PRECONDITION_FAILED {
            if complete_send {
                // An earlier send streamed every byte and lost the answer; the key now
                // holding these exact bytes IS that answer.
                return Ok(Pushed {
                    object,
                    http_status: status,
                    landed_precondition: true,
                    bytes_sent: payload_length,
                });
            }
            // The immutable key already holds bytes. Under a content-addressed key that
            // is the object being durable — the no-clobber admission, remotely.
            return Ok(Pushed {
                object,
                http_status: status,
                landed_precondition: true,
                bytes_sent: 0,
            });
        }
        if (200..300).contains(&status) {
            return Ok(Pushed {
                object,
                http_status: status,
                landed_precondition: false,
                bytes_sent: payload_length,
            });
        }
        return refuse(
            Code::TRANSFER_FAILED,
            format!("the object store answered HTTP {status} to the PUT"),
        );
    }
    Err(last_refusal.unwrap_or(Refusal {
        code: Code::TRANSFER_FAILED,
        detail: "the object store did not answer any send".into(),
    }))
}
