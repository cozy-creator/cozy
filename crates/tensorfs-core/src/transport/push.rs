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
//! status, `PUT_ATTEMPTS` times, exactly as the pull re-asks for one object. The write
//! silence bound is the ledger's floor — the same derived noise floor `egress.py` used —
//! never an invented constant. Authority stays where it was: `podweights` (or whoever
//! minted the grant) decided WHETHER these bytes may go; this module only moves them.

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

/// Read one verified object out of the store, whole. Objects are cut on the 64 MiB grid
/// and a manifest is capped harder, so one object in memory is bounded by construction —
/// which is what lets a re-send restart from byte zero with no scratch state.
fn verified_bytes(store: &Store, sha256: &str, manifest: bool) -> Result<(Vec<u8>, ObjectRef)> {
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
        Ok((bytes, object))
    } else {
        let file = store.open_verified(sha256)?;
        let length = file.len();
        let mut bytes = vec![
            0u8;
            usize::try_from(length).map_err(|_| Refusal {
                code: Code::SIZE_CAP,
                detail: format!("sha256:{sha256} exceeds this platform's address space"),
            })?
        ];
        file.read_exact_at(&mut bytes, 0).map_err(|error| Refusal {
            code: Code::IO_FAILED,
            detail: format!("read sha256:{sha256}: {error}"),
        })?;
        Ok((
            bytes,
            ObjectRef {
                sha256: sha256.to_string(),
                length,
            },
        ))
    }
}

/// Stream one verified store object to one exact presigned URL.
#[allow(clippy::too_many_arguments)]
pub fn push_object(
    store: &Store,
    sha256: &str,
    manifest: bool,
    url: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    content_type: &str,
) -> Result<Pushed> {
    let client = shared_client();
    let (bytes, object) = verified_bytes(store, sha256, manifest)?;
    let mut complete_send = false;
    let mut last_refusal: Option<Refusal> = None;
    for attempt in 1..=PUT_ATTEMPTS {
        let last = attempt == PUT_ATTEMPTS;
        let checked = policy.check(url)?;
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
        headers.extend(credential.headers(&checked.host));
        let sent = http::request(
            client,
            &checked,
            "PUT",
            &headers,
            Some(&bytes),
            deadline,
            ledger,
            false,
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
        let body = response.read_capped(64 * 1024)?;
        let _ = body;
        if crate::transport::pull::RETRYABLE_STATUS.contains(&status) && !last {
            complete_send = true;
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
                    bytes_sent: bytes.len() as u64,
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
                bytes_sent: bytes.len() as u64,
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
