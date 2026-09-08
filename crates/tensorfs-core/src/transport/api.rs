//! Provider-plane reads over the one transport (the seam tfs-052's handlers stand on).
//!
//! An API listing is untrusted input like any other: bounded, policy-fenced on every hop,
//! credential asked per host so an owner's key cannot leak onto a CDN. Nothing here
//! admits bytes — a listing names members; the fetch plane moves them.

use crate::err::{refuse, Code, Refusal, Result};
use crate::transport::http::{self, Client};
use crate::transport::hub::CredentialProvider;
use crate::transport::ledger::{Deadline, Ledger};
use crate::transport::policy::SourcePolicy;
use crate::transport::pull::absolute_location;
use std::sync::OnceLock;

fn shared_client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(Client::new)
}

/// One bounded GET of a caller-validated API URL, redirect hops re-checked, following
/// `Link: <...>; rel="next"` pagination by handing the next URL back to the caller.
pub fn api_get(
    url: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    max_bytes: u64,
    deadline: Deadline,
    ledger: &Ledger,
) -> Result<(Vec<u8>, Option<String>)> {
    let response = open_following(url, policy, credential, deadline, ledger)?;
    if response.status != 200 {
        return refuse(
            Code::TRANSFER_FAILED,
            format!("origin answered HTTP {} to an API read", response.status),
        );
    }
    let next = response.header("link").and_then(next_link);
    let cap = usize::try_from(max_bytes).unwrap_or(usize::MAX);
    let body = response.read_capped(cap)?;
    Ok((body, next))
}

/// The exact byte length of one origin object, learned from a one-byte ranged GET —
/// `Content-Range: bytes 0-0/<total>` — for providers whose listings declare a digest but
/// not an exact length. Falls back to Content-Length when the origin ignores ranges.
pub fn probe_length(
    url: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
) -> Result<u64> {
    let response = open_following_with(
        url,
        policy,
        credential,
        deadline,
        ledger,
        &[("range".to_string(), "bytes=0-0".to_string())],
    )?;
    let total = match response.status {
        206 => response
            .header("content-range")
            .and_then(|value| value.trim().rsplit_once('/'))
            .and_then(|(_, total)| total.trim().parse::<u64>().ok()),
        200 => response
            .header("content-length")
            .and_then(|value| value.trim().parse::<u64>().ok()),
        other => {
            return refuse(
                Code::TRANSFER_FAILED,
                format!("origin answered HTTP {other} to a length probe"),
            )
        }
    };
    total.ok_or(Refusal {
        code: Code::TRANSFER_FAILED,
        detail: "origin did not state the object's total length".into(),
    })
}

/// What one prefix read learned. `bytes` is `None` when the origin would not serve a
/// range for an object larger than the ask — the only honest answer there is "not cheaply
/// readable", because reading it whole is the transfer a prefix read exists to avoid.
pub struct Prefix {
    pub bytes: Option<Vec<u8>>,
    /// The object's total length as the ORIGIN states it. A caller that already pinned a
    /// length has two opinions here and must require that they agree.
    pub total: Option<u64>,
}

/// The leading `want` bytes of one origin object, by ranged GET.
///
/// This is the read a conversion plan is actually made of. A safetensors header is
/// `8 + <declared>` bytes at offset zero and a `*.safetensors.index.json` is a small whole
/// document, so the facts that decide whether a 210 GB selection can convert at all live in
/// the first few kilobytes of each member. `probe_length` beside it asks the same question
/// with `want` of one byte and keeps only the total; this keeps the bytes.
///
/// The 200 arm reads nothing it was not promised. An origin that ignores the range and
/// starts sending a 5 GiB shard is answered by dropping the body unread — a prefix read
/// that silently became a whole-object download would be the exact cost it exists to
/// remove. Only an object the origin itself declares to be no larger than the ask is read
/// from a 200, which is how a small member on a range-less origin still arrives.
pub fn fetch_prefix(
    url: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    want: u64,
    deadline: Deadline,
    ledger: &Ledger,
) -> Result<Prefix> {
    if want == 0 {
        return refuse(Code::COUNT_CAP, "a prefix read asks for at least one byte");
    }
    let response = open_following_with(
        url,
        policy,
        credential,
        deadline,
        ledger,
        &[("range".to_string(), format!("bytes=0-{}", want - 1))],
    )?;
    let cap = usize::try_from(want).unwrap_or(usize::MAX);
    match response.status {
        206 => {
            let total = response
                .header("content-range")
                .and_then(|value| value.trim().rsplit_once('/'))
                .and_then(|(_, total)| total.trim().parse::<u64>().ok());
            let bytes = response.read_capped(cap)?;
            Ok(Prefix {
                bytes: Some(bytes),
                total,
            })
        }
        200 => {
            let total = response
                .header("content-length")
                .and_then(|value| value.trim().parse::<u64>().ok());
            match total {
                Some(total) if total <= want => Ok(Prefix {
                    bytes: Some(response.read_capped(cap)?),
                    total: Some(total),
                }),
                // The body is dropped here, unread, and that is the point.
                _ => Ok(Prefix { bytes: None, total }),
            }
        }
        other => refuse(
            Code::TRANSFER_FAILED,
            format!("origin answered HTTP {other} to a prefix read"),
        ),
    }
}

fn open_following<'l>(
    url: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &'l Ledger,
) -> Result<http::Response<'l>> {
    open_following_with(url, policy, credential, deadline, ledger, &[])
}

fn open_following_with<'l>(
    url: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &'l Ledger,
    extra: &[(String, String)],
) -> Result<http::Response<'l>> {
    let mut target = url.to_string();
    for _hop in 0..=policy.max_redirects {
        let checked = policy.check(&target)?;
        let mut headers = credential.headers(&checked.host);
        headers.extend_from_slice(extra);
        let response = http::request(
            shared_client(),
            &checked,
            "GET",
            &headers,
            None,
            deadline,
            ledger,
            // The one plane tfs-106 deliberately left as it was. A foreign source read is
            // not serving a hub call and has the source lane's own per-part retry above it,
            // so arming a plane here would change a path this lane did not measure. It is
            // therefore still bounded by the caller's deadline alone, and a foreign source
            // that accepts a connection and stops sending to a caller who set no deadline
            // still waits. Named here so the next reader finds a decision, not an oversight.
            http::Judge::Deadline,
            None,
            Code::TRANSFER_FAILED,
        )?;
        if (300..400).contains(&response.status) {
            let Some(location) = response.header("location").map(str::to_string) else {
                return refuse(Code::REDIRECT_REFUSED, "origin redirected with no Location");
            };
            // RFC 9110 §10.2.2 permits a relative reference here and HuggingFace uses
            // one for every plain git blob — which is exactly the class resolution must
            // read (`*.safetensors.index.json`). tfs-060 fixed this in the object path's
            // opener and left the API path's own copy behind; this is that same fix.
            // The fence is unchanged: the resolved reference goes back through
            // `policy.check` on the next hop like any other.
            let Some(absolute) = absolute_location(&checked, &location) else {
                return refuse(
                    Code::REDIRECT_REFUSED,
                    "origin redirected with an unusable Location",
                );
            };
            target = absolute;
            continue;
        }
        return Ok(response);
    }
    refuse(
        Code::REDIRECT_REFUSED,
        format!(
            "more than {} redirect hop(s); a longer chain is a loop or an evasion",
            policy.max_redirects
        ),
    )
}

fn next_link(header: &str) -> Option<String> {
    for part in header.split(',') {
        let part = part.trim();
        let (target, params) = part.split_once('>')?;
        if params.contains("rel=\"next\"") || params.contains("rel=next") {
            return Some(target.trim_start_matches('<').to_string());
        }
    }
    None
}
