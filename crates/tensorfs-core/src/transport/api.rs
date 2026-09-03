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
            false,
            None,
            Code::TRANSFER_FAILED,
        )?;
        if (300..400).contains(&response.status) {
            let Some(location) = response.header("location").map(str::to_string) else {
                return refuse(Code::REDIRECT_REFUSED, "origin redirected with no Location");
            };
            if !location.contains("://") {
                return refuse(
                    Code::REDIRECT_REFUSED,
                    "origin redirected to a non-absolute URL",
                );
            }
            target = location;
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
