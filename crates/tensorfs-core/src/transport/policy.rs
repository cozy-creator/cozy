//! The ONE address predicate, moved here WITH the transport (tfs-049, cr-012's lesson).
//!
//! cozy-runtime's `internal/egress.py` states the hazard this file exists against: *"the
//! failure mode this module exists against is not 'the SSRF check was wrong', it is 'there
//! were two downloaders and only one had the check'."* The byte mover moved into tensorfs,
//! so the predicate moved with it — `blocked()` below is the port of `egress.blocked`, one
//! class list, and cr-090 retargets the `one-address-predicate` fence at this file.
//!
//! **The allowlist is a deployment-wide DECLARATION and resolution is fail-closed.** A host
//! absent from the allowlist refuses. A host that resolves to a private, loopback,
//! link-local, multicast, reserved or unspecified address refuses — EVERY address it
//! resolves to, not the first, because a name with one public and one 169.254.169.254
//! answer is the metadata-service bypass with an extra step. An empty allowlist is no
//! egress at all; nothing widens it at request time.
//!
//! **The connection is made to the addresses that were checked.** `egress.py` named its own
//! DNS-rebinding limit — validate at resolve time, connect by hostname, window of one
//! connect — as work rather than papering over it. Here the checked `SocketAddr`s are what
//! the socket connects to, with SNI/Host still carrying the hostname, so a name that
//! answers publicly during the check cannot answer privately during the connect.
//!
//! `allow_local` is the dev carve-out and it is explicit: loopback/private addresses and
//! plaintext http, admitted only when the caller declared a local deployment (a loopback
//! hub, a test origin). "I trust this host" and "I accept a private address" stay separate
//! statements — conflating them is how a localhost dev setting becomes a production
//! metadata read.

use crate::err::{refuse, Code, Result};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use std::sync::{Arc, Mutex};

/// The address classes no downloader in this tree connects to — the port of
/// `egress._CLASSES`, same names, same order. Returns why `address` is not fetchable, or
/// `""` for a public routable address.
pub fn blocked(address: IpAddr) -> String {
    let mut named: Vec<&'static str> = Vec::new();
    match canonical(address) {
        IpAddr::V4(v4) => {
            if v4_private(v4) {
                named.push("private");
            }
            if v4.is_loopback() {
                named.push("loopback");
            }
            if v4.is_link_local() {
                named.push("link-local");
            }
            if v4.is_multicast() {
                named.push("multicast");
            }
            if v4_reserved(v4) {
                named.push("reserved");
            }
            if v4.is_unspecified() {
                named.push("unspecified");
            }
        }
        IpAddr::V6(v6) => {
            if v6_unique_local(v6) {
                named.push("private");
            }
            if v6.is_loopback() {
                named.push("loopback");
            }
            if v6_link_local(v6) {
                named.push("link-local");
            }
            if v6.is_multicast() {
                named.push("multicast");
            }
            // FAIL-CLOSED arm: outside 2000::/3 and not already named is not a space this
            // predicate can vouch for, so it refuses as reserved rather than defaulting
            // open. Stricter than the Python original in places, looser nowhere.
            if named.is_empty() && !v6.is_unspecified() && (v6.segments()[0] & 0xe000) != 0x2000 {
                named.push("reserved");
            }
            if v6.is_unspecified() {
                named.push("unspecified");
            }
        }
    }
    named.join(", ")
}

/// v4-mapped and NAT64 spellings of a v4 address are judged AS that v4 address:
/// `::ffff:169.254.169.254` is the metadata service, whatever family the socket claims.
fn canonical(address: IpAddr) -> IpAddr {
    if let IpAddr::V6(v6) = address {
        if let Some(v4) = v6.to_ipv4_mapped() {
            return IpAddr::V4(v4);
        }
        let s = v6.segments();
        if s[0] == 0x64 && s[1] == 0xff9b && s[2..6] == [0, 0, 0, 0] {
            return IpAddr::V4(Ipv4Addr::new(
                (s[6] >> 8) as u8,
                s[6] as u8,
                (s[7] >> 8) as u8,
                s[7] as u8,
            ));
        }
    }
    address
}

fn v4_private(v4: Ipv4Addr) -> bool {
    let o = v4.octets();
    v4.is_private()                                    // 10/8, 172.16/12, 192.168/16
        || o[0] == 0                                   // 0.0.0.0/8 "this network"
        || (o[0] == 100 && (o[1] & 0xc0) == 64)        // 100.64/10 CGNAT
        || (o[0] == 192 && o[1] == 0 && o[2] == 0)     // 192.0.0/24 protocol assignments
        || (o[0] == 198 && (o[1] & 0xfe) == 18)        // 198.18/15 benchmarking
        || v4.is_documentation()
}

fn v4_reserved(v4: Ipv4Addr) -> bool {
    v4.octets()[0] >= 240 // 240/4, includes 255.255.255.255
}

fn v6_unique_local(v6: Ipv6Addr) -> bool {
    (v6.segments()[0] & 0xfe00) == 0xfc00 // fc00::/7
}

fn v6_link_local(v6: Ipv6Addr) -> bool {
    (v6.segments()[0] & 0xffc0) == 0xfe80 // fe80::/10
}

/// EVERY answer for `host`, through one resolver call. A caller that resolved for itself
/// could check a different answer set than it connects over; here the returned addresses
/// are both the checked set and the connect set.
pub fn resolve(host: &str, port: u16) -> std::io::Result<Vec<SocketAddr>> {
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Ok(vec![SocketAddr::new(ip, port)]);
    }
    Ok((host, port).to_socket_addrs()?.collect())
}

/// One URL, checked: the exact origin the socket may open, and nothing else about it.
#[derive(Debug, Clone)]
pub struct CheckedUrl {
    pub https: bool,
    pub host: String,
    pub port: u16,
    /// Path plus query, exactly as given — a presigned URL's query IS its authorization.
    pub target: String,
    /// The checked answers. The connection is made to these, never re-resolved.
    pub addrs: Vec<SocketAddr>,
}

impl CheckedUrl {
    /// The URL this check admitted, as a string a later ask can start from.
    pub fn url(&self) -> String {
        let scheme = if self.https { "https" } else { "http" };
        let host = if self.host.contains(':') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        format!("{scheme}://{host}:{}{}", self.port, self.target)
    }
}

/// The deployment's declared egress bound. Carried by every caller that hands this module
/// a URL; enforced on the first URL and on every redirect hop, because the hops are only
/// visible here.
#[derive(Debug, Clone, Default)]
pub struct SourcePolicy {
    /// Exact host names, case-insensitive, trailing dot trimmed. An entry starting with a
    /// dot (`.example.com`) admits any subdomain; a bare entry admits only itself. Empty
    /// means NO egress at all.
    pub allowed_hosts: Vec<String>,
    /// Loopback/private addresses and plaintext http, for a deployment that DECLARED it
    /// is local (a loopback dev hub, a test origin). Never inferred here.
    pub allow_local: bool,
    /// Redirect hop budget. 0 refuses every redirect — a presigned GET has no business
    /// redirecting; a provider handler (tfs-052) that needs hops declares them.
    pub max_redirects: u32,
    /// Answers this policy has already resolved, shared by its clones. One pull asks the
    /// resolver once per host instead of once per object; every answer is still checked.
    pub resolved: Resolved,
}

/// Answers by host and port, each behind its own lock so one lookup serves every caller.
type Answers = HashMap<(String, u16), Arc<Mutex<Option<Vec<SocketAddr>>>>>;

/// One resolution per host and port for as long as the policy that made it lives. Callers
/// that ask for a host while it resolves wait for that answer: a pull's 512 requests sent
/// as many lookups to a pod's Docker resolver at once, which answered some after 4-8 s.
#[derive(Clone, Default)]
pub struct Resolved(Arc<Mutex<Answers>>);

impl std::fmt::Debug for Resolved {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Resolved")
    }
}

impl Resolved {
    fn lookup(&self, host: &str, port: u16) -> std::io::Result<Vec<SocketAddr>> {
        let slot = Arc::clone(
            self.0
                .lock()
                .unwrap()
                .entry((host.to_string(), port))
                .or_default(),
        );
        let mut answer = slot.lock().unwrap();
        if let Some(addrs) = answer.as_ref() {
            return Ok(addrs.clone());
        }
        let addrs = resolve(host, port)?;
        if !addrs.is_empty() {
            *answer = Some(addrs.clone());
        }
        Ok(addrs)
    }
}

impl SourcePolicy {
    pub fn allows_host(&self, host: &str) -> bool {
        let host = host.trim_end_matches('.').to_ascii_lowercase();
        self.allowed_hosts.iter().any(|entry| {
            let entry = entry.trim_end_matches('.').to_ascii_lowercase();
            match entry.strip_prefix('.') {
                Some(suffix) => {
                    !suffix.is_empty()
                        && host
                            .strip_suffix(suffix)
                            .is_some_and(|head| head.ends_with('.'))
                }
                None => !entry.is_empty() && host == entry,
            }
        })
    }

    /// The whole front door: scheme, shape, allowlist, resolution, and the class check on
    /// every answer. Everything this module fetches or pushes passes here first, and every
    /// redirect hop passes here again.
    pub fn check(&self, url: &str) -> Result<CheckedUrl> {
        let (scheme, host, port, target) = split_url(url)?;
        let https = scheme == "https";
        if !https && !self.allow_local {
            return refuse(
                Code::SOURCE_NOT_ALLOWED,
                format!(
                    "{host}: plaintext http to a host outside a declared-local deployment — \
                     artifact bytes travel over https"
                ),
            );
        }
        if !self.allows_host(&host) {
            return refuse(
                Code::SOURCE_NOT_ALLOWED,
                format!(
                    "{host:?} is not in this deployment's declared allowed-hosts \
                     ({:?}). The allowlist is a deployment-wide declaration; nothing \
                     widens it at request time",
                    self.allowed_hosts
                ),
            );
        }
        let port = port.unwrap_or(if https { 443 } else { 80 });
        let addrs = self
            .resolved
            .lookup(&host, port)
            .map_err(|error| crate::err::Refusal {
                code: if crate::descriptors::exhausted(&error) {
                    Code::FD_HEADROOM
                } else {
                    Code::HUB_UNREACHABLE
                },
                detail: format!(
                    "{host}: name resolution failed ({error}); an address this module cannot \
                 check is not an address it will connect to"
                ),
            })?;
        if addrs.is_empty() {
            return refuse(Code::HUB_UNREACHABLE, format!("{host} resolved to nothing"));
        }
        if !self.allow_local {
            for addr in &addrs {
                let why = blocked(addr.ip());
                if !why.is_empty() {
                    return refuse(
                        Code::SOURCE_NOT_ALLOWED,
                        format!(
                            "{host} resolves to {}, which is {why} and not a public \
                             address. EVERY answer is checked, not the first: a name with \
                             one public answer and one 169.254.169.254 is the \
                             metadata-service bypass with an extra step",
                            addr.ip()
                        ),
                    );
                }
            }
        }
        Ok(CheckedUrl {
            https,
            host,
            port,
            target,
            addrs,
        })
    }
}

/// Whether every answer for the URL's host is loopback — the one fact that lets a caller
/// DECLARE a local deployment from its own hub base (`pull` does exactly this, as
/// cozy-runtime's `local_hub` did).
pub fn is_loopback_origin(url: &str) -> bool {
    let Ok((_, host, port, _)) = split_url(url) else {
        return false;
    };
    match resolve(&host, port.unwrap_or(443)) {
        Ok(addrs) => {
            !addrs.is_empty() && addrs.iter().all(|a| blocked(a.ip()).contains("loopback"))
        }
        Err(_) => false,
    }
}

/// The host a base URL names — what a caller scopes a credential to, and what naming a
/// hub base declares.
pub fn base_host(url: &str) -> Result<String> {
    split_url(url).map(|(_, host, _, _)| host)
}

/// The small strict split this module needs. Userinfo and fragments are refused outright —
/// a capability URL carrying either is not one clean origin.
fn split_url(url: &str) -> Result<(String, String, Option<u16>, String)> {
    if url.is_empty() || url.len() > 16 * 1024 || !url.is_ascii() {
        return refuse(
            Code::SOURCE_NOT_ALLOWED,
            "source URL is empty, over 16 KiB, or not ASCII",
        );
    }
    if url.contains('#') {
        return refuse(
            Code::SOURCE_NOT_ALLOWED,
            "source URL carries a fragment, which is not sent to a server",
        );
    }
    let Some((scheme, rest)) = url.split_once("://") else {
        return refuse(Code::SOURCE_NOT_ALLOWED, "source URL is not absolute");
    };
    let scheme = scheme.to_ascii_lowercase();
    if scheme != "http" && scheme != "https" {
        return refuse(
            Code::SOURCE_NOT_ALLOWED,
            format!("{scheme}: not an allowed transport scheme (http/https only)"),
        );
    }
    let split_at = rest.find(['/', '?']).unwrap_or(rest.len());
    let (authority, target) = rest.split_at(split_at);
    let target = if target.is_empty() || target.starts_with('?') {
        format!("/{target}")
    } else {
        target.to_string()
    };
    if authority.is_empty() || authority.contains('@') {
        return refuse(
            Code::SOURCE_NOT_ALLOWED,
            "source URL authority is empty or carries userinfo; credentials belong in no host",
        );
    }
    let (host_raw, port_raw) = match authority.strip_prefix('[') {
        Some(bracketed) => {
            let Some((host, after)) = bracketed.split_once(']') else {
                return refuse(
                    Code::SOURCE_NOT_ALLOWED,
                    "source URL IPv6 authority is unterminated",
                );
            };
            match after.strip_prefix(':') {
                Some(p) => (host.to_string(), Some(p.to_string())),
                None if after.is_empty() => (host.to_string(), None),
                None => {
                    return refuse(
                        Code::SOURCE_NOT_ALLOWED,
                        "source URL authority is malformed after the IPv6 bracket",
                    )
                }
            }
        }
        None => match authority.rsplit_once(':') {
            Some((host, port)) => (host.to_string(), Some(port.to_string())),
            None => (authority.to_string(), None),
        },
    };
    let port = match port_raw {
        None => None,
        Some(text) => Some(text.parse::<u16>().map_err(|_| crate::err::Refusal {
            code: Code::SOURCE_NOT_ALLOWED,
            detail: format!("source URL port {text:?} is not a 16-bit number"),
        })?),
    };
    let host = host_raw.trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty() {
        return refuse(Code::SOURCE_NOT_ALLOWED, "source URL has no host");
    }
    Ok((scheme, host, port, target))
}
