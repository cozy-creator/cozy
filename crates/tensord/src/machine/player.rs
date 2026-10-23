//! The machine's direct player endpoint: the WebRTC listener a browser reaches with its play
//! link, and the addresses Status reports for it. No relay: a viewer behind NAT is refused.
use crate::api::v1;
use std::{
    collections::BTreeMap,
    io,
    net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener},
    path::Path,
};

/// Where a browser reaches the listener: this computer's own interfaces, a provider's (or the
/// launcher's) mapping of its port, or a mapping that could not be read, which stops playback
/// alone.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Reach {
    #[default]
    Interfaces,
    Mapped(SocketAddr),
    Invalid(String),
}

/// The listener on every interface: its capability is bound to the viewer's DTLS certificate,
/// while the control API stays where the grant puts it. A rental listens on its granted port;
/// this computer's machine keeps the port it first took, so a play link survives restarts.
pub fn listen(state: &Path, granted: Option<u16>, local: bool) -> io::Result<Option<TcpListener>> {
    if granted.is_none() && !local {
        return Ok(None);
    }
    super::net::kept_listener(&state.join("webrtc-port"), Ipv4Addr::UNSPECIFIED.into(), granted).map(Some)
}

/// The launch environment's mapping of a granted port: `COZY_WEBRTC_PUBLIC_ADDRESS`, else
/// RunPod's or vast.ai's public IP and external port for it.
pub fn reach(granted: Option<u16>, env: &BTreeMap<String, String>) -> Reach {
    let ipv4 = |address: SocketAddr, what: &str| match address.is_ipv4() && address.port() != 0 {
        true => Reach::Mapped(address),
        false => Reach::Invalid(format!("{what} must name an IPv4 address and a nonzero port")),
    };
    if let Some(address) = env.get("COZY_WEBRTC_PUBLIC_ADDRESS") {
        return match address.parse() {
            Ok(address) => ipv4(address, "COZY_WEBRTC_PUBLIC_ADDRESS"),
            Err(_) => Reach::Invalid("COZY_WEBRTC_PUBLIC_ADDRESS must be host:port".into()),
        };
    }
    let Some(port) = granted else { return Reach::Interfaces };
    for (host, mapped) in [("RUNPOD_PUBLIC_IP", "RUNPOD_TCP_PORT"), ("PUBLIC_IPADDR", "VAST_TCP_PORT")] {
        let mapped = format!("{mapped}_{port}");
        let Some(external) = env.get(&mapped) else { continue };
        return match (env.get(host).map(|ip| ip.parse::<IpAddr>()), external.parse::<u16>()) {
            (Some(Ok(ip)), Ok(external)) => ipv4(SocketAddr::new(ip, external), &mapped),
            _ => Reach::Invalid(format!("{mapped} needs a port and {host} an IP address")),
        };
    }
    Reach::Interfaces
}

/// Status's player endpoint: the listener's port, where it is reached, and the fingerprint of
/// the certificate its DTLS presents.
pub fn endpoint(port: u16, reach: &Reach, cert_der: &[u8]) -> v1::WebRtc {
    let (addresses, unavailable_reason) = match reach {
        Reach::Mapped(address) => (vec![address.to_string()], String::new()),
        Reach::Invalid(reason) => (vec![], reason.clone()),
        Reach::Interfaces => (interfaces(port), String::new()),
    };
    v1::WebRtc {
        port: port.into(),
        addresses,
        fingerprint: tensorfs::sha256::hex_digest(cert_der),
        unavailable_reason,
    }
}

/// This computer's IPv4 addresses (LAN, Tailscale; loopback last) at `port`.
fn interfaces(port: u16) -> Vec<String> {
    let mut found: Vec<Ipv4Addr> = addresses()
        .into_iter()
        .filter(|ip| !ip.is_unspecified() && !ip.is_multicast())
        .collect();
    found.sort_by_key(|ip| (ip.is_loopback(), *ip));
    found.dedup();
    found.into_iter().map(|ip| SocketAddr::new(ip.into(), port).to_string()).collect()
}

#[cfg(unix)]
fn addresses() -> Vec<Ipv4Addr> {
    (nix::ifaddrs::getifaddrs().into_iter().flatten())
        .filter_map(|interface| interface.address?.as_sockaddr_in().map(|v4| v4.ip()))
        .collect()
}

/// Where this computer's default route leaves from, and loopback: Windows has no getifaddrs,
/// and a UDP socket's connect only consults the route, sending nothing.
#[cfg(not(unix))]
fn addresses() -> Vec<Ipv4Addr> {
    let routed = std::net::UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).and_then(|socket| {
        socket.connect((Ipv4Addr::new(192, 0, 2, 1), 9))?;
        socket.local_addr()
    });
    let routed = routed.ok().and_then(|address| match address.ip() {
        std::net::IpAddr::V4(v4) => Some(v4),
        std::net::IpAddr::V6(_) => None,
    });
    routed.into_iter().chain([Ipv4Addr::LOCALHOST]).collect()
}
