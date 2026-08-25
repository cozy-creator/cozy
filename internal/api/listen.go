package api

import (
	"net"
	"strconv"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// LOOPBACK-ONLY, and it is a REFUSAL rather than a default.
//
// The design's words are exact: "a non-localhost/LAN bind is a DEFERRED door gated on
// TLS, durable authentication, and its own threat review, NEVER A FLAG." So this package
// owns the only `net.Listen("tcp", …)` in the binary (the `api` fence family proves it),
// and it refuses any address whose IP is not a loopback address. A future pod profile
// (cl-014) binds TLS on a public interface through its OWN listener — a different
// function with its own review — not by relaxing this one.
//
// IPv6 is bound DELIBERATELY and tested, not assumed: `localhost` resolves to ::1 first
// on most modern systems, so a v4-only server is one a browser reaches intermittently
// depending on the resolver, and `[::1]` in the Host allowlist would be a promise the
// server could not keep.

// Listeners binds 127.0.0.1 and [::1] on one port. It answers with every listener it
// bound and the canonical address (the v4 one, which is what the service publishes).
//
// An IPv6 bind that fails is NOT fatal: a host with IPv6 disabled is ordinary, and
// refusing to start there would be a worse bug than serving v4 only. What it returns is
// the truth about which families are live, so `doctor` can say so.
func Listeners(port int) (v4 net.Listener, v6 net.Listener, addr string, e *exit.Error) {
	addr = "127.0.0.1:" + strconv.Itoa(port)
	v4, err := listenLoopback("tcp4", addr)
	if err != nil {
		return nil, nil, "", exit.New(exit.Conflict, "%s is held by another process: %s", addr, err).
			WithRemedy("choose another port with --port, or stop what holds it")
	}
	if l, err := listenLoopback("tcp6", "[::1]:"+strconv.Itoa(port)); err == nil {
		v6 = l
	}
	return v4, v6, addr, nil
}

// listenLoopback is the ONE bind. It checks the address it was handed rather than
// trusting its caller — the guarantee has to live where the syscall is.
func listenLoopback(network, address string) (net.Listener, error) {
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return nil, err
	}
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return nil, &net.AddrError{Err: "the local client API binds loopback addresses only; " +
			"the LAN door is deferred behind TLS and durable auth, and it is not a flag",
			Addr: address}
	}
	return net.Listen(network, address)
}
