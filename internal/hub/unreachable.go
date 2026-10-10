package hub

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"net"
	"net/url"
	"strings"
	"syscall"

	"github.com/cozy-creator/cozy/internal/exit"
)

// UnreachableName is a hub call that never got an HTTP answer: nothing listening, no
// route, no such host, a connection that could not be made, or one that died.
const UnreachableName = "hub.unreachable"

// Unanswered reports whether a problem is the hub failing to answer at all, as
// opposed to the hub answering with a refusal.
func Unanswered(problem *exit.Error) bool {
	return problem != nil && problem.ErrName() == UnreachableName
}

// TransportFailure maps an HTTP client error to the one spelling every hub
// carrier shares, so an unreachable hub never reads as an auth or catalog fault.
func TransportFailure(base string, err error) *exit.Error {
	if errors.Is(err, context.Canceled) {
		return exit.New(exit.Canceled, "the call to Tensorhub at %s was canceled", base)
	}
	if errors.Is(err, context.DeadlineExceeded) {
		// Only a caller's own explicit deadline (a command's --timeout) is ever on a Hub call.
		return exit.New(exit.Deadline, "the call to Tensorhub at %s reached its command's deadline", base)
	}
	if cause, dialed := connectFailure(err); dialed {
		return exit.Named(exit.Unavailable, UnreachableName, "Tensorhub unreachable at %s (%s)", base, cause).
			WithRemedy("start that Tensorhub, or set tensorhub_url to one that is running")
	}
	var netErr net.Error
	if errors.Is(err, syscall.ETIMEDOUT) || errors.As(err, &netErr) && netErr.Timeout() ||
		strings.Contains(err.Error(), "client connection lost") {
		return exit.Named(exit.Unavailable, UnreachableName,
			"the connection to Tensorhub at %s went dead: it stopped answering liveness probes", base).
			WithRemedy("retry; if it persists the network to that Tensorhub is dropping traffic")
	}
	return exit.Named(exit.Unavailable, UnreachableName, "Tensorhub at %s dropped the connection (%s)", base, innermost(err)).
		WithRemedy("retry; if it persists that Tensorhub or a proxy in front of it is dropping connections")
}

// connectFailure names why no connection was made, or reports false when one was.
func connectFailure(err error) (string, bool) {
	var dns *net.DNSError
	var certificate *tls.CertificateVerificationError
	var authority x509.UnknownAuthorityError
	var hostname x509.HostnameError
	var op *net.OpError
	switch {
	case errors.Is(err, syscall.ECONNREFUSED):
		return "connection refused", true
	case errors.As(err, &dns):
		if dns.IsNotFound {
			return "no such host " + dns.Name, true
		}
		return "DNS lookup of " + dns.Name + " failed", true
	case errors.Is(err, syscall.EHOSTUNREACH):
		return "host unreachable", true
	case errors.Is(err, syscall.ENETUNREACH):
		return "network unreachable", true
	case errors.As(err, &certificate), errors.As(err, &authority), errors.As(err, &hostname):
		return "TLS certificate not trusted: " + innermost(err), true
	case errors.As(err, &op) && op.Op == "dial":
		if op.Timeout() {
			return "connect timed out", true
		}
		return innermost(err), true
	}
	return "", false
}

func innermost(err error) string {
	var wrapped *url.Error
	if errors.As(err, &wrapped) {
		err = wrapped.Err
	}
	var op *net.OpError
	if errors.As(err, &op) && op.Err != nil {
		err = op.Err
	}
	return err.Error()
}
