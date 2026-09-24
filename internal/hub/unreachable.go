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

const (
	// UnreachableName is a hub call that never got an HTTP answer: nothing
	// listening, no route, no such host, or a connection that could not be made.
	UnreachableName = "hub.unreachable"
	// DeadlineName is a hub that accepted the connection and then did not answer.
	DeadlineName = "hub.deadline"
)

// Unanswered reports whether a problem is the hub failing to answer at all, as
// opposed to the hub answering with a refusal.
func Unanswered(problem *exit.Error) bool {
	if problem == nil {
		return false
	}
	name := problem.ErrName()
	return name == UnreachableName || name == DeadlineName
}

// TransportFailure maps an HTTP client error to the one spelling every hub
// carrier shares, so an unreachable hub never reads as an auth or catalog fault.
func TransportFailure(base string, err error) *exit.Error {
	if errors.Is(err, context.Canceled) {
		return exit.New(exit.Canceled, "the call to Tensorhub at %s was canceled", base)
	}
	if cause, dialed := connectFailure(err); dialed {
		return exit.Named(exit.Unavailable, UnreachableName, "Tensorhub unreachable at %s (%s)", base, cause).
			WithRemedy("start that Tensorhub, or set tensorhub_url to one that is running")
	}
	var netErr net.Error
	if errors.Is(err, context.DeadlineExceeded) || strings.Contains(err.Error(), "Client.Timeout") ||
		(errors.As(err, &netErr) && netErr.Timeout()) {
		return exit.Named(exit.Deadline, DeadlineName,
			"Tensorhub at %s accepted the connection but did not answer within %s", base, Timeout).
			WithRemedy("retry; if it persists the hub is up but not serving")
	}
	return exit.Named(exit.Unavailable, UnreachableName, "Tensorhub unreachable at %s (%s)", base, innermost(err)).
		WithRemedy("start that Tensorhub, or set tensorhub_url to one that is running")
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
