package hub

import (
	"context"
	"crypto/x509"
	"encoding/base64"
	"net/http"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// ExecutionAccess is account-authorized catalog/storage access delegated to one TLS
// identity. It is not a machine registration or a rental lifecycle capability.
type ExecutionAccess struct {
	Token       string            `json:"token"`
	ExpiresAt   time.Time         `json:"expires_at"`
	Environment map[string]string `json:"environment"`
	TrustRoot   []byte            `json:"-"`
}

func (c *Client) AuthorizeExecutionAccess(ctx context.Context, leaf []byte) (ExecutionAccess, *exit.Error) {
	var out ExecutionAccess
	certificate, err := x509.ParseCertificate(leaf)
	now := time.Now()
	if err != nil || now.Before(certificate.NotBefore) || !now.Before(certificate.NotAfter) {
		return out, exit.New(exit.Credential, "execution access requires the machine's valid pinned certificate")
	}
	ttl := min(int64(7*24*60*60), int64(time.Until(certificate.NotAfter).Seconds())-1)
	if ttl < 1 {
		return out, exit.New(exit.Credential, "the machine certificate is expiring")
	}
	if problem := c.do(ctx, call{method: http.MethodPost, path: "/v1/execution-access", auth: true,
		reason: "delegate execution access", trustRoot: &out.TrustRoot,
		body: map[string]any{"delegate_certificate_der_b64url": base64.RawURLEncoding.EncodeToString(leaf), "ttl_seconds": ttl}}, &out); problem != nil {
		return ExecutionAccess{}, problem
	}
	if out.Token == "" || !out.ExpiresAt.After(now) || out.ExpiresAt.After(certificate.NotAfter) || out.Environment["TENSORHUB_ORIGIN"] == "" {
		return ExecutionAccess{}, exit.Named(exit.Conflict, "hub.execution_access_invalid", "Tensorhub returned incomplete execution access")
	}
	for name := range out.Environment {
		switch name {
		case "TENSORHUB_ORIGIN", "TENSORHUB_PUBLIC_ORIGIN", "TENSORHUB_OBJECT_STORAGE_HOSTS":
		default:
			delete(out.Environment, name)
		}
	}
	return out, nil
}
