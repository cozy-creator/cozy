package hub

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// ExecutionEnvironment is where a machine reaches this Hub and its object storage, and the
// resource a run capability for it names (TENSORHUB_PUBLIC_ORIGIN).
type ExecutionEnvironment struct {
	Environment map[string]string `json:"environment"`
	TrustRoot   []byte            `json:"-"`
}

// ExecutionEnvironment is read once per credential and kept beside it, as the account is: a
// later run, in this process or the next, reads no Hub.
func (c *Client) ExecutionEnvironment(ctx context.Context) (ExecutionEnvironment, *exit.Error) {
	type kept struct {
		Environment map[string]string `json:"environment"`
		TrustRoot   []byte            `json:"trust_root,omitempty"`
	}
	path := c.keptPath("environments")
	var held kept
	if loadRelease(path, &held) && held.Environment["TENSORHUB_ORIGIN"] != "" && held.Environment["TENSORHUB_PUBLIC_ORIGIN"] != "" {
		return ExecutionEnvironment{Environment: held.Environment, TrustRoot: held.TrustRoot}, nil
	}
	out, problem := c.readExecutionEnvironment(ctx)
	if problem == nil {
		storeRelease(path, kept{Environment: out.Environment, TrustRoot: out.TrustRoot})
	}
	return out, problem
}

func (c *Client) readExecutionEnvironment(ctx context.Context) (ExecutionEnvironment, *exit.Error) {
	var out ExecutionEnvironment
	if problem := c.do(ctx, call{method: http.MethodGet, path: "/v1/execution-environment", auth: true, trustRoot: &out.TrustRoot}, &out); problem != nil {
		return ExecutionEnvironment{}, problem
	}
	if out.Environment["TENSORHUB_ORIGIN"] == "" || out.Environment["TENSORHUB_PUBLIC_ORIGIN"] == "" {
		return ExecutionEnvironment{}, exit.Named(exit.Conflict, "hub.execution_environment_invalid", "Tensorhub returned an incomplete execution environment")
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

// TokenEndpoint is where a machine trades a run capability for this Hub: its authorization
// server's token endpoint, from its protected-resource metadata (RFC 9728) and the issuer's
// OpenID metadata. The issuer is this Hub's AuthKit, read at its path on this client's origin.
func (c *Client) TokenEndpoint(ctx context.Context) (string, *exit.Error) {
	path := c.keptPath("token-endpoints")
	var endpoint string
	if loadRelease(path, &endpoint) && endpoint != "" {
		return endpoint, nil
	}
	endpoint, problem := c.readTokenEndpoint(ctx)
	if problem == nil {
		storeRelease(path, endpoint)
	}
	return endpoint, problem
}

func (c *Client) readTokenEndpoint(ctx context.Context) (string, *exit.Error) {
	var described struct {
		Servers []string `json:"authorization_servers"`
	}
	if problem := c.do(ctx, call{method: http.MethodGet, path: "/.well-known/oauth-protected-resource"}, &described); problem != nil {
		return "", problem
	}
	unavailable := exit.Named(exit.Conflict, "hub.token_endpoint_unavailable", "Tensorhub names no authorization server that takes run capabilities")
	if len(described.Servers) == 0 {
		return "", unavailable
	}
	issuer := strings.TrimRight(described.Servers[0], "/")
	parsed, err := url.Parse(issuer)
	if err != nil || parsed.Host == "" {
		return "", unavailable
	}
	var server struct {
		Issuer        string   `json:"issuer"`
		TokenEndpoint string   `json:"token_endpoint"`
		Grants        []string `json:"grant_types_supported"`
	}
	if problem := c.do(ctx, call{method: http.MethodGet, path: parsed.EscapedPath() + "/.well-known/openid-configuration"}, &server); problem != nil {
		return "", problem
	}
	endpoint, err := url.Parse(server.TokenEndpoint)
	if strings.TrimRight(server.Issuer, "/") != issuer || err != nil || endpoint.Host == "" ||
		!slices.Contains(server.Grants, "urn:ietf:params:oauth:grant-type:jwt-bearer") {
		return "", unavailable
	}
	return server.TokenEndpoint, nil
}

// LeafThumbprint is the RFC 7638 thumbprint of a machine leaf's P-256 key: the `cnf.jkt` a
// run capability binds to that machine.
func LeafThumbprint(leaf []byte) (string, *exit.Error) {
	certificate, err := x509.ParseCertificate(leaf)
	if err != nil {
		return "", exit.New(exit.Credential, "a run capability requires the machine's pinned certificate")
	}
	key, ok := certificate.PublicKey.(*ecdsa.PublicKey)
	if !ok || key.Curve != elliptic.P256() {
		return "", exit.New(exit.Credential, "the machine's certificate key is not P-256")
	}
	raw, err := key.Bytes()
	if err != nil {
		return "", exit.New(exit.Credential, "the machine's certificate key is invalid")
	}
	enc := base64.RawURLEncoding.EncodeToString
	sum := sha256.Sum256([]byte(`{"crv":"P-256","kty":"EC","x":"` + enc(raw[1:33]) + `","y":"` + enc(raw[33:]) + `"}`))
	return enc(sum[:]), nil
}
