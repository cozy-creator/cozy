package hub

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/url"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Machine grants (Tensorhub th-238): this client requests one for a machine's TLS key and
// approves it with its device-key sign-in; the machine redeems the code at the issuer's token
// endpoint and holds every token, so none passes through here.
const (
	machineClient   = "cozy-machine"
	machineRedirect = "http://127.0.0.1/cozy-machine/callback"
	authPrefix      = "/v1/auth"
)

// MachineGrant is one approved code for the machine to redeem within a minute.
type MachineGrant struct {
	Issuer, Code, Verifier, RedirectURI, Resource string
}

// ExecutionEnvironment is where a machine reaches this Hub and its object storage.
type ExecutionEnvironment struct {
	Environment map[string]string `json:"environment"`
	TrustRoot   []byte            `json:"-"`
}

func (c *Client) ExecutionEnvironment(ctx context.Context) (ExecutionEnvironment, *exit.Error) {
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

// GrantExecution approves reads of this account's private catalog for the machine whose
// leaf is given, for as long as this sign-in lasts.
func (c *Client) GrantExecution(ctx context.Context, leaf []byte, resource string) (MachineGrant, *exit.Error) {
	return c.grantMachine(ctx, leaf, resource, []any{map[string]string{"type": "tensorhub_execution"}}, false)
}

// GrantPublication approves publication into repositories for the machine whose leaf is
// given: machineID is its rental, or "" for a machine without one. It outlives this sign-in.
func (c *Client) GrantPublication(ctx context.Context, leaf []byte, resource, machineID string, repositories []PublicationRepository) (MachineGrant, *exit.Error) {
	return c.grantMachine(ctx, leaf, resource, []any{map[string]any{"type": "tensorhub_machine_publication", "machine_id": machineID,
		"repositories": repositories, "permissions": []string{"assessment", "checkpoint", "release"}}}, true)
}

func (c *Client) grantMachine(ctx context.Context, leaf []byte, resource string, details any, offline bool) (MachineGrant, *exit.Error) {
	jkt, problem := leafThumbprint(leaf)
	if problem != nil {
		return MachineGrant{}, problem
	}
	var metadata struct {
		Issuer string `json:"issuer"`
	}
	if problem := c.do(ctx, call{method: http.MethodGet, path: authPrefix + "/.well-known/openid-configuration"}, &metadata); problem != nil {
		return MachineGrant{}, problem
	}
	raw, err := json.Marshal(details)
	if err != nil || metadata.Issuer == "" || resource == "" {
		return MachineGrant{}, exit.Named(exit.Conflict, "hub.grant_unavailable", "Tensorhub does not grant machine access")
	}
	verifier, state := randomToken(), randomToken()
	challenge := sha256.Sum256([]byte(verifier))
	q := url.Values{"response_type": {"code"}, "client_id": {machineClient}, "redirect_uri": {machineRedirect}, "state": {state},
		"code_challenge": {base64.RawURLEncoding.EncodeToString(challenge[:])}, "code_challenge_method": {"S256"},
		"resource": {resource}, "dpop_jkt": {jkt}, "authorization_details": {string(raw)}}
	if offline {
		q.Set("scope", "offline_access")
	}
	var location string
	if problem := c.do(ctx, call{method: http.MethodGet, path: authPrefix + "/oauth2/authorize?" + q.Encode(), location: &location}, nil); problem != nil {
		return MachineGrant{}, problem
	}
	pending, err := url.Parse(location)
	if err != nil {
		return MachineGrant{}, exit.Named(exit.Conflict, "hub.grant_unreadable", "Tensorhub answered the machine grant request unreadably")
	}
	if refusal := pending.Query().Get("error"); refusal != "" {
		return MachineGrant{}, exit.Named(exit.Credential, "hub.grant_refused", "Tensorhub refused the machine grant: %s", pending.Query().Get("error_description"))
	}
	id := pending.Query().Get("authorization")
	if id == "" {
		return MachineGrant{}, exit.Named(exit.Conflict, "hub.grant_unreadable", "Tensorhub answered the machine grant request unreadably")
	}
	var approved struct {
		RedirectTo string `json:"redirect_to"`
	}
	if problem := c.do(ctx, call{method: http.MethodPost, path: authPrefix + "/v1/oauth2/authorizations/" + url.PathEscape(id) + "/approve", auth: true}, &approved); problem != nil {
		return MachineGrant{}, problem
	}
	back, err := url.Parse(approved.RedirectTo)
	if err != nil || !strings.HasPrefix(approved.RedirectTo, machineRedirect+"?") || back.Query().Get("state") != state || back.Query().Get("iss") != metadata.Issuer {
		return MachineGrant{}, exit.Named(exit.Conflict, "hub.grant_unreadable", "Tensorhub's approval answers another request")
	}
	if refusal := back.Query().Get("error"); refusal != "" {
		if description := back.Query().Get("error_description"); description != "" {
			refusal = description
		}
		return MachineGrant{}, exit.Named(exit.Credential, "hub.grant_refused", "Tensorhub refused the machine grant: %s", refusal)
	}
	code := back.Query().Get("code")
	if code == "" {
		return MachineGrant{}, exit.Named(exit.Conflict, "hub.grant_unreadable", "Tensorhub's approval carries no code")
	}
	return MachineGrant{Issuer: metadata.Issuer, Code: code, Verifier: verifier, RedirectURI: machineRedirect, Resource: resource}, nil
}

// leafThumbprint is the RFC 7638 thumbprint of a machine leaf's P-256 key: its DPoP key.
func leafThumbprint(leaf []byte) (string, *exit.Error) {
	certificate, err := x509.ParseCertificate(leaf)
	if err != nil {
		return "", exit.New(exit.Credential, "a machine grant requires the machine's pinned certificate")
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

func randomToken() string {
	var raw [32]byte
	_, _ = rand.Read(raw[:])
	return base64.RawURLEncoding.EncodeToString(raw[:])
}
