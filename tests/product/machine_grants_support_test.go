package producttest

import (
	"crypto/ecdsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// fakeGrants answers a fixture Hub's machine-grant routes (Tensorhub th-238): its execution
// environment, the OAuth code flow the CLI drives with its sign-in, and the token endpoint
// the machine redeems the code at. bearer, when set, is the sign-in the CLI must present;
// origin is where machines read the Hub, public its resource identity.
type fakeGrants struct {
	origin, public, issuer, bearer string
	// token is the access token every redemption answers: what the machine reads with.
	token string
	// refuse, when set, refuses redeeming the code of each request it names.
	refuse   func(asked url.Values) bool
	mu       sync.Mutex
	pending  []url.Values
	redeemed int
}

func newFakeGrants(server, public, bearer string) *fakeGrants {
	return &fakeGrants{origin: public, public: public, issuer: server + "/v1/auth", bearer: bearer,
		token: fixtureGrantToken(server+"/v1/auth", public)}
}

// serve answers r when it is a grant route.
func (g *fakeGrants) serve(w http.ResponseWriter, r *http.Request) bool {
	signedIn := g.bearer == "" || r.Header.Get("Authorization") == "Bearer "+g.bearer
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/v1/execution-environment":
		if !signedIn {
			http.Error(w, "account required", http.StatusUnauthorized)
			return true
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"environment": map[string]string{"TENSORHUB_ORIGIN": g.origin, "TENSORHUB_PUBLIC_ORIGIN": g.public}})
	case r.Method == http.MethodGet && r.URL.Path == "/v1/auth/.well-known/openid-configuration":
		_ = json.NewEncoder(w).Encode(map[string]string{"issuer": g.issuer})
	case r.Method == http.MethodGet && r.URL.Path == "/v1/auth/oauth2/authorize":
		g.mu.Lock()
		g.pending = append(g.pending, r.URL.Query())
		id := strconv.Itoa(len(g.pending) - 1)
		g.mu.Unlock()
		http.Redirect(w, r, g.issuer+"/authorize?authorization="+id, http.StatusSeeOther)
	case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/v1/auth/v1/oauth2/authorizations/") && strings.HasSuffix(r.URL.Path, "/approve"):
		if !signedIn {
			http.Error(w, "sign-in required", http.StatusUnauthorized)
			return true
		}
		id, _ := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/v1/auth/v1/oauth2/authorizations/"), "/approve"))
		g.mu.Lock()
		asked := g.pending[id]
		g.mu.Unlock()
		back := url.Values{"code": {"fixture-code-" + strconv.Itoa(id)}, "state": {asked.Get("state")}, "iss": {g.issuer}}
		_ = json.NewEncoder(w).Encode(map[string]string{"redirect_to": asked.Get("redirect_uri") + "?" + back.Encode()})
	case r.Method == http.MethodPost && r.URL.Path == "/v1/auth/oauth2/token":
		_ = r.ParseForm()
		g.mu.Lock()
		g.redeemed++
		var asked url.Values
		if id, err := strconv.Atoi(strings.TrimPrefix(r.PostForm.Get("code"), "fixture-code-")); err == nil && id < len(g.pending) {
			asked = g.pending[id]
		}
		g.mu.Unlock()
		if asked != nil && g.refuse != nil && g.refuse(asked) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"invalid_grant","error_description":"the stand-in hub grants nothing"}`))
			return true
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"access_token": g.token, "token_type": "DPoP",
			"expires_in": 900, "refresh_token": "fixture-refresh"})
	default:
		return false
	}
	return true
}

// requests are the authorization requests the CLI sent, oldest first.
func (g *fakeGrants) requests() []url.Values {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]url.Values{}, g.pending...)
}

// fixtureGrantToken is a syntactic RFC 9068 token, the same for the same Hub; the real Hub
// verifies it and its DPoP proof.
func fixtureGrantToken(issuer, resource string) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"typ":"at+jwt","alg":"ES256"}`))
	body, _ := json.Marshal(map[string]any{"iss": issuer, "sub": "fixture-account", "client_id": "cozy-machine", "aud": resource,
		"exp": 4102444800, "permissions": []string{"tensorhub:execution:access"}})
	return header + "." + base64.RawURLEncoding.EncodeToString(body) + "." + base64.RawURLEncoding.EncodeToString([]byte("fixture-signature"))
}

// leafJKT is the RFC 7638 thumbprint of a machine leaf's P-256 key, as dpop_jkt names it.
func leafJKT(t *testing.T, der []byte) string {
	t.Helper()
	certificate, err := x509.ParseCertificate(der)
	must(t, err)
	raw, err := certificate.PublicKey.(*ecdsa.PublicKey).Bytes()
	must(t, err)
	enc := base64.RawURLEncoding.EncodeToString
	sum := sha256.Sum256([]byte(`{"crv":"P-256","kty":"EC","x":"` + enc(raw[1:33]) + `","y":"` + enc(raw[33:]) + `"}`))
	return enc(sum[:])
}

// publicationRequest is whether asked requests a machine publication grant.
func publicationRequest(asked url.Values) bool {
	return strings.Contains(asked.Get("authorization_details"), "tensorhub_machine_publication")
}

// grantCall is whether a recorded "METHOD /path" call is one of the machine-grant routes a
// cold run makes: discovery, the code flow, and the machine's redemption.
func grantCall(call string) bool {
	_, path, _ := strings.Cut(call, " ")
	return path == "/v1/execution-environment" || strings.HasPrefix(path, "/v1/auth/")
}
