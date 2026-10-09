package producttest

import (
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeHubAccess answers a fixture Hub's side of a run's Hub access (Tensorhub th-241): the
// device-key sign-in `cozy auth login` left (it answers bearer), the execution environment,
// the current account with its AuthKit user id, the RFC 9728 and OpenID metadata naming the
// token endpoint, and, for a real machine, that JWT-bearer token endpoint. The CLI signs a
// capability offline with its device key: nothing asks the Hub to authorize it.
type fakeHubAccess struct {
	server, origin, public, bearer string
	deviceKeyID, userID           string
	// account is the signed-in account's name.
	account string
	device                        ed25519.PrivateKey
	// refuse, when set, answers every trade invalid_grant.
	refuse bool
	mu     sync.Mutex
	// traded are the capabilities the machine presented, in order.
	traded []string
}

// newFakeHubAccess serves server's account routes; machines read the Hub at public.
func newFakeHubAccess(server, public, bearer string) *fakeHubAccess {
	_, device, _ := ed25519.GenerateKey(rand.Reader)
	return &fakeHubAccess{server: server, origin: public, public: public, bearer: bearer,
		deviceKeyID: "dk-fixture", userID: "user-fixture", account: "proof", device: device}
}

// signIn writes home's device-key credential for the fixture Hub, as `cozy auth login` does.
func (a *fakeHubAccess) signIn(t *testing.T, home string) {
	t.Helper()
	sum := sha256.Sum256([]byte(a.server))
	path := filepath.Join(home, "auth", hex.EncodeToString(sum[:])+".json")
	must(t, os.MkdirAll(filepath.Dir(path), 0o700))
	record, err := json.Marshal(map[string]any{"version": 1, "hub": a.server, "email": "fixture@example.com",
		"device_key_id": a.deviceKeyID, "private_key": base64.RawURLEncoding.EncodeToString(a.device.Seed())})
	must(t, err)
	must(t, os.WriteFile(path, record, 0o600))
}

// serve answers r when it is one of the account routes the CLI calls.
func (a *fakeHubAccess) serve(w http.ResponseWriter, r *http.Request) bool {
	signedIn := a.bearer == "" || r.Header.Get("Authorization") == "Bearer "+a.bearer
	switch {
	case r.Method == http.MethodPost && r.URL.Path == "/v1/auth/v1/device-keys/login/begin":
		_ = json.NewEncoder(w).Encode(map[string]any{"challenge_id": "fixture-challenge",
			"challenge": base64.RawURLEncoding.EncodeToString(make([]byte, 32)), "expires_at": time.Now().Add(time.Minute)})
	case r.Method == http.MethodPost && r.URL.Path == "/v1/auth/v1/device-keys/login/finish":
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_set":  map[string]any{"access_token": a.bearer, "token_type": "Bearer", "expires_in": 600},
			"device_key": map[string]any{"id": a.deviceKeyID}})
	case r.Method == http.MethodGet && r.URL.Path == "/v1/execution-environment":
		if !signedIn {
			http.Error(w, "account required", http.StatusUnauthorized)
			return true
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"environment": map[string]string{"TENSORHUB_ORIGIN": a.origin, "TENSORHUB_PUBLIC_ORIGIN": a.public}})
	case r.Method == http.MethodGet && r.URL.Path == "/v1/accounts/current":
		if !signedIn {
			http.Error(w, "account required", http.StatusUnauthorized)
			return true
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"name": a.account, "user_id": a.userID})
	case r.Method == http.MethodGet && r.URL.Path == "/.well-known/oauth-protected-resource":
		_ = json.NewEncoder(w).Encode(map[string]any{"resource": a.public, "authorization_servers": []string{a.server + "/v1/auth"}})
	case r.Method == http.MethodGet && r.URL.Path == "/v1/auth/.well-known/openid-configuration":
		// The machine reaches the token endpoint where it reads the Hub.
		_ = json.NewEncoder(w).Encode(map[string]any{"issuer": a.server + "/v1/auth", "token_endpoint": a.origin + "/v1/auth/oauth2/token",
			"grant_types_supported": []string{"urn:ietf:params:oauth:grant-type:jwt-bearer"}})
	default:
		return false
	}
	return true
}

// serveMachine answers the token endpoint a machine trades a capability at, where it reads
// the Hub. It verifies nothing: TensorD's AuthKit tests and the Hub's own prove the trade;
// this records what the machine presented.
func (a *fakeHubAccess) serveMachine(w http.ResponseWriter, r *http.Request) bool {
	switch {
	case r.Method == http.MethodPost && r.URL.Path == "/v1/auth/oauth2/token":
		_ = r.ParseForm()
		var claims struct {
			Capability string `json:"capability"`
		}
		parts := strings.Split(r.PostForm.Get("assertion"), ".")
		if len(parts) == 3 {
			raw, _ := base64.RawURLEncoding.DecodeString(parts[1])
			_ = json.Unmarshal(raw, &claims)
		}
		a.mu.Lock()
		a.traded = append(a.traded, claims.Capability)
		refuse := a.refuse
		a.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if refuse || r.Header.Get("DPoP") == "" || r.PostForm.Get("client_id") != "tensord" {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"invalid_grant","reason":"refused","error_description":"the stand-in hub grants nothing"}`))
			return true
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"access_token": fixtureCapabilityToken(a.server+"/v1/auth", a.public), "token_type": "DPoP", "expires_in": 900})
	default:
		return false
	}
	return true
}

// refuseTrades answers every later trade invalid_grant.
func (a *fakeHubAccess) refuseTrades() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.refuse = true
}

// trades are the capabilities machines presented at the token endpoint, oldest first.
func (a *fakeHubAccess) trades() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]string(nil), a.traded...)
}

// capabilityOf verifies a run capability's device-key signature and answers its header and
// claims.
func (a *fakeHubAccess) capabilityOf(t *testing.T, jws string) (header, claims map[string]any) {
	t.Helper()
	parts := strings.Split(jws, ".")
	if len(parts) != 3 {
		t.Fatalf("not a compact JWS: %q", jws)
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[2])
	must(t, err)
	if !ed25519.Verify(a.device.Public().(ed25519.PublicKey), []byte(parts[0]+"."+parts[1]), signature) {
		t.Fatalf("the capability is not signed by the device key")
	}
	for i, out := range []*map[string]any{&header, &claims} {
		raw, err := base64.RawURLEncoding.DecodeString(parts[i])
		must(t, err)
		must(t, json.Unmarshal(raw, out))
	}
	return header, claims
}

// fixtureCapabilityToken is a syntactic RFC 9068 token; the real Hub verifies it and its
// DPoP proof.
func fixtureCapabilityToken(issuer, resource string) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"typ":"at+jwt","alg":"ES256"}`))
	body, _ := json.Marshal(map[string]any{"iss": issuer, "sub": "user-fixture", "client_id": "tensord", "aud": resource, "exp": 4102444800})
	return header + "." + base64.RawURLEncoding.EncodeToString(body) + "." + base64.RawURLEncoding.EncodeToString([]byte("fixture-signature"))
}

// leafJKT is the RFC 7638 thumbprint of a machine leaf's P-256 key: a capability's cnf.jkt.
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

// hubAccessCall is whether a recorded "METHOD /path" call is one a run's Hub access makes:
// the execution environment, the current account, device-key sign-in, or a machine's trade.
func hubAccessCall(call string) bool {
	_, path, _ := strings.Cut(call, " ")
	return path == "/v1/execution-environment" || path == "/v1/accounts/current" || strings.HasPrefix(path, "/v1/auth/") ||
		path == "/.well-known/oauth-protected-resource"
}

// strayTensor writes one F16 tensor's safetensors file in dir: a model a machine makes as it
// is, with no provider.
func strayTensor(t *testing.T, dir string) string {
	t.Helper()
	header := []byte(`{"stray.weight":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}`)
	file := filepath.Join(dir, "stray.safetensors")
	body := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	must(t, os.WriteFile(file, append(append(body, header...), 0, 0, 0, 0), 0o600))
	return file
}
