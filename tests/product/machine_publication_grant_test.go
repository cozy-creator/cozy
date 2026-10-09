package producttest

import (
	"context"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"math/big"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
)

func machineLeaf(t *testing.T, key any, public any) []byte {
	t.Helper()
	template := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "cozy-machine"},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(48 * time.Hour)}
	der, err := x509.CreateCertificate(rand.Reader, template, template, public, key)
	must(t, err)
	return der
}

// The CLI's half of a machine grant (Tensorhub th-238): the authorization request names the
// machine's leaf key, the exact scope and client, and only publication asks for an offline
// grant; the approval uses the account's sign-in and returns a code for the machine.
func TestMachineGrantsBindTheLeafKeyAndExplicitScope(t *testing.T) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	must(t, err)
	leaf := machineLeaf(t, key, &key.PublicKey)
	raw, err := key.PublicKey.Bytes()
	must(t, err)
	enc := base64.RawURLEncoding.EncodeToString
	thumbprint := sha256.Sum256([]byte(`{"crv":"P-256","kty":"EC","x":"` + enc(raw[1:33]) + `","y":"` + enc(raw[33:]) + `"}`))
	var grants *fakeGrants
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !grants.serve(w, r) {
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	grants = newFakeGrants(server.URL, "https://hub.example", "fixture")
	account := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("fixture")}, "")

	execution, problem := account.GrantExecution(context.Background(), leaf, "https://hub.example")
	fatal(t, problem)
	publication, problem := account.GrantPublication(context.Background(), leaf, "https://hub.example", "pr-proof",
		hub.PublicationRepositories([]string{"alice/a", "alice/z"}))
	fatal(t, problem)
	asked := grants.requests()
	if len(asked) != 2 {
		t.Fatalf("authorization requests: %v", asked)
	}
	for i, grant := range []hub.MachineGrant{execution, publication} {
		q := asked[i]
		challenge := sha256.Sum256([]byte(grant.Verifier))
		if q.Get("client_id") != "cozy-machine" || q.Get("redirect_uri") != grant.RedirectURI || q.Get("resource") != "https://hub.example" ||
			q.Get("dpop_jkt") != enc(thumbprint[:]) || q.Get("code_challenge_method") != "S256" || q.Get("code_challenge") != enc(challenge[:]) ||
			grant.Issuer != server.URL+"/v1/auth" || grant.Code == "" || grant.Resource != "https://hub.example" {
			t.Fatalf("grant %d is not bound to the leaf key and PKCE: %v %+v", i, q, grant)
		}
	}
	if asked[0].Get("scope") != "" || asked[0].Get("authorization_details") != `[{"type":"tensorhub_execution"}]` {
		t.Fatalf("execution access is not one online grant: %v", asked[0])
	}
	var detail []map[string]any
	must(t, json.Unmarshal([]byte(asked[1].Get("authorization_details")), &detail))
	if asked[1].Get("scope") != "offline_access" || len(detail) != 1 || detail[0]["type"] != "tensorhub_machine_publication" ||
		detail[0]["machine_id"] != "pr-proof" || len(detail[0]["repositories"].([]any)) != 2 || len(detail[0]["permissions"].([]any)) != 3 {
		t.Fatalf("publication scope changed: %v", asked[1])
	}
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	if _, problem := account.GrantExecution(context.Background(), machineLeaf(t, private, public), "https://hub.example"); problem == nil {
		t.Fatal("a leaf without a P-256 key was granted")
	}
	if _, problem := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("stranger")}, "").GrantExecution(context.Background(), leaf, "https://hub.example"); problem == nil {
		t.Fatal("a grant was approved without the account's sign-in")
	}
}
