package producttest

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"math/big"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/config"
)

func machineLeaf(t *testing.T, key any, public any) []byte {
	t.Helper()
	template := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "cozy-machine"},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(48 * time.Hour)}
	der, err := x509.CreateCertificate(rand.Reader, template, template, public, key)
	must(t, err)
	return der
}

// A run's capability (Tensorhub th-241) is signed offline by the device key `cozy auth login`
// enrolled, in AuthKit's envelope: it names the user, the Hub, the machine's leaf key, the run,
// its exact operations and an expiry, and nothing is asked of the Hub to make it.
func TestRunCapabilityIsSignedOfflineByTheDeviceKey(t *testing.T) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	must(t, err)
	jkt := leafJKT(t, machineLeaf(t, key, &key.PublicKey))
	home := t.TempDir()
	access := newFakeHubAccess("https://hub.invalid", "https://hub.example", "")
	cfg := config.Config{HubURL: access.server, Home: home}
	operations := []any{
		map[string]string{"type": "tensorhub_model_read", "model": "proof/private", "manifest": "sha256:" + strings.Repeat("ab", 32)},
		map[string]string{"type": "tensorhub_model_publish", "model": "proof/output"},
	}
	capability := accountauth.RunCapability{UserID: access.userID, Audience: access.public, Workload: jkt, Run: "run-1",
		Operations: operations, Expires: time.Now().Add(time.Hour)}
	if _, problem := accountauth.New(cfg).SignRunCapability(capability); problem == nil || problem.ErrName() != "auth.machine_key_missing" {
		t.Fatalf("a capability was signed without a device key: %v", problem)
	}
	access.signIn(t, home)
	jws, problem := accountauth.New(cfg).SignRunCapability(capability)
	fatal(t, problem)
	header, claims := access.capabilityOf(t, jws)
	if header["alg"] != "EdDSA" || header["typ"] != "authkit-capability+jwt" || header["kid"] != access.deviceKeyID || len(header) != 3 {
		t.Fatalf("header: %v", header)
	}
	cnf, _ := claims["cnf"].(map[string]any)
	ops, _ := claims["authorization_details"].([]any)
	jti, _ := claims["jti"].(string)
	if claims["sub"] != access.userID || claims["aud"] != access.public || cnf["jkt"] != jkt || claims["run"] != "run-1" ||
		len(ops) != 2 || len(jti) < 16 || int64(claims["exp"].(float64)) != capability.Expires.Unix() {
		t.Fatalf("claims: %v", claims)
	}
	for _, incomplete := range []func(*accountauth.RunCapability){
		func(c *accountauth.RunCapability) { c.UserID = "" },
		func(c *accountauth.RunCapability) { c.Workload = "" },
		func(c *accountauth.RunCapability) { c.Operations = nil },
		func(c *accountauth.RunCapability) { c.Expires = time.Now().Add(-time.Second) },
	} {
		changed := capability
		incomplete(&changed)
		if _, problem := accountauth.New(cfg).SignRunCapability(changed); problem == nil {
			t.Fatalf("an incomplete capability was signed: %+v", changed)
		}
	}
}
