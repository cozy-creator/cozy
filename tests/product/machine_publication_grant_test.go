package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"math/big"
	"reflect"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/hub"
)

// A publication grant names one machine, rented (pr-…) or owned (om-…), by its exact Host
// leaf, with explicit repositories, for at most seven days within the leaf's validity.
func TestMachinePublicationIntentIsExplicitBoundedAndCertificatePinned(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)
	pub, key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "private-worker"}, NotBefore: now.Add(-time.Hour), NotAfter: now.Add(48 * time.Hour)}
	der, err := x509.CreateCertificate(rand.Reader, template, template, pub, key)
	if err != nil {
		t.Fatal(err)
	}
	grant, problem := hub.PrepareMachinePublicationGrant("local", "om-proof", der, []string{"alice/z", "alice/a", "alice/z"}, now, now.Add(24*time.Hour))
	if problem != nil {
		t.Fatal(problem)
	}
	if grant.AuthorizationID == "" || grant.Machine != "local" || grant.MachineID != "om-proof" || grant.CertificateDER != base64.RawURLEncoding.EncodeToString(der) || grant.ExpiresAtUnix != now.Add(24*time.Hour).Unix() {
		t.Fatalf("grant changed its exact machine or lifetime: %+v", grant)
	}
	if !reflect.DeepEqual(grant.Repositories, []hub.PublicationRepository{{Org: "alice", Name: "a"}, {Org: "alice", Name: "z"}}) {
		t.Fatalf("scope is not exact and deduplicated: %+v", grant.Repositories)
	}
	for _, names := range [][]string{nil, {"local/model"}, {"alice/model@v1"}, {"alice/../model"}} {
		if _, problem := hub.PrepareMachinePublicationGrant("local", "om-proof", der, names, now, now.Add(time.Hour)); problem == nil {
			t.Fatalf("invalid scope admitted: %v", names)
		}
	}
	for _, until := range []time.Time{now, now.Add(49 * time.Hour), now.Add(8 * 24 * time.Hour)} {
		if _, problem := hub.PrepareMachinePublicationGrant("local", "om-proof", der, []string{"alice/a"}, now, until); problem == nil {
			t.Fatalf("invalid lifetime admitted: %v", until)
		}
	}
	for _, machine := range []struct {
		id   string
		leaf []byte
	}{{"", der}, {"om-proof", []byte("not a certificate")}} {
		if _, problem := hub.PrepareMachinePublicationGrant("local", machine.id, machine.leaf, []string{"alice/a"}, now, now.Add(time.Hour)); problem == nil {
			t.Fatalf("a grant without one machine identity and valid leaf was prepared: %q", machine.id)
		}
	}
}
