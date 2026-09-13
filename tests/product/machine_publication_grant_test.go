package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/pem"
	"math/big"
	"reflect"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/hub"
)

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
	rental := hub.Rental{ID: "pr-proof", WorkerID: "worker-proof", State: hub.RentalReady, CertPEM: string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}))}
	grant, problem := hub.PrepareMachinePublicationGrant(rental, []string{"alice/z", "alice/a", "alice/z"}, now, now.Add(24*time.Hour))
	if problem != nil {
		t.Fatal(problem)
	}
	if grant.AuthorizationID == "" || grant.RentalID != rental.ID || grant.CertificateDER != base64.RawURLEncoding.EncodeToString(der) || grant.ExpiresAtUnix != now.Add(24*time.Hour).Unix() {
		t.Fatalf("grant changed its exact machine or lifetime: %+v", grant)
	}
	if !reflect.DeepEqual(grant.Repositories, []hub.PublicationRepository{{Org: "alice", Name: "a"}, {Org: "alice", Name: "z"}}) {
		t.Fatalf("scope is not exact and deduplicated: %+v", grant.Repositories)
	}
	for _, names := range [][]string{nil, {"local/model"}, {"alice/model@v1"}, {"alice/../model"}} {
		if _, problem := hub.PrepareMachinePublicationGrant(rental, names, now, now.Add(time.Hour)); problem == nil {
			t.Fatalf("invalid scope admitted: %v", names)
		}
	}
	for _, until := range []time.Time{now, now.Add(49 * time.Hour), now.Add(8 * 24 * time.Hour)} {
		if _, problem := hub.PrepareMachinePublicationGrant(rental, []string{"alice/a"}, now, until); problem == nil {
			t.Fatalf("invalid lifetime admitted: %v", until)
		}
	}
	rental.State = hub.RentalReleased
	if _, problem := hub.PrepareMachinePublicationGrant(rental, []string{"alice/a"}, now, now.Add(time.Hour)); problem == nil {
		t.Fatal("released rental obtained new publication authority")
	}
}
