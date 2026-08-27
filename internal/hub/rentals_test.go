package hub

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

func TestRentalRequestBytesAreProviderNeutralAndStable(t *testing.T) {
	hash := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	got, e := RentalRequestBytes(" acme/h3/v1/generate ", " NVIDIA H200 ", "sha256:"+hash)
	if e != nil {
		t.Fatal(e)
	}
	want := []byte(`{"endpoint_ref":"acme/h3/v1/generate","accelerator_model":"NVIDIA H200","renter_token_sha256":["` + hash + `"]}`)
	if !bytes.Equal(got, want) {
		t.Fatalf("request bytes\n got: %s\nwant: %s", got, want)
	}
	for _, forbidden := range [][]byte{
		[]byte(`"provider"`), []byte(`"region"`), []byte(`"datacenter"`),
		[]byte(`"image"`), []byte(`"volume"`), []byte(`"pod_id"`),
	} {
		if bytes.Contains(got, forbidden) {
			t.Fatalf("provider placement field %s leaked into %s", forbidden, got)
		}
	}
}

func TestRentalViewCarriesABoundedControlSnapshotBeyondTheCatalogCap(t *testing.T) {
	want := bytes.Repeat([]byte{'x'}, maxBody+1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(wireRental{ID: "rnt-control", State: RentalReady,
			WorkerAddress: "pod.invalid:443", MediaAddress: "pod.invalid:444", CertPEM: "cert",
			TokenSHA256: []string{strings.Repeat("a", 64)}, ControlSnapshot: &ExactControlDocument{
				CanonicalBytes: want, Digest: "sha256:" + strings.Repeat("b", 64), Length: int64(len(want)),
			}})
	}))
	defer server.Close()
	c := &Client{base: server.URL, token: secret.New("admin"), http: server.Client()}
	got, e := c.Rental(t.Context(), "rnt-control")
	if e != nil || got.ControlSnapshot == nil || !bytes.Equal(got.ControlSnapshot.CanonicalBytes, want) {
		t.Fatalf("large rental control view = %#v, %v", got.ControlSnapshot, e)
	}
}

func TestRentalIDIsPortableOpaqueName(t *testing.T) {
	for _, id := range []string{"rnt-abc_123", "pi-0123456789abcdef", "rental.7"} {
		if e := validateRentalID(id); e != nil {
			t.Fatalf("%q refused: %v", id, e)
		}
	}
	for _, id := range []string{
		"", "../outside", `..\outside`, "%2e%2e", "a/b", "a:b", "CON", "nul.txt", "LPT1.log",
	} {
		if e := validateRentalID(id); e == nil {
			t.Fatalf("unsafe rental id %q was accepted", id)
		}
	}
}
