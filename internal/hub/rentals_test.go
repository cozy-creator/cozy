package hub

import (
	"bytes"
	"testing"
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
