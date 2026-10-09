package hub

import (
	"encoding/json"
	"testing"
)

func TestRentalIdentityFieldsAreOptionalAcrossHubVersions(t *testing.T) {
	for _, payload := range []string{
		`{"rental_id":"pr-proof","name":"morlock","state":"released"}`,
		`{"rental_id":"pr-proof","name":"morlock","state":"released","provider":"vast","provider_machine_id":"144163","provider_resource_id":"54878123","future_fact":true}`,
	} {
		var wire wireRental
		if err := json.Unmarshal([]byte(payload), &wire); err != nil {
			t.Fatal(err)
		}
		got := wire.rental()
		if got.ID != "pr-proof" || got.Name != "morlock" || got.State != "released" {
			t.Fatalf("additive identities changed the existing rental contract: %+v", got)
		}
		if wire.Provider == "" {
			if got.Provider != "" || got.ProviderMachineID != "" || got.ProviderResourceID != "" {
				t.Fatal("older Hub response acquired invented provider identity")
			}
		} else if got.Provider != "vast" || got.ProviderMachineID != "144163" || got.ProviderResourceID != "54878123" {
			t.Fatalf("observed identity was lost: %+v", got)
		}
	}
}
