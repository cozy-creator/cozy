package cli

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
)

func TestFullRentalIdentityForLocalOtherClientAndOlderHubs(t *testing.T) {
	for _, otherClient := range []bool{false, true} {
		for _, observed := range []bool{false, true} {
			rental := api.RentalSummary{ID: "pr-proof", MachineName: "morlock", State: "booting"}
			if observed {
				rental.Provider, rental.ProviderMachineID, rental.ProviderResourceID = "vast", "144163", "54878123"
			}
			inventory := api.RentalInventory{}
			if otherClient {
				inventory.Unrecorded = []api.RentalSummary{rental}
			} else {
				inventory.Rentals = []api.RentalSummary{rental}
			}
			list := renderRentalList(config.Config{}, inventory, true)
			var encoded bytes.Buffer
			if err := list.Emit(&encoded, output.Mode{JSON: true, Full: true}); err != nil {
				t.Fatal(err)
			}
			var document struct{ Rentals []map[string]any }
			if err := json.Unmarshal(encoded.Bytes(), &document); err != nil || len(document.Rentals) != 1 {
				t.Fatalf("rental output: %s %v", encoded.Bytes(), err)
			}
			for field, want := range map[string]string{"provider": "vast", "provider_machine_id": "144163", "provider_resource_id": "54878123"} {
				got, present := document.Rentals[0][field]
				if present != observed || observed && got != want {
					t.Fatalf("%s: got %v present=%v observed=%v otherClient=%v", field, got, present, observed, otherClient)
				}
			}
		}
	}
}

func TestFullEndedRentalRetainsObservedProviderIdentity(t *testing.T) {
	for _, observed := range []bool{false, true} {
		rental := hub.Rental{ID: "pr-proof", Name: "morlock", State: "released"}
		if observed {
			rental.Provider, rental.ProviderMachineID, rental.ProviderResourceID = "vast", "144163", "54878123"
		}
		ctx := &Context{Inv: &Invocation{Mode: output.Mode{JSON: true, Full: true}}}
		var encoded bytes.Buffer
		if err := endedRentalRecord(ctx, "http://127.0.0.1:8819", rental).Emit(&encoded, ctx.Mode()); err != nil {
			t.Fatal(err)
		}
		var document map[string]any
		if err := json.Unmarshal(encoded.Bytes(), &document); err != nil {
			t.Fatal(err)
		}
		for field, want := range map[string]string{"provider": "vast", "provider_machine_id": "144163", "provider_resource_id": "54878123"} {
			got, present := document[field]
			if present != observed || observed && got != want {
				t.Fatalf("ended identity %s: %s", field, encoded.Bytes())
			}
		}
	}
}
