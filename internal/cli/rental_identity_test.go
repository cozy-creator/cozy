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
			rental.ProviderRefusal = refusal
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
		for field, want := range map[string]string{"provider": "vast", "provider_machine_id": "144163", "provider_resource_id": "54878123",
			"provider_refusal": refusal} {
			got, present := document[field]
			if present != observed || observed && got != want {
				t.Fatalf("ended identity %s: %s", field, encoded.Bytes())
			}
		}
		// The refusal is shown without asking for every field.
		human := &Context{Inv: &Invocation{Mode: output.Mode{Human: true}}}
		encoded.Reset()
		if err := endedRentalRecord(human, "http://127.0.0.1:8819", rental).Emit(&encoded, human.Mode()); err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(encoded.Bytes(), []byte("vast.ai machine 136980")) != observed {
			t.Fatalf("the default ended record: %s", encoded.Bytes())
		}
	}
}

// refusal is how vast.ai's adapter recorded the create the rtx-3090 proof lost (2026-10-13).
const refusal = "provider.no_capacity: vast.ai machine 136980 has no on-demand 1x RTX 3090 ask"

// An unresolved acquisition is not a rental: it neither weakens the accrued total to "at
// least ... est." nor is drawn as a row, and two rentals on one host show their IDs.
func TestRentalBoardTotalsAndSharedHostNames(t *testing.T) {
	unmetered := func(id string) api.RentalSummary {
		return api.RentalSummary{ID: id, MachineName: "otogi", State: "ready", HourlyRateUSDMicros: 3_800_000,
			SpendUSDMicros: 1_000_000, SpendBasis: "unmetered"}
	}
	list := renderRentalList(config.Config{}, api.RentalInventory{MachinesRunning: 2, HourlySpendUSDMicros: 7_600_000,
		Rentals: []api.RentalSummary{unmetered("pr-a"), unmetered("pr-b")},
		Pending: []api.RentalSummary{{ID: "pr-ghost", State: "pending_acquisition", HourlyRateUSDMicros: 2_890_000}}}, false)
	if lead := list.Lead[1]; lead != "Current spend per hour: $7.60 · accrued $2.00 unmetered" {
		t.Fatalf("pending acquisition changed the spend line: %q", lead)
	}
	if len(list.Rows) != 2 || list.Fields[1] != "rental" {
		t.Fatalf("rows %d, fields %v: a shared host name must show each rental's ID", len(list.Rows), list.Fields)
	}
}
