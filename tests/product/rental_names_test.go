package producttest

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rentalid"
)

func TestRentalBookingAliasIsOpaqueAndReplayPreservesIt(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	author := func(name string) ([]byte, string, *exit.Error) {
		if !strings.HasPrefix(name, "lease-") || !rentalid.ValidMachineName(name) {
			t.Fatalf("booking invented a host name: %q", name)
		}
		body, _ := json.Marshal(map[string]string{"name": name})
		return body, "digest", nil
	}
	first, replay, problem := store.BeginRentalOperation(records.RentalOperation{Key: "first", Hub: "https://hub.invalid", HourlyRateUSDMicros: 1}, author)
	fatal(t, problem)
	if replay {
		t.Fatal("new operation replayed")
	}
	again, replay, problem := store.BeginRentalOperation(records.RentalOperation{Key: "first"}, func(string) ([]byte, string, *exit.Error) {
		t.Fatal("replay minted another alias")
		return nil, "", nil
	})
	fatal(t, problem)
	if !replay || string(first.RequestBody) != string(again.RequestBody) {
		t.Fatal("replay changed its original request")
	}
	second, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: "second", Hub: "https://hub.invalid", HourlyRateUSDMicros: 1}, author)
	fatal(t, problem)
	if string(first.RequestBody) == string(second.RequestBody) {
		t.Fatal("independent bookings shared an alias")
	}
}
