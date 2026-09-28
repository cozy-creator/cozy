package producttest

import (
	"encoding/base64"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentalEndKeepsRecordedMachineWhenHubCannotConfirmRelease(t *testing.T) {
	root, hubURL, stand := rentalEndRoot(t, "release-missing-record")
	const id = "pr-3234567890abcdef1234"
	store, key := releaseConfirmationRecord(t, root, hubURL, id)
	code, out := runCozy(t, root, "rental", "end", id, "--json")
	if code == 0 || !strings.Contains(out, "rental.hub_record_missing") || strings.Contains(out, `"state":"ended"`) {
		t.Fatalf("404 was promoted to destruction: exit=%d %s", code, out)
	}
	row, problem := store.RentalRow(id)
	fatal(t, problem)
	op, problem := store.RentalOperation(key)
	fatal(t, problem)
	if row == nil || op.State == "released" || stand.releases(id) != 0 {
		t.Fatal("unconfirmed rental or operation was forgotten/released")
	}
}

func releaseConfirmationRecord(t *testing.T, root, hubURL, id string) (*records.Store, string) {
	t.Helper()
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	machine := ""
	op, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: "release-confirmation", Hub: hubURL, Reason: "cozy rental new cpu", HourlyRateUSDMicros: 100000}, func(name string) ([]byte, string, *exit.Error) {
		machine = name
		body, e := hub.RentalRequestBytes(name, "cpu", 1, strings.Repeat("ab", 32), base64.RawURLEncoding.EncodeToString(make([]byte, 32)), hub.DeclaredWorkload{}, nil, "")
		return body, "digest-" + name, e
	})
	fatal(t, problem)
	fatal(t, store.AdvanceRentalOperation(op.Key, id, "attached"))
	fatal(t, store.RecordRental(records.Rental{ID: id, MachineName: machine, SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, State: "ready", ReadyAt: time.Now().Add(-time.Hour).UTC().Format(time.RFC3339Nano), Hub: hubURL, Address: "127.0.0.1:1", CertPath: filepath.Join(root, id+".pem")}))
	return store, op.Key
}
