package producttest

import (
	"encoding/base64"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
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

func TestManagedReleaseRequiresPositiveHubDestruction(t *testing.T) {
	for _, phase := range []string{"delete", "observe"} {
		t.Run(phase, func(t *testing.T) {
			root, hubURL, stand := rentalEndRoot(t, "managed-release-"+phase)
			const id = "pr-4234567890abcdef1234"
			store, key := releaseConfirmationRecord(t, root, hubURL, id)
			row, problem := store.RentalRow(id)
			fatal(t, problem)
			stand.add(id, row.MachineName)
			original := stand.server.Config.Handler
			var asked atomic.Bool
			stand.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/rentals/"+id {
					if r.Method == http.MethodDelete {
						asked.Store(true)
						if phase == "delete" {
							http.NotFound(w, r)
							return
						}
					}
					if r.Method == http.MethodGet && phase == "observe" && asked.Load() {
						http.NotFound(w, r)
						return
					}
				}
				original.ServeHTTP(w, r)
			})
			startDaemonProcess(t, root)
			deadline := time.Now().Add(15 * time.Second)
			for time.Now().Before(deadline) {
				current, p := store.RentalRow(id)
				fatal(t, p)
				log, _ := os.ReadFile(filepath.Join(root, "daemon.log"))
				if asked.Load() && (current == nil || strings.Contains(string(log), "has not confirmed release")) {
					break
				}
				time.Sleep(25 * time.Millisecond)
			}
			current, p := store.RentalRow(id)
			fatal(t, p)
			op, p := store.RentalOperation(key)
			fatal(t, p)
			log, _ := os.ReadFile(filepath.Join(root, "daemon.log"))
			if !asked.Load() || current == nil || op.State == "released" || !strings.Contains(string(log), "has not confirmed release") {
				t.Fatalf("%s404 was treated as destruction: requested=%v rental=%+v operation=%+v\n%s", phase, asked.Load(), current, op, log)
			}
		})
	}
}

func releaseConfirmationRecord(t *testing.T, root, hubURL, id string) (*records.Store, string) {
	t.Helper()
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	machine := ""
	op, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: "release-confirmation", Hub: hubURL, Reason: "cozy rental new cpu", HourlyRateUSDMicros: 100000}, 5000000, 10000, func(name string) ([]byte, string, *exit.Error) {
		machine = name
		body, e := hub.RentalRequestBytes(name, "cpu", strings.Repeat("ab", 32), base64.RawURLEncoding.EncodeToString(make([]byte, 32)), hub.DeclaredWorkload{}, nil)
		return body, "digest-" + name, e
	}, nil)
	fatal(t, problem)
	fatal(t, store.AdvanceRentalOperation(op.Key, id, "attached"))
	fatal(t, store.RecordRental(records.Rental{ID: id, MachineName: machine, SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, State: "ready", ReadyAt: time.Now().Add(-time.Hour).UTC().Format(time.RFC3339Nano), Hub: hubURL, Address: "127.0.0.1:1", CertPath: filepath.Join(root, id+".pem")}))
	return store, op.Key
}
