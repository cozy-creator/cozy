package producttest

import (
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

// The Hub commits failed, like released, only after proving the pod absent at its provider,
// so the daemon forgets such a rental on its first look: its paid operation closes, its row
// goes, and the Hub is not asked about it again. Failed rentals once stayed on the owner's
// books for weeks, each with an open paid operation that `cozy rental list` showed as "a pod
// may be provisioning and billing", and every poll re-read them all (2026-10-13).
func TestHubEndedRentalsAreForgottenOnFirstLook(t *testing.T) {
	h := newFakeRentalHub(t, 0)
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+"\ntensorhub_token: rental-idle-test\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	ended := map[string]bool{}
	for i := range 30 {
		id := fmt.Sprintf("pr-ended-%04d", i)
		op, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: "ask-" + id, Hub: h.server.URL,
			Reason: "cozy rental new cpu", HourlyRateUSDMicros: 1}, func(name string) ([]byte, string, *exit.Error) {
			body, problem := hub.RentalRequestBytes(name, "cpu", 1, strings.Repeat("ab", 32),
				base64.RawURLEncoding.EncodeToString(make([]byte, 32)), hub.DeclaredWorkload{}, nil, "", "")
			return body, "digest-" + name, problem
		})
		fatal(t, problem)
		fatal(t, store.AdvanceRentalOperation(op.Key, id, "attached"))
		// Half were already recorded failed by an older daemon; half fail now.
		local, remote := "failed", "failed"
		if i%2 == 1 {
			local = "ready"
		}
		if i%3 == 0 {
			remote = "released"
		}
		fatal(t, store.RecordRental(records.Rental{ID: id, MachineName: op.Key, AcceleratorModel: "CPU", AcceleratorCount: 1,
			HourlyRateUSDMicros: 1, State: local, Hub: h.server.URL}))
		h.add(id, op.Key)
		h.setState(id, remote, "readiness.provider_lost")
		ended[id] = true
	}
	store.Close()

	startDaemonProcess(t, root)
	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	deadline := time.Now().Add(60 * time.Second)
	for {
		rows, problem := store.Rentals()
		fatal(t, problem)
		open, problem := store.ActiveRentalOperations()
		fatal(t, problem)
		if len(rows) == 0 && len(open) == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the daemon kept %d ended rentals and %d open paid operations", len(rows), len(open))
		}
		time.Sleep(200 * time.Millisecond)
	}
	if code, out := runCozy(t, root, "rental", "list", "--json", "--full"); code != 0 || strings.Contains(out, "pr-ended-") ||
		strings.Contains(out, "unresolved_acquisitions") {
		t.Fatalf("rental list kept an ended rental [%d]: %s", code, out)
	}
	asked := func() int {
		h.mu.Lock()
		defer h.mu.Unlock()
		n := 0
		for id, reads := range h.byID {
			if ended[id] {
				n += reads
			}
		}
		return n
	}
	before := asked()
	time.Sleep(12 * time.Second) // longer than one Hub reconcile cadence
	if after := asked(); after != before {
		t.Fatalf("the daemon asked the Hub about forgotten rentals %d more times", after-before)
	}
	for id := range ended {
		if h.releases(id) != 0 {
			t.Fatalf("forgetting %s issued a provider release", id)
		}
	}
}
