package producttest

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// Tensorhub never replaces a machine its provider stopped (owner, 2026-10-14): the rental is
// degraded with readiness.provider_stopped until the provider resumes it or it idles out.
// The listing says so and what to do, a later stop observation with other provider facts
// does not stall Calcifer's reconcile, and new work buys around the stopped machine.
func TestRentalStoppedByProviderIsShownAndBoughtAround(t *testing.T) {
	h := newFakeRentalHub(t, 0)
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+"\ntensorhub_token: rental-idle-test\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	const id = "pr-stopped-0001"
	op, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: "ask-" + id, Hub: h.server.URL,
		Reason: "cozy rental new cpu", HourlyRateUSDMicros: 100_000}, func(name string) ([]byte, string, *exit.Error) {
		body, problem := hub.RentalRequestBytes(name, "cpu", 1, strings.Repeat("ab", 32),
			base64.RawURLEncoding.EncodeToString(make([]byte, 32)), hub.DeclaredWorkload{}, nil, "", "")
		return body, "digest-" + name, problem
	})
	fatal(t, problem)
	fatal(t, store.AdvanceRentalOperation(op.Key, id, "attached"))
	// An earlier look recorded the first stop with fewer provider facts than the Hub now states.
	fatal(t, store.RecordRental(records.Rental{ID: id, MachineName: op.Key, SKU: "cpu", AcceleratorModel: "CPU",
		AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, State: "degraded", Hub: h.server.URL,
		Failure: records.RentalFailure{Code: hub.RentalProviderStopped, ContainerState: "stopped"}}))
	store.Close()
	h.add(id, op.Key)
	h.setState(id, "degraded", hub.RentalProviderStopped)
	h.mu.Lock()
	h.rentals[id]["failure"] = map[string]any{"code": hub.RentalProviderStopped, "provider": "runpod",
		"provider_resource_id": "pod-7", "provider_state": "stopped", "container_state": "exited"}
	h.mu.Unlock()
	h.setRate(id, 120_000)

	startCalciferProcess(t, root)
	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for deadline := time.Now().Add(60 * time.Second); ; time.Sleep(200 * time.Millisecond) {
		row, problem := store.RentalRow(id)
		fatal(t, problem)
		if row != nil && row.HourlyRateUSDMicros == 120_000 {
			if row.Failure.Code != hub.RentalProviderStopped || row.Failure.ContainerState != "stopped" {
				t.Fatalf("the first recorded stop was rewritten: %+v", row.Failure)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("Calcifer did not reconcile the stopped rental: %+v", row)
		}
	}
	for _, args := range [][]string{{"rental", "list"}, {"rental", "show", id}} {
		code, out := runCozy(t, root, args...)
		if code != 0 || !strings.Contains(out, "stopped by its provider") || !strings.Contains(out, "cozy rental new cpu") {
			t.Fatalf("cozy %s does not say the provider stopped it [%d]: %s", strings.Join(args, " "), code, out)
		}
	}
	if code, out := runCozy(t, root, "rental", "list"); code != 0 || !strings.Contains(out, "stopped by provider") {
		t.Fatalf("the state column does not say stopped by provider [%d]: %s", code, out)
	}

	for _, tc := range []struct {
		state, code string
		buy         bool
	}{{"degraded", hub.RentalProviderStopped, true}, {"booting", "", false}} {
		candidate := orchestrator.PlacementCandidate{GPUs: 1}
		row := records.Rental{AcceleratorModel: "NVIDIA H100", AcceleratorCount: 1, State: tc.state,
			Failure: records.RentalFailure{Code: tc.code}}
		rental.Standing(&candidate, []records.ModelRef{{Manifest: "weights"}}, row,
			80, true, true, false, rental.Constraints{}, rental.Disk{})
		if bought := strings.HasPrefix(candidate.Verdict, orchestrator.VerdictExcluded); bought != tc.buy {
			t.Fatalf("%s %s: verdict %q, want buy-around=%t", tc.state, tc.code, candidate.Verdict, tc.buy)
		}
	}
}
