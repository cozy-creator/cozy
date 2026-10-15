package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// A rental name resolves only on the Hub --tensorhub names. Without it, a name that live
// rentals on two Hubs share is refused, naming both. The Klein bench's `--rental=nayuta
// --tensorhub tensorhub` instead ran on another session's CPU nayuta on the local Hub.
func TestARentalNameResolvesOnItsHubAndIsRefusedAcrossHubs(t *testing.T) {
	root, _, standA := rentalEndRoot(t, "rental-name-hubs")
	const onA, onB = "pr-a5a5a5a5a5a5a5a5a5a5", "pr-b5b5b5b5b5b5b5b5b5b5"
	standA.add(onA, "nayuta")
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	gpu := map[string]any{"rental_id": onB, "name": "nayuta", "state": "ready",
		"requested_accelerator_model": "NVIDIA GeForce RTX 4090", "accelerator_count": 1, "hourly_rate_usd_micros": 1_400_000}
	hubB := httptest.NewServer(machineKeyLogin("machine-b", public, "token-b", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Header.Get("Authorization") != "Bearer token-b":
			w.WriteHeader(http.StatusUnauthorized)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rentals":
			_ = json.NewEncoder(w).Encode(map[string]any{"rentals": []any{gpu}})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rentals/"+onB:
			_ = json.NewEncoder(w).Encode(gpu)
		default:
			http.NotFound(w, r)
		}
	})))
	defer hubB.Close()
	plantMachineKey(t, root, hubB.URL, "machine-b", private)
	configPath := filepath.Join(root, config.FileName)
	written, err := os.ReadFile(configPath)
	must(t, err)
	must(t, os.WriteFile(configPath, append(written, []byte("hubs:\n  second: "+hubB.URL+"\n")...), 0o600))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down") })

	// At the Hubs: each answers for its own nayuta.
	if code, out := runCozy(t, root, "rental", "show", "nayuta", "--tensorhub=second", "--json"); code != 0 || !strings.Contains(out, onB) || strings.Contains(out, onA) {
		t.Fatalf("--tensorhub=second did not resolve its own nayuta [%d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "rental", "show", "nayuta", "--json"); code == 0 ||
		!strings.Contains(out, "rental.machine_name_ambiguous") || !strings.Contains(out, onA) || !strings.Contains(out, onB) {
		t.Fatalf("a name two Hubs share was not refused naming both [%d]: %s", code, out)
	}

	// Work sent to nayuta goes to the named Hub's rental, never the other.
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	code, out := runCozy(t, root, "package", "install", "proof/anything", "--rental=nayuta", "--tensorhub=second", "--no-wait", "--json")
	var sent struct{ ID string }
	if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &sent) != nil || sent.ID == "" {
		t.Fatalf("the install was not sent [%d]: %s", code, out)
	}
	link, problem := store.MachineExecution(sent.ID)
	fatal(t, problem)
	if link == nil || link.MachineID != onB {
		t.Fatalf("work for --rental=nayuta --tensorhub=second went to %+v, not %s", link, onB)
	}
	if code, out := runCozy(t, root, "package", "install", "proof/anything", "--rental=nayuta", "--no-wait", "--json"); code == 0 ||
		!strings.Contains(out, "rental.machine_name_ambiguous") || !strings.Contains(out, onA) || !strings.Contains(out, onB) {
		t.Fatalf("work for a name two Hubs share was not refused naming both [%d]: %s", code, out)
	}
}
