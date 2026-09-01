package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestManagedRentalHubRecordLossFailsClosed(t *testing.T) {
	const rentalID = "pr-lost-hub-record"
	var posts atomic.Int64
	hubServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rentals/"+rentalID:
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"code":"rental.not_found","message":"no rental","remedy":"reconcile provider state"}}`))
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rental-skus":
			_ = json.NewEncoder(w).Encode([]map[string]any{{
				"name": "h200", "accelerator_model": "H200", "compute_capability": "9.0",
				"vram_gb": 141, "minimum_ram_per_gpu_gb": 128,
				"price_usd_micros_per_hour": 6_000_000,
			}})
		case r.Method == http.MethodPost && r.URL.Path == "/v1/rentals":
			posts.Add(1)
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"rental_id": "pr-wrong-replacement", "state": "failed",
				"requested_accelerator_model": "H200", "hourly_rate_usd_micros": 6_000_000,
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer hubServer.Close()

	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"), []byte(
		"tensorhub_url: "+hubServer.URL+"\n"+
			"tensorhub_token: proof-token\nport: 0\n"+
			"rentals:\n  max_hourly_spend_usd: 10.00\n"), 0o600))
	t.Cleanup(func() { terminateTestDaemon(t, root) })
	project := weightlessProject(t)
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("fixture install [exit %d]\n%s", code, out)
	}

	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	want := records.Rental{
		ID: rentalID, MachineName: "lost-hub-record", SKU: "h200",
		AcceleratorModel: "H200", HourlyRateUSDMicros: 6_000_000,
		ManagedRequestID: "req-original", State: "ready", Hub: hubServer.URL,
	}
	const operationKey = "managed-rental-req-original"
	if _, replay, problem := store.BeginRentalOperation(records.RentalOperation{
		Key: operationKey, RequestDigest: "sha256:durable-intent", RequestBody: []byte(`{}`),
		Hub: hubServer.URL, HourlyRateUSDMicros: want.HourlyRateUSDMicros,
		ManagedRequestID: want.ManagedRequestID,
	}, 10_000_000); problem != nil || replay {
		t.Fatalf("record rental operation: replay=%v problem=%v", replay, problem)
	}
	fatal(t, store.AdvanceRentalOperation(operationKey, want.ID, "attached"))
	fatal(t, store.RecordRental(want))
	credentials := map[string][]byte{
		layout.RentalMediaToken(want.ID):      []byte("media-token\n"),
		layout.RentalCreatorIdentity(want.ID): []byte("creator-private-key\n"),
		layout.RentalCert(want.ID):            []byte("pinned-certificate\n"),
	}
	must(t, os.MkdirAll(layout.Rentals, 0o700))
	for path, body := range credentials {
		must(t, os.WriteFile(path, body, 0o600))
	}
	store.Close()

	if code, out := runCozy(t, root, "rental", "--json"); code != 1 ||
		!strings.Contains(out, "rental.hub_record_missing") ||
		!strings.Contains(out, "reconcile Tensorhub with its provider") {
		t.Fatalf("Hub record loss was not a typed conflict [exit %d]\n%s", code, out)
	}
	code, out := runCozy(t, root, "run", localWeightlessRef+"/tile", "size=32", "seed=7",
		"--rental-only", "--idempotency-key", "db-loss-replacement", "--json")
	if code != 1 || !strings.Contains(out, "rental.hub_record_missing") {
		t.Fatalf("subsequent rental acquisition did not fail closed [exit %d]\n%s", code, out)
	}

	store, problem = records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	stored, problem := store.RentalRow(want.ID)
	if problem != nil || stored == nil || stored.State != want.State ||
		stored.ManagedRequestID != want.ManagedRequestID {
		t.Fatalf("durable rental after Hub 404 = %+v, %v", stored, problem)
	}
	operation, problem := store.RentalOperation(operationKey)
	if problem != nil || operation == nil || operation.State != "attached" || operation.RentalID != want.ID {
		t.Fatalf("durable rental operation after Hub 404 = %+v, %v", operation, problem)
	}
	count, burn, problem := store.RentalFleetTotals()
	if problem != nil || count != 1 || burn != want.HourlyRateUSDMicros {
		t.Fatalf("fleet reservation after Hub 404 = %d/%d, %v", count, burn, problem)
	}
	request, problem := store.RequestByIdempotencyKey("db-loss-replacement")
	if problem != nil || request == nil || request.Worker != "" {
		t.Fatalf("replacement request after Hub 404 = %+v, %v", request, problem)
	}
	events, problem := store.EventsAfter(request.ID, 0, 100)
	if problem != nil || !eventHasError(events, "rental.hub_record_missing") {
		t.Fatalf("replacement request events after Hub 404 = %+v, %v", events, problem)
	}
	for path, wantBody := range credentials {
		got, err := os.ReadFile(path)
		if err != nil || !bytes.Equal(got, wantBody) {
			t.Fatalf("credential %s after Hub 404 = %q, %v", filepath.Base(path), got, err)
		}
	}
	if got := posts.Load(); got != 0 {
		t.Fatalf("Hub record loss allowed %d replacement rental POSTs", got)
	}
}
