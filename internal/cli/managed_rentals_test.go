package cli

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestRentalFleetUsesLockedRatesAndReusesCheapestIdleMachine(t *testing.T) {
	layout, store := rentalTestStore(t)
	for _, row := range []records.Rental{
		{ID: "rental-expensive", MachineName: "expensive", SKU: "gpu-large",
			AcceleratorModel: "GPU", HourlyRateUSDMicros: 900_000,
			State: "failed", Hub: "https://another-hub.invalid"},
		{ID: "rental-cheap", MachineName: "cheap", SKU: "gpu-small",
			AcceleratorModel: "GPU", HourlyRateUSDMicros: 250_000,
			State: "ready", Hub: "https://another-hub.invalid"},
	} {
		if problem := store.RecordRental(row); problem != nil {
			t.Fatal(problem)
		}
	}
	request := records.Request{
		ID: "req-rental-reuse", IdemKey: "rental-reuse", BodyDigest: "sha256:body",
		Package: "cozy/test", Entrypoint: "run", PlanID: "sha256:plan",
		Payload: []byte("{}"), Rental: true,
	}
	if _, _, problem := store.Submit(request); problem != nil {
		t.Fatal(problem)
	}
	ctx := &Context{Inv: &Invocation{}, Out: &bytes.Buffer{}, Err: &bytes.Buffer{}, Cfg: config.Config{
		Home: layout.Root, HubURL: "http://127.0.0.1:1", RentalsMaxHourlySpendUSDMicros: 2_000_000,
	}}
	fleet := &managedRentals{ctx: ctx, layout: layout, store: store}
	line, problem := fleet.status()
	if problem != nil {
		t.Fatal(problem)
	}
	if line != "rentals: 2 remote machines running · $1.15/hour of $2/hour" {
		t.Fatalf("fleet line = %q", line)
	}
	id, after, problem := fleet.acquire(request)
	if problem != nil {
		t.Fatal(problem)
	}
	if id != "rental-cheap" || after != line {
		t.Fatalf("reuse = %q, %q", id, after)
	}
	stored, problem := store.RequestRow(request.ID)
	if problem != nil || stored == nil || stored.Worker != id {
		t.Fatalf("request assignment = %+v, %v", stored, problem)
	}
}

func TestRentalFleetRefusesPurchaseBeyondCeilingBeforePOST(t *testing.T) {
	posts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/rental-skus":
			_ = json.NewEncoder(w).Encode([]map[string]any{{
				"name": "gpu", "accelerator_model": "GPU", "compute_capability": "9.0",
				"vram_gb": 80, "price_usd_micros_per_hour": 750_000,
			}})
		case "/v1/rentals":
			posts++
			http.Error(w, "unexpected paid POST", http.StatusInternalServerError)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	layout, store := rentalTestStore(t)
	request := records.Request{
		ID: "req-rental-cap", IdemKey: "rental-cap", BodyDigest: "sha256:body",
		Package: "cozy/test", Entrypoint: "run", PlanID: "sha256:plan",
		Payload: []byte("{}"), Rental: true,
	}
	if _, _, problem := store.Submit(request); problem != nil {
		t.Fatal(problem)
	}
	ctx := &Context{Inv: &Invocation{}, Out: &bytes.Buffer{}, Err: &bytes.Buffer{}, Cfg: config.Config{
		Home: layout.Root, HubURL: server.URL, RentalsMaxHourlySpendUSDMicros: 500_000,
	}}
	_, _, problem := (&managedRentals{ctx: ctx, layout: layout, store: store}).acquire(request)
	if problem == nil || problem.ErrName() != "rental.fleet_spend_cap" || posts != 0 {
		t.Fatalf("cap refusal = %v, paid POSTs = %d", problem, posts)
	}
}

func TestManagedRentalAcquiresCheapestSKUAndLocksReturnedRate(t *testing.T) {
	posts := 0
	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rental-skus":
			_ = json.NewEncoder(w).Encode([]map[string]any{
				{"name": "expensive", "accelerator_model": "GPU X", "compute_capability": "9.0", "vram_gb": 80, "price_usd_micros_per_hour": 900_000},
				{"name": "cheap", "accelerator_model": "GPU C", "compute_capability": "8.9", "vram_gb": 24, "price_usd_micros_per_hour": 300_000},
			})
		case r.Method == http.MethodPost && r.URL.Path == "/v1/rentals":
			posts++
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Error(err)
			}
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"rental_id": "pr-managed-proof", "state": "pending_acquisition",
				"requested_accelerator_model": "GPU C", "hourly_rate_usd_micros": 300_000,
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rentals/pr-managed-proof":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"rental_id": "pr-managed-proof", "state": "ready",
				"requested_accelerator_model": "GPU C", "hourly_rate_usd_micros": 300_000,
				"worker_address": "127.0.0.1:9443", "media_address": "127.0.0.1:9444",
				"worker_id": "worker-proof", "worker_boot_id": "boot-proof", "cert_pem": "test-cert",
				"creator_public_key": request["creator_public_key"],
				"media_token_sha256": []any{request["media_token_sha256"]},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	layout, store := rentalTestStore(t)
	requested := records.Request{
		ID: "req-managed-proof", IdemKey: "managed-proof", BodyDigest: "sha256:body",
		Package: "cozy/test", Entrypoint: "run", PlanID: "sha256:plan",
		Payload: []byte("{}"), Rental: true,
	}
	if _, _, problem := store.Submit(requested); problem != nil {
		t.Fatal(problem)
	}
	ctx := &Context{Inv: &Invocation{}, Out: &bytes.Buffer{}, Err: &bytes.Buffer{}, Cfg: config.Config{
		Home: layout.Root, HubURL: server.URL, HubToken: secret.New("test-token"),
		RentalsMaxHourlySpendUSDMicros: 1_000_000,
	}}
	id, line, problem := (&managedRentals{ctx: ctx, layout: layout, store: store}).acquire(requested)
	if problem != nil {
		t.Fatal(problem)
	}
	if posts != 1 || request["sku"] != "cheap" || id != "pr-managed-proof" ||
		line != "rentals: 1 remote machines running · $0.3/hour of $1/hour" {
		t.Fatalf("acquisition posts=%d request=%v id=%q line=%q", posts, request, id, line)
	}
	row, problem := store.RentalRow(id)
	if problem != nil || row == nil || row.HourlyRateUSDMicros != 300_000 ||
		row.ManagedRequestID != requested.ID {
		t.Fatalf("locked rental row = %+v, %v", row, problem)
	}
}

func TestManualRentalIsNeverAutoReleased(t *testing.T) {
	layout, store := rentalTestStore(t)
	row := records.Rental{
		ID: "rental-manual", MachineName: "manual", SKU: "gpu",
		AcceleratorModel: "GPU", HourlyRateUSDMicros: 100_000,
		State: "ready", Hub: "https://another-hub.invalid",
	}
	if problem := store.RecordRental(row); problem != nil {
		t.Fatal(problem)
	}
	ctx := &Context{Inv: &Invocation{}, Out: &bytes.Buffer{}, Err: &bytes.Buffer{}, Cfg: config.Config{
		Home: layout.Root, HubURL: "http://127.0.0.1:1", RentalsMaxHourlySpendUSDMicros: 1_000_000,
	}}
	line, problem := (&managedRentals{ctx: ctx, layout: layout, store: store}).release(row.ID)
	if problem != nil || !strings.Contains(line, "1 remote machines") {
		t.Fatalf("manual release decision = %q, %v", line, problem)
	}
	if kept, problem := store.RentalRow(row.ID); problem != nil || kept == nil {
		t.Fatalf("manual rental was auto-released: %+v, %v", kept, problem)
	}
}

func TestManagedRentalReleasesOnlyAfterItsAssignedQueueDrains(t *testing.T) {
	deletes := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodDelete && r.URL.Path == "/v1/rentals/pr-release-proof":
			deletes++
			w.WriteHeader(http.StatusNoContent)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rentals/pr-release-proof":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"rental_id": "pr-release-proof", "state": "released",
				"requested_accelerator_model": "GPU", "hourly_rate_usd_micros": 200_000,
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	layout, store := rentalTestStore(t)
	row := records.Rental{
		ID: "pr-release-proof", MachineName: "release-proof", SKU: "gpu",
		AcceleratorModel: "GPU", HourlyRateUSDMicros: 200_000,
		ManagedRequestID: "req-origin", State: "ready", Hub: server.URL,
	}
	if problem := store.RecordRental(row); problem != nil {
		t.Fatal(problem)
	}
	ctx := &Context{Inv: &Invocation{}, Out: &bytes.Buffer{}, Err: &bytes.Buffer{}, Cfg: config.Config{
		Home: layout.Root, HubURL: server.URL, HubToken: secret.New("test-token"),
		RentalsMaxHourlySpendUSDMicros: 1_000_000,
	}}
	line, problem := (&managedRentals{ctx: ctx, layout: layout, store: store}).release(row.ID)
	if problem != nil || deletes != 1 || line != "rentals: 0 remote machines running · $0/hour of $1/hour" {
		t.Fatalf("managed release deletes=%d line=%q problem=%v", deletes, line, problem)
	}
	if kept, problem := store.RentalRow(row.ID); problem != nil || kept != nil {
		t.Fatalf("released managed rental remains: %+v, %v", kept, problem)
	}
}

func rentalTestStore(t *testing.T) (home.Layout, *records.Store) {
	t.Helper()
	layout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(filepath.Join(layout.Root, "records.db"))
	if problem != nil {
		t.Fatal(problem)
	}
	t.Cleanup(func() { store.Close() })
	return layout, store
}
