package producttest

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// The real daemon reaps an overdue owned rental through the Hub and leaves a
// freshly ready one alive. Boundary timing is covered by the clock-driven tests.
func TestRentalIdleRelease(t *testing.T) {
	root := t.TempDir()
	peer := newFakeRentalHub(t, 0)
	peer.publishListing()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+peer.server.URL+"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for _, item := range []struct {
		id, machine string
		ready       time.Time
	}{{"rental-idle-old", "heron", time.Now().Add(-16 * time.Minute)}, {"rental-idle-new", "otter", time.Now()}, {"rental-idle-retained", "curlew", time.Now().Add(-time.Hour)}} {
		peer.add(item.id, item.machine)
		fatal(t, store.RecordRental(records.Rental{ID: item.id, MachineName: item.machine, SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, State: "ready", Hub: peer.server.URL, Address: "127.0.0.1:1", CertPath: filepath.Join(root, item.id+".pem"), ReadyAt: item.ready.UTC().Format(time.RFC3339Nano)}))
	}
	retained := recordPrivateTransaction(t, store, "idle-expiry", "rental-idle-retained")
	changed, problem := store.BlockRetainedWork(retained.ID, "fixture", "retained bytes are not active work")
	fatal(t, problem)
	if !changed {
		t.Fatal("retained fixture did not settle")
	}
	db, err := sql.Open("sqlite", filepath.Join(root, "creator.sqlite"))
	must(t, err)
	_, err = db.Exec(`UPDATE request_events SET at=? WHERE request_id=? AND type='request.blocked'`, time.Now().Add(-16*time.Minute).UTC().Format(time.RFC3339Nano), retained.ID)
	must(t, err)
	must(t, db.Close())
	startDaemonProcess(t, root)
	awaitRentalGone(t, store, "rental-idle-retained", 20*time.Second, filepath.Join(root, "daemon.log"))
	if peer.releases("rental-idle-retained") != 1 {
		t.Fatal("retained custody vetoed idle expiry")
	}
	awaitRentalGone(t, store, "rental-idle-old", 20*time.Second, filepath.Join(root, "daemon.log"))
	if peer.releases("rental-idle-old") != 1 || peer.releases("rental-idle-new") != 0 {
		t.Fatal("fixed idle sweep released the wrong rental")
	}
	fresh, problem := store.RentalRow("rental-idle-new")
	fatal(t, problem)
	if fresh == nil {
		t.Fatal("fresh rental disappeared")
	}
}

func awaitRentalGone(t *testing.T, store *records.Store, id string, within time.Duration, logPath string) {
	t.Helper()
	deadline := time.Now().Add(within)
	for time.Now().Before(deadline) {
		row, problem := store.RentalRow(id)
		fatal(t, problem)
		if row == nil {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("rental %s was not released within %s\n%s", id, within, tail(logPath))
}

func awaitLog(t *testing.T, logPath, substr string, within time.Duration) {
	t.Helper()
	deadline := time.Now().Add(within)
	for time.Now().Before(deadline) {
		if log, _ := os.ReadFile(logPath); strings.Contains(string(log), substr) {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("the daemon log never said %q within %s\n%s", substr, within, tail(logPath))
}

// reservePort finds a loopback port the test can bind later, so a hub can be absent and
// then present at the one address the daemon was configured with.
func reservePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow the test reserves the loopback port its stand-in hub will answer on; the product binds through internal/api
	must(t, err)
	port := l.Addr().(*net.TCPAddr).Port
	must(t, l.Close())
	return port
}

// fakeRentalHub answers the two rental routes the idle release uses — read one rental,
// release one rental — the way Tensorhub does: a DELETE moves the pod to `released`, and
// every later read says so.
type fakeRentalHub struct {
	mu              sync.Mutex
	rentals         map[string]map[string]any
	inventories     map[string]json.RawMessage
	packageReleases map[string]any
	skus            []map[string]any
	rent            func(map[string]any) map[string]any
	// spendCap stands in for Tensorhub's owner fleet cap: a paid ask whose SKU
	// total would take the live rentals' burn past it is refused, buying nothing.
	spendCap int64
	released map[string]int
	server   *httptest.Server
	// publishes is whether this stand-in hub carries th-199's account listing.
	publishes bool
	// reads counts rental reads, so a test can plant a transition after the daemon's first look.
	reads int
}

func (h *fakeRentalHub) rentalReads() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.reads
}

func newFakeRentalHub(t *testing.T, port int) *fakeRentalHub {
	t.Helper()
	h := &fakeRentalHub{rentals: map[string]map[string]any{}, released: map[string]int{}, publishes: true}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/packages/{owner}/{name}/releases/{release}", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		release, ok := h.packageReleases[r.PathValue("owner")+"/"+r.PathValue("name")+"@"+r.PathValue("release")]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_ = json.NewEncoder(w).Encode(release)
	})
	mux.HandleFunc("POST /v1/packages/{owner}/{name}/download", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		pkg, release := r.PathValue("owner")+"/"+r.PathValue("name"), r.URL.Query().Get("release")
		if _, ok := h.packageReleases[pkg+"@"+release]; !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_ = json.NewEncoder(w).Encode(rentalDownloadPlan(pkg, release))
	})
	// Ordinary fixtures provide the real account census. Tests of an unavailable
	// route must opt into that refusal explicitly.
	mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		h.reads++
		if !h.publishes || r.Header.Get("Authorization") != "Bearer rental-idle-test" {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		rows := []map[string]any{}
		for _, row := range h.rentals {
			if state, _ := row["state"].(string); state == "released" || state == "failed" {
				continue
			}
			rows = append(rows, row)
		}
		sort.Slice(rows, func(i, j int) bool {
			return fmt.Sprint(rows[i]["rental_id"]) < fmt.Sprint(rows[j]["rental_id"])
		})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"rentals": rows})
	})
	mux.HandleFunc("GET /v1/rentals/{id}", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		h.reads++
		row, ok := h.rentals[r.PathValue("id")]
		if !ok || r.Header.Get("Authorization") != "Bearer rental-idle-test" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(row)
	})
	mux.HandleFunc("GET /v1/rentals/{id}/image-inventory", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		raw := h.inventories[r.PathValue("id")]
		if len(raw) == 0 || r.Header.Get("Authorization") != "Bearer rental-idle-test" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]json.RawMessage{"image_inventory": raw})
	})
	mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(catalogRows(h.skus))
	})
	mux.HandleFunc("POST /v1/rentals", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		if r.Header.Get("Authorization") != "Bearer rental-idle-test" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var request map[string]any
		if json.NewDecoder(r.Body).Decode(&request) != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if h.spendCap > 0 && h.liveBurnLocked()+h.skuTotalLocked(request["sku"]) > h.spendCap {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusPaymentRequired)
			_, _ = w.Write([]byte(`{"error":{"code":"rental.fleet_spend_cap","message":"the rental would exceed the owner hourly spend cap"}}`))
			return
		}
		if h.rent == nil {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		row := h.rent(request)
		id, _ := row["rental_id"].(string)
		h.rentals[id] = row
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(row)
	})
	mux.HandleFunc("DELETE /v1/rentals/{id}", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		row, ok := h.rentals[r.PathValue("id")]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		row["state"] = "released"
		h.released[r.PathValue("id")]++
		w.WriteHeader(http.StatusNoContent)
	})
	l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port)) //cozy:allow the stand-in HUB binds where the daemon was told the hub is, on loopback; the product binds through internal/api
	must(t, err)
	h.server = &httptest.Server{Listener: l, Config: &http.Server{Handler: mux}}
	h.server.Start()
	t.Cleanup(h.close)
	return h
}

func micros(value any) int64 {
	switch v := value.(type) {
	case int:
		return int64(v)
	case int64:
		return v
	case float64:
		return int64(v)
	}
	return 0
}

func (h *fakeRentalHub) liveBurnLocked() int64 {
	var burn int64
	for _, row := range h.rentals {
		if state := row["state"]; state != "released" && state != "failed" {
			burn += micros(row["hourly_rate_usd_micros"])
		}
	}
	return burn
}

func (h *fakeRentalHub) skuTotalLocked(name any) int64 {
	for _, sku := range h.skus {
		if sku["name"] == name {
			return micros(sku["price_usd_micros_per_hour"]) + micros(sku["storage_usd_micros_per_hour"])
		}
	}
	return 0
}

// publishListing enables the account listing for tests that control availability.
func (h *fakeRentalHub) publishListing() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.publishes = true
}

// setSKUs is the fake hub's product catalog, the shape Tensorhub serves it:
// GPU list price and the spec-derived storage adder, decomposed (th-126).
// catalogRows groups flat per-count SKU maps into the hub's listing: one row per name,
// its counts under `widths`.
func catalogRows(skus []map[string]any) []map[string]any {
	var out []map[string]any
	index := map[any]int{}
	for _, sku := range skus {
		width := map[string]any{"accelerator_count": 1}
		row := map[string]any{}
		for key, value := range sku {
			switch key {
			case "accelerator_count", "price_usd_micros_per_hour", "storage_usd_micros_per_hour":
				width[key] = value
			default:
				row[key] = value
			}
		}
		i, seen := index[row["name"]]
		if !seen {
			i = len(out)
			index[row["name"]] = i
			row["widths"] = []map[string]any{}
			out = append(out, row)
		}
		out[i]["widths"] = append(out[i]["widths"].([]map[string]any), width)
	}
	return out
}

func (h *fakeRentalHub) setSKUs(skus ...map[string]any) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.skus = append([]map[string]any(nil), skus...)
}

func (h *fakeRentalHub) add(id, machine string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.rentals[id] = map[string]any{
		"rental_id": id, "name": machine, "state": "ready",
		"requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 100_000,
	}
}

// setRate moves one rental's served hourly rate, the way Tensorhub does when a
// readback reconciles the quote to the provider's actual billed total (th-120).
func (h *fakeRentalHub) setRate(id string, usdMicros int64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.rentals[id]["hourly_rate_usd_micros"] = usdMicros
}

// setState moves one rental to a hub lifecycle state, with the failure Tensorhub reports
// alongside a terminal one. It is how a pod DYING reaches this daemon: the hub owns
// provider reclaim, so a rental that fails after it was serving is served as `failed` with
// the code that reclaimed it, and there is no other channel that carries the fact.
func (h *fakeRentalHub) setState(id, state, failureCode string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	row, ok := h.rentals[id]
	if !ok {
		return
	}
	row["state"] = state
	if failureCode != "" {
		row["failure"] = map[string]any{"code": failureCode}
	}
}

func (h *fakeRentalHub) releases(id string) int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.released[id]
}

func (h *fakeRentalHub) close() {
	h.mu.Lock()
	server := h.server
	h.server = nil
	h.mu.Unlock()
	if server != nil {
		server.Close()
	}
}

// Listing assertions use the public JSON fields; adding a display column must not
// make lifecycle proofs depend on its position or the machine's elapsed uptime.
type rentalListingRow struct {
	ID          string `json:"rental_id"`
	Machine     string `json:"machine"`
	State       string `json:"state"`
	Failure     string `json:"failure_code"`
	Running     *int   `json:"running"`
	Queued      *int   `json:"queued"`
	IdleSeconds *int   `json:"idle_s"`
	ReleaseDue  string `json:"release_due_at"`
}

func listedRental(t *testing.T, root, id string) rentalListingRow {
	t.Helper()
	code, out := runCozy(t, root, "rental", "list", "--json", "--full")
	var document struct {
		Rentals []rentalListingRow `json:"rentals"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &document) != nil {
		t.Fatalf("rental list failed [exit %d]: %s", code, out)
	}
	for _, row := range document.Rentals {
		if row.ID == id {
			return row
		}
	}
	t.Fatalf("rental %s absent from listing: %s", id, out)
	return rentalListingRow{}
}
