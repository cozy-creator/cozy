package producttest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

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

// reservePort finds a loopback port nothing listens on, for a hub that must be absent.
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
	mu      sync.Mutex
	rentals map[string]map[string]any
	// bindings is the account's bindings revision the rental listing and a rental's authority
	// poll state; listedBindings is the one the last listing carried.
	bindings, listedBindings int64
	packageReleases          map[string]any
	skus                     []map[string]any
	rent                     func(map[string]any) map[string]any
	// quote answers POST /v1/rental-quotes; nil quotes the listed price of the SKU asked for.
	quote func(map[string]any) (int, string)
	// spendCap stands in for Tensorhub's owner fleet cap: a paid ask whose SKU
	// total would take the live rentals' burn past it is refused, buying nothing.
	spendCap int64
	released map[string]int
	server   *httptest.Server
	mux      *http.ServeMux // the routes, however server's handler is wrapped
	// publishes is whether this stand-in hub carries th-199's account listing.
	publishes bool
	// reads counts rental reads, so a test can plant a transition after the daemon's first look.
	reads int
	// software is the target pair GET /v1/software answers; unset, none.
	software map[string]string
	// providers is each `?provider=` the catalog was asked for ("" for none). A named one is
	// answered with the catalog tagged as that provider's, as Tensorhub does.
	providers []string
}

func newFakeRentalHub(t *testing.T, port int) *fakeRentalHub {
	t.Helper()
	h := &fakeRentalHub{rentals: map[string]map[string]any{}, released: map[string]int{}, publishes: true}
	mux := http.NewServeMux()
	h.mux = mux
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
	mux.HandleFunc("GET /v1/software", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]string{"runtime": h.software["runtime"], "tensorfs": h.software["tensorfs"]})
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
		// As Tensorhub: money in flight, or with ?state=all the ended ones too; ?name= keeps the
		// rentals that bore one name.
		history, name := r.URL.Query().Get("state") == "all", r.URL.Query().Get("name")
		rows := []map[string]any{}
		for _, row := range h.rentals {
			if state, _ := row["state"].(string); !history && (state == "released" || state == "failed") {
				continue
			}
			if name != "" && row["name"] != name {
				continue
			}
			rows = append(rows, row)
		}
		sort.Slice(rows, func(i, j int) bool {
			return fmt.Sprint(rows[i]["rental_id"]) < fmt.Sprint(rows[j]["rental_id"])
		})
		w.Header().Set("Content-Type", "application/json")
		h.listedBindings = h.bindings
		_ = json.NewEncoder(w).Encode(map[string]any{"rentals": rows, "bindings_revision": h.bindings})
	})
	mux.HandleFunc("GET /v1/rentals/{id}", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		h.reads++
		row, ok := h.rentals[r.PathValue("id")]
		if !ok || r.Header.Get("Authorization") != "Bearer rental-idle-test" {
			// As Tensorhub answers an id it does not hold for this caller.
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"code":"rental.not_found","message":"no rental","remedy":"POST /v1/rentals creates one"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(row)
	})
	mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		provider := r.URL.Query().Get("provider")
		h.providers = append(h.providers, provider)
		rows := catalogRows(h.skus)
		for _, row := range rows {
			if provider != "" {
				row["provider"] = provider
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(rows)
	})
	mux.HandleFunc("POST /v1/rental-quotes", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		quote := h.quote
		h.mu.Unlock()
		if quote == nil {
			listedRentalQuote(mux)(w, r)
			return
		}
		var request map[string]any
		if json.NewDecoder(r.Body).Decode(&request) != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		status, body := quote(request)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
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

// rentUntilRecorded runs `cozy rental new <args>` against a stand-in whose rental never
// becomes ready. Once the paid ask is recorded locally as rental `id`, the watch is
// interrupted: an interrupted acquisition stays open exactly as a --timeout leaves it, and
// no clock decides when that is. It returns what runCozy would.
func rentUntilRecorded(t *testing.T, root, id string, args ...string) (int, string) {
	t.Helper()
	cmd := exec.Command(cozyBin, append([]string{"rental", "new"}, args...)...)
	cmd.Env = childEnv(t, root)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	must(t, cmd.Start())
	exited := make(chan struct{})
	go func() { _ = cmd.Wait(); close(exited) }()
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	deadline := time.After(30 * time.Second)
	for waiting := true; waiting; {
		if row, problem := store.RentalRow(id); problem == nil && row != nil {
			_ = cmd.Process.Signal(os.Interrupt)
			waiting = false
			continue
		}
		select {
		case <-exited:
			waiting = false
		case <-deadline:
			_ = cmd.Process.Kill()
			t.Fatalf("rental %s was never recorded:\n%s%s", id, stdout.String(), stderr.String())
		case <-time.After(20 * time.Millisecond):
		}
	}
	<-exited
	if slices.Contains(args, "--json") {
		return cmd.ProcessState.ExitCode(), stdout.String()
	}
	return cmd.ProcessState.ExitCode(), stdout.String() + stderr.String()
}

// port is where this stand-in listens. It binds first and the config names it after,
// so no other process can take the address in between.
func (h *fakeRentalHub) port() int { return h.server.Listener.Addr().(*net.TCPAddr).Port }

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
