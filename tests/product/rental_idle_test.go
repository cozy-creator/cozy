package producttest

import (
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

// TestRentalIdleRelease is the owner's ruling as behaviour: a rental this daemon owns is
// ended once it has had no work for rentals.idle_release_s, however it was acquired; work
// pinned to it — or the unsettled request it was bought for (cl-113) — is what keeps it;
// and a release the hub did not confirm is asked again until it is. Every arm is the real daemon process on a real root, deciding from its real
// records and releasing through the real hub client against a hub that answers the rental
// routes; the grace is the one product knob.
func TestRentalIdleRelease(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "rental-idle")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	port := reservePort(t)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			"tensorhub_token: rental-idle-test\n"+
			"rentals:\n  max_hourly_spend_usd: 1.00\n  idle_release_s: 2\n"+
			"daemon:\n  idle_shutdown_s: 0\n"), 0o600))
	logPath := filepath.Join(root, "daemon.log")

	hub := newFakeRentalHub(t, port)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	plant := func(id, machine string) {
		t.Helper()
		hub.add(id, machine)
		fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
			ID: id, MachineName: machine, SKU: "cpu", AcceleratorModel: "CPU",
			HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL,
			Address: "127.0.0.1:1", CertPath: filepath.Join(root, id+".pem"),
		}))
	}

	// (a) A manual rental — no managing request, never ran anything — has an idle clock
	// from the moment it was recorded ready, and goes once that clock passes the grace.
	// The daemon says so once before, and once after.
	plant("rental-idle-manual", "heron")
	daemon := startDaemonProcess(t, root)
	awaitRentalGone(t, store, "rental-idle-manual", 15*time.Second, logPath)
	if hub.releases("rental-idle-manual") != 1 {
		t.Fatalf("the hub saw %d release(s) of the manual rental, wanted 1\n%s",
			hub.releases("rental-idle-manual"), tail(logPath))
	}
	log, _ := os.ReadFile(logPath)
	if strings.Count(string(log), "rental rental-idle-manual (heron) idle since") != 1 ||
		!strings.Contains(string(log), "rental rental-idle-manual (heron) released after") {
		t.Fatalf("the idle release did not say why, exactly once\n%s", tail(logPath))
	}

	// (b) Work pinned to a rental is what keeps it: a queued request holds the pod past
	// several graces, the listing says so, and settling the request is what lets it go.
	plant("rental-idle-busy", "otter")
	if _, _, problem := store.Submit(records.Request{
		ID: "req-rental-idle", IdemKey: "idem-rental-idle", BodyDigest: "sha256:" + strings.Repeat("ab", 32),
		Package: "fake/idle", Entrypoint: "generate", Payload: []byte("{}"),
		Rental: true, Worker: "rental-idle-busy",
	}); problem != nil {
		t.Fatal(problem.Message)
	}
	time.Sleep(6 * time.Second)
	if row, problem := store.RentalRow("rental-idle-busy"); problem != nil || row == nil || row.State != "ready" {
		t.Fatalf("a rental with a queued request was released: %+v %v\n%s", row, problem, tail(logPath))
	}
	if hub.releases("rental-idle-busy") != 0 {
		t.Fatalf("the hub saw a release of a busy rental\n%s", tail(logPath))
	}
	busy := listedRental(t, root, "rental-idle-busy")
	if busy.Machine != "otter" || busy.State != "ready" || busy.Running == nil || *busy.Running != 0 || busy.Queued == nil || *busy.Queued != 1 || busy.IdleSeconds != nil || busy.ReleaseDue != "" {
		t.Fatalf("the listing does not show queued work holding the rental: %+v", busy)
	}
	if r := daemon.call(t, "POST", "/v1/requests/req-rental-idle/cancel", nil); r.Status != http.StatusOK {
		t.Fatalf("cancel of the queued request: %s", r.brief())
	}
	awaitRentalGone(t, store, "rental-idle-busy", 15*time.Second, logPath)

	// (b2) The buy itself is a debt (cl-113): a rental bought for a request is owed by
	// that request from the moment the paid row exists — before dispatch pins, so the
	// request row still says worker='' — and the sweep must not release it while its
	// buyer is queued. On the code this arm was written against, the pod went the moment
	// it was ready: Spent saw a managed rental with no settled attempt and reaped a
	// healthy pod its buyer was still waiting for (observed live, pr-b192da1a).
	hub.add("rental-idle-owed", "curlew")
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
		ID: "rental-idle-owed", MachineName: "curlew", SKU: "cpu", AcceleratorModel: "CPU",
		HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL,
		Address: "127.0.0.1:1", CertPath: filepath.Join(root, "rental-idle-owed.pem"),
		ManagedRequestID: "req-rental-owed",
	}))
	if _, _, problem := store.Submit(records.Request{
		ID: "req-rental-owed", IdemKey: "idem-rental-owed", BodyDigest: "sha256:" + strings.Repeat("cd", 32),
		Package: "fake/owed", Entrypoint: "generate", Payload: []byte("{}"),
		Rental: true,
	}); problem != nil {
		t.Fatal(problem.Message)
	}
	time.Sleep(6 * time.Second)
	if row, problem := store.RentalRow("rental-idle-owed"); problem != nil || row == nil || row.State != "ready" {
		t.Fatalf("a rental owed by its still-queued buyer was released: %+v %v\n%s", row, problem, tail(logPath))
	}
	if hub.releases("rental-idle-owed") != 0 {
		t.Fatalf("the hub saw a release of an owed rental\n%s", tail(logPath))
	}
	owed := listedRental(t, root, "rental-idle-owed")
	if owed.State != "ready" || owed.Running == nil || *owed.Running != 0 || owed.Queued == nil || *owed.Queued != 0 || owed.IdleSeconds != nil || owed.ReleaseDue != "" {
		t.Fatalf("the listing shows an idle countdown on an owed rental: %+v", owed)
	}
	if r := daemon.call(t, "POST", "/v1/requests/req-rental-owed/cancel", nil); r.Status != http.StatusOK {
		t.Fatalf("cancel of the owing request: %s", r.brief())
	}
	awaitRentalGone(t, store, "rental-idle-owed", 15*time.Second, logPath)

	// (b3) QUEUED WORK THAT IS PINNED TO NOBODY still holds the fleet (cl-121). The pin is
	// routing's own output, so a --rental request belongs to no machine between the moment
	// some rental has its plan STAGED (`rentalHeld`, which makes `selectOrStart` return
	// without pinning) and the moment that placement is DISPATCHABLE (which is when `route`
	// pins it). On the code this arm was written against, that request counted toward NO
	// rental — `RentalRunCounts` counts `worker=<rental>` and `Owed` covers only the one
	// buyer — so a warm machine idled out from under work that was waiting for it, and the
	// next request paid a full cold acquisition (178-271 s and a 6.93 GB re-download of
	// bytes the released machine already held). This rental neither bought the request nor
	// holds its pin: that is the point, because releasing ANY machine on this evidence is
	// wrong in the expensive direction.
	plant("rental-idle-unpinned", "kestrel")
	if _, _, problem := store.Submit(records.Request{
		ID: "req-rental-unpinned", IdemKey: "idem-rental-unpinned",
		BodyDigest: "sha256:" + strings.Repeat("ef", 32),
		Package:    "fake/unpinned", Entrypoint: "generate", Payload: []byte("{}"),
		Rental: true,
	}); problem != nil {
		t.Fatal(problem.Message)
	}
	time.Sleep(6 * time.Second)
	if row, problem := store.RentalRow("rental-idle-unpinned"); problem != nil || row == nil ||
		row.State != "ready" {
		t.Fatalf("a rental was released while unpinned --rental work was queued: %+v %v\n%s",
			row, problem, tail(logPath))
	}
	if hub.releases("rental-idle-unpinned") != 0 {
		t.Fatalf("the hub saw a release while unpinned work was queued\n%s", tail(logPath))
	}
	// The listing has to agree with the mechanism: no work of its OWN (0 queued, 0 running)
	// and no countdown, because the fleet is not idle even though this machine is.
	unpinned := listedRental(t, root, "rental-idle-unpinned")
	if unpinned.State != "ready" || unpinned.Running == nil || *unpinned.Running != 0 || unpinned.Queued == nil || *unpinned.Queued != 0 || unpinned.IdleSeconds != nil || unpinned.ReleaseDue != "" {
		t.Fatalf("the listing shows an idle countdown while unpinned work is queued: %+v", unpinned)
	}
	// And settling the unpinned request is what lets it go: the hold is the WORK, never a
	// permanent exemption.
	if r := daemon.call(t, "POST", "/v1/requests/req-rental-unpinned/cancel", nil); r.Status != http.StatusOK {
		t.Fatalf("cancel of the unpinned request: %s", r.brief())
	}
	awaitRentalGone(t, store, "rental-idle-unpinned", 15*time.Second, logPath)

	// (c) A release the hub does not confirm is retried at the next observation, not left
	// until a rental command or a restart: with the hub gone the daemon says so once, keeps
	// asking, and the pod is released as soon as the hub answers again.
	hub.close()
	plant("rental-idle-retry", "puffin")
	awaitLog(t, logPath, "rental rental-idle-retry release deferred:", 15*time.Second)
	time.Sleep(3 * time.Second)
	log, _ = os.ReadFile(logPath)
	if n := strings.Count(string(log), "rental rental-idle-retry release deferred:"); n != 1 {
		t.Fatalf("the deferred release was said %d times, wanted once\n%s", n, tail(logPath))
	}
	if row, problem := store.RentalRow("rental-idle-retry"); problem != nil || row == nil {
		t.Fatalf("a rental the hub never confirmed released was forgotten: %+v %v", row, problem)
	}
	hub = newFakeRentalHub(t, port)
	hub.add("rental-idle-retry", "puffin")
	awaitRentalGone(t, store, "rental-idle-retry", 15*time.Second, logPath)
	if hub.releases("rental-idle-retry") != 1 {
		t.Fatalf("the returned hub saw %d release(s), wanted 1\n%s", hub.releases("rental-idle-retry"), tail(logPath))
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
	mu       sync.Mutex
	rentals  map[string]map[string]any
	skus     []map[string]any
	rent     func(map[string]any) map[string]any
	released map[string]int
	server   *httptest.Server
	// publishes is whether this stand-in hub carries th-199's account listing.
	publishes bool
}

func newFakeRentalHub(t *testing.T, port int) *fakeRentalHub {
	t.Helper()
	h := &fakeRentalHub{rentals: map[string]map[string]any{}, released: map[string]int{}}
	mux := http.NewServeMux()
	// th-199's enumeration door, the half a pre-th-199 hub does not have: `publishes`
	// off leaves `POST /v1/rentals` to answer a GET with net/http's own 405, exactly as
	// an un-upgraded Tensorhub does.
	mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
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
		row, ok := h.rentals[r.PathValue("id")]
		if !ok || r.Header.Get("Authorization") != "Bearer rental-idle-test" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(row)
	})
	mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(h.skus)
	})
	mux.HandleFunc("POST /v1/rentals", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		if h.rent == nil || r.Header.Get("Authorization") != "Bearer rental-idle-test" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var request map[string]any
		if json.NewDecoder(r.Body).Decode(&request) != nil {
			w.WriteHeader(http.StatusBadRequest)
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

// publishListing turns th-199's account listing on. It is off by default so every
// proof written before the route still runs against the hub it was written for.
func (h *fakeRentalHub) publishListing() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.publishes = true
}

// setSKUs is the fake hub's product catalog, the shape Tensorhub serves it:
// GPU list price and the spec-derived storage adder, decomposed (th-126).
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
