package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
)

// rental show says the deadline the machine's Status names now, never the listing's or a
// guessed one: none when it names none or cannot answer.
func TestRentalShowUsesOnlyTheWorkerDeadline(t *testing.T) {
	for _, state := range []string{"named", "none", "unreachable"} {
		t.Run(state, func(t *testing.T) {
			layout, lock, pid, _ := compatibilityOwner(t)
			deadline := time.Date(2026, 10, 10, 4, 6, 31, 571000000, time.UTC)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/":
				case "/v1/local/rentals":
					_, _ = w.Write([]byte(`{"rentals":[{"rental_id":"rental-proof","machine":"touji","state":"ready","release_due_at":"2026-10-10T04:10:00Z","activity":{"idle_since_at":"2026-10-10T03:55:00Z"}}]}`))
				case "/v1/local/machines/rental-proof/status":
					if state == "unreachable" {
						w.WriteHeader(http.StatusServiceUnavailable)
						_, _ = w.Write([]byte(`{"error":{"class":"operational","code":"unavailable","message":"worker unavailable"}}`))
						return
					}
					ms := int64(0)
					if state == "named" {
						ms = deadline.UnixMilli()
					}
					_ = json.NewEncoder(w).Encode(map[string]any{"phase": "ready", "idle_deadline_unix_ms": ms})
				default:
					t.Errorf("unexpected route %s", r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer server.Close()
			publishCompatibilityOwner(t, layout, lock, pid, strings.TrimPrefix(server.URL, "http://"), "")
			out, err := compatibilityCLI(t, layout.Root, "rental", "show", "touji", "--json", "--full")
			if err != nil {
				t.Fatalf("show: %v %s", err, out)
			}
			var document map[string]any
			must(t, json.Unmarshal([]byte(out), &document))
			got, present := document["release_due_at"]
			if state == "named" {
				if !present || got != deadline.Format(time.RFC3339Nano) {
					t.Fatalf("lost worker's actual deadline: %s", out)
				}
			} else if present {
				t.Fatalf("invented an expiry for %s worker: %s", state, out)
			}
		})
	}
}

func TestRentalListUsesDaemonAPIWithoutOpeningSQLite(t *testing.T) {
	layout, lock, pid, done := compatibilityOwner(t)
	var reads atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/":
		case "/v1/local/rentals":
			reads.Add(1)
			_, _ = w.Write([]byte(`{"machines_running":1,"hourly_spend_usd_micros":100000,"rentals":[{"rental_id":"retained","machine":"shelly","state":"ready","hourly_rate_usd_micros":100000,"accelerator_count":1,"activity":{"running":1,"queued":2}}],"unrecorded":[],"pending":[],"future_fact":"ignored"}`))
		default:
			t.Errorf("unexpected client route: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	publishCompatibilityOwner(t, layout, lock, pid, strings.TrimPrefix(server.URL, "http://"), "schema=999\n")
	// This file cannot be opened by SQLite. A client that still couples listing to
	// its records reader fails before the daemon can provide a valid inventory.
	before := []byte("newer daemon's private storage format")
	must(t, os.WriteFile(layout.DB, before, 0600))
	output, err := compatibilityCLI(t, layout.Root, "rental", "list", "--json")
	if err != nil || !strings.Contains(output, `"machine":"shelly"`) || !strings.Contains(output, `"running":1`) || !strings.Contains(output, `"queued":2`) || reads.Load() != 1 {
		t.Fatalf("rental list did not use the daemon inventory: %v %s (reads=%d)", err, output, reads.Load())
	}
	after, err := os.ReadFile(layout.DB)
	must(t, err)
	if string(before) != string(after) {
		t.Fatal("API listing changed daemon-owned storage")
	}
	select {
	case <-done:
		t.Fatal("API listing stopped the active owner")
	default:
	}
}

func TestRentalInventoryHumanDrainingKeepsRawStates(t *testing.T) {
	layout, lock, pid, _ := compatibilityOwner(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/local/rentals" {
			_, _ = w.Write([]byte(`{"machines_running":2,"hourly_spend_usd_micros":200000,"rentals":[],"unrecorded":[{"rental_id":"remote-id","machine":"remote-machine","state":"release_requested","hourly_rate_usd_micros":100000}],"pending":[{"machine":"pending-machine","state":"release_requested","operation":"pending-id","hourly_rate_usd_micros":100000}]}`))
		}
	}))
	defer server.Close()
	publishCompatibilityOwner(t, layout, lock, pid, strings.TrimPrefix(server.URL, "http://"), "")
	output, err := compatibilityCLI(t, layout.Root, "rental", "list", "--no-watch")
	if err != nil || strings.Count(output, "draining") != 2 || strings.Contains(output, "release_requested") || !strings.Contains(output, "Rentals end themselves after 15 minutes idle") {
		t.Fatalf("inventory lost human state or daemon-owned idle policy: %v %s", err, output)
	}
	output, err = compatibilityCLI(t, layout.Root, "rental", "list", "--json")
	if err != nil || strings.Count(output, `"state":"release_requested"`) != 2 || strings.Contains(output, `"state":"draining"`) {
		t.Fatalf("human projection changed raw API states: %v %s", err, output)
	}
}

func TestRentalInventoryAPIKeepsUnknownActivityAndPrivateFieldsAbsent(t *testing.T) {
	var purchases, releases atomic.Int32
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			purchases.Add(1)
		}
		if r.Method == http.MethodDelete {
			releases.Add(1)
		}
		if r.URL.Path == "/v1/rentals" {
			_, _ = w.Write([]byte(`{"rentals":[{"rental_id":"unrecorded","name":"remote","state":"ready","hourly_rate_usd_micros":100000,"accelerator_count":1}]}`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer hub.Close()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.URL+"\ntensorhub_token: test\n"), 0600))
	live := startDaemonProcess(t, root)
	if code, output := runCozy(t, root, "rental", "list", "--json"); code != 0 || !strings.Contains(output, `"recorded":false`) {
		t.Fatalf("real daemon inventory failed: %d %s", code, output)
	}
	// The CLI proof above reads the real server; its HTTP read model also keeps
	// the unavailable activity absent instead of inventing idle zeroes.
	response := live.call(t, "GET", "/v1/local/rentals", nil)
	if response.Status != http.StatusOK {
		t.Fatalf("inventory route: %s", response.brief())
	}
	var doc struct {
		Unrecorded []map[string]any `json:"unrecorded"`
	}
	must(t, json.Unmarshal(response.Body, &doc))
	if len(doc.Unrecorded) != 1 {
		t.Fatalf("missing unrecorded machine: %s", response.Body)
	}
	for _, key := range []string{"activity", "running", "queued", "cert_path", "token", "request_body"} {
		if _, exists := doc.Unrecorded[0][key]; exists {
			t.Errorf("inventory exposes %s", key)
		}
	}
	if denied := live.call(t, "GET", "/v1/local/rentals", nil, "Authorization", ""); denied.Status != http.StatusUnauthorized {
		t.Fatalf("inventory is not authenticated: %s", denied.brief())
	}
	if purchases.Load() != 0 || releases.Load() != 0 {
		t.Fatal("listing changed paid resources")
	}
}
