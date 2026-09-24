package producttest

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentalListUsesDaemonAPIWithoutOpeningSQLite(t *testing.T) {
	layout, lock, pid, done := compatibilityOwner(t)
	var reads atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/":
		case "/v1/local/rentals":
			reads.Add(1)
			_, _ = w.Write([]byte(`{"machines_running":1,"hourly_spend_usd_micros":100000,"idle_release_s":900,"rentals":[{"rental_id":"retained","machine":"shelly","state":"ready","hourly_rate_usd_micros":100000,"accelerator_count":1,"activity":{"running":1,"queued":2}}],"unrecorded":[],"pending":[],"future_fact":"ignored"}`))
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
			_, _ = w.Write([]byte(`{"machines_running":2,"hourly_spend_usd_micros":200000,"idle_release_s":900,"rentals":[],"unrecorded":[{"rental_id":"remote-id","machine":"remote-machine","state":"release_requested","hourly_rate_usd_micros":100000}],"pending":[{"machine":"pending-machine","state":"release_requested","operation":"pending-id","hourly_rate_usd_micros":100000}]}`))
		}
	}))
	defer server.Close()
	publishCompatibilityOwner(t, layout, lock, pid, strings.TrimPrefix(server.URL, "http://"), "")
	output, err := compatibilityCLI(t, layout.Root, "rental", "list", "--no-watch")
	if err != nil || strings.Count(output, "draining") != 2 || strings.Contains(output, "release_requested") || !strings.Contains(output, "Idle machines shut down after 15 minutes") {
		t.Fatalf("inventory lost human state or daemon-owned idle policy: %v %s", err, output)
	}
	output, err = compatibilityCLI(t, layout.Root, "rental", "list", "--json")
	if err != nil || strings.Count(output, `"state":"release_requested"`) != 2 || strings.Contains(output, `"state":"draining"`) {
		t.Fatalf("human projection changed raw API states: %v %s", err, output)
	}
}

func TestRentalSchemaGuardPreservesActiveOwnerAndRetainedRows(t *testing.T) {
	layout, lock, pid, done := compatibilityOwner(t)
	path, db := retainedWidthDatabase(t, true)
	before := map[string]string{}
	for _, table := range []string{"rentals", "requests", "attempts", "byte_outputs", "worker_processes", "native_artifact_retentions"} {
		before[table] = tableRows(t, db, table)
	}
	must(t, db.Close())
	must(t, os.Rename(path, layout.DB))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()
	publishCompatibilityOwner(t, layout, lock, pid, strings.TrimPrefix(server.URL, "http://"), "schema=33\n")
	output, err := compatibilityCLI(t, layout.Root, "rental", "ssh-info", "shelly", "--json")
	if err == nil || !strings.Contains(output, "records_schema_upgrade_required") {
		t.Fatalf("older live owner's schema was not preserved: %v %s", err, output)
	}
	db, err = sql.Open("sqlite", layout.DB)
	must(t, err)
	defer db.Close()
	var version int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	if version != 33 {
		t.Fatalf("live owner's schema changed to %d", version)
	}
	for table, want := range before {
		if got := tableRows(t, db, table); got != want {
			t.Fatalf("retained %s rows changed", table)
		}
	}
	select {
	case <-done:
		t.Fatal("schema guard stopped the active owner")
	default:
	}
}

func TestRentalReaderStartsDaemonToMigrateAnUnownedRoot(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "creator.sqlite")
	db, err := sql.Open("sqlite", path)
	must(t, err)
	ddl, err := os.ReadFile("testdata/records-schema33-before-rental-width.sql")
	must(t, err)
	_, err = db.Exec(string(ddl))
	must(t, err)
	must(t, db.Close())
	output, err := compatibilityCLI(t, root, "rental", "ssh-info", "missing", "--json")
	if err == nil || !strings.Contains(output, "development rental is not attached") {
		t.Fatalf("reader did not pass the daemon migration: %v %s", err, output)
	}
	store, problem := records.Open(path)
	fatal(t, problem)
	store.Close()
	if state := daemon.Probe(config.Config{Home: root}); !state.Up {
		t.Fatal("migration did not leave the owning daemon running")
	}
}

func TestRentalListOldDaemonFallbackRequiresCompatibleStore(t *testing.T) {
	for _, newer := range []bool{false, true} {
		t.Run(map[bool]string{false: "compatible", true: "newer-store"}[newer], func(t *testing.T) {
			layout, lock, pid, done := compatibilityOwner(t)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/" {
					return
				}
				w.WriteHeader(http.StatusNotFound)
				_, _ = w.Write([]byte(`{"error":{"code":"unknown_route","message":"older daemon"}}`))
			}))
			defer server.Close()
			hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/rentals" {
					t.Errorf("unexpected Hub route: %s", r.URL.Path)
				}
				_, _ = w.Write([]byte(`{"rentals":[]}`))
			}))
			defer hub.Close()
			must(t, os.WriteFile(filepath.Join(layout.Root, config.FileName), []byte("tensorhub_url: "+hub.URL+"\ntensorhub_token: test\n"), 0600))
			st, problem := records.Open(layout.DB)
			fatal(t, problem)
			st.Close()
			if newer {
				db, err := sql.Open("sqlite", layout.DB)
				must(t, err)
				_, err = db.Exec(`PRAGMA user_version=999`)
				must(t, err)
				db.Close()
			}
			publishCompatibilityOwner(t, layout, lock, pid, strings.TrimPrefix(server.URL, "http://"), "")
			output, err := compatibilityCLI(t, layout.Root, "rental", "list", "--json")
			if newer {
				if err == nil || !strings.Contains(output, "records_schema_newer") {
					t.Fatalf("unsafe old-route fallback: %v %s", err, output)
				}
			} else if err != nil || !strings.Contains(output, `"machines_running":0`) {
				t.Fatalf("compatible old daemon fallback failed: %v %s", err, output)
			}
			select {
			case <-done:
				t.Fatal("fallback stopped the owner")
			default:
			}
		})
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
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.URL+"\ntensorhub_token: test\nrentals:\n"), 0600))
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
