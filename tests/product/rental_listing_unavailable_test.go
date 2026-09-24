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

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestUnavailableRentalListingNeverClaimsZeroOrAuthorizesPurchase(t *testing.T) {
	for _, status := range []int{http.StatusServiceUnavailable, http.StatusNotFound, http.StatusMethodNotAllowed} {
		t.Run(http.StatusText(status), func(t *testing.T) { proveUnavailableRentalListing(t, status) })
	}
}

func proveUnavailableRentalListing(t *testing.T, unavailableStatus int) {
	t.Helper()
	var unavailable atomic.Bool
	unavailable.Store(true)
	var purchases, releases atomic.Int32
	var released atomic.Bool
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, r *http.Request) {
		if unavailable.Load() {
			w.WriteHeader(unavailableStatus)
			if unavailableStatus == http.StatusServiceUnavailable {
				_, _ = w.Write([]byte(`{"error":{"code":"rental.list_unavailable","message":"account rental census unavailable"}}`))
			} else {
				_, _ = w.Write([]byte("account rental census unavailable"))
			}
			return
		}
		_, _ = w.Write([]byte(`{"rentals":[]}`))
	})
	mux.HandleFunc("GET /v1/rentals/{id}", func(w http.ResponseWriter, r *http.Request) {
		state := "ready"
		if released.Load() {
			state = "released"
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"rental_id": r.PathValue("id"), "name": "isao", "state": state, "requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 100000})
	})
	mux.HandleFunc("DELETE /v1/rentals/pr-existing", func(w http.ResponseWriter, r *http.Request) {
		releases.Add(1)
		released.Store(true)
		w.WriteHeader(http.StatusNoContent)
	})
	mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`[{"name":"cpu","accelerator_model":"CPU","accelerator_count":1,"price_usd_micros_per_hour":100000}]`))
	})
	mux.HandleFunc("POST /v1/rentals", func(w http.ResponseWriter, r *http.Request) {
		purchases.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\ntensorhub_token: test\nrentals:\n  max_hourly_spend_usd: 10\n"), 0600))
	for _, known := range []bool{false, true} {
		if known {
			st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			fatal(t, st.RecordRental(records.Rental{ID: "pr-existing", MachineName: "isao", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, State: "ready", Address: "127.0.0.1:1", CertPath: "fixture", Hub: server.URL}))
			st.Close()
		}
		for _, jsonMode := range []bool{false, true} {
			args := []string{"rental", "list"}
			if jsonMode {
				args = append(args, "--json")
			}
			code, out := runCozy(t, root, args...)
			if code == 0 || !strings.Contains(out, "account rental census unavailable") || strings.Contains(out, "machines_running") || strings.Contains(out, "Current spend") {
				t.Fatalf("unknown census reported fleet totals: %d %s", code, out)
			}
		}
		code, out := runCozy(t, root, "rental", "new", "cpu", "--json")
		if code == 0 || !strings.Contains(out, "account rental census unavailable") || purchases.Load() != 0 {
			t.Fatalf("unknown account spend authorized purchase: %d %s", code, out)
		}
	}
	st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	row, problem := st.RentalRow("pr-existing")
	fatal(t, problem)
	if row == nil || row.State != "ready" {
		t.Fatal("unavailable listing removed known rental")
	}
	st.Close()
	unavailable.Store(false)
	code, out := runCozy(t, root, "rental", "list", "--json")
	if code != 0 || !strings.Contains(out, `"machines_running":1`) {
		t.Fatalf("recovered census did not restore real known totals: %d %s", code, out)
	}
	st, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	_, problem = st.ForgetRental("pr-existing")
	fatal(t, problem)
	st.Close()
	code, out = runCozy(t, root, "rental", "list", "--json")
	if code != 0 || !strings.Contains(out, `"machines_running":0`) || !strings.Contains(out, `"hourly_spend_usd_micros":0`) {
		t.Fatalf("verified empty account did not show zero: %d %s", code, out)
	}
	// Account totals are not needed to release one locally recorded identity.
	st, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	fatal(t, st.RecordRental(records.Rental{ID: "pr-existing", MachineName: "isao", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, State: "ready", Address: "127.0.0.1:1", CertPath: "fixture", Hub: server.URL}))
	st.Close()
	unavailable.Store(true)
	code, out = runCozy(t, root, "rental", "end", "pr-existing", "--json")
	if code != 0 || !strings.Contains(out, `"changed":true`) || releases.Load() != 1 || purchases.Load() != 0 {
		t.Fatalf("unavailable census blocked known-rental release: %d %s, releases=%d", code, out, releases.Load())
	}
}

func TestUnreachableRentalListingReturnsTypedFailure(t *testing.T) {
	server := httptest.NewServer(http.NotFoundHandler())
	url := server.URL
	server.Close()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+url+"\n"), 0600))
	code, out := runCozy(t, root, "rental", "list", "--json")
	if code == 0 || !strings.Contains(out, `"error"`) || strings.Contains(out, `"machines_running":0`) {
		t.Fatalf("closed Hub port became zero fleet: %d %s", code, out)
	}
}
