package producttest

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
)

// A rented ingest (`cozy model upload <source> <repo> --rental-only`) runs as a script, and
// the rental bought for it declares the ingest's planned source bytes, so the Hub sizes the
// pod's disk to the ingest rather than to the script-run default. The stand-in Hub captures
// the paid request and refuses it; nothing is rented.
func TestRentedIngestDeclaresItsPlannedSourceBytes(t *testing.T) {
	probe := &http.Client{Timeout: 20 * time.Second}
	response, err := probe.Get("https://civitai.com/api/v1/model-versions/128078")
	if err != nil {
		t.Skipf("provider unreachable from this runner: %v", err)
	}
	var version struct {
		Files []struct {
			Name    string  `json:"name"`
			Primary bool    `json:"primary"`
			SizeKB  float64 `json:"sizeKB"`
		} `json:"files"`
	}
	must(t, json.NewDecoder(response.Body).Decode(&version))
	response.Body.Close()
	var planned int64
	for _, file := range version.Files {
		if file.Primary {
			planned = int64(file.SizeKB * 1024)
		}
	}

	var mu sync.Mutex
	var posts [][]byte
	var unexpected []string
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.RentalProducts([]hub.RentalSKU{{Name: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1,
			PriceUSDMicrosPerHour: 100_000, BaseWorkerProfile: "python3.12-cpu-linux-x86"}}))
	})
	mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{"rentals":[]}`)) })
	mux.HandleFunc("POST /v1/rentals", func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		mu.Lock()
		posts = append(posts, raw)
		mu.Unlock()
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"code":"proof.no_paid_create","message":"captured without renting"}}`))
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		unexpected = append(unexpected, r.Method+" "+r.URL.String())
		mu.Unlock()
		http.NotFound(w, r)
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	root, err := os.MkdirTemp(scratchBase, "ingest-sizing-")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(root) })
	// An operator identity publishes to its named org without a Hub account read.
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+
		"\ntensorhub_token: operator-proof\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	startDaemonProcess(t, root)

	cmd := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "model", "upload", "civitai://128078", "proof/sdxl",
		"--source-profile", "civitai/sdxl/single-file/1", "--rental-only", "--json")
	cmd.Env = childEnv(t, root)
	out, _ := cmd.CombinedOutput()
	deadline := time.Now().Add(90 * time.Second)
	var paid []byte
	for len(paid) == 0 {
		mu.Lock()
		if len(posts) > 0 {
			paid = append([]byte(nil), posts[0]...)
		}
		seen := append([]string(nil), unexpected...)
		mu.Unlock()
		if time.Now().After(deadline) {
			t.Fatalf("no rental was asked for [exit %d]: %s\nunserved Hub routes: %v\n%s", cmd.ProcessState.ExitCode(), out, seen, tail(filepath.Join(root, "daemon.log")))
		}
		time.Sleep(50 * time.Millisecond)
	}
	request, problem := hub.ParseRentalRequestBytes(paid)
	fatal(t, problem)
	if request.PlannedSourceBytes != planned {
		t.Fatalf("the ingest's rental declared %d planned source bytes, want the source's %d: %s", request.PlannedSourceBytes, planned, paid)
	}
}
