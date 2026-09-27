package producttest

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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
	planned := civitaiPrimaryBytes(t, 128078)
	root, paid := rentalCapture(t, "")
	out, code := rentedUpload(t, root, "civitai://128078", "proof/sdxl", "--source-profile", "civitai/sdxl/single-file/1")
	if request := paid(t, out, code); request.PlannedSourceBytes != planned {
		t.Fatalf("the ingest's rental declared %d planned source bytes, want the source's %d", request.PlannedSourceBytes, planned)
	}
}

// The host and the rental install TensorFS independently, so the host's may know fewer
// profiles. A profile this host does not recognize is the rental's to decide: the ingest
// still asks for its rental, and only the rental's own TensorFS may refuse it.
func TestHostTensorFSUnawareOfAProfileDefersToTheRental(t *testing.T) {
	planned := civitaiPrimaryBytes(t, 128078)
	root, paid := rentalCapture(t, "{\"entries\":[],\"source_profiles\":[]}")

	// Nothing named, and this host recognizes nothing: it cannot choose, and says how the
	// rental can.
	out, code := rentedUpload(t, root, "civitai://128078", "proof/sdxl")
	if code == 0 || !strings.Contains(string(out), "UNREGISTERED_FINGERPRINT") || !strings.Contains(string(out), "--source-profile") {
		t.Fatalf("an unrecognized source with no named profile must say how to defer [exit %d]: %s", code, out)
	}

	out, code = rentedUpload(t, root, "civitai://128078", "proof/sdxl",
		"--source-profile", "civitai/101055/128078/single-file-fp16")
	if request := paid(t, out, code); request.PlannedSourceBytes != planned {
		t.Fatalf("the deferred ingest declared %d planned source bytes, want the source's %d", request.PlannedSourceBytes, planned)
	}
	if !strings.Contains(string(out), "the rental's TensorFS decides") {
		t.Fatalf("a deferred profile must say who decides: %s", out)
	}
}

func civitaiPrimaryBytes(t *testing.T, version int) int64 {
	t.Helper()
	probe := &http.Client{Timeout: 20 * time.Second}
	response, err := probe.Get(fmt.Sprintf("https://civitai.com/api/v1/model-versions/%d", version))
	if err != nil {
		t.Skipf("provider unreachable from this runner: %v", err)
	}
	defer response.Body.Close()
	var document struct {
		Files []struct {
			Primary bool    `json:"primary"`
			SizeKB  float64 `json:"sizeKB"`
		} `json:"files"`
	}
	must(t, json.NewDecoder(response.Body).Decode(&document))
	for _, file := range document.Files {
		if file.Primary {
			return int64(file.SizeKB * 1024)
		}
	}
	t.Fatalf("civitai version %d has no primary file", version)
	return 0
}

// rentalCapture starts a daemon against a stand-in Hub that captures the first paid rental
// request and refuses it. A non-empty registry replaces this host's TensorFS registry.
func rentalCapture(t *testing.T, registry string) (string, func(*testing.T, []byte, int) hub.RentalRequest) {
	t.Helper()
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
	t.Cleanup(server.Close)
	root, err := os.MkdirTemp(scratchBase, "ingest-sizing-")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(root) })
	// An operator identity publishes to its named org without a Hub account read.
	cfg := "tensorhub_url: " + server.URL + "\ntensorhub_token: operator-proof\ndaemon:\n  idle_shutdown_s: 0\n"
	if registry != "" {
		path := filepath.Join(root, "host-registry.json")
		must(t, os.WriteFile(path, []byte(registry), 0o600))
		cfg += "tensorfs_registry: " + path + "\n"
	}
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(cfg), 0o600))
	startDaemonProcess(t, root)
	return root, func(t *testing.T, out []byte, code int) hub.RentalRequest {
		t.Helper()
		deadline := time.Now().Add(90 * time.Second)
		for {
			mu.Lock()
			seen, first := append([]string(nil), unexpected...), []byte(nil)
			if len(posts) > 0 {
				first = append(first, posts[0]...)
			}
			mu.Unlock()
			if first != nil {
				request, problem := hub.ParseRentalRequestBytes(first)
				fatal(t, problem)
				return request
			}
			if time.Now().After(deadline) {
				t.Fatalf("no rental was asked for [exit %d]: %s\nunserved Hub routes: %v\n%s", code, out, seen, tail(filepath.Join(root, "daemon.log")))
			}
			time.Sleep(50 * time.Millisecond)
		}
	}
}

func rentedUpload(t *testing.T, root string, args ...string) ([]byte, int) {
	t.Helper()
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin, "model", "upload"},
		append(args, "--rental-only", "--json")...)...)
	cmd.Env = childEnv(t, root)
	out, _ := cmd.CombinedOutput()
	return out, cmd.ProcessState.ExitCode()
}
