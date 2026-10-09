package producttest

import (
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
)

// `cozy model download` without --rental installs into this computer's machine through the
// same durable queue and Host preparation a rental uses: the daemon dials the real local
// Host, never a classic worker, and the Host's own answer settles the installation.
func TestLocalModelDownloadRunsOnThisComputersMachine(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the pod-supervisor this computer's machine runs")
	}
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv lays out the machine root")
	}
	h := newMachineHub(t)
	manifest := "sha256:" + strings.Repeat("a", 64)
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models/resolve" {
			_ = json.NewEncoder(w).Encode(map[string]any{"model": "proof/model", "manifest_id": manifest,
				"manifest_length": 128, "bytes": 4096, "components": []string{"transformer"}})
			return
		}
		served.ServeHTTP(w, r)
	})
	serveOtherBytes(h, manifest)
	root, err := os.MkdirTemp("", "czd")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			log, _ := os.ReadFile(filepath.Join(root, "machine", "host.log"))
			t.Logf("evidence retained at %s\nlocal Host log:\n%s", root, log)
		} else {
			_ = removeAllForce(root)
		}
	})
	install := []string{"machine", "install", "--host", *machineHostBinary}
	if *machineRuntimeWheel != "" {
		install = append(install, "--runtime-wheel", *machineRuntimeWheel, "--tensorfs-wheel", *machineTensorFSWheel)
	}
	if code, out := runCozy(t, root, install...); code != 0 {
		t.Fatalf("machine install [exit %d]\n%s", code, out)
	}
	// The caller's own checkpoint named by digest is a private read: the run carries a
	// capability its device key signs (th-241).
	h.hubAccess.signIn(t, root)
	code, out := runCozy(t, root, "model", "download", "proof/model#"+manifest, "--json")
	var accepted struct {
		ID     string `json:"id"`
		Rental string `json:"rental"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &accepted) != nil || accepted.Rental != machines.Local || accepted.ID == "" {
		t.Fatalf("local model download was not accepted for this computer's machine [exit %d]\n%s", code, out)
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	var settled *records.RentalInstall
	deadline := time.Now().Add(3 * time.Minute)
	for settled == nil && time.Now().Before(deadline) {
		row, problem := store.RentalInstall(accepted.ID)
		fatal(t, problem)
		if row != nil && (row.State == "succeeded" || row.State == "failed") {
			settled = row
		}
		time.Sleep(250 * time.Millisecond)
	}
	if settled == nil {
		t.Fatal("the local machine never settled the installation")
	}
	t.Logf("local installation settled: %s %s: %s", settled.State, settled.ErrorCode, settled.Error)
	// The stand-in hub serves bytes that are not the model, so this computer's machine refuses
	// the download by its own name. What must never appear is a rental-only refusal or a
	// classic local worker.
	if settled.State != "failed" || strings.HasPrefix(settled.ErrorCode, "rental.") ||
		!strings.HasPrefix(settled.ErrorCode, "machine") && !strings.Contains(settled.ErrorCode, "download") {
		t.Fatalf("the installation did not reach the local Host: %+v", settled)
	}
	requests, problem := store.Requests("", "", 10)
	fatal(t, problem)
	if len(requests) != 0 {
		t.Fatalf("a model download submitted a run: %+v", requests)
	}
}

// serveOtherBytes has the machine's Hub doors (behind their private CA) resolve manifest to
// a closure of one object and presign both onto bytes that do not hash to them: the Host's
// fetch runs end to end and its TensorFS refuses what arrives.
func serveOtherBytes(h *machineHub, manifest string) {
	lengths := map[string]int{strings.TrimPrefix(manifest, "sha256:"): 128, strings.Repeat("b", 64): 64}
	doors := h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/tensorfs/closure":
			var asked struct{ Lane string }
			_ = json.NewDecoder(r.Body).Decode(&asked)
			_ = json.NewEncoder(w).Encode(map[string]any{"complete": true, "lane": asked.Lane, "model": "proof/model",
				"manifest":            map[string]any{"length": 128, "sha256": strings.TrimPrefix(manifest, "sha256:")},
				"objects":             []any{map[string]any{"length": 64, "sha256": strings.Repeat("b", 64)}},
				"presign_max_digests": 64, "release": "", "scope": "runtime"})
		case r.URL.Path == "/v1/tensorfs/presign":
			var asked struct{ Digests []string }
			_ = json.NewDecoder(r.Body).Decode(&asked)
			urls := map[string]string{}
			for _, digest := range asked.Digests {
				urls[digest] = h.worker.URL + "/o/" + digest
			}
			now := time.Now().Unix()
			_ = json.NewEncoder(w).Encode(map[string]any{"expires_at_unix": now + 3600, "server_time_unix": now, "urls": urls})
		case strings.HasPrefix(r.URL.Path, "/o/") && lengths[strings.TrimPrefix(r.URL.Path, "/o/")] > 0:
			_, _ = w.Write(make([]byte, lengths[strings.TrimPrefix(r.URL.Path, "/o/")]))
		default:
			doors.ServeHTTP(w, r)
		}
	})
}
