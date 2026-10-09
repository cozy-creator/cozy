package producttest

import (
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
)

// A local file, a local/ alias and a Hub checkpoint into a local/ alias are warm runs on this
// computer's machine: the file is written to it and made there, the alias is read from its
// store, the checkpoint is downloaded into its store. Each ends with the machine's own answer:
// the stand-in hub's private doors refuse the download, the alias is absent, and the file,
// made into a model, has its publication refused.
func TestLocalTransfersAreWarmRunsOnThisComputersMachine(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the machine this computer runs")
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
	root, err := os.MkdirTemp("", "czt")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	h.hubAccess.signIn(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("evidence retained at %s\nmachine log tail:\n%s", root, tail(filepath.Join(root, "machine", "host.log")))
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
	refused := func(t *testing.T, what, want string, args ...string) {
		t.Helper()
		code, out := runCozy(t, root, append(args, "--await", "--json")...)
		t.Logf("%s [exit %d]:\n%s", what, code, out)
		if code == 0 || !strings.Contains(out, want) {
			t.Fatalf("%s did not end with the machine's own answer %q [exit %d]\n%s", what, want, code, out)
		}
	}
	// The owner's checkpoint named by digest is a private read: the machine trades its
	// capability, then refuses the bytes that arrive by its own name.
	refused(t, "a checkpoint into a local alias", "model_download_failed", "model", "download", "proof/model#"+manifest, "local/tiny")
	refused(t, "an absent alias's upload", "this machine holds no local/absent", "model", "upload", "local/absent", "proof/model")

	// One F16 tensor: written, made as it is, then published under the run's capability,
	// which the Hub now refuses. An upload that ends before publishing trades nothing.
	h.hubAccess.refuseTrades()
	refused(t, "a local file's upload", "capability_refused", "model", "upload", strayTensor(t, root), "proof/model")
	if trades := h.hubAccess.trades(); len(trades) != 2 {
		t.Fatalf("want two trades, the download's read and the file's upload, got %d", len(trades))
	}
}
