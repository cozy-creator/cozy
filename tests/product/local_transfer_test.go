package producttest

import (
	"encoding/binary"
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
	// The machine redeems its publication grant at the Hub, which grants nothing.
	h.oauth.refuse = publicationRequest
	root, err := os.MkdirTemp("", "czt")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
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
	refused(t, "a checkpoint into a local alias", "model_download_failed", "model", "download", "proof/model#"+manifest, "local/tiny")
	refused(t, "an absent alias's upload", "this machine holds no local/absent", "model", "upload", "local/absent", "proof/model")

	// One F16 tensor: written, made as it is, then published under the machine's own grant.
	header := []byte(`{"stray.weight":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}`)
	file := filepath.Join(root, "stray.safetensors")
	body := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	must(t, os.WriteFile(file, append(append(body, header...), 0, 0, 0, 0), 0o600))
	refused(t, "a local file's upload", "publication_unauthorized", "model", "upload", file, "proof/model")
	publications := 0
	for _, asked := range h.oauth.requests() {
		if publicationRequest(asked) {
			publications++
		}
	}
	if publications != 2 {
		t.Fatalf("want a publication grant for each upload, got %d", publications)
	}
}
