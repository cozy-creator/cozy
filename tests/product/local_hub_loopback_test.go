package producttest

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
)

// A Hub its owner reaches on loopback runs on this computer, so this computer's machine reads
// it there too. With the public origin the Hub declares for remote pods down (a tunnel after a
// reboot), a run still reads its Model's catalog rows and pulls its weights.
func TestALoopbackHubServesItsMachineWhileItsPublicOriginIsDown(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: this computer's machine runs the call")
	}
	const public = "https://public-origin.invalid"
	header, err := base64.StdEncoding.DecodeString(probeHeader)
	must(t, err)
	ref := func(raw []byte) map[string]any {
		digest := sha256.Sum256(raw)
		return map[string]any{"length": len(raw), "sha256": hex.EncodeToString(digest[:])}
	}
	manifest, err := json.Marshal(map[string]any{"entries": []any{map[string]any{"blob": ref(header), "kind": "cozytensors", "path": "model.cozytensors"}}})
	must(t, err)
	vocab := []byte("vocab\n")
	objects := map[string][]byte{}
	for _, raw := range [][]byte{manifest, header, vocab} {
		objects[ref(raw)["sha256"].(string)] = raw
	}

	h := newMachineHub(t)
	access := newFakeHubAccess(h.server.URL, public, "rental-idle-test")
	var mu sync.Mutex
	var reads, credentialed []string
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		reads = append(reads, r.Method+" "+r.URL.Path)
		// A public run's machine reads with no credential (th-241).
		if r.Header.Get("DPoP") != "" || strings.HasPrefix(r.URL.Path, "/v1/tensorfs/") && r.Header.Get("Authorization") != "" {
			credentialed = append(credentialed, r.Method+" "+r.URL.Path)
		}
		mu.Unlock()
		if access.serve(w, r) {
			return
		}
		switch {
		case r.URL.Path == "/v1/models/proof/probe":
			_, _ = w.Write([]byte(`{"releases":[{"release":"1.0.0","lanes":[{"lane":"bf16","bytes":1}]}]}`))
		case r.URL.Path == "/v1/models/resolve" && r.URL.Query().Get("ref") == "proof/probe@1.0.0" && r.URL.Query().Get("lane") == "bf16":
			_ = json.NewEncoder(w).Encode(map[string]any{"model": "proof/probe", "release": "1.0.0", "lane": "bf16",
				"manifest_id": "sha256:" + ref(manifest)["sha256"].(string), "manifest_length": len(manifest)})
		case r.URL.Path == "/v1/tensorfs/closure":
			_ = json.NewEncoder(w).Encode(map[string]any{"complete": true, "lane": "bf16", "model": "proof/probe", "release": "1.0.0",
				"manifest": ref(manifest), "objects": []any{ref(header), ref(vocab)}, "presign_max_digests": 64, "scope": "runtime"})
		case r.URL.Path == "/v1/tensorfs/presign":
			var asked struct{ Digests []string }
			_ = json.NewDecoder(r.Body).Decode(&asked)
			urls := map[string]string{}
			for _, digest := range asked.Digests {
				urls[digest] = h.server.URL + "/o/" + digest
			}
			now := time.Now().Unix()
			_ = json.NewEncoder(w).Encode(map[string]any{"expires_at_unix": now + 3600, "server_time_unix": now, "urls": urls})
		case strings.HasPrefix(r.URL.Path, "/o/") && objects[strings.TrimPrefix(r.URL.Path, "/o/")] != nil:
			_, _ = w.Write(objects[strings.TrimPrefix(r.URL.Path, "/o/")])
		default:
			served.ServeHTTP(w, r)
		}
	})

	root, err := os.MkdirTemp("", "czl")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if !t.Failed() {
			_ = removeAllForce(root)
		}
	})
	provisionMachine(t, root)
	if code, out := runCozy(t, root, "package", "install", probeProject(t, "proof/probe@1.0.0/bf16"), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	run := func() string {
		t.Helper()
		code, out := runCozy(t, root, "run", parityPackage+"/touch", "value=1", "model.source=proof/probe", "--await", "--json")
		mu.Lock()
		defer mu.Unlock()
		seen := strings.Join(reads, "\n")
		if code != 0 || !strings.Contains(out, `"value":2`) {
			t.Fatalf("run [exit %d]\n%s\nthe Hub saw:\n%s", code, out, seen)
		}
		return seen
	}
	seen := run()
	for _, read := range []string{"GET /v1/models/proof/probe", "GET /v1/models/resolve", "POST /v1/tensorfs/closure", "POST /v1/tensorfs/presign",
		"GET /o/" + ref(header)["sha256"].(string)} {
		if !strings.Contains(seen, read) {
			t.Errorf("the machine did not read %q at the Hub's loopback origin; it saw:\n%s", read, seen)
		}
	}

	// A second native run still reaches the local Hub while its public origin is down.
	run()
	mu.Lock()
	defer mu.Unlock()
	if len(credentialed) != 0 {
		t.Fatalf("a public run's machine presented a credential: %q", credentialed)
	}
}
