package producttest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
)

// The actual CPU machine resolves a multi-lane model input, downloads real
// TensorFS objects and executes a job that never loads the Model. This complements
// the serving-wire regression without claiming accelerator serving qualification.
func TestNativeCPUModelInputKeepsAuthoredLaneAndExplicitOverrideAtOneHub(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires real CPU native machine fixture")
	}
	manifest, objects := probeCheckpoint(t)
	ref := func(raw []byte) map[string]any {
		sum := sha256.Sum256(raw)
		return map[string]any{"sha256": hex.EncodeToString(sum[:]), "length": len(raw)}
	}
	manifestRef := ref(manifest)
	var closure []any
	for digest, body := range objects {
		if digest != manifestRef["sha256"] {
			closure = append(closure, ref(body))
		}
	}
	var mu sync.Mutex
	var lanes []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := "http://" + r.Host
		encode := func(value any) { _ = json.NewEncoder(w).Encode(value) }
		switch {
		case r.URL.Path == "/v1/accounts/current":
			encode(map[string]any{"name": "proof"})
		case r.URL.Path == "/v1/execution-access":
			if r.Header.Get("Authorization") != "Bearer fixture-account" {
				http.Error(w, "wrong account", 403)
				return
			}
			encode(map[string]any{"token": "fixture-execution", "expires_at": time.Now().Add(time.Hour), "environment": map[string]string{"TENSORHUB_ORIGIN": origin}})
		case r.URL.Path == "/v1/models/resolve":
			lane := r.URL.Query().Get("lane")
			if r.Header.Get("Authorization") != "Bearer fixture-execution" {
				t.Error("model resolution did not use machine execution access")
				http.Error(w, "wrong authority", 403)
				return
			}
			if r.URL.Query().Get("ref") != "proof/probe@1.0.0" || lane != "bf16" && lane != "fp32" {
				http.Error(w, "multiple lanes require the authored or explicit selection", 404)
				return
			}
			mu.Lock()
			lanes = append(lanes, lane)
			mu.Unlock()
			encode(map[string]any{"model": "proof/probe", "release": "1.0.0", "lane": lane, "bytes": 4096, "manifest_id": "sha256:" + manifestRef["sha256"].(string), "manifest_length": len(manifest)})
		case r.URL.Path == "/v1/tensorfs/closure":
			encode(map[string]any{"complete": true, "model": "proof/probe", "release": "1.0.0", "lane": "bf16", "manifest": manifestRef, "objects": closure, "presign_max_digests": 64, "scope": "runtime"})
		case r.URL.Path == "/v1/tensorfs/presign":
			var asked struct{ Digests []string }
			_ = json.NewDecoder(r.Body).Decode(&asked)
			urls := map[string]string{}
			for _, digest := range asked.Digests {
				urls[digest] = origin + "/o/" + digest
			}
			encode(map[string]any{"urls": urls, "server_time_unix": time.Now().Unix(), "expires_at_unix": time.Now().Add(time.Hour).Unix()})
		case strings.HasPrefix(r.URL.Path, "/o/") && objects[strings.TrimPrefix(r.URL.Path, "/o/")] != nil:
			_, _ = w.Write(objects[strings.TrimPrefix(r.URL.Path, "/o/")])
		case r.URL.Path == "/v1/rentals":
			encode(map[string]any{"rentals": []any{}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	root, err := os.MkdirTemp("", "czlad-")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\ntensorhub_token: fixture-account\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Log("CPU ladder evidence", root)
		} else {
			_ = removeAllForce(root)
		}
	})
	args := []string{"machine", "install", "--host", *machineHostBinary}
	if *machineRuntimeWheel != "" {
		args = append(args, "--runtime-wheel", *machineRuntimeWheel, "--tensorfs-wheel", *machineTensorFSWheel)
	}
	if code, out := runCozy(t, root, args...); code != 0 {
		t.Fatalf("machine install: %d %s", code, out)
	}
	project := probeProject(t, "proof/probe@1.0.0/bf16")
	for _, which := range []string{"authored", "explicit"} {
		mu.Lock()
		before := len(lanes)
		mu.Unlock()
		args := []string{"run", project + "/touch", "value=1", "--await", "--json"}
		want := "bf16"
		if which == "explicit" {
			args = append(args, "model.source=proof/probe@1.0.0/fp32")
			want = "fp32"
		}
		if code, out := runCozy(t, root, args...); code != 0 || !strings.Contains(out, `"value":2`) {
			t.Fatalf("%s model input: %d %s", which, code, out)
		}
		mu.Lock()
		got := append([]string(nil), lanes[before:]...)
		mu.Unlock()
		if len(got) != 1 || got[0] != want {
			t.Fatalf("%s selected model lane %v, expected %s at configured Hub", which, got, want)
		}
	}
}
