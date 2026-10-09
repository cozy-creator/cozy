package producttest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// A multi-lane model input keeps its lane: the authored default's ladder picks bf16 and an
// explicit override fp32. The run names them (th-245); the actual CPU machine resolves each
// release lane with the Hub's closure, anonymously, downloads real TensorFS objects and
// executes a job that never loads the Model. This complements the serving-wire regression
// without claiming accelerator serving qualification.
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
	var machineReads, asked []string
	var access *fakeHubAccess
	digest := "sha256:" + manifestRef["sha256"].(string)
	card, err := json.Marshal(map[string]any{"model": map[string]string{"org": "proof", "name": "probe"},
		"releases": []any{map[string]any{"release": "1.0.0", "lanes": []any{
			map[string]any{"lane": "bf16", "manifest_id": digest, "components": []string{}, "bytes": 4096},
			map[string]any{"lane": "fp32", "manifest_id": digest, "components": []string{}, "bytes": 4096}}}}})
	must(t, err)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := "http://" + r.Host
		encode := func(value any) { _ = json.NewEncoder(w).Encode(value) }
		if access.serve(w, r) {
			return
		}
		if strings.HasPrefix(r.URL.Path, "/v1/tensorfs/") {
			mu.Lock()
			machineReads = append(machineReads, r.URL.Path+" "+r.Header.Get("Authorization"))
			mu.Unlock()
		}
		switch {
		case r.URL.Path == "/v1/models/proof/probe":
			_, _ = w.Write(card)
		case strings.HasPrefix(r.URL.Path, "/v1/models/proof/probe/releases/1.0.0/lanes/") && strings.HasSuffix(r.URL.Path, "/manifest"):
			_, _ = w.Write(manifest) // the CLI sizes a pinned job input by its manifest
		case r.URL.Path == "/v1/tensorfs/closure":
			var named struct{ Ref, Lane string }
			_ = json.NewDecoder(r.Body).Decode(&named)
			mu.Lock()
			asked = append(asked, named.Ref+" "+named.Lane)
			mu.Unlock()
			encode(map[string]any{"complete": true, "model": "proof/probe", "release": "1.0.0", "lane": named.Lane, "manifest": manifestRef, "objects": closure, "presign_max_digests": 64, "scope": "runtime"})
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
	access = newFakeHubAccess(server.URL, server.URL, "fixture-account")
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
		args := []string{"run", project + "/touch", "value=1", "--await", "--json", "--idempotency-key=" + which}
		if which == "explicit" {
			args = append(args, "model.source=proof/probe@1.0.0/fp32")
		}
		if code, out := runCozy(t, root, args...); code != 0 || !strings.Contains(out, `"value":2`) {
			t.Fatalf("%s model input: %d %s", which, code, out)
		}
		store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
		fatal(t, problem)
		request, problem := store.RequestByIdempotencyKey(which)
		store.Close()
		fatal(t, problem)
		if request == nil || len(request.Models) != 1 {
			t.Fatalf("%s run recorded no one model: %+v", which, request)
		}
		model := request.Models[0]
		switch which {
		case "authored":
			if model.Pinned() || len(model.Ladder) != 1 || model.Ladder[0].Lane != "bf16" || model.Ladder[0].Manifest != digest {
				t.Fatalf("the authored default is not its ladder bound to bf16's checkpoint: %+v", model)
			}
		case "explicit":
			if model.Lane != "fp32" || model.Manifest != digest {
				t.Fatalf("the explicit override is not fp32's checkpoint: %+v", model)
			}
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(machineReads) == 0 || slices.ContainsFunc(machineReads, func(read string) bool { return !strings.HasSuffix(read, " ") }) {
		t.Fatalf("the machine's content reads are absent or carried a credential: %q", machineReads)
	}
	if !slices.Contains(asked, "proof/probe@1.0.0 bf16") || !slices.Contains(asked, "proof/probe@1.0.0 fp32") ||
		slices.ContainsFunc(asked, func(ref string) bool { return strings.HasPrefix(ref, "proof/probe@sha256:") }) {
		t.Fatalf("the machine did not resolve each lane by name: %q", asked)
	}
}
