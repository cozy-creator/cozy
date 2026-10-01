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
	var mu sync.Mutex
	var reads []string
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		reads = append(reads, r.Method+" "+r.URL.Path)
		mu.Unlock()
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/execution-access":
			if r.Header.Get("Authorization") != "Bearer rental-idle-test" {
				http.Error(w, "account required", http.StatusUnauthorized)
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"token": executionGrantToken(public, "fixture-account", 1), "expires_at": time.Now().Add(time.Hour),
				"environment": map[string]string{"TENSORHUB_ORIGIN": public, "TENSORHUB_PUBLIC_ORIGIN": public}})
		case r.URL.Path == "/v1/accounts/current":
			_, _ = w.Write([]byte(`{"name":"proof"}`))
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

	// A grant an older Cozy cached for the public origin is minted again for loopback.
	cache := filepath.Join(root, "machine", "execution-access.json")
	raw, err := os.ReadFile(cache)
	must(t, err)
	var cached map[string]map[string]any
	must(t, json.Unmarshal(raw, &cached))
	for _, grant := range cached {
		grant["origin"] = public
		grant["environment"].(map[string]any)["TENSORHUB_ORIGIN"] = public
	}
	raw, err = json.Marshal(cached)
	must(t, err)
	must(t, os.WriteFile(cache, raw, 0o600))
	if minted := strings.Count(run(), "POST /v1/execution-access"); minted != 2 {
		t.Fatalf("a grant cached for the public origin was minted %d times, want once more", minted-1)
	}
	raw, err = os.ReadFile(cache)
	must(t, err)
	if !strings.Contains(string(raw), `"origin":"`+h.server.URL+`"`) || strings.Contains(string(raw), `"origin":"`+public+`"`) {
		t.Fatalf("the renewed grant is cached at the wrong origin: %s", raw)
	}

	raw, err = os.ReadFile(filepath.Join(root, "machine", "root", "var", "lib", "cozy", "machine", "hub-access.json"))
	must(t, err)
	var held struct {
		Hubs []struct {
			Origin      string            `json:"origin"`
			Environment map[string]string `json:"environment"`
		} `json:"hubs"`
	}
	must(t, json.Unmarshal(raw, &held))
	if len(held.Hubs) != 1 || held.Hubs[0].Origin != h.server.URL || held.Hubs[0].Environment["TENSORHUB_ORIGIN"] != h.server.URL ||
		held.Hubs[0].Environment["TENSORHUB_PUBLIC_ORIGIN"] != public {
		t.Fatalf("the machine holds Hub access at %+v, want %s with public origin %s", held.Hubs, h.server.URL, public)
	}
}
