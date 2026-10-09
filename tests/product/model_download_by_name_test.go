package producttest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
)

// `cozy model download` names the model (th-245): this computer's machine resolves its release
// lane with the Hub's closure, anonymously, and downloads the checkpoint the closure names, by
// release and lane, and by the name alone, which reads the release's default lane once its
// owner names one. No hash travels and no capability is traded.
func TestAModelDownloadNamesItsReleaseAndLane(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: this computer's machine downloads the model")
	}
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv lays out the machine root")
	}
	h := newMachineHub(t)
	manifest, objects := probeCheckpoint(t)
	sum := sha256.Sum256(manifest)
	digest := hex.EncodeToString(sum[:])
	row := func(raw []byte) map[string]any {
		sum := sha256.Sum256(raw)
		return map[string]any{"length": len(raw), "sha256": hex.EncodeToString(sum[:])}
	}
	var closure []any
	for hash, raw := range objects {
		if hash != digest {
			closure = append(closure, row(raw))
		}
	}
	var mu sync.Mutex
	var asked []string
	doors := h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/tensorfs/closure":
			var named struct{ Ref, Lane string }
			_ = json.NewDecoder(r.Body).Decode(&named)
			mu.Lock()
			asked = append(asked, strings.TrimSpace(named.Ref+" "+named.Lane+" "+r.Header.Get("Authorization")))
			mu.Unlock()
			if named.Lane == "" {
				named.Lane = "bf16"
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"complete": true, "lane": named.Lane, "model": "proof/probe", "release": "1.0.0",
				"manifest": row(manifest), "objects": closure, "presign_max_digests": 64, "scope": "runtime"})
		case r.URL.Path == "/v1/tensorfs/presign":
			var ask struct{ Digests []string }
			_ = json.NewDecoder(r.Body).Decode(&ask)
			urls := map[string]string{}
			for _, hash := range ask.Digests {
				urls[hash] = h.access.URL + "/o/" + hash
			}
			now := time.Now().Unix()
			_ = json.NewEncoder(w).Encode(map[string]any{"expires_at_unix": now + 3600, "server_time_unix": now, "urls": urls})
		case strings.HasPrefix(r.URL.Path, "/o/") && objects[strings.TrimPrefix(r.URL.Path, "/o/")] != nil:
			_, _ = w.Write(objects[strings.TrimPrefix(r.URL.Path, "/o/")])
		default:
			doors.ServeHTTP(w, r)
		}
	})
	// Two lanes of one checkpoint; the release's default lane (th-245) is what a ref with no
	// lane reads, and the owner sets it with `model publish --default-lane`.
	defaultLane, published := "", ""
	account := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		lane := func(name string) map[string]any {
			return map[string]any{"lane": name, "manifest_id": "sha256:" + digest, "components": []string{}, "bytes": len(closure)}
		}
		release := map[string]any{"release": "1.0.0", "revision": 1, "default_lane": defaultLane, "lanes": []any{lane("bf16"), lane("fp8-pruned")}}
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/models/proof/probe":
			_ = json.NewEncoder(w).Encode(map[string]any{"model": map[string]string{"org": "proof", "name": "probe"}, "releases": []any{release}})
		case r.Method == http.MethodPost && r.URL.Path == "/v1/models/proof/probe/releases/1.0.0":
			var body struct {
				DefaultLane string `json:"default_lane"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			published, defaultLane = body.DefaultLane, body.DefaultLane
			_ = json.NewEncoder(w).Encode(map[string]any{"release": "1.0.0", "revision": 1, "changed": true, "default_lane": defaultLane,
				"lanes": []any{map[string]string{"lane": "bf16", "checkpoint_id": "sha256:" + digest}, map[string]string{"lane": "fp8-pruned", "checkpoint_id": "sha256:" + digest}}})
		default:
			account.ServeHTTP(w, r)
		}
	})
	root, err := os.MkdirTemp("", "czn")
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
	install := []string{"machine", "install", "--host", *machineHostBinary}
	if *machineRuntimeWheel != "" {
		install = append(install, "--runtime-wheel", *machineRuntimeWheel, "--tensorfs-wheel", *machineTensorFSWheel)
	}
	if code, out := runCozy(t, root, install...); code != 0 {
		t.Fatalf("machine install [exit %d]\n%s", code, out)
	}
	h.hubAccess.signIn(t, root)
	download := func(ref string) {
		t.Helper()
		if code, out := runCozy(t, root, "model", "download", ref, "--await", "--json"); code != 0 {
			t.Fatalf("model download %s [exit %d]\n%s", ref, code, out)
		}
	}
	download("proof/probe@1.0.0/bf16")
	if code, out := runCozy(t, root, "model", "publish", "proof/probe", "--release", "1.0.0", "--default-lane", "fp8-pruned", "--json"); code != 0 || published != "fp8-pruned" || !strings.Contains(out, `"default_lane":"fp8-pruned"`) {
		t.Fatalf("model publish --default-lane [exit %d] sent %q\n%s", code, published, out)
	}
	download("proof/probe")
	download("proof/probe@1.0.0")
	mu.Lock()
	defer mu.Unlock()
	t.Logf("the machine asked the closure for: %q", asked)
	if len(asked) < 2 || asked[0] != "proof/probe@1.0.0 bf16" || !slices.Contains(asked, "proof/probe@1.0.0 fp8-pruned") || slices.ContainsFunc(asked, func(ask string) bool {
		return strings.HasPrefix(ask, "proof/probe@sha256:") || strings.Contains(ask, "DPoP")
	}) {
		t.Fatalf("the machine did not resolve the model by name and default lane, anonymously: %q", asked)
	}
	if trades := h.hubAccess.trades(); len(trades) != 0 {
		t.Fatalf("a public model download traded a capability: %d", len(trades))
	}
	if code, out := runCozy(t, root, "model", "list", "--json"); code != 0 || !strings.Contains(out, "proof/probe") || !strings.Contains(out, "bf16") {
		t.Fatalf("the machine does not hold the downloaded model [exit %d]\n%s", code, out)
	}
}
