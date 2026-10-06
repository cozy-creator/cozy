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

	"github.com/cozy-creator/cozy/internal/machines"
)

// probeHeader is proof/probe's canonical CozyTensors header, as TensorFS produced it.
const probeHeader = "hW1jb3p5dGVuc29ycy8xgYJkdW5ldFgceyJoaWRkZW5fc2l6ZSI6NCwibGF5ZXJzIjoxfYGFc3Rv" +
	"a2VuaXplci92b2NhYi50eHRYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63Bmp0ZXh0" +
	"L3BsYWlugYJYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63BoGEjAMLAgEABAUIBwYJC" +
	"oCBg2V2YWx1ZYEAgQCBglggR2b5Mbu3DtQx40s05Pn8XdqBdOlvQ6is152Hme/NQ9IZAgSBgmR1bm" +
	"V0gYVmd2VpZ2h0AYEBAIGEZXZhbHVlAYEBRAAAAAA="

// probeCheckpoint is proof/probe@1.0.0/bf16 as TensorFS made it: its manifest and its
// objects (the canonical CozyTensors header and its asset), by sha256.
func probeCheckpoint(t *testing.T) ([]byte, map[string][]byte) {
	t.Helper()
	header, err := base64.StdEncoding.DecodeString(probeHeader)
	must(t, err)
	digest := func(raw []byte) string {
		sum := sha256.Sum256(raw)
		return hex.EncodeToString(sum[:])
	}
	manifest, err := json.Marshal(map[string]any{"entries": []any{map[string]any{
		"blob": map[string]any{"length": len(header), "sha256": digest(header)}, "kind": "cozytensors", "path": "model.cozytensors"}}})
	must(t, err)
	objects := map[string][]byte{}
	for _, raw := range [][]byte{manifest, header, []byte("vocab\n")} {
		objects[digest(raw)] = raw
	}
	return manifest, objects
}

// seedProbe lands proof/probe@1.0.0/bf16 in each named machine's store (this computer's, or a
// rental's name) as a user does, with `cozy model download` from the stand-in Hub, and
// answers the Hub's resolution of it.
func seedProbe(t *testing.T, h *machineHub, root string, venues ...string) map[string]any {
	t.Helper()
	manifest, objects := probeCheckpoint(t)
	ref := func(raw []byte) map[string]any {
		sum := sha256.Sum256(raw)
		return map[string]any{"length": len(raw), "sha256": hex.EncodeToString(sum[:])}
	}
	var closure []any
	bytes := 0
	for digest, raw := range objects {
		if digest != ref(manifest)["sha256"] {
			closure = append(closure, ref(raw))
			bytes += len(raw)
		}
	}
	resolved := map[string]any{"model": "proof/probe", "release": "1.0.0", "lane": "bf16", "bytes": bytes,
		"manifest_id": "sha256:" + ref(manifest)["sha256"].(string), "manifest_length": len(manifest)}
	doors := h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/tensorfs/closure":
			_ = json.NewEncoder(w).Encode(map[string]any{"complete": true, "lane": "bf16", "model": "proof/probe", "release": "1.0.0",
				"manifest": ref(manifest), "objects": closure, "presign_max_digests": 64, "scope": "runtime"})
		case r.URL.Path == "/v1/tensorfs/presign":
			var asked struct{ Digests []string }
			_ = json.NewDecoder(r.Body).Decode(&asked)
			urls := map[string]string{}
			for _, digest := range asked.Digests {
				urls[digest] = h.access.URL + "/o/" + digest
			}
			now := time.Now().Unix()
			_ = json.NewEncoder(w).Encode(map[string]any{"expires_at_unix": now + 3600, "server_time_unix": now, "urls": urls})
		case strings.HasPrefix(r.URL.Path, "/o/") && objects[strings.TrimPrefix(r.URL.Path, "/o/")] != nil:
			_, _ = w.Write(objects[strings.TrimPrefix(r.URL.Path, "/o/")])
		default:
			doors.ServeHTTP(w, r)
		}
	})
	account := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/models/resolve" && strings.HasPrefix(r.URL.Query().Get("ref"), "proof/probe"):
			_ = json.NewEncoder(w).Encode(resolved)
		default:
			account.ServeHTTP(w, r)
		}
	})
	for _, venue := range venues {
		args := []string{"model", "download", "proof/probe#" + resolved["manifest_id"].(string), "--await", "--json"}
		if venue != machines.Local {
			args = append(args, "--rental="+venue)
		}
		if code, out := runCozy(t, root, args...); code != 0 {
			t.Fatalf("seeding proof/probe on %s [exit %d]\n%s", venue, code, out)
		}
	}
	h.server.Config.Handler, h.worker.Config.Handler = account, doors
	return resolved
}

// probeProject is the parity package with a CPU job reading one Model slot by its authored
// default lane: a derive-only Manifest capability, so the job never loads it.
func probeProject(t *testing.T, lane string) string {
	t.Helper()
	return probeProjectOn(t, machines.Source{RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel}, lane)
}

// probeProjectOn is probeProject locked against source's wheels, or the published Runtime.
func probeProjectOn(t *testing.T, source machines.Source, lane string) string {
	t.Helper()
	project := parityProjectOn(t, source)
	must(t, os.WriteFile(filepath.Join(project, "machine_parity.py"), []byte(`import msgspec
from cozy_runtime.author import App, Loader, Model


class Nothing:
    pass


class Probe(Model[Nothing]):
    def load(self, loader: Loader) -> None:
        raise RuntimeError("a job never loads its Model")


class TouchRequest(msgspec.Struct, forbid_unknown_fields=True):
    value: int


class TouchResult(msgspec.Struct):
    value: int


app = App()


@app.job(defaults={"source": [{"gpu": "*", "lane": "`+lane+`"}]})
def touch(payload: TouchRequest, source: Probe) -> TouchResult:
    return TouchResult(value=payload.value + 1)
`), 0o600))
	return project
}

// A warm run of a published release makes no Hub content request, from the CLI or machine:
// the machine read the owner's binding and resolved its lane at its own Hub on the cold run
// and kept both. Rental authority still refreshes independently of content reuse.
func TestAWarmRunReadsNothingAtAnyHub(t *testing.T) {
	h, root, _, _ := parityMachines(t)
	resolved := seedProbe(t, h, root, machines.Local, "tessa")
	// This proof measures content reuse after the machine can describe releases.
	// A still-booting Runtime legitimately falls back to the account catalog,
	// which this fixture deliberately does not serve.
	machine := machines.NewHost(filepath.Join(root, "machine"), machineStore(root), nil)
	_, problem := machine.Ensure(t.Context(), "", nil, true)
	fatal(t, problem)
	eventually(t, root, "native machine readiness", func() bool {
		frame, problem := machine.ReadStatus(t.Context())
		return problem == nil && frame != nil && frame.Phase == "ready"
	})
	var mu sync.Mutex
	var seen []string
	count := func(hub string, handler http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			authority := r.Method == http.MethodGet && r.URL.Path == "/v1/worker/rental/authorized-keys"
			if !strings.HasPrefix(r.URL.Path, "/v1/rentals") && !authority {
				mu.Lock()
				seen = append(seen, hub+" "+r.Method+" "+r.URL.Path)
				mu.Unlock()
			}
			handler.ServeHTTP(w, r)
		})
	}
	publishParityRelease(t, h, root, probeProject(t, "proof/probe@1.0.0/bf16"))
	binding := `{"bindings":[{"slot":"touch.models.source","model":"proof/probe","release":"1.0.0","revision":1,"ladder":[{"gpu":"*","lane":"bf16"}]}]}`
	doors := h.worker.Config.Handler
	h.worker.Config.Handler = count("machine", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/packages/"+parityPublished+"/bindings":
			_, _ = w.Write([]byte(binding))
		case r.URL.Path == "/v1/models/resolve" && r.URL.Query().Get("ref") == "proof/probe@1.0.0" && r.URL.Query().Get("lane") == "bf16":
			_ = json.NewEncoder(w).Encode(resolved)
		default:
			doors.ServeHTTP(w, r)
		}
	}))
	h.server.Config.Handler = count("account", h.server.Config.Handler)

	for _, venue := range []struct {
		name string
		args []string
	}{{"local", nil}, {"rental", []string{"--rental=tessa"}}} {
		for _, key := range []string{"cold", "warm"} {
			mu.Lock()
			seen = nil
			mu.Unlock()
			code, out := runCozy(t, root, append([]string{"run", parityPublished + "/touch", "value=1", "--await", "--json"}, venue.args...)...)
			if code != 0 || !strings.Contains(out, `"value":2`) {
				t.Fatalf("%s touch on %s [exit %d]\n%s", key, venue.name, code, out)
			}
			mu.Lock()
			calls := append([]string(nil), seen...)
			mu.Unlock()
			t.Logf("%s run on %s: %v", key, venue.name, calls)
			if key == "cold" && !strings.Contains(strings.Join(calls, "\n"), "machine GET /v1/models/resolve") {
				t.Fatalf("the cold run on %s never resolved its Model at the machine's Hub: %v", venue.name, calls)
			}
			if key == "warm" && len(calls) != 0 {
				t.Fatalf("the warm run on %s made %d Hub requests; want none: %v", venue.name, len(calls), calls)
			}
		}
	}

	// The owner's change to the package reaches every machine this computer knows, over the
	// machine connection: each reads the package and its Model once more, then nothing again.
	h.mux.HandleFunc("DELETE /v1/packages/"+parityPublished+"/releases/"+parityVersion, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"release":"` + parityVersion + `","state":"yanked","changed":true}`))
	})
	if code, out := runCozy(t, root, "package", "yank", parityPublished, "--version", parityVersion); code != 0 || strings.Contains(out, "keeps what it read") || strings.Contains(out, "no machine was told") {
		t.Fatalf("yank did not reach every machine [exit %d]\n%s", code, out)
	}
	for venue, args := range map[string][]string{"local": nil, "tessa": {"--rental=tessa"}} {
		for _, key := range []string{"changed", "warm again"} {
			mu.Lock()
			seen = nil
			mu.Unlock()
			if code, out := runCozy(t, root, append([]string{"run", parityPublished + "/touch", "value=1", "--await", "--json"}, args...)...); code != 0 || !strings.Contains(out, `"value":2`) {
				t.Fatalf("%s touch on %s [exit %d]\n%s", key, venue, code, out)
			}
			mu.Lock()
			calls := strings.Join(seen, "\n")
			mu.Unlock()
			read := strings.Contains(calls, "machine GET /v1/packages/"+parityPublished+"/bindings") && strings.Contains(calls, "machine GET /v1/models/resolve")
			if (key == "changed") != read || key == "warm again" && calls != "" {
				t.Fatalf("the %s run on %s read: %q", key, venue, calls)
			}
		}
	}

	// Unpublished code is no different. Its org-relative default names the caller's account,
	// read once and kept, which the root carries as its owner; the machine resolves the slot.
	h.mux.HandleFunc("GET /v1/accounts/current", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"name":"proof"}`))
	})
	if code, out := runCozy(t, root, "package", "install", probeProject(t, "probe@1.0.0/bf16"), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	for venue, args := range map[string][]string{"local": nil, "tessa": {"--rental=tessa"}} {
		for _, key := range []string{"cold", "warm"} {
			mu.Lock()
			seen = nil
			mu.Unlock()
			if code, out := runCozy(t, root, append([]string{"run", "local/machine-parity/touch", "value=1", "--await", "--json"}, args...)...); code != 0 || !strings.Contains(out, `"value":2`) {
				t.Fatalf("%s local/ touch on %s [exit %d]\n%s", key, venue, code, out)
			}
			mu.Lock()
			calls := strings.Join(seen, "\n")
			mu.Unlock()
			t.Logf("%s local/ run on %s: %q", key, venue, calls)
			if key == "cold" && !strings.Contains(calls, "machine GET /v1/models/resolve") || key == "warm" && calls != "" {
				t.Fatalf("the %s local/ run on %s read: %q", key, venue, calls)
			}
		}
	}
}
