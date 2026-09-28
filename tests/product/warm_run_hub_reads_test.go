package producttest

import (
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// seedCheckpoint lands proof/probe@1.0.0/bf16 in a machine's TensorFS store with the machine's
// own Python, as a cold pull would: the canonical CozyTensors header TensorFS produced, its
// asset, and the released lane. It answers the manifest digest and length.
const seedCheckpoint = `# //cozy:allow a fixture lands the one checkpoint a cold pull would, in the machine's own store
import base64, hashlib, json, os, sys, tempfile
import tensorfs
root = sys.argv[1]
store = tensorfs.Store.open(root) if os.path.isdir(root) else tensorfs.Store.init(root)
header = base64.b64decode(
    "hW1jb3p5dGVuc29ycy8xgYJkdW5ldFgceyJoaWRkZW5fc2l6ZSI6NCwibGF5ZXJzIjoxfYGFc3Rv"
    "a2VuaXplci92b2NhYi50eHRYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63Bmp0ZXh0"
    "L3BsYWlugYJYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63BoGEjAMLAgEABAUIBwYJC"
    "oCBg2V2YWx1ZYEAgQCBglggR2b5Mbu3DtQx40s05Pn8XdqBdOlvQ6is152Hme/NQ9IZAgSBgmR1bm"
    "V0gYVmd2VpZ2h0AYEBAIGEZXZhbHVlAYEBRAAAAAA=")
ref = lambda raw: {"length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
with tempfile.TemporaryDirectory() as scratch:
    for name, raw in (("header.cbor", header), ("vocab.txt", b"vocab\n")):
        path = os.path.join(scratch, name)
        open(path, "wb").write(raw)
        store.put_file(path, "sha256:" + ref(raw)["sha256"], len(raw))
raw = json.dumps({"entries": [{"blob": ref(header), "kind": "cozytensors", "path": "model.cozytensors"}]},
    sort_keys=True, separators=(",", ":")).encode()
manifest = "sha256:" + hashlib.sha256(raw).hexdigest()
store.put_manifest(raw, manifest, len(raw))
operation = store.begin_operation("seed", "proof", "probe")
operation.hold_manifest(manifest, len(raw))
operation.commit_release(None, "1.0.0", "bf16", manifest, len(raw))
print(json.dumps({"manifest_id": manifest, "manifest_length": len(raw)}))
`

// probeProject is the parity package with a CPU job reading one Model slot by its authored
// default lane: a derive-only Manifest capability, so the job never loads it.
func probeProject(t *testing.T, lane string) string {
	t.Helper()
	project := parityProject(t)
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

// A warm run of a published release makes no Hub request, from the CLI or from the machine:
// the machine read the owner's binding and resolved its lane at its own Hub on the cold run
// and kept both. On this computer's machine and on a rental, with the real Host and Runtime.
func TestAWarmRunReadsNothingAtAnyHub(t *testing.T) {
	h, root, _, _ := parityMachines(t)
	python := filepath.Join(root, "machine", "root", "opt", "cozy", "python", "bin", "python")
	var resolved string
	for _, store := range []string{filepath.Join(root, "tensorfs"), filepath.Join(h.provider, "var", "lib", "tensorfs")} {
		out, err := exec.Command(python, "-I", "-c", seedCheckpoint, store).CombinedOutput()
		if err != nil {
			t.Fatalf("seeding %s: %v\n%s", store, err, out)
		}
		resolved = strings.TrimSpace(string(out))
	}
	var mu sync.Mutex
	var seen []string
	count := func(hub string, handler http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !strings.HasPrefix(r.URL.Path, "/v1/rentals") { // the fleet's own reconciliation
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
			var body map[string]any
			must(t, json.Unmarshal([]byte(resolved), &body))
			body["model"], body["release"], body["lane"] = "proof/probe", "1.0.0", "bf16"
			_ = json.NewEncoder(w).Encode(body)
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
