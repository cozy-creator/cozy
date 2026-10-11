package producttest

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"slices"
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
// answers the Hub's resolution of it. The machines' Hub keeps answering the probe's closure,
// for any spelling of it, and records each ref and lane asked (h.closures).
func seedProbe(t *testing.T, h *machineHub, root string, venues ...string) map[string]any {
	t.Helper()
	// A checkpoint of the caller's own named by digest is a private read: the run carries a
	// capability its device key signs (th-241).
	h.hubAccess.signIn(t, root)
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
			var asked struct{ Ref, Lane string }
			_ = json.NewDecoder(r.Body).Decode(&asked)
			h.mu.Lock()
			h.closures = append(h.closures, strings.TrimSpace(asked.Ref+" "+asked.Lane))
			h.mu.Unlock()
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
	h.server.Config.Handler = account
	return resolved
}

// probeModel answers proof/probe's model card and its bf16 lane's manifest, as the CLI reads
// them to bind each rung of a ladder to its exact checkpoint and size the job (th-241). It
// answers false for any other request.
func probeModel(t *testing.T, resolved map[string]any) func(http.ResponseWriter, *http.Request) bool {
	t.Helper()
	manifest, _ := probeCheckpoint(t)
	card, err := json.Marshal(map[string]any{"model": map[string]string{"org": "proof", "name": "probe"},
		"releases": []any{map[string]any{"release": "1.0.0", "lanes": []any{map[string]any{"lane": "bf16",
			"manifest_id": resolved["manifest_id"], "components": []string{}, "bytes": resolved["bytes"]}}}}})
	must(t, err)
	return func(w http.ResponseWriter, r *http.Request) bool {
		switch r.URL.Path {
		case "/v1/models/proof/probe":
			_, _ = w.Write(card)
		case "/v1/models/proof/probe/releases/1.0.0/lanes/bf16/manifest":
			_, _ = w.Write(manifest)
		default:
			return false
		}
		return true
	}
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

// A warm run of a published release makes no Hub request, from the CLI or machine. The cold
// run names the release and the model (th-245): the CLI reads the owner's binding once per
// catalog revision, and each machine installs the release by name and resolves the model's
// release lane with the Hub's closure, never by hash and never reading a binding or model
// card. Rental authority still refreshes independently of content reuse.
func TestAWarmRunReadsNothingAtAnyHub(t *testing.T) {
	h, root, _, _ := parityMachines(t)
	resolved := seedProbe(t, h, root, machines.Local, "tessa")
	// This proof measures content reuse once the machine is ready.
	machine := machines.NewHost(filepath.Join(root, "machine"), machineStore(root), nil)
	_, problem := machine.Ensure(t.Context(), nil)
	fatal(t, problem)
	eventually(t, root, "native machine readiness", func() bool {
		frame, problem := machine.ReadStatus(t.Context())
		return problem == nil && frame != nil && frame.Phase == "ready"
	})
	var mu sync.Mutex
	var seen []string
	count := func(hub string, handler http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Run or no run, the daemon re-reads its rentals and a rental's machine its own
			// authority, each on its own clock.
			background := r.Method == http.MethodGet && (strings.HasPrefix(r.URL.Path, "/v1/rentals") || r.URL.Path == "/v1/worker/rental/authorized-keys")
			if !background {
				mu.Lock()
				seen = append(seen, hub+" "+r.Method+" "+r.URL.Path)
				mu.Unlock()
			}
			handler.ServeHTTP(w, r)
		})
	}
	publishParityRelease(t, h, root, probeProject(t, "proof/probe@1.0.0/bf16"))
	binding := `{"bindings":[{"slot":"touch.models.source","model":"proof/probe","release":"1.0.0","revision":1,"ladder":[{"gpu":"*","lane":"bf16"}]}]}`
	h.worker.Config.Handler = count("machine", h.worker.Config.Handler)
	account, probe := h.server.Config.Handler, probeModel(t, resolved)
	h.server.Config.Handler = count("account", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/packages/"+parityPublished+"/bindings":
			_, _ = w.Write([]byte(binding))
		case probe(w, r):
		default:
			account.ServeHTTP(w, r)
		}
	}))
	catalog := func(calls string) bool {
		return strings.Contains(calls, "machine GET /v1/models/") || strings.Contains(calls, "machine GET /v1/packages/"+parityPublished+"/bindings")
	}
	installs := func(calls string) bool {
		return strings.Contains(calls, "machine GET /v1/packages/"+parityPublished+"/releases/")
	}
	byName := func(venue string) {
		t.Helper()
		h.mu.Lock()
		asked := slices.Clone(h.closures)
		h.closures = nil
		h.mu.Unlock()
		if !slices.Contains(asked, "proof/probe@1.0.0 bf16") || slices.ContainsFunc(asked, func(ref string) bool {
			return strings.HasPrefix(ref, "proof/probe@sha256:")
		}) {
			t.Fatalf("the cold run on %s did not resolve proof/probe@1.0.0 bf16 by name: %q", venue, asked)
		}
	}
	h.mu.Lock()
	h.closures = nil
	h.mu.Unlock()

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
			if catalog(strings.Join(calls, "\n")) {
				t.Fatalf("the %s run on %s read the catalog from its machine: %v", key, venue.name, calls)
			}
			if key == "cold" {
				if !installs(strings.Join(calls, "\n")) {
					t.Fatalf("the cold run on %s did not install the release by name: %v", venue.name, calls)
				}
				byName(venue.name)
			}
			if key == "warm" && len(calls) != 0 {
				t.Fatalf("the warm run on %s made %d Hub requests; want none: %v", venue.name, len(calls), calls)
			}
		}
	}

	// A new daemon is no colder: what the last one read (the Hub's execution environment,
	// this computer's short bearer) is kept on disk, not in a process.
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}
	for venue, args := range map[string][]string{"local": nil, "tessa": {"--rental=tessa"}} {
		mu.Lock()
		seen = nil
		mu.Unlock()
		if code, out := runCozy(t, root, append([]string{"run", parityPublished + "/touch", "value=1", "--await", "--json"}, args...)...); code != 0 || !strings.Contains(out, `"value":2`) {
			t.Fatalf("touch on %s under a new daemon [exit %d]\n%s", venue, code, out)
		}
		mu.Lock()
		calls := append([]string(nil), seen...)
		mu.Unlock()
		if len(calls) != 0 {
			t.Fatalf("the warm run on %s under a new daemon made %d Hub requests; want none: %v", venue, len(calls), calls)
		}
	}

	// The owner's change to the package moves the catalog revision: the CLI reads the binding
	// once more for whichever machine runs next, then nothing again; no machine reads it, and
	// each resolves its model once more by name.
	h.mux.HandleFunc("DELETE /v1/packages/"+parityPublished+"/releases/"+parityVersion, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"release":"` + parityVersion + `","state":"yanked","changed":true}`))
	})
	if code, out := runCozy(t, root, "package", "yank", parityPublished, "--version", parityVersion); code != 0 || strings.Contains(out, "keeps what it read") || strings.Contains(out, "no machine was told") {
		t.Fatalf("yank did not reach every machine [exit %d]\n%s", code, out)
	}
	changed := true
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
			read := strings.Contains(calls, "account GET /v1/packages/"+parityPublished+"/bindings")
			if changed != read || catalog(calls) || installs(calls) || !strings.Contains(calls, "machine POST /v1/tensorfs/closure") && key == "changed" || key == "warm again" && calls != "" {
				t.Fatalf("the %s run on %s read: %q", key, venue, calls)
			}
			changed = false
		}
	}

	// Unpublished code is no different. Its org-relative default names the caller's account,
	// read once and kept; the CLI binds the default's rungs and the machine reads no model.
	if code, out := runCozy(t, root, "package", "install", probeProject(t, "probe@1.0.0/bf16")); code != 0 {
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
			if catalog(calls) || key == "warm" && calls != "" {
				t.Fatalf("the %s local/ run on %s read: %q", key, venue, calls)
			}
		}
	}
}
