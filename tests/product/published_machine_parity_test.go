package producttest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/machines"
)

const (
	parityPublished = "proof/machine-parity"
	parityVersion   = "0.0.1"
)

// A published release on a known machine is one message to that machine: `cozy run` makes no
// Hub request, on this computer's machine or on a rental, with the real Host and Runtime on
// both. Each machine reads the release, its lock and its interface at its own Hub.
func TestPublishedRunOnAKnownMachineReadsNoHub(t *testing.T) {
	h, root, _, store := parityMachines(t)
	python := filepath.Join(root, "machine", "root", "opt", "cozy", "python", "bin", "python")
	if exec.Command(python, "-c", "import cozy_runtime.internal.worker.machine_release_catalog").Run() != nil {
		t.Skip("the machine's Runtime predates releases the machine installs itself (0.18.67)")
	}
	publishParityRelease(t, h, root)
	var mu sync.Mutex
	var seen []string
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/v1/rentals") { // the fleet's own reconciliation
			mu.Lock()
			seen = append(seen, r.Method+" "+r.URL.Path)
			mu.Unlock()
		}
		served.ServeHTTP(w, r)
	})
	for _, venue := range []struct {
		name string
		args []string
	}{{"local", nil}, {"rental", []string{"--rental=tessa"}}} {
		for _, key := range []string{"cold", "warm"} {
			mu.Lock()
			seen = nil
			mu.Unlock()
			idem := "published-" + venue.name + "-" + key
			code, out := runCozy(t, root, append([]string{"run", parityPublished + "/add", "value=41", "--await", "--json", "--idempotency-key", idem}, venue.args...)...)
			if code != 0 || !strings.Contains(out, `"value":42`) {
				t.Fatalf("published add, %s on %s [exit %d]\n%s", key, venue.name, code, out)
			}
			request, problem := store.RequestByIdempotencyKey(idem)
			fatal(t, problem)
			link, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			want := map[string]string{"local": machines.Local, "rental": parityRental}[venue.name]
			if link == nil || link.MachineID != want || !link.Collected {
				t.Fatalf("published add on %s was not a collected execution on %s: %+v", venue.name, want, link)
			}
			mu.Lock()
			calls := append([]string(nil), seen...)
			mu.Unlock()
			// This computer's machine registers with its hub once, on its first launch.
			if venue.name == "local" && key == "cold" && len(calls) == 1 && calls[0] == "POST /v1/machines" {
				calls = nil
			}
			if len(calls) != 0 {
				t.Fatalf("the %s run on %s made %d Hub requests; want none: %v", key, venue.name, len(calls), calls)
			}
		}
	}
}

// publishParityRelease publishes the parity project as proof/machine-parity@0.0.1 at the
// machines' own Hub: the interface the machine's Runtime describes from source, and the
// locked closure every machine installs. The wheel is served beside it, not by a Hub.
func publishParityRelease(t *testing.T, h *machineHub, root string) {
	t.Helper()
	// A release pins the published closure: the machine's own Runtime wheel is no release.
	project := parityProject(t)
	pyproject := filepath.Join(project, "pyproject.toml")
	raw, err := os.ReadFile(pyproject)
	must(t, err)
	if head, tail, sourced := strings.Cut(string(raw), "[tool.uv.sources]"); sourced {
		_, rest, _ := strings.Cut(tail, "[build-system]")
		must(t, os.WriteFile(pyproject, []byte(head+"[build-system]"+rest), 0o600))
		if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
			t.Fatalf("locking the published parity release: %v\n%s", err, out)
		}
	}
	dist := t.TempDir()
	if out, err := exec.Command("uv", "build", "--wheel", "--project", project, "-o", dist).CombinedOutput(); err != nil {
		t.Fatalf("building the parity wheel: %v\n%s", err, out)
	}
	name := "machine_parity-" + parityVersion + "-py3-none-any.whl"
	wheel, err := os.ReadFile(filepath.Join(dist, name))
	must(t, err)
	files := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(wheel) }))
	t.Cleanup(files.Close)
	sum := sha256.Sum256(wheel)
	runtime := filepath.Join(root, "machine", "root", "opt", "cozy", "python", "bin", "cozy-runtime")
	iface, err := exec.Command(runtime, "--json", "--dir", project, "describe").Output()
	must(t, err)
	closure, err := exec.Command("uv", "export", "--project", project, "--frozen", "--format", "requirements-txt",
		"--no-dev", "--no-emit-project", "--no-header").Output()
	must(t, err)
	locked := string(closure) + "machine-parity @ " + files.URL + "/" + name + " --hash=sha256:" + hex.EncodeToString(sum[:]) + "\n"
	path := "/v1/packages/" + parityPublished
	releases := map[string][]byte{path: []byte(`{"releases":[{"release":"` + parityVersion + `"}]}`),
		path + "/releases/" + parityVersion + "/locked-requirements": []byte(locked)}
	detail, err := json.Marshal(map[string]any{"release": map[string]string{"release": parityVersion},
		"package_interface": json.RawMessage(iface), "requires_python": ">=3.12,<3.13"})
	must(t, err)
	releases[path+"/releases/"+parityVersion] = detail
	doors := h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if body, ok := releases[r.URL.Path]; ok {
			_, _ = w.Write(body)
			return
		}
		doors.ServeHTTP(w, r)
	})
}
