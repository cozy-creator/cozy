package producttest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machines"
)

const (
	parityPublished = "proof/machine-parity"
	parityVersion   = "0.0.1"
)

// A published release on a known machine is one message to that machine, with the real Host
// and Runtime on this computer's machine and on a rental. The run names the release (th-245):
// the CLI reads its card once per catalog revision for its own checks, each machine reads the
// card and lock at its Hub by name once to install it, and after that no run reads a Hub.
func TestPublishedRunOnAKnownMachineReadsNoHub(t *testing.T) {
	h, root, _, store := parityMachines(t)
	publishParityRelease(t, h, root, parityProject(t))
	var mu sync.Mutex
	var seen, machineReads []string
	doors := h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/packages/") {
			mu.Lock()
			machineReads = append(machineReads, r.Method+" "+r.URL.Path)
			mu.Unlock()
		}
		doors.ServeHTTP(w, r)
	})
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/v1/rentals") { // the fleet's own reconciliation
			mu.Lock()
			seen = append(seen, r.Method+" "+r.URL.Path)
			mu.Unlock()
		}
		served.ServeHTTP(w, r)
	})
	release := "/v1/packages/" + parityPublished + "/releases/" + parityVersion
	installs := []string{"GET " + release, "GET " + release + "/locked-requirements"}
	carded := false
	run := func(venue, key string, args ...string) {
		t.Helper()
		mu.Lock()
		seen, machineReads = nil, nil
		mu.Unlock()
		idem := "published-" + venue + "-" + key
		code, out := runCozy(t, root, append([]string{"run", parityPublished + "/add", "value=41", "--await", "--json", "--idempotency-key", idem}, args...)...)
		if code != 0 || !strings.Contains(out, `"value":42`) {
			t.Fatalf("published add, %s on %s [exit %d]\n%s", key, venue, code, out)
		}
		request, problem := store.RequestByIdempotencyKey(idem)
		fatal(t, problem)
		link, problem := store.MachineExecution(request.ID)
		fatal(t, problem)
		want := map[string]string{"local": machines.Local, "rental": parityRental}[venue]
		if link == nil || link.MachineID != want || !link.Collected {
			t.Fatalf("published add on %s was not a collected execution on %s: %+v", venue, want, link)
		}
		mu.Lock()
		calls, reads := append([]string(nil), seen...), append([]string(nil), machineReads...)
		mu.Unlock()
		if key == "cold" {
			calls = slices.DeleteFunc(calls, hubAccessCall)
			if slices.Sort(reads); !slices.Equal(slices.Compact(reads), installs) {
				t.Fatalf("the cold run on %s did not install the release by name: %v", venue, reads)
			}
		} else if len(reads) != 0 {
			t.Fatalf("the %s run on %s read a package at its Hub: %v", key, venue, reads)
		}
		if !carded {
			if len(calls) == 0 || slices.ContainsFunc(calls, func(call string) bool {
				return !strings.Contains(call, " /v1/packages/"+parityPublished)
			}) {
				t.Fatalf("the first run on %s did not read only the release's card: %v", venue, calls)
			}
			carded, calls = true, nil
		}
		if len(calls) != 0 {
			t.Fatalf("the %s run on %s made %d Hub requests; want none: %v", key, venue, len(calls), calls)
		}
	}
	for _, venue := range []struct {
		name string
		args []string
	}{{"local", nil}, {"rental", []string{"--rental=tessa"}}} {
		for _, key := range []string{"cold", "warm"} {
			run(venue.name, key, venue.args...)
		}
	}
	// A stopped machine boots for the next run and keeps its installation: its startup check
	// finds no update pending, so the run reads no Hub either.
	for i := range 3 {
		if code, out := runCozy(t, root, "machine", "stop"); code != 0 {
			t.Fatalf("machine stop [exit %d]\n%s", code, out)
		}
		run("local", fmt.Sprintf("restarted-%d", i))
	}
}

// publishParityRelease publishes a parity project as proof/machine-parity@0.0.1 at the
// machines' own Hub: the interface the machine's Runtime describes from source, and the
// locked closure every machine installs. The wheel is served beside it, not by a Hub.
func publishParityRelease(t *testing.T, h *machineHub, root, project string) {
	t.Helper()
	publishParityReleaseAt(t, h, root, project, parityVersion)
}

func publishParityReleaseAt(t *testing.T, h *machineHub, root, project, release string) {
	t.Helper()
	// A release pins the published closure: the machine's own Runtime wheel is no release.
	pyproject := filepath.Join(project, "pyproject.toml")
	raw, err := os.ReadFile(pyproject)
	must(t, err)
	if head, tail, sourced := strings.Cut(string(raw), "[tool.uv.sources]"); sourced {
		_, rest, _ := strings.Cut(tail, "[build-system]")
		must(t, os.WriteFile(pyproject, []byte(head+"[build-system]"+rest), 0o600))
		if out, err := exec.Command("uv", "lock", "--refresh-package", hostruntime.Distribution, "--project", project).CombinedOutput(); err != nil {
			t.Fatalf("locking the published parity release: %v\n%s", err, out)
		}
	}
	dist := t.TempDir()
	if out, err := exec.Command("uv", "build", "--wheel", "--project", project, "-o", dist).CombinedOutput(); err != nil {
		t.Fatalf("building the parity wheel: %v\n%s", err, out)
	}
	name := "machine_parity-" + release + "-py3-none-any.whl"
	wheel, err := os.ReadFile(filepath.Join(dist, name))
	must(t, err)
	files := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(wheel) }))
	t.Cleanup(files.Close)
	sum := sha256.Sum256(wheel)
	runtime := filepath.Join(filepath.Dir(machinePython(t)), "cozy-runtime") //cozy:allow the fixture describes its package as the publisher would
	iface, err := exec.Command(runtime, "--json", "--dir", project, "describe").Output()
	must(t, err)
	closure, err := exec.Command("uv", "export", "--project", project, "--frozen", "--format", "requirements-txt",
		"--no-dev", "--no-emit-project", "--no-header").Output()
	must(t, err)
	locked := string(closure) + "machine-parity @ " + files.URL + "/" + name + " --hash=sha256:" + hex.EncodeToString(sum[:]) + "\n"
	path := "/v1/packages/" + parityPublished
	releases := map[string][]byte{path: []byte(`{"releases":[{"release":"` + release + `"}]}`),
		path + "/releases/" + release + "/locked-requirements": []byte(locked)}
	detail, err := json.Marshal(map[string]any{"release": map[string]string{"release": release},
		"package_interface": json.RawMessage(iface), "requires_python": ">=3.12,<3.13"})
	must(t, err)
	releases[path+"/releases/"+release] = detail
	// The CLI reads the release's card for its own checks; a machine reads the card and locked
	// requirements by name (th-245); `cozy package install` reads its download plan and project index.
	exact := func(raw []byte) hub.ExactDocument {
		return hub.ExactDocument{CanonicalBytes: raw, Digest: fmt.Sprintf("sha256:%x", sha256.Sum256(raw)), Length: int64(len(raw))}
	}
	document := func(name string) hub.ExactDocument {
		raw, err := os.ReadFile(filepath.Join(project, name))
		must(t, err)
		return exact(raw)
	}
	plan, err := json.Marshal(hub.PackageDownloadPlan{Release: release, PackageConfig: document("package.toml"),
		PackageInterface: exact(iface), Pyproject: document("pyproject.toml"), UVLock: document("uv.lock"),
		Downloads: []hub.PackageInstallDownload{{Kind: "project_wheel", Path: name, Distribution: "machine-parity", Version: release,
			Digest: fmt.Sprintf("sha256:%x", sum), Length: int64(len(wheel)), Tags: []string{"py3-none-any"}, ImportRoots: []string{"machine_parity"}}}})
	must(t, err)
	doors := h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if body, ok := releases[r.URL.Path]; ok && r.Method == http.MethodGet {
			_, _ = w.Write(body)
			return
		}
		doors.ServeHTTP(w, r)
	})
	account := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if body, ok := releases[r.URL.Path]; ok && r.Method == http.MethodGet {
			_, _ = w.Write(body)
			return
		}
		switch {
		case r.URL.Path == path+"/download" && (r.URL.Query().Get("release") == release || r.URL.Query().Get("release") == ""):
			_, _ = w.Write(plan)
		case r.URL.Path == "/v1/index/proof/simple/machine-parity/":
			_, _ = fmt.Fprintf(w, `<a href="%s/%s#sha256=%x">%s</a>`, files.URL, name, sum, name)
		default:
			account.ServeHTTP(w, r)
		}
	})
}
