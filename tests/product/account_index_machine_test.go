package producttest

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/machines"
)

const indexAccount = "author"

// accountIndex is one account's Tensorhub package index serving the one dependency wheel,
// counting the files it serves. Closed, it answers as a Hub the caller cannot reach.
func accountIndex(wheel []byte, open *atomic.Bool, served *atomic.Int64) http.Handler {
	digest := fmt.Sprintf("%x", sha256.Sum256(wheel))
	filename := orgRelativeWheelName("1.0.0")
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/index/"+indexAccount+"/simple/"+orgRelativeDependency+"/", func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprintf(w, `<a href="/v1/index/%s/files/%s/%s#sha256=%s">%s</a><br>`, indexAccount, digest, filename, digest, filename)
	})
	mux.HandleFunc("GET /v1/index/"+indexAccount+"/files/"+digest+"/"+filename, func(w http.ResponseWriter, _ *http.Request) {
		served.Add(1)
		_, _ = w.Write(wheel)
	})
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !open.Load() {
			http.Error(w, "unreachable from here", http.StatusBadGateway)
			return
		}
		mux.ServeHTTP(w, r)
	})
}

// indexProject is an editable package whose one Hub dependency comes from its author's account
// index, locked as `cozy package lock` locks it: against the author's own Hub.
func indexProject(t *testing.T, laptopHub string) string {
	t.Helper()
	project := filepath.Join(t.TempDir(), "index-probe")
	must(t, os.MkdirAll(project, 0o700))
	sources := "[tool.uv.sources]\norg-relative-dep={index=\"tensorhub\"}\n"
	if *machineRuntimeWheel != "" {
		sources += fmt.Sprintf("cozy-runtime={path=%q}\ntensorfs={path=%q}\n", *machineRuntimeWheel, *machineTensorFSWheel)
	}
	authored := `[project]
name="index-probe"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=` + machines.RuntimeFloor + `", "msgspec>=0.19", "org-relative-dep>=1.0,<2"]
[project.entry-points."cozy.application"]
default="index_probe:app"
` + sources + `[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["index_probe.py"]
`
	for name, body := range map[string]string{
		"pyproject.toml": authored,
		"package.toml":   "[application]\nobject=\"index_probe:app\"\n",
		"index_probe.py": `import msgspec
import org_relative_dep
from cozy_runtime.author import App


class AddRequest(msgspec.Struct, forbid_unknown_fields=True):
    value: int


class AddResult(msgspec.Struct):
    value: int


app = App()


@app.job
def add(payload: AddRequest) -> AddResult:
    return AddResult(value=payload.value + org_relative_dep.VALUE)
`,
	} {
		must(t, os.WriteFile(filepath.Join(project, name), []byte(body), 0o600))
	}
	// The owned copy Creator locks: the authored project plus the account index it writes.
	staged := t.TempDir()
	bound := strings.TrimRight(authored, "\n") + "\n" + fmt.Sprintf("\n[[tool.uv.index]]\nname = %q\nurl = %q\nexplicit = true\n",
		"tensorhub", laptopHub+"/v1/index/"+indexAccount+"/simple/")
	must(t, os.WriteFile(filepath.Join(staged, "pyproject.toml"), []byte(bound), 0o600))
	if out, err := exec.Command("uv", "lock", "--project", staged).CombinedOutput(); err != nil {
		t.Fatalf("locking the index probe: %v\n%s", err, out)
	}
	lock, err := os.ReadFile(filepath.Join(staged, "uv.lock"))
	must(t, err)
	must(t, os.WriteFile(filepath.Join(project, "uv.lock"), lock, 0o600))
	return project
}

// A package's account-index dependency installs on every machine through that machine's own
// Hub grant. The lock names the author's Hub, which no machine reaches; each machine resolves
// the same account path at its grant origin, the rental exactly as this computer's machine.
func TestAccountIndexResolvesAtEachMachinesOwnHub(t *testing.T) {
	h, root, _, _ := parityMachines(t)
	wheel := orgRelativeWheel(t, "1.0.0", "VALUE = 7\n")
	var laptopOpen, machineOpen atomic.Bool
	laptopOpen.Store(true)
	machineOpen.Store(true)
	var laptopServed, machineServed atomic.Int64
	h.mux.HandleFunc("GET /v1/accounts/current", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"name":"` + indexAccount + `"}`))
	})
	h.mux.Handle("GET /v1/index/", accountIndex(wheel, &laptopOpen, &laptopServed))
	machineIndex, doors := accountIndex(wheel, &machineOpen, &machineServed), h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/index/") {
			machineIndex.ServeHTTP(w, r)
			return
		}
		doors.ServeHTTP(w, r)
	})

	project := indexProject(t, h.server.URL)
	if lock, err := os.ReadFile(filepath.Join(project, "uv.lock")); err != nil ||
		!bytes.Contains(lock, []byte(h.server.URL+"/v1/index/"+indexAccount+"/files/")) {
		t.Fatalf("the lock does not name the author's Hub: %v", err)
	}
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	if laptopServed.Load() == 0 {
		t.Fatal("the editable install did not resolve its dependency at the author's Hub")
	}
	laptopOpen.Store(false)
	for _, venue := range []struct {
		name string
		args []string
	}{{"local", nil}, {"rental", []string{"--rental=tessa"}}} {
		args := append([]string{"run", "local/index-probe/add", "value=1", "--await", "--json"}, venue.args...)
		if code, out := runCozy(t, root, args...); code != 0 || !strings.Contains(out, `"value":8`) {
			t.Fatalf("add on the %s machine [exit %d]\n%s", venue.name, code, out)
		}
	}
	if machineServed.Load() == 0 {
		t.Fatal("no machine fetched the dependency at its own Hub")
	}
}
