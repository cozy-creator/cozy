package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
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
		fmt.Fprintf(w, `<a href="/v1/index/%s/%s/1.0.0/%s#sha256=%s">%s</a><br>`, indexAccount, orgRelativeDependency, filename, digest, filename)
	})
	mux.HandleFunc("GET /v1/index/"+indexAccount+"/"+orgRelativeDependency+"/1.0.0/"+filename, func(w http.ResponseWriter, _ *http.Request) {
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
// index, locked as `cozy package lock` locks it: against the author's own Hub. module, when set,
// is its code; dependencies are added to its own.
func indexProject(t *testing.T, laptopHub, module string, dependencies ...string) string {
	t.Helper()
	project := filepath.Join(t.TempDir(), "index-probe")
	must(t, os.MkdirAll(project, 0o700))
	extra, metadata := "", ""
	for _, dependency := range dependencies {
		extra += fmt.Sprintf(", %q", dependency)
		// A direct reference builds, as any editable install of it does, once its backend allows it.
		metadata = "[tool.hatch.metadata]\nallow-direct-references=true\n"
	}
	sources := "[tool.uv.sources]\norg-relative-dep={index=\"tensorhub\"}\n"
	if *machineRuntimeWheel != "" {
		sources += fmt.Sprintf("cozy-runtime={path=%q}\ntensorfs={path=%q}\n", *machineRuntimeWheel, *machineTensorFSWheel)
	}
	authored := `[project]
name="index-probe"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=` + runtimeFloor + `", "msgspec>=0.19", "org-relative-dep>=1.0,<2"` + extra + `]
[project.entry-points."cozy.application"]
default="index_probe:app"
` + sources + `[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["index_probe.py"]
` + metadata
	files := map[string]string{
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
	}
	if module != "" {
		files["index_probe.py"] = module
	}
	for name, body := range files {
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

// A package's account-index dependency is part of its captured closure: the editable install
// resolves it at the author's Hub that its lock names, and its exact bytes travel with every
// run to every machine. No machine reads a Hub for it, so it installs on this computer's
// machine and on a rental alike while the author's Hub is unreachable.
func TestAccountIndexDependencyTravelsWithItsCapture(t *testing.T) {
	h, root, _, _ := parityMachines(t)
	// Rental authority is current-attempt scoped, not a blanket fixture grant.
	for _, valid := range []bool{false, true} {
		h.mu.Lock()
		worker, token, key := h.authorityWorker, h.authorityToken, h.authorityKey
		h.mu.Unlock()
		if !valid {
			token = "not-the-provider-token"
		}
		request, err := http.NewRequest(http.MethodGet, h.worker.URL+"/v1/worker/rental/authorized-keys", nil)
		must(t, err)
		request.Header.Set("X-Cozy-Worker-ID", worker)
		request.Header.Set("X-Cozy-Worker-Token", token)
		response, err := h.worker.Client().Do(request)
		must(t, err)
		if !valid {
			response.Body.Close()
			if response.StatusCode != http.StatusForbidden {
				t.Fatal("unbound provider received fixture authority")
			}
			continue
		}
		var authority struct {
			Worker string   `json:"worker_id"`
			Keys   []string `json:"authorized_keys"`
			Lease  int      `json:"lease_seconds"`
		}
		err = json.NewDecoder(response.Body).Decode(&authority)
		response.Body.Close()
		must(t, err)
		if response.StatusCode != http.StatusOK || authority.Worker != worker || len(authority.Keys) != 1 || authority.Keys[0] != key || authority.Lease <= 0 {
			t.Fatal("fixture authority differs from current provider grant")
		}
	}
	wheel := orgRelativeWheel(t, "1.0.0", "VALUE = 7\n")
	var laptopOpen, machineOpen atomic.Bool
	laptopOpen.Store(true)
	machineOpen.Store(true)
	var laptopServed, machineServed atomic.Int64
	h.hubAccess.account = indexAccount
	h.mux.Handle("GET /v1/index/", accountIndex(wheel, &laptopOpen, &laptopServed))
	machineIndex, doors := accountIndex(wheel, &machineOpen, &machineServed), h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/index/") {
			machineIndex.ServeHTTP(w, r)
			return
		}
		doors.ServeHTTP(w, r)
	})

	project := indexProject(t, h.server.URL, "")
	if lock, err := os.ReadFile(filepath.Join(project, "uv.lock")); err != nil ||
		!bytes.Contains(lock, []byte(h.server.URL+"/v1/index/"+indexAccount+"/"+orgRelativeDependency+"/1.0.0/")) {
		t.Fatalf("the lock does not name the author's Hub: %v", err)
	}
	if code, out := runCozy(t, root, "package", "install", project); code != 0 {
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
	if served := machineServed.Load(); served != 0 {
		t.Fatalf("a machine fetched the captured dependency from a Hub %d times", served)
	}
}

// A local package calls a job of another package it takes from its author's account index. That
// job's Model slot declares a default ladder and nobody bound or chose a model for it. Each
// machine gives the slot its declared default: no binding is read at any Hub and the caller
// names nothing (run 5327 failed `model_choice_absent` on exactly this).
func TestAnIndexCalleesUnchosenSlotTakesItsDeclaredDefault(t *testing.T) {
	h, root, _, _ := parityMachines(t)
	h.hubAccess.account = indexAccount
	seedProbe(t, h, root, machines.Local, "tessa")
	wheel := orgRelativeWheel(t, "1.0.0", `import msgspec
from cozy_runtime.author import App, Context, Loader, Model, invocable


class Nothing:
    pass


class Probe(Model[Nothing]):
    def load(self, loader: Loader) -> None:
        raise RuntimeError("a job never loads its Model")


class Touched(msgspec.Struct, frozen=True):
    value: int


@invocable(defaults={"source": [{"gpu": "*", "lane": "proof/probe@1.0.0/bf16"}]})
async def touch(ctx: Context, *, value: int, source: Probe) -> Touched:
    return Touched(value + 7)


app = App()
app.job(touch)
`, "[cozy.application]\ndefault = org_relative_dep:app\n")
	var open atomic.Bool
	open.Store(true)
	var served atomic.Int64
	h.mux.Handle("GET /v1/index/", accountIndex(wheel, &open, &served))
	var bindings atomic.Int64
	account := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/bindings") {
			bindings.Add(1)
		}
		account.ServeHTTP(w, r)
	})
	project := indexProject(t, h.server.URL, `import msgspec
from cozy_runtime.author import App
from org_relative_dep import touch


class AddRequest(msgspec.Struct, forbid_unknown_fields=True):
    value: int


class AddResult(msgspec.Struct):
    value: int


app = App()


@app.job
async def add(payload: AddRequest) -> AddResult:
    return AddResult(value=(await touch(value=payload.value)).value)
`)
	if code, out := runCozy(t, root, "package", "install", project); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	h.mu.Lock()
	h.closures = nil
	h.mu.Unlock()
	for _, venue := range []struct {
		name string
		args []string
	}{{"local", nil}, {"rental", []string{"--rental=tessa"}}} {
		args := append([]string{"run", "local/index-probe/add", "value=1", "--await", "--json"}, venue.args...)
		if code, out := runCozy(t, root, args...); code != 0 || !strings.Contains(out, `"value":8`) {
			t.Fatalf("add on the %s machine [exit %d]\n%s", venue.name, code, out)
		}
		h.mu.Lock()
		asked := slices.Clone(h.closures)
		h.closures = nil
		h.mu.Unlock()
		if !slices.Contains(asked, "proof/probe@1.0.0 bf16") {
			t.Fatalf("the %s machine did not resolve the callee's declared default by name: %q", venue.name, asked)
		}
	}
	if read := bindings.Load(); read != 0 {
		t.Fatalf("an unpublished package read %d Hub bindings; it has none", read)
	}
}

// `cozy run ./project` of a package that names `index = "tensorhub"` and declares no index: the
// CLI writes the caller's account index into what it locks and installs, locked or not and with
// a Git dependency it carries as a wheel (minimax-h3's shape), so the author never spells a Hub
// URL, and both machines run it.
func TestARunOfAnAuthoredDirectorySuppliesItsAccountIndex(t *testing.T) {
	h, root, _, _ := parityMachines(t)
	h.hubAccess.account = indexAccount
	wheel := orgRelativeWheel(t, "1.0.0", "VALUE = 7\n")
	var open atomic.Bool
	open.Store(true)
	var served atomic.Int64
	h.mux.Handle("GET /v1/index/", accountIndex(wheel, &open, &served))
	repository, commit := serveGitProject(t)
	locked := indexProject(t, h.server.URL, "")
	unlocked := indexProject(t, h.server.URL, "")
	must(t, os.Remove(filepath.Join(unlocked, "uv.lock")))
	withGit := indexProject(t, h.server.URL, "", "cozy-fixture-pinned @ git+"+repository+"@"+commit)
	for _, project := range []string{withGit, locked, unlocked} {
		for _, venue := range []struct {
			name string
			args []string
		}{{"local", nil}, {"rental", []string{"--rental=tessa"}}} {
			args := append([]string{"run", project + "/add", "value=1", "--await", "--json"}, venue.args...)
			if code, out := runCozy(t, root, args...); code != 0 || !strings.Contains(out, `"value":8`) {
				t.Fatalf("%s on the %s machine [exit %d]\n%s", project, venue.name, code, out)
			}
		}
		if raw, _ := os.ReadFile(filepath.Join(project, "pyproject.toml")); bytes.Contains(raw, []byte("[[tool.uv.index]]")) {
			t.Fatalf("the run wrote an index into the authored pyproject:\n%s", raw)
		}
	}
	if _, err := os.Stat(filepath.Join(unlocked, "uv.lock")); err == nil {
		t.Fatal("a one-off run wrote a lock into the authored tree")
	}
}
