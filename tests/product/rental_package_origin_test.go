package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// A stored package release is metadata, not a second CLI Hub configuration. Its source
// and the selected rental's owner can both differ from the user's selected package Hub.
func TestNamedRentalKeepsConfiguredHubDespiteCachedInstall(t *testing.T) {
	root, _, _, _, _ := runModelCatalog(t)
	selected := configuredHub(t, root)
	installedHere(t, root, selected, "proof/quantize", "1.0.0")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	_, installed, problem := store.ActivePackage(selected, "proof/quantize")
	fatal(t, problem)
	iface, err := os.ReadFile(launch.PackageInterfacePath(installed.Dir))
	must(t, err)
	installed.ID += "-other-origin"
	installed.Dir = filepath.Join(root, "installs", installed.ID)
	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(installed.Dir)), 0o700))
	must(t, os.WriteFile(launch.PackageInterfacePath(installed.Dir), iface, 0o444))
	installed.Hub = "http://127.0.0.1:2"
	_, problem = store.Activate(*installed)
	fatal(t, problem)
	fatal(t, store.RecordRental(records.Rental{ID: "pr-selected", MachineName: "tessa", State: "ready",
		Hub: "http://127.0.0.1:1", AcceleratorModel: "CPU", AcceleratorCount: 1,
		Address: "127.0.0.1:1", CertPath: "unused", HourlyRateUSDMicros: 100_000}))

	request, _, out := submitRun(t, root, "selected-source", "run", "proof/quantize/quantize", "steps=7",
		"model.dits=proof/source@1.0.0/bf16", "model.shared=proof/source@1.0.0/bf16", "--rental=tessa", "--json")
	if request == nil || request.Hub != selected || request.RequestedRental != "pr-selected" {
		t.Fatalf("configured package Hub and selected machine did not survive submission: %+v\n%s", request, out)
	}
}

// The owner's report: the list showed fidika/minimax-h3 (installed from tensorhub.com) while
// the current hub was local, where his account is paul. A run of it on a local hub rental
// finds nothing at local, uses no other hub's install, and says so: the hub it searched,
// where this computer got that package, the same name at local, and both working commands,
// never the hub's publisher remedy.
func TestAMissingPackageNamesItsHubAndWhereItExists(t *testing.T) {
	root, _, _, _, _ := runModelCatalog(t)
	prod := configuredHub(t, root)
	installedHere(t, root, prod, "proof/quantize", "1.0.0")
	var reads atomic.Int32
	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/packages/") {
			reads.Add(1)
		}
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":{"code":"package.not_found","message":"no package proof/quantize","remedy":"publish its first immutable release"}}`))
	}))
	defer local.Close()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: local\nhubs:\n  local: "+local.URL+
		"\n  prod: "+prod+"\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.RecordRental(records.Rental{ID: "pr-local-hub", MachineName: "tessa", State: "ready",
		Hub: local.URL, AcceleratorModel: "CPU", AcceleratorCount: 1, Address: "127.0.0.1:1", CertPath: "unused", HourlyRateUSDMicros: 100_000}))
	mine := records.PackageInstall{ID: "mine-quantize", Package: "mine/quantize", Major: 1, Version: "1.0.3", SourceKind: "tensorhub",
		SourceRef: "mine/quantize@1.0.3", Hub: local.URL, Verified: true, Dir: filepath.Join(root, "installs", "mine-quantize")}
	_, problem = store.Activate(mine)
	fatal(t, problem)
	code, out := runCozy(t, root, "package", "info", "proof/quantize")
	if code == 0 || !strings.Contains(out, "Error: no package proof/quantize on hub local (") ||
		!strings.Contains(out, "Next: cozy package info proof/quantize --tensorhub=prod\n") {
		t.Fatalf("human error does not name the hub and the working command [exit %d]\n%s", code, out)
	}
	for _, placement := range [][]string{{"--rental=tessa"}, nil} {
		before := reads.Load()
		args := append([]string{"run", "Proof/quantize/quantize", "--describe", "--json"}, placement...)
		code, out := runCozy(t, root, args...)
		if code == 0 || !strings.Contains(out, "package.not_found") || reads.Load() == before {
			t.Fatalf("another hub's install satisfied a missing package instead of the current hub [exit %d, placement %v]\n%s", code, placement, out)
		}
		typed := strings.Join(append([]string{"run", "Proof/quantize/quantize", "--describe", "--json"}, placement...), " ")
		for _, want := range []string{
			"no package proof/quantize on hub local (" + local.URL + ")",
			"proof/quantize@1.0.0 is installed from hub prod (" + prod + ")",
			"mine/quantize@1.0.3 is installed from hub local (" + local.URL + ")",
			`"cozy ` + typed + ` --tensorhub=prod"`,
			`"cozy ` + strings.Replace(typed, "Proof/quantize", "mine/quantize", 1) + `"`,
		} {
			if !strings.Contains(out, want) {
				t.Fatalf("missing %q [placement %v]\n%s", want, placement, out)
			}
		}
		if strings.Contains(out, "publish its first") {
			t.Fatalf("a reader was told to publish [placement %v]\n%s", placement, out)
		}
	}
}

// The same contract with real CLI, daemon, machine and Runtime. The package exists only on
// the selected source Hub. Its execution environment advertises a separate remotely
// reachable catalog address (the role a local Hub's public/ngrok URL has), the machine reads
// the public package there with no credential (th-241), and the rental's Hub never receives
// package requests.
func TestNamedRentalReadsSelectedPackageHub(t *testing.T) {
	rentalHub, root, layout, store := parityMachines(t)
	source := newMachineHub(t)
	publishParityRelease(t, source, root, parityProject(t))
	configure := func(current string) {
		t.Helper()
		must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+current+
			"\ntensorhub_token: rental-idle-test\nhubs:\n  source: "+source.server.URL+"\n  rental: "+rentalHub.server.URL+
			"\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	}
	configure("source")
	var mu sync.Mutex
	var catalogReads []string
	wrongReads := []string{}
	catalog := source.access.Config.Handler
	source.access.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/packages/") {
			mu.Lock()
			catalogReads = append(catalogReads, r.URL.Path)
			mu.Unlock()
			if r.Header.Get("Authorization") != "" || r.Header.Get("DPoP") != "" || r.Header.Get("X-Cozy-Worker-Token") != "" {
				t.Error("a public package was read with a credential")
				http.Error(w, "public reads are anonymous", http.StatusForbidden)
				return
			}
		}
		catalog.ServeHTTP(w, r)
	})
	ownerCatalog := rentalHub.worker.Config.Handler
	rentalHub.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/packages/") {
			mu.Lock()
			wrongReads = append(wrongReads, r.URL.Path)
			mu.Unlock()
		}
		ownerCatalog.ServeHTTP(w, r)
	})
	run := func(key string, extra ...string) {
		t.Helper()
		args := append([]string{"run", parityPublished + "/add", "value=41", "--rental=tessa", "--await", "--json", "--idempotency-key", key}, extra...)
		code, out := runCozy(t, root, args...)
		if code != 0 || !strings.Contains(out, `"value":42`) {
			t.Fatalf("%s [exit %d]\n%s", key, code, out)
		}
		request, problem := store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		link, problem := store.MachineExecution(request.ID)
		fatal(t, problem)
		if request.Release != parityVersion || request.Hub != source.server.URL || request.RequestedRental != parityRental || link == nil || link.MachineID != parityRental || !link.Collected {
			t.Fatalf("source or machine changed: request=%+v link=%+v", request, link)
		}
	}
	run("source-cold") // The rental describes the selected Hub's newest release itself.

	// Reproduce cached install provenance from a different Hub without changing its release.
	response, err := source.worker.Client().Get(source.worker.URL + "/v1/packages/" + parityPublished + "/releases/" + parityVersion)
	must(t, err)
	var detail struct {
		Interface json.RawMessage `json:"package_interface"`
	}
	must(t, json.NewDecoder(response.Body).Decode(&detail))
	response.Body.Close()
	installed := records.PackageInstall{ID: "source-routing", Package: parityPublished, Major: 0, Version: parityVersion,
		SourceKind: "tensorhub", SourceRef: parityPublished + "@" + parityVersion, Hub: rentalHub.server.URL,
		Verified: true, Dir: layout.InstallDir("source-routing")}
	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(installed.Dir)), 0o700))
	must(t, os.WriteFile(launch.PackageInterfacePath(installed.Dir), detail.Interface, 0o444))
	_, problem := store.Activate(installed)
	fatal(t, problem)
	run("source-cached")

	// An explicit source selects the same content independently of the configured rental Hub.
	configure("rental")
	// Static credentials are origin-bound; this source's machine key authorizes the override.
	fixtureExecutionAccess(t, root, source.server, source.worker.Config.Handler)
	run("source-explicit", "--tensorhub=source")

	// This computer's machine holds the source package too; neither machine may use it
	// implicitly when a different selected Hub has none.
	if code, out := runCozy(t, root, "run", parityPublished+"/add", "value=41", "--tensorhub=source", "--await", "--json"); code != 0 || !strings.Contains(out, `"value":42`) {
		t.Fatalf("local source package setup [exit %d]\n%s", code, out)
	}
	missing := newMachineHub(t)
	var missingReads atomic.Int32
	missingCatalog := missing.worker.Config.Handler
	missing.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/packages/"+parityPublished) {
			missingReads.Add(1)
		}
		missingCatalog.ServeHTTP(w, r)
	})
	fixtureExecutionAccess(t, root, missing.server, missing.worker.Config.Handler)
	configure(missing.server.URL)
	for _, placement := range [][]string{{"--rental=tessa"}, nil} {
		before := missingReads.Load()
		args := append([]string{"run", parityPublished + "/add", "value=41", "--await", "--json"}, placement...)
		if code, out := runCozy(t, root, args...); code == 0 || missingReads.Load() == before {
			t.Fatalf("the machine reused another Hub's package when its selected source had none [exit %d, placement %v, source reads %d]\n%s", code, placement, missingReads.Load(), out)
		}
	}

	// Identical package/version spelling at another source means different code and interface.
	other := newMachineHub(t)
	project := parityProject(t)
	codePath := filepath.Join(project, "machine_parity.py")
	raw, err := os.ReadFile(codePath)
	must(t, err)
	changed := strings.ReplaceAll(string(raw), "payload.value + 1", "payload.value + 100")
	changed = strings.ReplaceAll(changed, "def echo(", "def other_echo(")
	must(t, os.WriteFile(codePath, []byte(changed), 0o600))
	publishParityRelease(t, other, root, project)
	fixtureExecutionAccess(t, root, other.server, other.worker.Config.Handler)
	configure(other.server.URL)
	code, out := runCozy(t, root, "run", parityPublished+"/add", "value=41", "--rental=tessa", "--await", "--json", "--idempotency-key=other-source")
	if code != 0 || !strings.Contains(out, `"value":141`) {
		t.Fatalf("same package/version at another Hub reused the earlier source's code [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", parityPublished+"/other_echo", "value=41", "--rental=tessa", "--await", "--json"); code != 0 || !strings.Contains(out, `"value":82`) {
		t.Fatalf("same package/version at another Hub reused the earlier source's interface [exit %d]\n%s", code, out)
	}
	otherRequest, problem := store.RequestByIdempotencyKey("other-source")
	fatal(t, problem)
	if otherRequest.Release != parityVersion || otherRequest.Hub != other.server.URL {
		t.Fatalf("the client pinned or redirected the other source: %+v", otherRequest)
	}

	// The machine names the newest release once per catalog revision: a newer one reaches a
	// run once this client's yank moves the revision, and the output it added is collected.
	moving := newMachineHub(t)
	publishParityRelease(t, moving, root, parityProject(t))
	newer := parityProject(t)
	pyproject := filepath.Join(newer, "pyproject.toml")
	raw, err = os.ReadFile(pyproject)
	must(t, err)
	must(t, os.WriteFile(pyproject, []byte(strings.ReplaceAll(string(raw), `version="0.0.1"`, `version="0.0.2"`)), 0o600))
	newerCode := filepath.Join(newer, "machine_parity.py")
	must(t, os.WriteFile(newerCode, []byte(`from typing import Annotated
import msgspec
from cozy_runtime.author import App, AssetBound, FileAsset, Outputs
class AddRequest(msgspec.Struct):
    value: int
class AddResult(msgspec.Struct):
    value: int
    report: Annotated[FileAsset, AssetBound(max_bytes=1024, media_types=("text/plain",))]
app = App()
@app.job
def add(payload: AddRequest, out: Outputs) -> AddResult:
    return AddResult(payload.value + 100, out.save_bytes(b"newer release output\n", media_type="text/plain"))
`), 0o600))
	publishParityReleaseAt(t, moving, root, newer, "0.0.2")
	var newest atomic.Value
	newest.Store("0.0.1")
	var newestReads atomic.Int32
	movingCatalog := moving.worker.Config.Handler
	moving.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/packages/"+parityPublished {
			newestReads.Add(1)
			_ = json.NewEncoder(w).Encode(map[string]any{"releases": []map[string]string{{"release": newest.Load().(string)}}})
			return
		}
		movingCatalog.ServeHTTP(w, r)
	})
	moving.mux.HandleFunc("DELETE /v1/packages/"+parityPublished+"/releases/"+parityVersion, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"release":"` + parityVersion + `","state":"yanked","changed":true}`))
	})
	fixtureExecutionAccess(t, root, moving.server, moving.worker.Config.Handler)
	configure(moving.server.URL)
	moved := func(key, want string, reads int32) {
		t.Helper()
		code, out := runCozy(t, root, "run", parityPublished+"/add", "value=41", "--rental=tessa", "--await", "--json", "--idempotency-key="+key)
		if code != 0 || !strings.Contains(out, want) || newestReads.Load() != reads {
			t.Fatalf("%s [exit %d, newest reads %d]\n%s", key, code, newestReads.Load(), out)
		}
	}
	moved("kept-release", `"value":42`, 1)
	newest.Store("0.0.2")
	moved("kept-release-warm", `"value":42`, 1)
	if code, out := runCozy(t, root, "package", "yank", parityPublished, "--version", parityVersion); code != 0 {
		t.Fatalf("yank [exit %d]\n%s", code, out)
	}
	moved("newer-release", `"value":141`, 2)
	movingRequest, problem := store.RequestByIdempotencyKey("newer-release")
	fatal(t, problem)
	products, problem := store.Products(movingRequest.ID)
	fatal(t, problem)
	heldReport := false
	for _, product := range products {
		if product.Output == "report" && product.Path != "" {
			report, err := os.ReadFile(product.Path)
			must(t, err)
			heldReport = string(report) == "newer release output\n"
		}
	}
	if !heldReport || movingRequest.Release != "0.0.2" {
		t.Fatalf("the newer release's output was not collected: %+v %+v", movingRequest, products)
	}

	row, problem := store.RentalRow(parityRental)
	fatal(t, problem)
	mu.Lock()
	defer mu.Unlock()
	if len(catalogReads) == 0 || len(wrongReads) != 0 {
		t.Fatalf("catalog routing: catalog=%v wrong=%v", catalogReads, wrongReads)
	}
	if row.Hub != rentalHub.server.URL {
		t.Fatal("running another source changed the rental's lifecycle authority")
	}
}
