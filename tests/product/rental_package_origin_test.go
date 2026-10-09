package producttest

import (
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
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
	_, installed, problem := store.ActivePackage("proof/quantize")
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

// A published install from another Hub is never converted into an implicit local source.
func TestForeignPublishedInstallDoesNotSatisfyMissingSelectedPackage(t *testing.T) {
	root, _, _, _, _ := runModelCatalog(t)
	original := configuredHub(t, root)
	installedHere(t, root, original, "proof/quantize", "1.0.0")
	var reads atomic.Int32
	missing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/packages/") {
			reads.Add(1)
		}
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":{"code":"package.not_found","message":"no package at this Hub"}}`))
	}))
	defer missing.Close()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+missing.URL+"\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.RecordRental(records.Rental{ID: "pr-old-hub", MachineName: "tessa", State: "ready",
		Hub: original, AcceleratorModel: "CPU", AcceleratorCount: 1, Address: "127.0.0.1:1", CertPath: "unused", HourlyRateUSDMicros: 100_000}))
	for _, placement := range [][]string{{"--rental=tessa"}, nil} {
		before := reads.Load()
		args := append([]string{"run", "proof/quantize/quantize", "--describe", "--json"}, placement...)
		code, out := runCozy(t, root, args...)
		if code == 0 || !strings.Contains(out, "package.not_found") || reads.Load() == before {
			t.Fatalf("foreign install satisfied a missing package instead of consulting the selected Hub [exit %d, placement %v]\n%s", code, placement, out)
		}
		if !strings.Contains(out, "selected Hub "+missing.URL) || !strings.Contains(out, "installed proof/quantize@1.0.0 came from "+original) || !strings.Contains(out, "--tensorhub="+original) {
			t.Fatalf("missing package hid selected Hub or installed provenance [placement %v]\n%s", placement, out)
		}
	}
}

// The same contract with real CLI, daemon, machine and Runtime. The package exists only on
// the selected source Hub. Its execution grant advertises a separate remotely reachable
// catalog address (the role a local Hub's public/ngrok URL has), and the rental's Hub never
// receives package requests or the source Hub's account credential.
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
	var catalogReads, grantLeaves []string
	wrongReads := []string{}
	grantToken := executionGrantToken(source.server.URL, "fixture-account", 1)
	catalog := source.access.Config.Handler
	source.access.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/packages/") {
			mu.Lock()
			catalogReads = append(catalogReads, r.URL.Path)
			mu.Unlock()
			if r.Header.Get("Authorization") != "Bearer "+grantToken || r.Header.Get("X-Cozy-Worker-Token") != "" {
				t.Error("source catalog was not read with its delegated execution access")
				http.Error(w, "execution access required", http.StatusForbidden)
				return
			}
		}
		catalog.ServeHTTP(w, r)
	})
	account := source.server.Config.Handler
	source.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/execution-access" {
			var body struct {
				Leaf string `json:"delegate_certificate_der_b64url"`
			}
			if json.NewDecoder(r.Body).Decode(&body) != nil {
				t.Error("invalid execution access request")
			}
			mu.Lock()
			grantLeaves = append(grantLeaves, body.Leaf)
			mu.Unlock()
		}
		account.ServeHTTP(w, r)
	})
	ownerCatalog := rentalHub.worker.Config.Handler
	rentalHub.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/packages/") || r.Header.Get("Authorization") == "Bearer "+grantToken {
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
		if request.Release != "" || request.Hub != source.server.URL || request.RequestedRental != parityRental || link == nil || link.MachineID != parityRental || !link.Collected {
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
	if otherRequest.Release != "" || otherRequest.Hub != other.server.URL {
		t.Fatalf("the client pinned or redirected the other source: %+v", otherRequest)
	}

	// A release can change after schema description but before execution. The machine,
	// not the schema lookup, selects the release for the actual bare invocation.
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
	var newestReads atomic.Int32
	movingCatalog := moving.worker.Config.Handler
	moving.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/packages/"+parityPublished {
			release := "0.0.1"
			if newestReads.Add(1) > 1 {
				release = "0.0.2"
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"releases": []map[string]string{{"release": release}}})
			return
		}
		movingCatalog.ServeHTTP(w, r)
	})
	fixtureExecutionAccess(t, root, moving.server, moving.worker.Config.Handler)
	configure(moving.server.URL)
	code, out = runCozy(t, root, "run", parityPublished+"/add", "value=41", "--rental=tessa", "--await", "--json", "--idempotency-key=machine-selected-release")
	if code != 0 || !strings.Contains(out, `"value":141`) || newestReads.Load() < 2 {
		t.Fatalf("schema lookup pinned the machine's actual release [exit %d, newest reads %d]\n%s", code, newestReads.Load(), out)
	}
	movingRequest, problem := store.RequestByIdempotencyKey("machine-selected-release")
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
	if !heldReport {
		t.Fatalf("the actual release's new output was lost because the advisory schema had no files: %+v", products)
	}

	row, problem := store.RentalRow(parityRental)
	fatal(t, problem)
	certificate, err := os.ReadFile(row.CertPath)
	must(t, err)
	leaf, _ := pem.Decode(certificate)
	if leaf == nil {
		t.Fatal("rental has no pinned certificate")
	}
	mu.Lock()
	defer mu.Unlock()
	if len(catalogReads) == 0 || len(grantLeaves) == 0 || grantLeaves[0] != base64.RawURLEncoding.EncodeToString(leaf.Bytes) || len(wrongReads) != 0 {
		t.Fatalf("catalog/grant routing: catalog=%v grants=%d wrong=%v", catalogReads, len(grantLeaves), wrongReads)
	}
	if row.Hub != rentalHub.server.URL {
		t.Fatal("running another source changed the rental's lifecycle authority")
	}
}
