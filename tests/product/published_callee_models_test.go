package producttest

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Runs 1254/1255 of paul/minimax-h3 long_form (2026-09-27). The CPU job calls
// paul/qwen-image-2's generate_image, a published dependency locked in its uv.lock. The
// fleet selected the callee's model for the machine it rented, and the machine execution
// prepared the ROOT with that row: rental.download_set_model_invalid, 5 s after submit.
// One preparation is one package and that package's slots (Runtime package_prepare), so
// the callee's model rides the callee's own preparation.
func TestRentedJobPreparesPublishedCalleeModelsWithTheCallee(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	facts := publishCalleeRelease(t, h)

	root := ladderRoot(t, h)
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	identity, problem := rental.PendingCreatorIdentity(layout, "rented-callee")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	machine := &runtimeMachine{blocker: "none"}
	pod := &fakePod{controlKey: public, machine: machine, deviceCount: 4,
		preparedPlacement: func(download []byte, pkg, release string) *pb.Placement {
			placement := podPlacement(download, pkg, release, "")
			placement.PackageInterface = facts[pkg].iface
			return placement
		}}
	connection, certPath := startFakePod(t, root, pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	h.mu.Lock()
	h.rentals[podRental] = map[string]any{"rental_id": podRental, "name": "tessa", "state": "ready",
		"requested_accelerator_model": "fake-4090", "accelerator_count": 4, "hourly_rate_usd_micros": 1,
		"worker_address": connection.Addr, "media_address": connection.Media.Addr}
	h.mu.Unlock()
	fatal(t, rental.Attach(layout, store, records.Rental{ID: podRental, MachineName: "tessa", SKU: "fake-x", State: "ready",
		AcceleratorModel: "fake-4090", AcceleratorCount: 4, HourlyRateUSDMicros: 1, Hub: h.server.URL,
		Address: connection.Addr, MediaAddress: connection.Media.Addr,
		ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}, string(cert), connection.Media.Token, identity))
	store.Close()
	startDaemonProcess(t, root)

	const key = "rented-callee-models"
	if code, out := runCozy(t, root, "run", ladderPackage+"/long_form", "--rental=tessa", "--json", "--idempotency-key", key); code != 0 {
		t.Fatalf("the rented job was refused [exit %d]: %s", code, out)
	}
	store, problem = records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	waitFor(t, root, "the Runtime submission or a terminal run", func() bool {
		row, _ := store.RequestByIdempotencyKey(key)
		return machine.submitted() != nil || row != nil && (row.State == "failed" || row.State == "cancelled")
	})
	if machine.submitted() == nil {
		_, show := runCozy(t, root, "run", "show", "1", "--json")
		t.Fatalf("the job ended before its submission:\n%s\n%s", show, tail(filepath.Join(root, "daemon.log")))
	}
	row, problem := store.RequestByIdempotencyKey(key)
	fatal(t, problem)
	var callee []string
	for _, model := range row.Models {
		if model.Package == calleePackage {
			callee = append(callee, model.Package+" "+model.BindingSlot()+" "+model.Manifest)
		}
	}
	if len(callee) != 1 || strings.HasSuffix(callee[0], " ") {
		t.Fatalf("the fleet did not pin one model for the published callee: %+v", row.Models)
	}

	pod.mu.Lock()
	prepares := append([]*pb.PreparePackageSetCall(nil), pod.prepares...)
	pod.mu.Unlock()
	delivered := map[string][]string{}
	for _, call := range prepares {
		var set pb.DownloadDelegation
		must(t, canonical.Unmarshal(call.PackageSet.DownloadDelegation, &set))
		pkg := set.Packages[0].Package
		for _, model := range set.Models {
			delivered[pkg] = append(delivered[pkg], model.Package+" "+model.Slot+" "+model.Manifest)
		}
		if _, known := delivered[pkg]; !known {
			delivered[pkg] = nil
		}
	}
	want := map[string][]string{ladderPackage: nil, calleePackage: callee}
	if fmt.Sprint(delivered) != fmt.Sprint(want) {
		t.Fatalf("each package's preparation carries its own selections; got %v, want %v", delivered, want)
	}
	var capture pb.MachineExecutionCapture
	must(t, canonical.Unmarshal(machine.submitted().CaptureCanonicalBytes, &capture))
	if len(capture.InstalledPackages) != 2 {
		t.Fatalf("the capture does not hold the prepared callee: %+v", capture.InstalledPackages)
	}
	// The callee's default is not an input of the CPU root.
	var spec pb.InvocationSpec
	must(t, canonical.Unmarshal(machine.submitted().Offer.InvocationSpecCanonicalBytes, &spec))
	for _, input := range spec.Inputs {
		if strings.HasPrefix(input.InputId, "model:") {
			t.Fatalf("the root job was handed its callee's model as an input: %s", input.InputId)
		}
	}
}

const calleePackage = "proof/child"

type calleeFacts struct {
	iface  []byte
	locked string
}

// publishCalleeRelease replaces proof/h3@1.0.0 with a torch-free release whose CPU
// `long_form` job has no model of its own and whose lock pins proof/child@0.1.0 from the
// org index. The callee's `generate` entrypoint holds the ladder's model slot.
func publishCalleeRelease(t *testing.T, h *ladderHub) map[string]calleeFacts {
	t.Helper()
	return publishCalleeReleaseOf(t, h, `{"application":"h3:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[{"models":[],"name":"long_form","publishes":false,"request":{"fields":[]},"result":{"fields":[]}}]}`)
}

// publishCalleeReleaseOf publishes the same pair with the root's own interface.
func publishCalleeReleaseOf(t *testing.T, h *ladderHub, root string) map[string]calleeFacts {
	t.Helper()
	childWheel := []byte("child wheel")
	childDigest := strings.TrimPrefix(mustSpell(childWheel), "sha256:")
	rootIface, err := canonical.NormalizeJCS([]byte(root))
	must(t, err)
	childIface, err := canonical.NormalizeJCS([]byte(`{"application":"child:app","entrypoints":[{"invocable":{"context":"ctx","defaults":{},"enum_members":{},"export":"generate","module":"child","parameters":["steps"],"type_names":{}},"models":[{"class":"H3","component_use":{"condition_text":["text_encoder"],"decode_video":["video_vae"],"sample_fl2va":["fl2va_dit"]},"path":"generate.models.model"}],"name":"generate","request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[]}`))
	must(t, err)
	index := h.server.URL + "/v1/index/proof/simple/"
	rootLock := "version = 1\nrequires-python = \">=3.12\"\n\n" +
		"[[package]]\nname = \"child\"\nversion = \"0.1.0\"\nsource = { registry = \"" + index + "\" }\n" +
		"wheels = [{ url = \"" + h.server.URL + "/v1/index/proof/files/" + childDigest + "/child-0.1.0-py3-none-any.whl\", hash = \"sha256:" + childDigest + "\" }]\n\n" +
		"[[package]]\nname = \"h3\"\nversion = \"1.0.0\"\nsource = { editable = \".\" }\ndependencies = [{ name = \"child\" }]\n"
	release := func(name, version string, iface []byte, pyproject, lock string) (hub.PackageReleaseDetail, hub.PackageDownloadPlan) {
		exact := func(raw []byte) hub.ExactDocument {
			return hub.ExactDocument{CanonicalBytes: raw, Digest: mustSpell(raw), Length: int64(len(raw))}
		}
		var detail hub.PackageReleaseDetail
		detail.PackageInterface = iface
		detail.Release.Release = version
		detail.Release.PackageInterfaceDigest = mustSpell(iface)
		detail.Release.PackageInterfaceLength = int64(len(iface))
		detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25"}
		wheel := []byte(name + " wheel")
		filename := name + "-" + version + "-py3-none-any.whl"
		return detail, hub.PackageDownloadPlan{Release: version,
			PackageConfig:    exact([]byte("[application]\nobject = \"" + name + ":app\"\n")),
			PackageInterface: exact(iface),
			Pyproject:        exact([]byte(pyproject)),
			UVLock:           exact([]byte(lock)),
			Downloads: []hub.PackageInstallDownload{{Kind: "project_wheel", Path: filename,
				Distribution: name, Version: version, Digest: mustSpell(wheel), Length: int64(len(wheel)),
				Tags: []string{"py3-none-any"}, ImportRoots: []string{name}}}}
	}
	rootDetail, rootPlan := release("h3", "1.0.0", rootIface,
		"[project]\nname = \"h3\"\nversion = \"1.0.0\"\nrequires-python = \">=3.12\"\ndependencies = [\"child==0.1.0\"]\n", rootLock)
	childDetail, childPlan := release("child", "0.1.0", childIface,
		"[project]\nname = \"child\"\nversion = \"0.1.0\"\nrequires-python = \">=3.12\"\ndependencies = []\n",
		"version = 1\nrequires-python = \">=3.12\"\n\n[[package]]\nname = \"child\"\nversion = \"0.1.0\"\nsource = { editable = \".\" }\n")
	pin := func(pkg, release string) string {
		return string(testPrepareFacts(pkg, release).LockedRequirements)
	}
	facts := map[string]calleeFacts{
		ladderPackage: {iface: rootIface, locked: pin(ladderPackage, "1.0.0") + "child==0.1.0 --hash=sha256:" + childDigest + "\n"},
		calleePackage: {iface: childIface, locked: pin(calleePackage, "0.1.0")},
	}
	inventory, err := json.Marshal(map[string]any{"format": "tensorhub.image_inventory/1",
		"profile": "python3.12-cpu-linux-x86", "python": "3.12.8",
		"distributions": []map[string]string{{"name": runtimeDistribution, "version": "0.18.41"}}})
	must(t, err)
	fallback := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body any
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/proof/h3/releases/1.0.0":
			body = rootDetail
		case r.Method == http.MethodPost && r.URL.Path == "/v1/packages/proof/h3/download" && r.URL.Query().Get("release") == "1.0.0":
			body = rootPlan
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/"+calleePackage:
			body = hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "child"}, Releases: []hub.ReleaseSummary{{Release: "0.1.0"}}}
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/"+calleePackage+"/releases/0.1.0":
			body = childDetail
		case r.Method == http.MethodPost && r.URL.Path == "/v1/packages/"+calleePackage+"/download" && r.URL.Query().Get("release") == "0.1.0":
			body = childPlan
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/"+calleePackage+"/bindings":
			h.mu.Lock()
			body = map[string]any{"bindings": append([]hub.PackageBindingRow{}, h.bindings...)}
			h.mu.Unlock()
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rentals/"+podRental+"/prepare-facts":
			selected, known := facts[r.URL.Query().Get("package")]
			if !known {
				http.NotFound(w, r)
				return
			}
			body = hub.PrepareFactsView{Application: "app", ModelSlotPaths: []string{ladderSlot},
				ImageInventory: inventory, LockedRequirements: selected.locked}
		default:
			fallback.ServeHTTP(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(body)
	})
	return facts
}
