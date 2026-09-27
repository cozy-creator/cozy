package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func TestPublishedMachineJobPreservesInstallationIdentity(t *testing.T) {
	buildID := "published-installation"
	raw, digest, err := canonical.Identity(&pb.MachineExecutionCapture{RootInstallationId: buildID,
		InstalledPackages: []*pb.InstalledPackage{{InstallationId: buildID, Package: "alice/ops", Release: "1.0.0", PackageInterface: []byte("{}")}}})
	must(t, err)
	request := records.Request{ID: "published-root", IdemKey: "published-root", Kind: "job", Package: "alice/ops", Release: "1.0.0", PlanID: childDigest("9"), Payload: []byte(`{}`), Org: "local"}
	plan := &orchestrator.JobPlan{Function: "main", DescriptorID: request.PlanID, InstallationID: buildID}
	submitted, problem := orchestrator.MachineJobSubmission(request, localpackage.ExecutionCapture{Canonical: raw, Digest: digest}, plan, nil)
	fatal(t, problem)
	var spec pb.InvocationSpec
	must(t, canonical.Unmarshal(submitted.Offer.InvocationSpecCanonicalBytes, &spec))
	if spec.GetJob().InstallationId != buildID || submitted.PreparedState.GetJob().InstallationId != buildID || request.LocalInstallationID != "" {
		t.Fatal("published code acquired a private revision identity")
	}
	plan.InstallationID = "other-installation"
	if _, problem := orchestrator.MachineJobSubmission(request, localpackage.ExecutionCapture{Canonical: raw, Digest: digest}, plan, nil); problem == nil {
		t.Fatal("a different prepared build was accepted")
	}
}

type publishedRouteResolver struct{ api.Resolver }

func (publishedRouteResolver) ResolveRemoteJob(hub, pkg, release, function string, models []orchestrator.ModelRef, deferred bool) (orchestrator.LogicalJob, *launch.Entrypoint, *exit.Error) {
	return orchestrator.LogicalJob{Package: pkg, Release: release, Function: function, DescriptorID: childDigest("9")}, &launch.Entrypoint{Name: function, Kind: "job", Request: launch.Struct{Fields: []launch.Field{}}, Result: launch.Struct{Fields: []launch.Field{}}}, nil
}

func (publishedRouteResolver) ResolveInstall(id string, _ []orchestrator.ModelRef) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return orchestrator.WorkerLaunchSpec{Placement: orchestrator.DesiredPlacement{Package: "alice/ops", Release: "1.0.0", InstallID: id}}, nil
}

func (publishedRouteResolver) JobsInstall(string) ([]launch.JobFacts, *exit.Error) {
	return []launch.JobFacts{{Name: "main", DescriptorID: childDigest("9"), Request: launch.Struct{Fields: []launch.Field{}}, Result: launch.Struct{Fields: []launch.Field{}}}}, nil
}

type publishedRouteObserver struct{}

func (publishedRouteObserver) Refresh(context.Context, records.Request) *exit.Error { return nil }
func (publishedRouteObserver) Control(context.Context, records.Request, string) *exit.Error {
	return nil
}

func TestPublishedMachineRoutingRequiresExplicitPin(t *testing.T) {
	o := hostOwner(t, "published-machine-routing")
	_, problem := o.store.Activate(records.PackageInstall{ID: "published-local-install", Package: "alice/ops", Major: 1, Version: "1.0.0", SourceKind: "tensorhub", Dir: t.TempDir(), Platform: "linux-x86_64"})
	fatal(t, problem)
	fatal(t, o.store.RecordRental(records.Rental{ID: "rental-pinned", MachineName: "otter", SKU: "cpu", AcceleratorModel: "CPU", State: "ready", Hub: "http://127.0.0.1:1", Address: "127.0.0.1:1", AcceleratorCount: 1, HourlyRateUSDMicros: 1}))
	const bearer = "published-machine-routing-fixture"
	credential := secret.New(bearer)
	handler, problem := api.New(api.Options{Orchestrator: o.c, Cfg: o.cfg, Creds: api.Credentials{CLI: credential}, Addr: "127.0.0.1:11111", Web: http.NotFoundHandler(), Packages: publishedRouteResolver{}, MachineExecutions: publishedRouteObserver{}}).Handler()
	fatal(t, problem)
	for _, arm := range []string{"default", "rental-only", "pinned", "local"} {
		body := map[string]any{"package": "alice/ops", "release": "1.0.0", "function": "main", "input": map[string]any{}, "rental": true}
		if arm == "local" {
			body["rental"] = false
			body["install_id"] = "published-local-install"
		} else if arm == "pinned" {
			body["requested_rental"] = "rental-pinned"
		} else if arm == "rental-only" {
			body["rental_required"] = true
		}
		raw, err := json.Marshal(body)
		must(t, err)
		request := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:11111/v1/local/jobs", bytes.NewReader(raw))
		request.RemoteAddr = "127.0.0.1:12345"
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Authorization", "Bearer "+bearer)
		request.Header.Set("Idempotency-Key", arm)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusAccepted {
			t.Fatalf("%s submission: %d %s", arm, response.Code, response.Body.String())
		}
		row, problem := o.store.RequestByIdempotencyKey(arm)
		fatal(t, problem)
		link, problem := o.store.MachineExecution(row.ID)
		fatal(t, problem)
		if (link != nil) != (arm == "pinned" || arm == "local") || row.LocalInstallationID != "" {
			t.Fatalf("%s changed published routing or code origin", arm)
		}
	}
}

var publishedMachineFixture = flag.String("published-machine-fixture", "", "exact public-shaped wheel/interface fixture for the owned actual Host proof")

type publishedPackageFixture struct {
	Package   string          `json:"package"`
	Release   string          `json:"release"`
	Wheel     string          `json:"wheel"`
	Interface json.RawMessage `json:"interface"`
}

type publishedHostFixture struct {
	Package publishedPackageFixture
	Layout  home.Layout
	Store   *records.Store
	Host    actualChildHost
	Path    string
	Restart func()
}

func startPublishedMachineHost(t *testing.T, configure ...func(*fakeRentalHub)) publishedHostFixture {
	t.Helper()
	if *publishedMachineFixture == "" {
		t.Skip("requires an exact wheel/interface fixture and owned actual Host")
	}
	var fixture publishedPackageFixture
	raw, err := os.ReadFile(*publishedMachineFixture)
	must(t, err)
	must(t, json.Unmarshal(raw, &fixture))
	iface, problem := launch.DecodePackageInterface(fixture.Interface)
	fatal(t, problem)
	wheel, err := os.ReadFile(fixture.Wheel)
	must(t, err)
	wheelDigest, err := canonical.Spell(canonical.Digest(wheel))
	must(t, err)
	wheelURL := publishedFixtureWheel(t, wheel, filepath.Base(fixture.Wheel))
	layout, store, host, path, restart := startActualChildHostConfigured(t, func(h *fakeRentalHub) {
		fallback := h.server.Config.Handler
		h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			switch r.URL.Path {
			case "/v1/packages/" + fixture.Package:
				org, name, _ := strings.Cut(fixture.Package, "/")
				_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: org, Name: name}, Releases: []hub.ReleaseSummary{{Release: fixture.Release}}})
			case "/v1/packages/" + fixture.Package + "/releases/" + fixture.Release:
				var detail hub.PackageReleaseDetail
				detail.Release.Release = fixture.Release
				detail.Release.PackageInterfaceDigest = assessmentDigest(iface.Raw)
				detail.Release.PackageInterfaceLength = int64(len(iface.Raw))
				detail.PackageInterface = iface.Raw
				detail.ExecutionRequirements = []string{"cozy-runtime>=0.17.2"}
				detail.RequiresPython = ">=3.12,<3.13"
				_ = json.NewEncoder(w).Encode(detail)
			case "/v1/rentals/rental-private-child-host/prepare-facts":
				if r.URL.Query().Get("package") != fixture.Package || r.URL.Query().Get("release") != fixture.Release {
					http.Error(w, "unknown exact release", 404)
					return
				}
				_, distribution, _ := strings.Cut(fixture.Package, "/")
				locked := fmt.Sprintf("--index-url https://pypi.org/simple\n%s @ %s --hash=%s\n", distribution, wheelURL, wheelDigest)
				_ = json.NewEncoder(w).Encode(hub.PrepareFactsView{Application: iface.Application, ModelSlotPaths: iface.ModelSlotPaths(), ImageInventory: h.inventories["rental-private-child-host"], LockedRequirements: locked})
			case "/wheels/" + filepath.Base(fixture.Wheel):
				w.Header().Set("Content-Type", "application/octet-stream")
				_, _ = w.Write(wheel)
			default:
				fallback.ServeHTTP(w, r)
			}
		})
		for _, update := range configure {
			update(h)
		}
	})
	return publishedHostFixture{fixture, layout, store, host, path, restart}
}

func TestPublishedMachineActualHostAndNewRootAfterRestart(t *testing.T) {
	proof := startPublishedMachineHost(t)
	fixture, layout, store, host, path, restart := proof.Package, proof.Layout, proof.Store, proof.Host, proof.Path, proof.Restart

	var build string
	for index := 0; index < 2; index++ {
		if index == 1 {
			restart() // The production supervisor restarts Host and Runtime together.
		}
		key := fmt.Sprintf("published-host-%d", index)
		code, out := runCozyPath(t, layout.Root, path, "run", fixture.Package+"/main", "--rental", "child-host", "--await", "--json", "--idempotency-key", key)
		if code != 0 {
			t.Fatalf("published root %d [%d]: %s", index, code, out)
		}
		request, problem := store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		if request == nil || request.State != "succeeded" || request.LocalInstallationID != "" || request.Release != fixture.Release {
			t.Fatalf("published request changed origin or failed: %+v", request)
		}
		link, problem := store.MachineExecution(request.ID)
		fatal(t, problem)
		if link == nil || !link.Collected || len(link.Receipt) == 0 {
			t.Fatal("published root has no collected Runtime receipt")
		}
		var submitted pb.MachineExecutionSubmit
		must(t, proto.Unmarshal(link.Submission, &submitted))
		var capture pb.MachineExecutionCapture
		must(t, canonical.Unmarshal(submitted.CaptureCanonicalBytes, &capture))
		if len(capture.InstalledPackages) != 1 || capture.InstalledPackages[0].Package != fixture.Package || capture.InstalledPackages[0].Release != fixture.Release {
			t.Fatal("published capture changed package origin")
		}
		current := capture.RootInstallationId
		if current == "" || capture.InstalledPackages[0].InstallationId != current || submitted.PreparedState.GetJob().InstallationId != current {
			t.Fatal("published installed identity changed during submission")
		}
		build = current
		attempts, problem := store.Attempts(request.ID)
		fatal(t, problem)
		children, problem := store.Children(request.ID)
		fatal(t, problem)
		if len(attempts) != 0 || len(children) != 0 {
			t.Fatal("Creator owns published attempts or children")
		}
		t.Logf("published root %s accepted/collected, build=%s, container=%s: %s", request.ID, build, host.Container, out)
	}
}
