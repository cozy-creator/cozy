package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
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
func (publishedRouteObserver) Withdraw(string)                                      {}
func (publishedRouteObserver) PruneOperationCache(context.Context, string) (uint32, uint64, bool, *exit.Error) {
	return 0, 0, false, exit.Unavailablef("no machine")
}
func (publishedRouteObserver) Control(context.Context, records.Request, string) *exit.Error {
	return nil
}

// Every rented published job is a Runtime execution, as is a local published install.
func TestPublishedMachineRoutingOwnsEveryRentedJob(t *testing.T) {
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
		if link == nil || row.LocalInstallationID != "" {
			t.Fatalf("%s changed published routing or code origin", arm)
		}
	}
}
