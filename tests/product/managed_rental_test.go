package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/mediawire"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestRentalLastSettlementDistinguishesWarmServingFromJobs(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.db"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.AttachWorker(records.WorkerProcess{
		InstanceID: "pr-warm", Package: "proof/package", WorkerID: "worker",
	}))
	serving := records.Request{
		ID: "req-serving", IdemKey: "serving", BodyDigest: "serving-body",
		Package: "proof/package", Entrypoint: "marco", Payload: []byte(`{}`),
		Worker: "pr-warm", Rental: true,
	}
	_, fresh, problem := store.Submit(serving)
	if problem != nil || !fresh {
		t.Fatalf("submit serving request: fresh=%v problem=%v", fresh, problem)
	}
	attempt, problem := store.Dispatch(records.Attempt{
		RequestID: serving.ID, InstanceID: "pr-warm", SessionID: "boot",
		InvocationDigest: "invocation", InvocationCanonical: []byte(`{}`),
	})
	fatal(t, problem)
	fatal(t, store.OfferDispatch(serving.ID, attempt, "boot"))
	applied, problem := store.AcceptTerminal(records.Terminal{
		RequestID: serving.ID, Attempt: attempt, SessionID: "boot",
		InvocationDigest: "invocation", TerminalID: "outcome", TerminalDigest: "terminal",
		Status: "SUCCEEDED", RequestState: "succeeded",
	})
	if problem != nil || !applied {
		t.Fatalf("accept serving terminal: applied=%v problem=%v", applied, problem)
	}
	queued, running, problem := store.RentalRunCounts("pr-warm")
	if problem != nil || queued != 0 || running != 1 {
		t.Fatalf("terminal awaiting ack counted as queued=%d running=%d problem=%v", queued, running, problem)
	}
	fatal(t, store.Closed(serving.ID, attempt))
	queued, running, problem = store.RentalRunCounts("pr-warm")
	if problem != nil || queued != 0 || running != 0 {
		t.Fatalf("acked terminal counted as queued=%d running=%d problem=%v", queued, running, problem)
	}
	last, found, problem := store.RentalLastSettlement("pr-warm")
	if problem != nil || !found || last.RequestID != serving.ID || last.Kind != "serving" || last.ClosedAt.IsZero() {
		t.Fatalf("serving settlement = %+v found=%v problem=%v", last, found, problem)
	}

	job := records.Request{
		ID: "req-job", IdemKey: "job", BodyDigest: "job-body", Kind: "job",
		Package: "proof/package", Entrypoint: "quantize", Payload: []byte(`{}`),
		Worker: "pr-warm", Rental: true,
	}
	_, fresh, problem = store.Submit(job)
	if problem != nil || !fresh {
		t.Fatalf("submit job: fresh=%v problem=%v", fresh, problem)
	}
	fatal(t, store.SettleRequest(job.ID, "failed"))
	last, found, problem = store.RentalLastSettlement("pr-warm")
	if problem != nil || !found || last.RequestID != job.ID || last.Kind != "job" || !last.ClosedAt.IsZero() {
		t.Fatalf("job settlement = %+v found=%v problem=%v", last, found, problem)
	}
}

func TestManagedRentalBatchAssignsQueuedCompatibilityClass(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.db"))
	fatal(t, problem)
	defer store.Close()
	for _, request := range []records.Request{
		{ID: "req-cpu-one", IdemKey: "cpu-one", BodyDigest: "cpu-one", Payload: []byte(`{}`), Rental: true},
		{ID: "req-cpu-two", IdemKey: "cpu-two", BodyDigest: "cpu-two", Payload: []byte(`{}`), Rental: true},
		{ID: "req-gpu", IdemKey: "gpu", BodyDigest: "gpu", Payload: []byte(`{}`), Rental: true,
			Models: []records.ModelRef{{Model: "proof/model"}}},
	} {
		_, fresh, problem := store.Submit(request)
		if problem != nil || !fresh {
			t.Fatalf("submit %s: fresh=%v problem=%v", request.ID, fresh, problem)
		}
	}
	fatal(t, store.AssignManagedRentalClass("rental-cpu", true))
	for _, id := range []string{"req-cpu-one", "req-cpu-two", "req-gpu"} {
		row, problem := store.RequestRow(id)
		fatal(t, problem)
		if (id == "req-gpu" && row.Worker != "") ||
			(id != "req-gpu" && row.Worker != "rental-cpu") {
			t.Fatalf("request %s assigned to %q", id, row.Worker)
		}
	}
}

func TestRentalRequestRequiresTheExplicitAcquisitionSeam(t *testing.T) {
	owner := hostOwner(t, "managed-rental-gate")
	requested := submission("sha256:plan", "proof/package", "rental-gate", map[string]any{})
	requested.Rental = true
	id, attempt, problem := owner.c.Submit(requested)
	fatal(t, problem)
	if attempt != 0 {
		t.Fatalf("rental request dispatched attempt %d without acquisition", attempt)
	}
	row := waitRequestState(t, owner, id, "failed")
	if !row.Rental || row.Worker != "" {
		t.Fatalf("rental request lost placement intent: %+v", row)
	}
	events, problem := owner.store.EventsAfter(id, 0, 100)
	fatal(t, problem)
	if !eventHasError(events, "rental.acquisition_unavailable") {
		t.Fatalf("rental request did not stop at the acquisition seam: %+v", events)
	}
	changedMode := requested
	changedMode.Rental = false
	if _, _, problem := owner.c.Submit(changedMode); problem == nil ||
		!strings.Contains(problem.Message, "different body") {
		t.Fatalf("changed placement mode reused one orchestrator identity: %v", problem)
	}

	local := submission("sha256:plan", "proof/package", "local-gate", map[string]any{})
	localID, attempt, problem := owner.c.Submit(local)
	fatal(t, problem)
	if attempt != 0 {
		t.Fatalf("local request dispatched attempt %d without local capacity", attempt)
	}
	waitRequestState(t, owner, localID, "failed")
	events, problem = owner.store.EventsAfter(localID, 0, 100)
	fatal(t, problem)
	if eventHasError(events, "rental.acquisition_unavailable") {
		t.Fatalf("local request entered rental acquisition: %+v", events)
	}
}

func TestRentalPermissionPrefersLocalAndRentalOnlySkipsIt(t *testing.T) {
	name := "rental-only-local"
	root := filepath.Join(os.TempDir(), "cozy-product-test", name)
	spec := fakeSpec(name, "0", "--arm", "output", "--cozy-home", root)
	var acquisitions atomic.Int64
	owner := hostOwnerConfigured(t, name, fixedLauncher{spec},
		func(options *orchestrator.Options) {
			options.RentalFleet = func() (string, *exit.Error) { return "rentals: proof", nil }
			options.AcquireManagedRental = func(records.Request) (string, string, *exit.Error) {
				acquisitions.Add(1)
				return "", "", exit.Named(exit.Unavailable, "force_rental_proof",
					"the proof rental seam was called")
			}
		})
	instance, _, problem := owner.c.EnsureWorker(spec)
	fatal(t, problem)
	fatal(t, owner.c.EnsurePlacementReady(instance, planIDOf(t, spec)))

	permitted := submission(planIDOf(t, spec), spec.Placement.Package,
		"rental-permitted-local", map[string]any{"message": "marco"})
	permitted.Rental = true
	localID, attempt, problem := owner.c.Submit(permitted)
	fatal(t, problem)
	if attempt != 1 {
		t.Fatalf("rental-permitted request did not use ready local capacity: attempt=%d", attempt)
	}
	if result, problem := owner.c.AwaitSettled(localID, 5*time.Second); problem != nil || result.Status != "SUCCEEDED" {
		t.Fatalf("rental-permitted local result=%+v problem=%v", result, problem)
	}
	if acquisitions.Load() != 0 {
		t.Fatalf("rental permission called the paid seam despite ready local capacity")
	}

	required := submission(planIDOf(t, spec), spec.Placement.Package,
		"rental-required-remote", map[string]any{"message": "marco"})
	required.Rental, required.RentalRequired = true, true
	remoteID, attempt, problem := owner.c.Submit(required)
	fatal(t, problem)
	if attempt != 0 || acquisitions.Load() != 1 {
		t.Fatalf("forced rental attempt=%d acquisitions=%d", attempt, acquisitions.Load())
	}
	row := waitRequestState(t, owner, remoteID, "failed")
	if !row.Rental || !row.RentalRequired || row.Worker != "" || row.Ordinal != 0 {
		t.Fatalf("forced rental crossed into local dispatch: %+v", row)
	}
}

func TestRentalOnlyRequirementSurvivesOwnerRestart(t *testing.T) {
	first := hostOwner(t, "rental-only-restart")
	recorded, fresh, problem := first.store.Submit(records.Request{
		ID: "req-rental-only-restart", IdemKey: "rental-only-restart",
		BodyDigest: "rental-only-restart-body", Package: "proof/package",
		Entrypoint: "marco", Payload: []byte(`{}`), Rental: true, RentalRequired: true,
	})
	if problem != nil || !fresh {
		t.Fatalf("record rental-only request: fresh=%v problem=%v", fresh, problem)
	}
	first.close()

	store, problem := records.Open(first.l.DB)
	fatal(t, problem)
	defer store.Close()
	acquired := make(chan records.Request, 1)
	restarted, problem := orchestrator.Open(orchestrator.Options{
		Cfg: first.cfg, Layout: first.l, Store: store, Yield: "smart", Log: io.Discard,
		RentalFleet: func() (string, *exit.Error) { return "rentals: restart", nil },
		AcquireManagedRental: func(request records.Request) (string, string, *exit.Error) {
			acquired <- request
			return "", "", exit.Named(exit.Capacity, "rental.restart_refusal",
				"the restart proof supplies no remote capacity")
		},
	})
	fatal(t, problem)
	go func() { _ = restarted.Serve() }()
	defer restarted.Close(time.Second)
	_, _, problem = restarted.Reconcile()
	fatal(t, problem)

	select {
	case request := <-acquired:
		if request.ID != recorded.ID || !request.Rental || !request.RentalRequired || request.Worker != "" {
			t.Fatalf("restart acquisition saw %+v", request)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("restart did not resume the rental-only request")
	}
	row := waitRequestState(t, &owner{store: store}, recorded.ID, "failed")
	if !row.RentalRequired || row.Worker != "" || row.Ordinal != 0 {
		t.Fatalf("restart fell back to local placement: %+v", row)
	}
}

func TestReadyManagedRentalDispatchesAgainAfterOutcomeAck(t *testing.T) {
	name := "managed-rental-warm-wake"
	root := filepath.Join(os.TempDir(), "cozy-product-test", name)
	spec := fakeSpec("warm@pr-warm", "0", "--arm", "output", "--cozy-home", root)
	var owner *owner
	owner = hostOwnerConfigured(t, name, nil, func(options *orchestrator.Options) {
		options.RentalFleet = func() (string, *exit.Error) { return "rentals: warm", nil }
		options.AcquireManagedRental = func(req records.Request) (string, string, *exit.Error) {
			assigned, problem := owner.store.AssignManagedRental(req.ID, "pr-warm")
			if problem != nil {
				return "", "", problem
			}
			if !assigned {
				t.Fatalf("request %s was not assigned to the warm rental", req.ID)
			}
			return "pr-warm", "rentals: reused warm machine", nil
		}
	})
	instance, _, problem := owner.c.EnsureWorker(spec)
	fatal(t, problem)
	fatal(t, owner.c.EnsurePlacementReady(instance, planIDOf(t, spec)))

	for run := 1; run <= 2; run++ {
		request := submission(planIDOf(t, spec), "fake/warm",
			fmt.Sprintf("warm-rental-reuse-%d", run), map[string]any{"message": "marco"})
		request.Rental = true
		request.RentalRequired = true
		id, attempt, problem := owner.c.Submit(request)
		fatal(t, problem)
		if attempt != 0 {
			t.Fatalf("run %d: unassigned rental request dispatched attempt %d before assignment", run, attempt)
		}
		result, problem := owner.c.AwaitSettled(id, 5*time.Second)
		if problem != nil || result.Status != "SUCCEEDED" || result.Attempt != 1 {
			t.Fatalf("run %d: warm rental did not dispatch after the prior outcome ack: result=%+v problem=%v",
				run, result, problem)
		}
		row, problem := owner.store.RequestRow(id)
		if problem != nil || row == nil || row.Worker != "pr-warm" || !row.RentalRequired {
			t.Fatalf("run %d: warm rental assignment = %+v problem=%v", run, row, problem)
		}
	}
}

func TestDynamicRemotePackagesShareOneWarmRental(t *testing.T) {
	spell := func(label string) string {
		digest := sha256.Sum256([]byte(label))
		value, _ := canonical.Spell(digest[:])
		return value
	}
	planID, releaseDigest := spell("dynamic plan"), spell("dynamic release")
	environmentDigest, configDigest := spell("dynamic environment"), spell("dynamic config")
	// Weightless functions with the same name have the same binding digest. Package is
	// therefore part of Creator's routing key; plan id alone cannot distinguish A from B.
	planIDB, releaseDigestB := planID, spell("dynamic release b")
	environmentDigestB, configDigestB := spell("dynamic environment b"), spell("dynamic config b")
	workerRoot := t.TempDir()
	workerLogPath := filepath.Join(workerRoot, "worker.log")
	workerLog, err := os.Create(workerLogPath)
	must(t, err)
	worker := exec.Command(fakeWorkerBin,
		"--socket", "127.0.0.1:0", "--out", workerRoot,
		"--arm", "dynamic-output", "--session", "boot-dynamic",
		"--dynamic-package", "paul/marco-polo", "--dynamic-plan", planID,
		"--dynamic-release", releaseDigest, "--dynamic-environment", environmentDigest,
		"--dynamic-config", configDigest,
		"--dynamic-package-b", "paul/marco-polo-b", "--dynamic-plan-b", planIDB,
		"--dynamic-release-b", releaseDigestB, "--dynamic-environment-b", environmentDigestB,
		"--dynamic-config-b", configDigestB)
	worker.Stdout, worker.Stderr = workerLog, workerLog
	must(t, worker.Start())
	t.Cleanup(func() {
		if worker.Process != nil {
			_ = worker.Process.Kill()
		}
		_ = worker.Wait()
		_ = workerLog.Close()
	})
	readWorkerLog := func() string {
		_ = workerLog.Sync()
		data, _ := os.ReadFile(workerLogPath)
		return string(data)
	}
	addrPath := filepath.Join(workerRoot, "control.addr")
	var controlAddr string
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); {
		if raw, err := os.ReadFile(addrPath); err == nil {
			controlAddr = strings.TrimSpace(string(raw))
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if controlAddr == "" {
		t.Fatalf("dynamic worker did not publish its address:\n%s", readWorkerLog())
	}
	revision := mediawire.ContractRev
	mediaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/health":
			_ = json.NewEncoder(w).Encode(mediawire.Health{Service: mediawire.Service, ContractRev: &revision})
		case r.Method == http.MethodPut && strings.HasPrefix(r.URL.Path, "/v1/inputs/"):
			_ = json.NewEncoder(w).Encode(map[string]any{"path": "/tmp/dynamic-payload"})
		case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/v1/outputs/"):
			_ = json.NewEncoder(w).Encode(map[string]any{"dir": "/tmp/dynamic-outputs"})
		case r.Method == http.MethodDelete && strings.HasPrefix(r.URL.Path, "/v1/attempts/"):
			_ = json.NewEncoder(w).Encode(map[string]any{})
		default:
			http.NotFound(w, r)
		}
	}))
	defer mediaServer.Close()
	mediaAddr := strings.TrimPrefix(mediaServer.URL, "http://")
	connection := &orchestrator.WorkerConnection{
		RentalID: "pr-dynamic", Addr: controlAddr, WorkerID: "local",
		WorkerBootID: "boot-dynamic", Media: &media.Spec{
			Addr: mediaAddr, Token: secret.New("dynamic-media-token"),
		},
	}
	signedSets := make(chan []string, 4)
	var owner *owner
	owner = hostOwnerConfigured(t, "dynamic-managed-rental-warm-wake", nil,
		func(options *orchestrator.Options) {
			options.Rentals = func(id string) (*orchestrator.RemoteTarget, *exit.Error) {
				if id != connection.RentalID {
					return nil, exit.New(exit.NotFound, "no rental %s", id)
				}
				return &orchestrator.RemoteTarget{Connection: connection}, nil
			}
			options.ObserveRental = func(orchestrator.RentalObservation) *exit.Error { return nil }
			options.RentalClaimProof = func(*orchestrator.WorkerConnection, uint64) ([]byte, *exit.Error) {
				return []byte("dynamic-claim"), nil
			}
			options.RentalPackageSet = func(_ *orchestrator.WorkerConnection,
				packages []*pb.DownloadPackageRef, models []*pb.DownloadModelRef,
			) ([]byte, []byte, *exit.Error) {
				names := make([]string, 0, len(packages))
				for _, row := range packages {
					names = append(names, row.Package)
				}
				signedSets <- names
				delegation := &pb.DownloadDelegation{
					ExpiresAtUnix: uint64(time.Now().Add(time.Hour).Unix()), RentalId: connection.RentalID,
					WorkerId: connection.WorkerID, WorkerBootId: connection.WorkerBootID,
					Packages: packages, Models: models,
				}
				data, _, err := canonical.Identity(delegation)
				if err != nil {
					return nil, nil, exit.Internalf("cannot mint test delegation: %s", err)
				}
				return data, bytes.Repeat([]byte{1}, 64), nil
			}
			options.RentalFleet = func() (string, *exit.Error) { return "rentals: dynamic", nil }
			options.AcquireManagedRental = func(req records.Request) (string, string, *exit.Error) {
				assigned, problem := owner.store.AssignManagedRental(req.ID, connection.RentalID)
				if problem != nil || !assigned {
					return "", "", problem
				}
				return connection.RentalID, "rentals: reused dynamic machine", nil
			}
		})

	runs := []struct {
		packageName, planID, releaseDigest, environmentDigest, configDigest string
	}{
		{"paul/marco-polo", planID, releaseDigest, environmentDigest, configDigest},
		{"paul/marco-polo-b", planIDB, releaseDigestB, environmentDigestB, configDigestB},
		{"paul/marco-polo", planID, releaseDigest, environmentDigest, configDigest},
	}
	for index, run := range runs {
		request := orchestrator.Submission{
			IdemKey: fmt.Sprintf("dynamic-rental-reuse-%d", index+1),
			Package: run.packageName, Entrypoint: "marco", PlanID: run.planID,
			Release: "1.0.4", ReleaseDigest: run.releaseDigest,
			Payload: []byte(`{"message":"marco"}`), Rental: true,
		}
		id, attempt, problem := owner.c.Submit(request)
		fatal(t, problem)
		if attempt != 0 {
			t.Fatalf("run %d: unassigned dynamic rental dispatched attempt %d", index+1, attempt)
		}
		result, problem := owner.c.AwaitSettled(id, 5*time.Second)
		if problem != nil || result.Status != "SUCCEEDED" || result.Attempt != 1 {
			t.Fatalf("run %d: dynamic rental did not return its warm seat: result=%+v problem=%v\nworker:\n%s",
				index+1, result, problem, readWorkerLog())
		}
		row, problem := owner.store.RequestRow(id)
		if problem != nil || row == nil || row.Worker != connection.RentalID ||
			row.EnvironmentDigest != run.environmentDigest || row.ConfigDigest != run.configDigest {
			t.Fatalf("run %d: worker-derived invocation identity was not bound: row=%+v problem=%v",
				index+1, row, problem)
		}
	}
	first, second := <-signedSets, <-signedSets
	if strings.Join(first, ",") != "paul/marco-polo" ||
		strings.Join(second, ",") != "paul/marco-polo,paul/marco-polo-b" {
		t.Fatalf("desired package sets replaced an incumbent: first=%v second=%v", first, second)
	}
	select {
	case extra := <-signedSets:
		t.Fatalf("returning to resident package A reissued desired state: %v", extra)
	default:
	}
}

func eventHasError(events []records.Event, name string) bool {
	for _, event := range events {
		if event.Payload["error_type"] == name {
			return true
		}
	}
	return false
}

func waitRequestState(t *testing.T, owner *owner, id, state string) *records.Request {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		row, problem := owner.store.RequestRow(id)
		fatal(t, problem)
		if row != nil && row.State == state {
			return row
		}
		time.Sleep(10 * time.Millisecond)
	}
	row, problem := owner.store.RequestRow(id)
	fatal(t, problem)
	t.Fatalf("request %s did not reach %s: %+v", id, state, row)
	return nil
}
