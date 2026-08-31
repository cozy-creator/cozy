package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
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

func TestReadyManagedRentalDispatchesWithoutAnotherWorkerReport(t *testing.T) {
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

	request := submission(planIDOf(t, spec), "fake/warm", "warm-rental-reuse", map[string]any{"message": "marco"})
	request.Rental = true
	id, attempt, problem := owner.c.Submit(request)
	fatal(t, problem)
	if attempt != 0 {
		t.Fatalf("unassigned rental request dispatched attempt %d before assignment", attempt)
	}
	result, problem := owner.c.AwaitSettled(id, 5*time.Second)
	if problem != nil || result.Status != "SUCCEEDED" || result.Attempt != 1 {
		t.Fatalf("warm rental did not dispatch without another report: result=%+v problem=%v", result, problem)
	}
	row, problem := owner.store.RequestRow(id)
	if problem != nil || row == nil || row.Worker != "pr-warm" {
		t.Fatalf("warm rental assignment = %+v problem=%v", row, problem)
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
