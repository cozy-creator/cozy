package producttest

import (
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Run855 was released between observing succeeded and collecting its outcome.
func TestMachineCompletionStartsIdleGraceBeforeOutcomeCollection(t *testing.T) {
	path := filepath.Join(t.TempDir(), "records.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	defer func() { store.Close() }()
	request, receipt := machineObserverRecord(t, store)
	old := time.Now().Add(-2 * time.Hour).UTC()
	row := idleRecord(t, store, "pr-owned-machine", old)
	db, err := sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	// Seed a genuinely old rental/request clock without waiting two hours.
	_, err = db.Exec(`UPDATE requests SET rental=1,worker=?,created_at=? WHERE id=?`, row.ID, old.Format(time.RFC3339Nano), request.ID)
	must(t, err)
	receipt.AcceptedAtMs = uint64(old.UnixMilli())
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "running"}
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
	if released, problem := store.ClaimRentalIdleRelease(row.ID, time.Now()); problem != nil || released {
		t.Fatalf("long-running work was released: %v %v", released, problem)
	}

	firstFinished := time.Now()
	state.State = "succeeded"
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if link.Collected || len(link.Outcome) != 0 {
		t.Fatal("fixture accidentally collected the result")
	}
	idle, problem := rental.ObserveIdle(store, row)
	fatal(t, problem)
	if idle.Since.Before(firstFinished) {
		t.Fatalf("completion kept stale rental clock: %+v", idle)
	}
	if released, problem := store.ClaimRentalIdleRelease(row.ID, time.Now()); problem != nil || released {
		t.Fatalf("newly completed result was released before collection: %v %v", released, problem)
	}
	baseline := idle.Since

	store.Close()
	store, problem = records.Open(path)
	fatal(t, problem)
	// Re-reading terminal state and collecting its delayed outcome are not work.
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
	spec, err := canonical.Spell(receipt.InvocationSpecDigest)
	must(t, err)
	raw, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: spec, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED})
	must(t, err)
	fatal(t, store.RecordMachineOutcome(request.ID, &pb.AttemptOutcome{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: receipt.InvocationSpecDigest, OutcomeId: "collected-later", OutcomeDigest: digest, OutcomeCanonicalBytes: raw}))
	idle, problem = rental.ObserveIdle(store, row)
	fatal(t, problem)
	if !idle.Since.Equal(baseline) {
		t.Fatalf("restart, polling or collection renewed grace: %s -> %s", baseline, idle.Since)
	}
	if released, problem := store.ClaimRentalIdleRelease(row.ID, baseline.Add(900*time.Second-time.Nanosecond)); problem != nil || released {
		t.Fatalf("rental was released before full idle grace: %v %v", released, problem)
	}
	if released, problem := store.ClaimRentalIdleRelease(row.ID, baseline.Add(900*time.Second)); problem != nil || !released {
		t.Fatalf("retained result disabled fixed idle expiration: %v %v", released, problem)
	}
}

func TestMachineInactiveStateBookkeepingDoesNotRenewIdleGrace(t *testing.T) {
	for _, finishedState := range []string{"failed", "blocked", "paused", "canceled"} {
		t.Run(finishedState, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "records.sqlite")
			store, problem := records.Open(path)
			fatal(t, problem)
			defer store.Close()
			request, receipt := machineObserverRecord(t, store)
			row := idleRecord(t, store, "pr-owned-machine", time.Now().Add(-2*time.Hour))
			db, err := sql.Open("sqlite", path)
			must(t, err)
			defer db.Close()
			_, err = db.Exec(`UPDATE requests SET rental=1,worker=? WHERE id=?`, row.ID, request.ID)
			must(t, err)
			fatal(t, store.AcceptMachineExecution(request.ID, receipt))
			state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "running"}
			fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
			before := time.Now()
			state.State = finishedState
			fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
			idle, problem := rental.ObserveIdle(store, row)
			fatal(t, problem)
			if idle.Since.Before(before) || idle.Due(time.Now()) {
				t.Fatalf("%s did not start fresh idle grace: %+v", finishedState, idle)
			}
			baseline := idle.Since
			if finishedState == "failed" {
				// A remote retry can start and fail between client observations.
				// The new attempt proves new work even though the state is unchanged.
				state.Generation++
				state.AttemptOrdinal++
				fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
				idle, problem = rental.ObserveIdle(store, row)
				fatal(t, problem)
				if !idle.Since.After(baseline) {
					t.Fatal("a completed new retry attempt did not start fresh idle grace")
				}
				baseline = idle.Since
			}
			// Canceling/releasing an already inactive result is bookkeeping, not work.
			state.Generation++
			state.State = "canceled"
			state.Collected = true
			fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
			idle, problem = rental.ObserveIdle(store, row)
			fatal(t, problem)
			if !idle.Since.Equal(baseline) {
				t.Fatalf("inactive %s cancellation renewed idle grace: %s -> %s", finishedState, baseline, idle.Since)
			}
		})
	}
}
