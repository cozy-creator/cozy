package producttest

import (
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

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
