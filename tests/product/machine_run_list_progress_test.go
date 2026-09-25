package producttest

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// The normal CLI reads machine progress already retained by the observer. There
// is no live fanout sample, as after a client restart or between worker ticks.
func TestRunListShowsCurrentMachineAttemptProgress(t *testing.T) {
	o := hostOwner(t, "machine-list-progress")
	request, receipt := machineObserverRecord(t, o.store)
	receipt.AcceptedAtMs = uint64(time.Now().Add(-time.Minute).UnixMilli())
	fatal(t, o.store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "running", Sequence: 1}
	progress := []byte(`{"type":"progress","payload":{"stage":"Shot 3 of 4 / denoise","stage_fraction":0.375,"overall_fraction":0.57,"position":3,"total":8,"step_ms":1000}}`)
	fatal(t, o.store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{NextAfter: 1, HeadSequence: 1, Events: []*pb.MachineExecutionEvent{{Sequence: 1, AttemptOrdinal: 1, AtMs: uint64(time.Now().UnixMilli()), Kind: "progress", BodyCanonicalBytes: progress}}}))
	stopAPI := publicationControlAPI(t, o)
	defer stopAPI()
	list := func() api.Lifecycle {
		t.Helper()
		code, output := runCozy(t, o.root, "run", "list", "--json", "--full")
		var response struct {
			Invocations []api.Lifecycle `json:"invocations"`
		}
		if code != 0 || json.Unmarshal([]byte(output), &response) != nil || len(response.Invocations) != 1 {
			t.Fatalf("run list failed [%d]: %s", code, output)
		}
		return response.Invocations[0]
	}
	row := list()
	if row.Status != "in_progress" || row.ProgressStage != "Shot 3 of 4 / denoise" || row.OverallFraction == nil || *row.OverallFraction != .57 || row.StageFraction == nil || *row.StageFraction != .375 || row.Position == nil || *row.Position != 3 || row.Total == nil || *row.Total != 8 {
		t.Fatalf("machine list dropped current progress: %+v", row)
	}
	if row.RemainingMS != nil {
		t.Fatal("one persisted sample invented an ETA")
	}
	if code, output := runCozy(t, o.root, "run", "list"); code != 0 || !strings.Contains(output, "57%") || !strings.Contains(output, "Shot 3 of 4") {
		t.Fatalf("human list omitted overall/shot progress [%d]: %s", code, output)
	}
	// A retry without a new progress sample must not inherit the previous attempt.
	state.AttemptOrdinal, state.Generation = 2, 2
	fatal(t, o.store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{NextAfter: 1, HeadSequence: 1}))
	row = list()
	if row.ProgressStage != "" || row.OverallFraction != nil || row.StageFraction != nil {
		t.Fatalf("retry inherited another attempt's progress: %+v", row)
	}
	state.Sequence = 2
	stageOnly := []byte(`{"type":"progress","payload":{"stage":"Shot 1 of 4 / denoise","stage_fraction":0.25,"step_ms":1000}}`)
	fatal(t, o.store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{NextAfter: 2, HeadSequence: 2, Events: []*pb.MachineExecutionEvent{{Sequence: 2, AttemptOrdinal: 2, AtMs: uint64(time.Now().UnixMilli()), Kind: "progress", BodyCanonicalBytes: stageOnly}}}))
	row = list()
	if row.ProgressStage == "" || row.StageFraction == nil || *row.StageFraction != .25 || row.OverallFraction != nil {
		t.Fatalf("stage-local fraction became whole-job progress: %+v", row)
	}
	state.State, state.Collected = "succeeded", true
	fatal(t, o.store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{NextAfter: 2, HeadSequence: 2}))
	row = list()
	if row.Status != "completed" || row.OverallFraction == nil || *row.OverallFraction != 1 {
		t.Fatalf("completed machine run lost100percent: %+v", row)
	}
}
