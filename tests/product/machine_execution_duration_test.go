package producttest

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestMachineExecutionDurationUsesRetainedAttemptIntervals(t *testing.T) {
	for _, test := range []struct {
		name     string
		state    string
		ordinal  uint64
		accepted uint64
		events   []*pb.MachineExecutionEvent
		metrics  *pb.AttemptMetrics
		want     int64
	}{
		{name: "historical terminal", state: "succeeded", ordinal: 1, accepted: 1789439087780,
			events: []*pb.MachineExecutionEvent{{AttemptOrdinal: 1, AtMs: 1789441319723, Kind: "outcome"}}, want: 2231943},
		{name: "paused day", state: "paused", ordinal: 1, accepted: 1000,
			events: []*pb.MachineExecutionEvent{{AttemptOrdinal: 1, AtMs: 6000, Kind: "outcome"}}, want: 5000},
		{name: "resume excludes paused day", state: "succeeded", ordinal: 2, accepted: 1000,
			events: []*pb.MachineExecutionEvent{
				{AttemptOrdinal: 1, AtMs: 6000, Kind: "outcome"},
				{AttemptOrdinal: 2, AtMs: 86406000, Kind: "control", BodyCanonicalBytes: []byte(`{"action":"resume"}`)},
				{AttemptOrdinal: 2, AtMs: 86410000, Kind: "outcome"},
			}, want: 9000},
		{name: "retry delay stays stopped", state: "retrying", ordinal: 1, accepted: 1000,
			events: []*pb.MachineExecutionEvent{{AttemptOrdinal: 1, AtMs: 6000, Kind: "outcome"}}, want: 5000},
		{name: "retained metric fallback", state: "succeeded", ordinal: 2, accepted: 1000,
			metrics: &pb.AttemptMetrics{RuntimeMs: 2345}, want: 2345},
		{name: "unverified metric is not timing", state: "succeeded", ordinal: 1, accepted: 1000,
			metrics: &pb.AttemptMetrics{RuntimeMs: 2345, UnverifiedFields: []string{"runtime_ms"}}, want: 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			o := hostOwner(t, fmt.Sprintf("machine-duration-%d", time.Now().UnixNano()))
			request, receipt := machineObserverRecord(t, o.store)
			receipt.AcceptedAtMs = test.accepted
			fatal(t, o.store.AcceptMachineExecution(request.ID, receipt))
			for i, event := range test.events {
				event.Sequence = uint64(i + 1)
				if event.BodyCanonicalBytes == nil {
					event.BodyCanonicalBytes = []byte(`{}`)
				}
			}
			state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId,
				WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId,
				AttemptOrdinal: test.ordinal, Generation: test.ordinal, State: test.state, Collected: true, Sequence: uint64(len(test.events))}
			fatal(t, o.store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{Events: test.events, NextAfter: state.Sequence, HeadSequence: state.Sequence}))
			if test.state == "succeeded" {
				invocation, err := canonical.Spell(receipt.InvocationSpecDigest)
				must(t, err)
				body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: request.ID,
					AttemptOrdinal: test.ordinal, InvocationSpecDigest: invocation,
					Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, Metrics: test.metrics})
				must(t, err)
				fatal(t, o.store.RecordMachineOutcome(request.ID, &pb.AttemptOutcome{RequestId: request.ID,
					AttemptOrdinal: test.ordinal, InvocationSpecDigest: receipt.InvocationSpecDigest,
					OutcomeId: "retained-terminal", OutcomeDigest: digest, OutcomeCanonicalBytes: body}))
			}
			closeAPI := publicationControlAPI(t, o)
			defer closeAPI()
			for range 2 {
				if got := listedMachineExecutionMS(t, o.root, request.ID); got != test.want {
					t.Fatalf("execution_ms=%d, want %d from retained Runtime intervals", got, test.want)
				}
			}
			if test.state == "succeeded" && test.want > 0 {
				code, output := runCozy(t, o.root, "run", "watch", request.ID, "--json", "--full")
				var document struct {
					Execution string `json:"execution"`
				}
				if code != 0 || json.Unmarshal([]byte(output), &document) != nil || document.Execution == "" || document.Execution == "0.0s" {
					t.Fatalf("terminal CLI lost retained duration [%d]: %s", code, output)
				}
			}
			attempts, problem := o.store.Attempts(request.ID)
			fatal(t, problem)
			if len(attempts) != 0 {
				t.Fatal("timing projection invented local execution attempts")
			}
		})
	}
}

func TestMachineExecutionDurationAdvancesOnlyItsCurrentActiveInterval(t *testing.T) {
	o := hostOwner(t, fmt.Sprintf("machine-duration-active-%d", time.Now().UnixNano()))
	request, receipt := machineObserverRecord(t, o.store)
	receipt.AcceptedAtMs = uint64(time.Now().Add(-5 * time.Second).UnixMilli())
	fatal(t, o.store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId,
		ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "running"}
	fatal(t, o.store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
	defer publicationControlAPI(t, o)()
	before := time.Now().UnixMilli() - int64(receipt.AcceptedAtMs)
	got := listedMachineExecutionMS(t, o.root, request.ID)
	after := time.Now().UnixMilli() - int64(receipt.AcceptedAtMs)
	if got < before || got > after {
		t.Fatalf("active duration %d outside observed clock interval %d..%d", got, before, after)
	}
}

func listedMachineExecutionMS(t *testing.T, root, id string) int64 {
	t.Helper()
	code, output := runCozy(t, root, "run", "list", "--json", "--full")
	var document struct {
		Invocations []struct {
			ID          string `json:"id"`
			ExecutionMS int64  `json:"execution_ms"`
		} `json:"invocations"`
	}
	if code != 0 || json.Unmarshal([]byte(output), &document) != nil {
		t.Fatalf("list Runtime-owned timing [%d]: %s", code, output)
	}
	for _, row := range document.Invocations {
		if row.ID == id {
			return row.ExecutionMS
		}
	}
	t.Fatalf("run %s missing from timing readback: %s", id, output)
	return 0
}
