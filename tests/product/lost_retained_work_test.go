package producttest

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func retainedLostObserver(t *testing.T, store *records.Store, label string) (records.Request, records.Event) {
	t.Helper()
	request, _, problem := store.Submit(records.Request{ID: "lost-" + label, IdemKey: "lost-" + label,
		BodyDigest: childDigest("1"), Package: "local/private-proof", Entrypoint: "main", Kind: "job",
		Payload: []byte(`{}`), Worker: "lost-rental", Rental: true, RetainWork: true, MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, request.Worker))
	capture, spec := []byte(`{"capture":"kept"}`), []byte(`{"invocation":"kept"}`)
	submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "workspace", SubmissionId: request.IdemKey,
		CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture),
		Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
	fatal(t, store.RecordMachineSubmission(request.ID, submission))
	receipt := &pb.MachineExecutionReceipt{RequestId: request.ID, SubmissionId: request.IdemKey,
		CaptureDigest: submission.CaptureDigest, InvocationSpecDigest: submission.Offer.InvocationSpecDigest,
		AcceptedAtMs: 1000, WorkerId: "worker", WorkerBootId: "boot", ExecutionWorkspaceId: "workspace"}
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	fatal(t, store.ObserveMachineExecution(request.ID, &pb.MachineExecutionState{RequestId: request.ID,
		WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId,
		Generation: 1, AttemptOrdinal: 1, State: "running"}, &pb.MachineExecutionEventPage{}))
	changed, problem := store.BlockRetainedWork(request.ID, "request.state_lost", "the retained rental and its local intermediate bytes are no longer available")
	fatal(t, problem)
	if !changed {
		t.Fatal("legacy lost-state fixture did not block")
	}
	events, problem := store.EventsAfter(request.ID, 0, 100)
	fatal(t, problem)
	current, problem := store.RequestByReference(request.ID)
	fatal(t, problem)
	return *current, events[len(events)-1]
}

func TestLostRetainedWorkReconcilesWithoutChangingHistoryOrCustody(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	request, original := retainedLostObserver(t, store, "historical-loss")
	before, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	waiting := recordPrivateTransaction(t, store, "temporary-failure", "healthy-rental")
	_, problem = store.BlockRetainedWork(waiting.ID, "source.not_ready", "input is not available yet")
	fatal(t, problem)
	store.Close()

	store, problem = records.OpenForDaemon(path)
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByReference(request.ID)
	fatal(t, problem)
	if row.State != "failed" || row.RetainWork || row.CreatedAt != request.CreatedAt || row.Ordinal != 1 {
		t.Fatalf("lost request changed identity or stayed resumable: %+v", row)
	}
	after, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if !bytes.Equal(before.Receipt, after.Receipt) || !bytes.Equal(before.Outcome, after.Outcome) ||
		!bytes.Equal(before.Submission, after.Submission) || after.Collected {
		t.Fatal("correction invented or changed an execution/custody receipt")
	}
	lost, problem := store.MachineExecutionLost(request.ID)
	fatal(t, problem)
	owed, problem := store.MachineExecutionOwesWork(request.ID)
	fatal(t, problem)
	if !lost || owed {
		t.Fatal("destroyed machine still owes an unreachable acknowledgement")
	}
	events, problem := store.EventsAfter(request.ID, 0, 100)
	fatal(t, problem)
	var prior records.Event
	for _, event := range events {
		if event.Seq == original.Seq {
			prior = event
		}
	}
	last := events[len(events)-1]
	if !reflect.DeepEqual(prior, original) || last.Type != "run.failed" || last.At != original.At ||
		last.Payload["original_error"] != original.Payload["error"] || last.Payload["reclassified_at"] == nil {
		t.Fatalf("historical cause/clock was rewritten: prior=%+v terminal=%+v", prior, last)
	}
	ended, problem := store.TerminalEventAt(request.ID)
	fatal(t, problem)
	intervals, problem := store.MachineExecutionIntervals(request.ID)
	fatal(t, problem)
	if ended != original.At || len(intervals) != 1 || intervals[0].Attempt != 1 || intervals[0].FinishedAt != original.At {
		t.Fatalf("correction invented later execution: ended=%s intervals=%+v", ended, intervals)
	}
	fatal(t, store.ReconcileLostRetainedWork())
	again, problem := store.EventsAfter(request.ID, 0, 100)
	fatal(t, problem)
	if !reflect.DeepEqual(events, again) {
		t.Fatal("reconciliation was not idempotent")
	}
	control, problem := store.RequestRow(waiting.ID)
	fatal(t, problem)
	if control.State != "blocked" || !control.RetainWork {
		t.Fatal("a recoverable stop was classified as permanent loss")
	}
	retry := request
	retry.ID, retry.IdemKey, retry.RetryOf, retry.MachineExecutionObserver = "retry-lost", "retry-lost", request.ID, true
	if _, _, problem := store.Submit(retry); problem == nil {
		t.Fatal("a permanently lost run was retryable")
	}
}

func TestLostRetainedWorkListAndWatchSkipHistoricalBlockedEvent(t *testing.T) {
	o := hostOwner(t, "lost-retained-watch")
	request, old := retainedLostObserver(t, o.store, "watch")
	changed, problem := o.store.FailLostRetainedWork(request.ID, "different-machine", "stale observation")
	fatal(t, problem)
	if changed {
		t.Fatal("a stale machine observation changed this request")
	}
	fatal(t, o.store.ReconcileLostRetainedWork())
	defer publicationControlAPI(t, o)()
	code, out := runCozy(t, o.root, "run", "list", "--json")
	var list struct {
		Invocations []struct {
			Status, ErrorType, Error string
			Number                   int64
		} `json:"invocations"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &list) != nil || len(list.Invocations) != 1 || list.Invocations[0].Status != "failed" {
		t.Fatalf("list did not expose permanent failure [%d]: %s", code, out)
	}
	code, out = runCozy(t, o.root, "run", "watch", strconv.FormatInt(request.Number, 10), "--json")
	var watched struct {
		Error struct {
			Code, Message string
			Details       map[string]any
		}
	}
	if code == 0 || json.Unmarshal([]byte(out), &watched) != nil || watched.Error.Code != "failed" ||
		!strings.Contains(watched.Error.Message, records.LostRetainedWorkMessage) || strings.Contains(out, "--retry") {
		t.Fatalf("watch stopped at old blocked state or proposed impossible retry [%d]: %s", code, out)
	}
	ended, problem := o.store.TerminalEventAt(request.ID)
	fatal(t, problem)
	if ended != old.At {
		t.Fatal("watch correction changed the historical execution end")
	}
}
