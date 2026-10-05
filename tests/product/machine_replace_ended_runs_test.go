package producttest

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// localRunOnOlderMachine records a run this computer's machine accepted and, with a status,
// ended with: the records an older machine leaves behind.
func localRunOnOlderMachine(t *testing.T, store *records.Store, label string, ended pb.OutcomeStatus) records.Request {
	t.Helper()
	request, _, problem := store.Submit(records.Request{ID: "older-" + label, IdemKey: "older-" + label, Package: "local/older",
		Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "local"))
	capture, spec := []byte(`{"capture":"`+label+`"}`), []byte(`{"invocation":"`+label+`"}`)
	submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "workspace", SubmissionId: request.IdemKey,
		CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture),
		Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
	fatal(t, store.RecordMachineSubmission(request.ID, submission))
	receipt := &pb.MachineExecutionReceipt{RequestId: request.ID, SubmissionId: request.IdemKey, CaptureDigest: submission.CaptureDigest,
		InvocationSpecDigest: submission.Offer.InvocationSpecDigest, AcceptedAtMs: 1000, WorkerId: "worker", WorkerBootId: "boot", ExecutionWorkspaceId: "workspace"}
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId,
		ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "running", Sequence: 1}
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{NextAfter: 1, HeadSequence: 1,
		Events: []*pb.MachineExecutionEvent{{Sequence: 1, AttemptOrdinal: 1, AtMs: 1001, Kind: "running", BodyCanonicalBytes: []byte(`{}`)}}}))
	if ended != pb.OutcomeStatus_OUTCOME_STATUS_UNSPECIFIED {
		state.State, state.Sequence = map[pb.OutcomeStatus]string{pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED: "succeeded", pb.OutcomeStatus_OUTCOME_STATUS_FAILED: "failed"}[ended], 2
		fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{NextAfter: 2, HeadSequence: 2}))
		body, err := canonical.Bytes(&pb.AttemptOutcomeBody{RequestId: request.ID, AttemptOrdinal: 1,
			InvocationSpecDigest: "sha256:" + hex.EncodeToString(receipt.InvocationSpecDigest), Status: ended, SafeMessage: label})
		must(t, err)
		fatal(t, store.RecordMachineOutcome(request.ID, &pb.AttemptOutcome{RequestId: request.ID, AttemptOrdinal: 1,
			InvocationSpecDigest: receipt.InvocationSpecDigest, OutcomeId: label, OutcomeDigest: canonical.Digest(body), OutcomeCanonicalBytes: body}))
	}
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	return *row
}

// `cozy machine install` over a machine that predates cozy.machine.v1 is held back only by
// runs that have not ended. Runs that ended on the older machine (failed, or completed and
// never collected) do not hold it: the machine is replaced, each keeps the state it ended in,
// its record says the older machine's own record of it is gone, and nothing waits on it.
func TestMachineInstallReplacesAnOlderMachineWhoseRunsHaveEnded(t *testing.T) {
	if *machineHostBinary == "" || *machineRuntimeWheel == "" || *machineTensorFSWheel == "" {
		t.Skip("requires -machine-host and a -machine-runtime-wheel/-machine-tensorfs-wheel pair")
	}
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv lays out the machine root")
	}
	root, err := os.MkdirTemp("", "czg")
	must(t, err)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("evidence retained at %s\nlocal machine log:\n%s", root, tail(filepath.Join(root, "machine", "host.log")))
		} else {
			_ = removeAllForce(root)
		}
	})
	// The older machine: installed, and not a cozy.machine.v1 machine.
	stubMachine(t, root, "#!/bin/sh\nexit 1\n")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	failed := localRunOnOlderMachine(t, store, "failed", pb.OutcomeStatus_OUTCOME_STATUS_FAILED)
	uncollected := localRunOnOlderMachine(t, store, "uncollected", pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED)
	running := localRunOnOlderMachine(t, store, "running", pb.OutcomeStatus_OUTCOME_STATUS_UNSPECIFIED)
	if failed.State != "failed" || uncollected.State != "succeeded" || running.State != "dispatching" {
		t.Fatalf("fixture states: %s, %s, %s", failed.State, uncollected.State, running.State)
	}
	store.Close()
	install := []string{"machine", "install", "--host", *machineHostBinary, "--runtime-wheel", *machineRuntimeWheel, "--tensorfs-wheel", *machineTensorFSWheel, "--json"}

	// The run still in flight holds the machine; the ended ones are not named.
	code, out := runCozy(t, root, install...)
	if code == 0 || !strings.Contains(out, "machine.holds_runs") || !strings.Contains(out, "run 3 ") || strings.Contains(out, "run 1") || strings.Contains(out, "runs ") {
		t.Fatalf("install over a machine holding one unended run [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "cancel", "--abandon", running.ID); code != 0 {
		t.Fatalf("abandon [exit %d]\n%s", code, out)
	}

	code, out = runCozy(t, root, install...)
	var installed struct {
		Replaced *struct{} `json:"replaced"`
	}
	if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &installed) != nil || installed.Replaced == nil {
		t.Fatalf("install over a machine whose runs have ended [exit %d]\n%s", code, out)
	}
	if !strings.Contains(out, "never collected, and gone with the older machine: the outputs of run 2 ") {
		t.Fatalf("the install does not say which completed run's outputs went with the older machine:\n%s", out)
	}
	if code, human := runCozy(t, root, "machine", "show"); code != 0 || strings.Contains(human, "older kind") {
		t.Fatalf("the replaced machine still reads as the older kind [exit %d]\n%s", code, human)
	}
	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for _, before := range []records.Request{failed, uncollected} {
		row, problem := store.RequestRow(before.ID)
		fatal(t, problem)
		lost, problem := store.MachineExecutionLost(before.ID)
		fatal(t, problem)
		owed, problem := store.MachineExecutionOwesWork(before.ID)
		fatal(t, problem)
		if row.State != before.State || !lost || owed {
			t.Fatalf("%s after the replacement: state %s (was %s), lost %v, owed %v", before.ID, row.State, before.State, lost, owed)
		}
	}
	if live, problem := store.MachineLiveRuns("local"); problem != nil || len(live) != 0 {
		t.Fatalf("runs still held by the replaced machine: %v %v", live, problem)
	}
}
