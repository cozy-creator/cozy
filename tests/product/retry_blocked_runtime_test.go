package producttest

import (
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"path/filepath"
	"strings"
	"testing"
)

func mustRetryNoError(t *testing.T, problem *exit.Error) {
	t.Helper()
	if problem != nil {
		t.Fatal(problem)
	}
}

func TestMachineRetryAfterBlockedPreparationKeepsUnsentHistory(t *testing.T) {
	for _, state := range []string{"unsent", "ambiguous", "canceled"} {
		t.Run(state, func(t *testing.T) {
			store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			mustRetryNoError(t, problem)
			defer store.Close()
			prior, _, problem := store.Submit(records.Request{ID: "job-blocked", IdemKey: "blocked", Kind: "job", Package: "local/example", Entrypoint: "main", Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("1", 64), RetainWork: true, MachineExecutionObserver: true})
			mustRetryNoError(t, problem)
			mustRetryNoError(t, store.LinkMachineExecution(prior.ID, "local"))
			if state == "ambiguous" {
				capture, spec := []byte(`{"capture":"immutable"}`), []byte(`{"invocation":"immutable"}`)
				mustRetryNoError(t, store.RecordMachineSubmission(prior.ID, &pb.MachineExecutionSubmit{
					ExpectedExecutionWorkspaceId: "persistent-workspace", SubmissionId: prior.IdemKey,
					CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture),
					Offer: &pb.AttemptOffer{RequestId: prior.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)},
				}))
			}
			changed, problem := store.BlockRetainedWork(prior.ID, "worker.protocol_incompatible", "upgrade the idle Runtime")
			mustRetryNoError(t, problem)
			if !changed {
				t.Fatal("preparation did not become blocked")
			}
			if state == "canceled" {
				_, problem = store.CancelMachineBeforeAcceptance(prior.ID)
				mustRetryNoError(t, problem)
			}
			retry, fresh, problem := store.Submit(records.Request{ID: "job-retry", IdemKey: "retry", Kind: "job", Package: prior.Package, Entrypoint: "main", Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("2", 64), RetainWork: true, RetryOf: prior.ID, MachineExecutionObserver: true})
			if state != "unsent" {
				if problem == nil {
					t.Fatal("retry duplicated ambiguous acceptance or revived canceled work")
				}
				return
			}
			mustRetryNoError(t, problem)
			if !fresh || retry.RetryOf != prior.ID || retry.ReuseScope != prior.ReuseScope {
				t.Fatal("retry lost retained predecessor lineage")
			}
			old, problem := store.RequestRow(prior.ID)
			mustRetryNoError(t, problem)
			link, problem := store.MachineExecution(prior.ID)
			mustRetryNoError(t, problem)
			if old.State != "blocked" || len(link.Submission) != 0 || len(link.Receipt) != 0 {
				t.Fatal("retry changed old history or invented Runtime acceptance")
			}
		})
	}
}

func TestAcceptedMachineRetryStillRequiresRetainedAuthority(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	mustRetryNoError(t, problem)
	defer store.Close()
	prior, _, problem := store.Submit(records.Request{ID: "job-accepted", IdemKey: "accepted", Kind: "job", Package: "local/example", Entrypoint: "main", Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("3", 64), RetainWork: true, MachineExecutionObserver: true})
	mustRetryNoError(t, problem)
	mustRetryNoError(t, store.LinkMachineExecution(prior.ID, "local"))
	capture, spec := []byte(`{"capture":"immutable"}`), []byte(`{"invocation":"immutable"}`)
	submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "workspace", SubmissionId: prior.IdemKey, CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture), Offer: &pb.AttemptOffer{RequestId: prior.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
	mustRetryNoError(t, store.RecordMachineSubmission(prior.ID, submission))
	receipt := &pb.MachineExecutionReceipt{RequestId: prior.ID, SubmissionId: prior.IdemKey, CaptureDigest: submission.CaptureDigest, InvocationSpecDigest: submission.Offer.InvocationSpecDigest, AcceptedAtMs: 1000, WorkerId: "worker", WorkerBootId: "boot", ExecutionWorkspaceId: "workspace"}
	mustRetryNoError(t, store.AcceptMachineExecution(prior.ID, receipt))
	mustRetryNoError(t, store.ObserveMachineExecution(prior.ID, &pb.MachineExecutionState{RequestId: prior.ID, WorkerId: "worker", WorkerBootId: "boot", ExecutionWorkspaceId: "workspace", Generation: 1, AttemptOrdinal: 1, State: "failed", Collected: true}, &pb.MachineExecutionEventPage{}))
	request := records.Request{ID: "job-retry-accepted", IdemKey: "retry-accepted", Kind: "job", Package: prior.Package, Entrypoint: "main", Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("4", 64), RetainWork: true, RetryOf: prior.ID, MachineExecutionObserver: true}
	_, _, problem = store.Submit(request)
	mustRetryNoError(t, problem)
	mustRetryNoError(t, store.RecordMachineControl(prior.ID, &pb.MachineExecutionControl{Execution: &pb.MachineExecutionQuery{RequestId: prior.ID, ExpectedExecutionWorkspaceId: "workspace"}, CommandId: "cancel", ExpectedGeneration: 1, Action: pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL}))
	request.ID, request.IdemKey = "job-retry-after-cancel", "retry-after-cancel"
	if _, _, problem := store.Submit(request); problem == nil {
		t.Fatal("retry bypassed pending Runtime cancellation")
	}
}
