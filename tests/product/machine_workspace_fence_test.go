package producttest

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func TestMachineSubmissionWorkspaceSurvivesRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	request, receipt := machineObserverRecord(t, store)
	store.Close()
	store, problem = records.Open(path)
	fatal(t, problem)
	defer store.Close()
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	var submission pb.MachineExecutionSubmit
	must(t, proto.Unmarshal(link.Submission, &submission))
	if submission.ExpectedExecutionWorkspaceId != "persistent-workspace" || len(link.Receipt) != 0 {
		t.Fatal("restart lost pre-acceptance workspace identity")
	}
	submission.ExpectedExecutionWorkspaceId = "replacement-journal"
	if problem := store.RecordMachineSubmission(request.ID, &submission); problem == nil {
		t.Fatal("ambiguous submission rebound to replacement journal")
	}
	changed := proto.Clone(receipt).(*pb.MachineExecutionReceipt)
	changed.ExecutionWorkspaceId = "replacement-journal"
	if problem := store.AcceptMachineExecution(request.ID, changed); problem == nil {
		t.Fatal("accepted a receipt from replacement journal with identical boot ID")
	}
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
}

func TestMachineSubmissionRequiresWorkspaceBeforeFreeze(t *testing.T) {
	store, request, _ := machineObserverFixture(t)
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	var submission pb.MachineExecutionSubmit
	must(t, proto.Unmarshal(link.Submission, &submission))
	request.ID, request.IdemKey = "unfenced-request", "unfenced-submission"
	request.MachineExecutionObserver = true
	_, _, problem = store.Submit(request)
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "local"))
	submission.Offer.RequestId, submission.SubmissionId = request.ID, request.IdemKey
	submission.ExpectedExecutionWorkspaceId = ""
	if problem := store.RecordMachineSubmission(request.ID, &submission); problem == nil {
		t.Fatal("accepted an unfenced machine submission")
	}
}
