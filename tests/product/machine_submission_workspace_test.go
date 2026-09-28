package producttest

import (
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// A submission recorded before execution workspaces were fenced names none. Its receipt
// is accepted in the workspace the machine reports; a submission that names a workspace
// still refuses a receipt from another.
func TestMachineReceiptForSubmissionWithoutWorkspace(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	defer store.Close()
	request, receipt := machineObserverRecord(t, store)
	foreign := proto.Clone(receipt).(*pb.MachineExecutionReceipt)
	foreign.ExecutionWorkspaceId = "another-workspace"
	if problem := store.AcceptMachineExecution(request.ID, foreign); problem == nil {
		t.Fatal("a receipt from another workspace was accepted for a fenced submission")
	}
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	var submission pb.MachineExecutionSubmit
	must(t, proto.Unmarshal(link.Submission, &submission))
	submission.ExpectedExecutionWorkspaceId = ""
	older, err := proto.Marshal(&submission)
	must(t, err)
	db, err := sql.Open("sqlite", path)
	must(t, err)
	_, err = db.Exec(`UPDATE machine_executions SET submission=? WHERE request_id=?`, older, request.ID)
	must(t, err)
	must(t, db.Close())
	fatal(t, store.AcceptMachineExecution(request.ID, foreign))
	fatal(t, store.AcceptMachineExecution(request.ID, foreign))
}
