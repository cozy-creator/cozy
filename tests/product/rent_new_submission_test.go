package producttest

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestFreshRentalPublicSubmissionRetainsIdentity(t *testing.T) {
	st, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer st.Close()
	owner, problem := orchestrator.Open(orchestrator.Options{Store: st})
	if problem != nil {
		t.Fatal(problem)
	}
	defer owner.Close(0)
	submission := orchestrator.Submission{IdemKey: "fresh", Package: "proof/model", Entrypoint: "run", Payload: []byte(`{}`), RentNew: true}
	first, fresh, problem := owner.RecordSubmission(submission)
	if problem != nil || !fresh || !first.RentNew || !first.Rental || !first.RentalRequired {
		t.Fatalf("fresh submission: %+v %v %v", first, fresh, problem)
	}
	replay, fresh, problem := owner.RecordSubmission(submission)
	if problem != nil || fresh || replay.ID != first.ID {
		t.Fatalf("replay changed request: %+v %v %v", replay, fresh, problem)
	}
	submission.RentNew = false
	submission.RentalRequired = true
	if _, _, problem := owner.RecordSubmission(submission); problem == nil {
		t.Fatal("fresh intent changed under an existing idempotency key")
	}
	submission.RentNew = true
	submission.IdemKey = "conflicting"
	submission.RequestedRental = "pr-existing"
	if _, _, problem := owner.RecordSubmission(submission); problem == nil {
		t.Fatal("fresh plus existing rental accepted")
	}
	if row, problem := st.RequestByIdempotencyKey("conflicting"); problem != nil || row != nil {
		t.Fatalf("rejected submission left durable work: %+v %v", row, problem)
	}
}
