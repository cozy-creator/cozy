package orchestrator

import "testing"

func TestFreshRentalIntentAndReplayIdentity(t *testing.T) {
	submission := Submission{IdemKey: "fresh", Package: "proof/model", Entrypoint: "run", Payload: []byte(`{}`), RentNew: true}
	row, _, problem := requestRecord(submission)
	if problem != nil {
		t.Fatal(problem)
	}
	if !row.RentNew || !row.RentalRequired || !row.Rental {
		t.Fatalf("fresh intent did not imply remote: %+v", row)
	}
	// A recorded replay can already carry its own assigned worker.
	submission.Worker = "pr-owned"
	replay, _, problem := requestRecord(submission)
	if problem != nil || replay.BodyDigest != row.BodyDigest {
		t.Fatalf("assignment changed replay identity: %+v %v", replay, problem)
	}
	submission.Worker = ""
	submission.RentNew = false
	submission.RentalRequired = true
	ordinary, _, problem := requestRecord(submission)
	if problem != nil || ordinary.BodyDigest == row.BodyDigest {
		t.Fatalf("fresh and reusable requests share identity: %+v %v", ordinary, problem)
	}
	submission.RentNew = true
	submission.RequestedRental = "pr-existing"
	if _, _, problem := requestRecord(submission); problem == nil {
		t.Fatal("fresh plus existing rental accepted")
	}
}
