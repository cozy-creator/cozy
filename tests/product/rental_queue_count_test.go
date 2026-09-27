package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// Rented inference and jobs wait in Runtime's GPU queue once submitted
// (TestRentedInferenceIsARuntimeExecution). The rented work this daemon still queues per
// machine is a model transfer, and only runs on the selected rental count toward its place.
func TestQueuePositionCountsOnlyTheSelectedRental(t *testing.T) {
	o := hostOwner(t, "rental-queue-count", func(options *orchestrator.Options) {
		options.Rentals = func(string) (*orchestrator.RemoteTarget, *exit.Error) {
			return nil, exit.Unavailablef("fixture rental is attaching")
		}
	})
	submit := func(key, worker, requested string) string {
		t.Helper()
		id, _, problem := o.c.Submit(orchestrator.Submission{
			IdemKey: key, Kind: "job", Package: "acme/weightless", Entrypoint: "convert", Release: "1.0.0",
			PlanID: childDigest("c"), Payload: []byte(`{}`), Worker: worker, RequestedRental: requested,
			Rental: true, RentalRequired: true,
			ModelTransfer: &records.ModelTransferIntent{Kind: "model-upload", Destination: "acme/model",
				Outputs: []records.ModelTransferOutput{{Name: "model"}}},
		})
		fatal(t, problem)
		return id
	}
	first := submit("first-niwaka", "pr-niwaka", "pr-niwaka")
	submit("other-rental", "pr-other", "pr-other")
	second := submit("second-niwaka", "pr-niwaka", "pr-niwaka")
	// An automatically selected request joins that machine's queue too.
	third := submit("third-niwaka", "pr-niwaka", "")
	for id, want := range map[string]int{first: 1, second: 2, third: 3} {
		if position, depth := o.c.QueueState(id); position != want || depth != 3 {
			t.Fatalf("%s queue position = %d/%d; want %d/3", id, position, depth, want)
		}
	}
	fatal(t, o.c.CancelQueued(second, "test client"))
	if position, depth := o.c.QueueState(third); position != 2 || depth != 2 {
		t.Fatalf("cancelled run still consumes queue position: %d/%d", position, depth)
	}
	if position, depth := o.c.QueueState(second); position != 0 || depth != 0 {
		t.Fatalf("cancelled run has a queue: %d/%d", position, depth)
	}
}
