package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// A worker that accepted the job but cannot allocate its contained executor has
// answered before attempt one. Keep private work while making that answer visible.
func TestJobExecutorRefusalSettlesBeforeAttempt(t *testing.T) {
	for _, arm := range []struct {
		name, want string
		retained   bool
		ready      bool
	}{
		{"retained", "blocked", true, false},
		{"ordinary", "failed", false, false},
		{"historical_fault_with_capacity", "succeeded", true, true},
	} {
		t.Run(arm.name, func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			var offers atomic.Int64
			pod := &fakePod{controlKey: public, serve: true, jobReady: true}
			pod.onJobReady = func(frame *pb.WorkerFrame, send func(*pb.WorkerFrame) error) error {
				report := frame.GetObservedState()
				if !arm.ready {
					report.ConvergedRevision = 0
					report.JobCapacity = &pb.JobCapacity{}
					report.AvailableAttemptSlots = 0
				}
				report.Faults = []*pb.Fault{{Kind: pb.FaultKind_FAULT_KIND_LOCAL_SAFETY_REFUSAL,
					Subject: podWorkerID, Reason: "job_executor_absent", Detail: "executor containment unavailable before allocation"}}
				return send(frame)
			}
			pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
				if offer := frame.GetAttemptOffer(); offer != nil {
					offers.Add(1)
					if arm.ready {
						if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptAccepted{AttemptAccepted: &pb.AttemptAccepted{
							RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch, WorkerBootId: offer.WorkerBootId,
							RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal, InvocationSpecDigest: offer.InvocationSpecDigest}}}); err != nil {
							return true, err
						}
						return true, send(privateAttemptOutcome(offer, pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED,
							pb.CauseCode_CAUSE_CODE_UNSPECIFIED, pb.CauseOrigin_CAUSE_ORIGIN_UNSPECIFIED))
					}
				}
				return false, nil
			}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "job-executor-refusal-"+arm.name, rentalWiring(connection, private))
			id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: arm.name,
				Package: "cozy/h3-package", Entrypoint: "prepare", PlanID: childDigest("1"), Release: "1.0.7",
				Kind: "job", Payload: []byte(`{}`), Worker: podRental, Rental: true, RentalRequired: true, RetainWork: arm.retained})
			fatal(t, problem)
			waitUntil(t, "executor refusal becomes the request answer", func() bool {
				row, problem := o.store.RequestRow(id)
				fatal(t, problem)
				return row.State == arm.want
			})
			row, problem := o.store.RequestRow(id)
			fatal(t, problem)
			wantOffers := int64(0)
			if arm.ready {
				wantOffers = 1
			}
			if row.Ordinal != wantOffers || row.RetainWork != arm.retained || offers.Load() != wantOffers {
				t.Fatal("executor refusal invented an attempt or lost retained ownership")
			}
			if !arm.ready && !strings.Contains(tail(o.root+"/orchestrator.log"), "delegated cgroup") {
				t.Fatal("executor refusal did not explain the required host setup")
			}
		})
	}
}
