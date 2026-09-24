package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func TestAcceptedModelCacheMissReensuresTheSameSelectionOnce(t *testing.T) {
	for _, missAgain := range []bool{false, true} {
		t.Run(map[bool]string{false: "refetched", true: "broken-ensure"}[missAgain], func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			pod := &fakePod{controlKey: public, latch: &pb.Fault{Kind: pb.FaultKind_FAULT_KIND_ARTIFACT_FETCH_FAILED, Reason: "model_materialization_required", Detail: "native object absent"}}
			var desires atomic.Int64
			var observed func(uint64)
			second := make(chan struct{})
			pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
				desired := frame.GetDesiredState()
				if desired == nil || desired.GetPlacementSet() == nil {
					return false, nil
				}
				n := desires.Add(1)
				if n == 1 {
					for index, change := range []func(*pb.ObservedWorkerState){
						func(r *pb.ObservedWorkerState) { r.AcceptedDesiredStateRevision-- },
						func(r *pb.ObservedWorkerState) { r.AcceptedPlacementSetDigest = bytes.Repeat([]byte{7}, 32) },
						func(r *pb.ObservedWorkerState) { r.Faults[0].Subject = "unrelated-placement" },
						func(r *pb.ObservedWorkerState) { r.Faults[0].Reason = "tensorfs_refused" },
					} {
						frame := pod.report(desired, 1)
						state := frame.GetObservedState()
						change(state)
						state.AdmissionEpoch = uint64(100 + index)
						if err := send(frame); err != nil {
							return true, err
						}
						observed(state.AdmissionEpoch)
						if desires.Load() != 1 {
							t.Error("historical or unrelated fault caused a model re-ensure")
						}
					}
					return true, send(pod.report(desired, 1))
				}
				if n == 2 {
					defer close(second)
					if !missAgain {
						return true, send(pod.served(desired, 1))
					}
					for i := 0; i < orchestrator.StillFactor+2; i++ {
						if err := send(pod.report(desired, 1)); err != nil {
							return true, err
						}
					}
					return true, nil
				}
				t.Error("native miss after completed ensure started an unlimited third preparation")
				return true, nil
			}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "cache-reensure", rentalWiring(connection, private))
			instance, _, _, problem := o.c.EnsureRental(podRental)
			fatal(t, problem)
			observed = func(epoch uint64) {
				waitUntil(t, "owner consumed unrelated miss report", func() bool { w := o.c.Worker(instance); return w != nil && w.AdmissionEpoch == epoch })
			}
			// This is the published inference path: the exact checkpoint rides the
			// package preparation selection. A missing repository therefore causes a
			// deterministic re-prepare of the same model before dispatch can resume.
			model := &pb.DownloadModelRef{Package: "cozy/h3-package", Slot: "tile.models.model",
				Model: "proof/h3", Release: "1.0.0", Lane: "bf16", Manifest: childDigest("8")}
			fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{Package: "cozy/h3-package", Release: "1.0.7"}}, []*pb.DownloadModelRef{model}))
			select {
			case <-second:
			case <-time.After(5 * time.Second):
				t.Fatal("typed model absence did not rerun Host prepare")
			}
			if !missAgain {
				waitUntil(t, "refetched model converged", func() bool { w := o.c.Worker(instance); return w != nil && w.ConvergedRevision == w.DesiredRevision })
			} else {
				time.Sleep(100 * time.Millisecond)
			}
			pod.mu.Lock()
			defer pod.mu.Unlock()
			if len(pod.prepares) != 2 || !bytes.Equal(pod.prepares[0].PackageSet.DownloadDelegation, pod.prepares[1].PackageSet.DownloadDelegation) {
				t.Fatal("re-ensure changed the selected inputs", len(pod.prepares))
			}
			for _, prepare := range pod.prepares {
				document, err := canonical.Read(prepare.PackageSet.DownloadDelegation, &pb.DownloadDelegation{})
				must(t, err)
				if len(document.List("models")) != 1 || document.List("models")[0].Str("manifest") != model.Manifest {
					t.Fatalf("model checkpoint was not carried into package preparation: %s", prepare.PackageSet.DownloadDelegation)
				}
			}
			if desires.Load() != 2 {
				t.Fatal("cache miss was unbounded", desires.Load())
			}
		})
	}
}

func TestJobModelCacheMissRepreparesBeforeBudgetedRetry(t *testing.T) {
	for _, broken := range []bool{false, true} {
		t.Run(map[bool]string{false: "refetched", true: "budget-exhausted"}[broken], func(t *testing.T) { proveJobModelCacheMiss(t, broken) })
	}
}

func proveJobModelCacheMiss(t *testing.T, broken bool) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	const pkg = "cozy/h3-package"
	const plan = "sha256:3535353535353535353535353535353535353535353535353535353535353535"
	manifest := childDigest("8")
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	var prepares, offers atomic.Int64
	var latest *pb.WorkerFrame
	pod.preparedPlacement = func(raw []byte, name, version string) *pb.Placement {
		prepares.Add(1)
		doc, err := canonical.Read(raw, &pb.DownloadDelegation{})
		must(t, err)
		if len(doc.List("models")) != 1 || doc.List("models")[0].Str("manifest") != manifest {
			t.Error("retry changed exact model selection")
		}
		return podPlacement(raw, name, version, "h3-package")
	}
	pod.onJobReady = func(frame *pb.WorkerFrame, send func(*pb.WorkerFrame) error) error {
		frame.GetObservedState().ConvergedRevision = 0
		latest = proto.Clone(frame).(*pb.WorkerFrame)
		return send(frame)
	}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if frame.GetOutcomeAck() != nil && latest != nil {
			return true, send(proto.Clone(latest).(*pb.WorkerFrame))
		}
		return false, nil
	}
	firstPreparation := int64(0)
	pod.answerOffer = func(offer *pb.AttemptOffer) (*pb.AttemptOutcome, error) {
		n := offers.Add(1)
		if n == 1 {
			firstPreparation = prepares.Load()
		} else if prepares.Load() <= firstPreparation {
			t.Error("retry reused PREPARED instead of ensuring model bytes")
		}
		identity, err := canonical.Spell(offer.InvocationSpecDigest)
		if err != nil {
			return nil, err
		}
		result := &pb.AttemptOutcomeBody{RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal, InvocationSpecDigest: identity,
			Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, ExecutionStarted: true, Cause: &pb.OutcomeCause{Origin: pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME}}
		if n == 1 || broken {
			result.Status = pb.OutcomeStatus_OUTCOME_STATUS_REFUSED
			result.ExecutionStarted = false
			result.SafeMessage = "model_materialization_required: cache was evicted"
			result.Cause = &pb.OutcomeCause{Code: pb.CauseCode_CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE, Origin: pb.CauseOrigin_CAUSE_ORIGIN_WORKER}
		}
		body, digest, err := canonical.Identity(result)
		return &pb.AttemptOutcome{RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch, WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
			InvocationSpecDigest: offer.InvocationSpecDigest, OutcomeId: fmt.Sprintf("model-miss-%d", n), OutcomeDigest: digest, OutcomeCanonicalBytes: body}, err
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "job-model-reensure", rentalWiring(connection, private))
	id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "job-model-reensure", Package: pkg, Release: "1.0.7", Entrypoint: "four-lane", PlanID: plan, Kind: "job", Org: "paul", Payload: []byte("{}"), Worker: podRental, Rental: true, RentalRequired: true,
		Models: []orchestrator.ModelRef{{Package: pkg, Slot: "source", BindingPath: "four-lane.models.source", Model: "proof/model", Release: "1.0.0", Lane: "native", Manifest: manifest, ManifestLength: 164, Bytes: 4096}}})
	fatal(t, problem)
	wantState, wantAttempts, wantRequeues := "succeeded", 2, 1
	if broken {
		wantState = "failed"
		wantAttempts = orchestrator.MaxRequeues + 1
		wantRequeues = orchestrator.MaxRequeues
	}
	waitUntil(t, "same logical job settles after bounded model re-ensure", func() bool {
		row, e := o.store.RequestRow(id)
		fatal(t, e)
		return row != nil && row.State == wantState
	})
	rows, problem := o.store.Attempts(id)
	fatal(t, problem)
	if len(rows) != wantAttempts || offers.Load() != int64(wantAttempts) {
		t.Fatal("retry changed request or failed to create exactly one new attempt", len(rows), offers.Load())
	}
	req, problem := o.store.RequestRow(id)
	fatal(t, problem)
	if req.Requeues != int64(wantRequeues) {
		t.Fatal("cache miss bypassed the existing durable retry budget", req.Requeues)
	}
}
