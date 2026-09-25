package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestRequestBindingChangesOnlyBeforeAnyAttempt(t *testing.T) {
	for _, phase := range []string{"submitted", "canceled", "dispatching", "offered", "dispatch_aborted", "refused_retry"} {
		t.Run(phase, func(t *testing.T) {
			store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			set, invocation := servingPlacementFixture(t)
			spec, err := canonical.Read(invocation, &pb.InvocationSpec{})
			must(t, err)
			binding := spec.Sub("serving").Str("entrypoint_binding_digest")
			old := "sha256:" + strings.Repeat("e", 64)
			request, _, problem := store.Submit(records.Request{ID: phase, IdemKey: phase,
				BodyDigest: assessmentDigest([]byte(phase)), Package: "proof/model", Release: "1.0.0",
				Entrypoint: "generate", PlanID: old, Payload: []byte(`{}`)})
			fatal(t, problem)
			fatal(t, store.BindRequestPlan(request.ID, binding))
			if phase == "canceled" {
				changed, problem := store.CancelQueuedRequest(request.ID, nil)
				fatal(t, problem)
				if !changed {
					t.Fatal("request did not cancel")
				}
			} else if phase != "submitted" {
				fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "instance",
					Package: "proof/model", WorkerID: "worker", Devices: []string{"cpu"}}))
				ordinal, problem := store.Dispatch(records.Attempt{RequestID: request.ID,
					InstanceID: "instance", SessionID: "boot", InvocationDigest: assessmentDigest(invocation),
					InvocationCanonical: invocation, ServingPlacementSet: set})
				fatal(t, problem)
				switch phase {
				case "dispatch_aborted":
					fatal(t, store.AbortDispatch(request.ID, ordinal, "boot", "transport closed before offer"))
				case "offered", "refused_retry":
					fatal(t, store.OfferDispatch(request.ID, ordinal, "boot"))
					if phase == "refused_retry" {
						_, problem = store.AcceptTerminal(records.Terminal{RequestID: request.ID,
							Attempt: ordinal, SessionID: "boot", InvocationDigest: assessmentDigest(invocation),
							TerminalID: "refused", TerminalDigest: assessmentDigest([]byte("refused")),
							Status: "REFUSED", Cause: "NO_CAPACITY", RequestState: "requeue_pending"})
						fatal(t, problem)
						fatal(t, store.Closed(request.ID, ordinal))
						_, started, _, problem := store.BeginRequeue(request.ID, 3, false)
						fatal(t, problem)
						if !started {
							t.Fatal("refused attempt did not return to the queue")
						}
					}
				}
			}
			problem = store.BindRequestPlan(request.ID, old)
			want := binding
			if phase == "submitted" {
				fatal(t, problem)
				want = old
			} else if problem == nil || problem.ErrName() != "request_invocation_identity_changed" {
				t.Fatalf("%s permitted a different binding: %v", phase, problem)
			}
			// Repeated observation of the same binding remains idempotent in every state.
			fatal(t, store.BindRequestPlan(request.ID, want))
			row, problem := store.RequestRow(request.ID)
			fatal(t, problem)
			if row.PlanID != want || row.Release != request.Release || row.Package != request.Package {
				t.Fatalf("binding update changed frozen request facts: %+v", row)
			}
		})
	}
}
