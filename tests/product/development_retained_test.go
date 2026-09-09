package producttest

import (
	"context"
	"io"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestDevelopmentHoldPreservesOnlyVerifiedSettledRetention(t *testing.T) {
	for _, mode := range []string{"succeeded", "paused", "blocked", "host", "unknown", "active",
		"wrong-spec", "wrong-rental", "unclosed", "corrupt-body", "body-request", "body-attempt", "body-spec", "duplicate", "generic",
		"missing-outcome-id", "missing-outcome-digest", "wrong-outcome-id", "wrong-outcome-digest"} {
		t.Run(mode, func(t *testing.T) {
			f := developmentFixtureAt(t)
			address, stop := serveIdleHoldPeer(t, f.peer, f.cert, "127.0.0.1:0")
			defer stop()
			f.attach(t, address)
			instance := (orchestrator.WorkerLaunchSpec{Connection: &orchestrator.WorkerConnection{RentalID: f.rentalID}}).InstanceID()
			fatal(t, f.store.SpawnWorker(records.WorkerProcess{InstanceID: instance, Package: "proof/app", WorkerID: f.peer.workerID}))
			worker := f.rentalID
			if mode == "wrong-rental" {
				worker = "another-rental"
			}
			_, _, problem := f.store.Submit(records.Request{ID: "retained", IdemKey: "retained", BodyDigest: childDigest("a"),
				Package: "proof/app", Entrypoint: "prepare", Kind: "job", RetainWork: true, Worker: worker, Rental: true, Payload: []byte(`{}`)})
			fatal(t, problem)
			spec := childDigest("b")
			ordinal, problem := f.store.Dispatch(records.Attempt{RequestID: "retained", InstanceID: instance, SessionID: "original-boot", InvocationDigest: spec, InvocationCanonical: []byte(`{}`)})
			fatal(t, problem)
			fatal(t, f.store.OfferDispatch("retained", ordinal, "original-boot"))
			fatal(t, f.store.Accepted("retained", ordinal, "original-boot"))
			body := &pb.AttemptOutcomeBody{RequestId: "retained", AttemptOrdinal: uint64(ordinal), InvocationSpecDigest: spec, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED}
			if mode == "body-request" {
				body.RequestId = "someone-else"
			}
			if mode == "body-attempt" {
				body.AttemptOrdinal++
			}
			if mode == "body-spec" {
				body.InvocationSpecDigest = childDigest("c")
			}
			raw := assessmentDocument(t, body)
			digest := assessmentDigest(raw)
			if mode == "corrupt-body" {
				digest = childDigest("d")
			}
			state := "succeeded"
			if mode == "paused" || mode == "blocked" {
				state = mode
			}
			_, problem = f.store.AcceptTerminal(records.Terminal{RequestID: "retained", Attempt: ordinal, SessionID: "original-boot", InvocationDigest: spec,
				TerminalID: "original-outcome", TerminalDigest: digest, Body: raw, Status: "SUCCEEDED", RequestState: state})
			fatal(t, problem)
			if mode != "unclosed" {
				fatal(t, f.store.Closed("retained", ordinal))
			}
			invocation, _ := canonical.Raw(spec)
			outcomeDigest, _ := canonical.Raw(digest)
			held := &pb.HeldAttempt{RequestId: "retained", AttemptOrdinal: uint64(ordinal), InvocationSpecDigest: invocation,
				State: pb.AttemptState_ATTEMPT_STATE_OUTCOME_PENDING_ACK, OutcomeId: "original-outcome", OutcomeDigest: outcomeDigest}
			switch mode {
			case "missing-outcome-id":
				held.OutcomeId = ""
			case "missing-outcome-digest":
				held.OutcomeDigest = nil
			case "wrong-outcome-id":
				held.OutcomeId = "another-outcome"
			case "wrong-outcome-digest":
				held.OutcomeDigest, _ = canonical.Raw(childDigest("f"))
			}
			if mode == "unknown" {
				held.RequestId = "unknown"
			}
			if mode == "active" {
				held.State = pb.AttemptState_ATTEMPT_STATE_RUNNING
			}
			if mode == "wrong-spec" {
				held.InvocationSpecDigest, _ = canonical.Raw(childDigest("e"))
			}
			f.peer.settled = []*pb.HeldAttempt{held}
			f.peer.settledAtHost = mode == "host"
			if mode == "duplicate" {
				f.peer.settled = append(f.peer.settled, held)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			if mode == "generic" {
				target, problem := rental.Resolver(f.layout, f.store)(f.rentalID)
				fatal(t, problem)
				control, problem := orchestrator.DialIdleControl(ctx, target.Connection, rental.ClaimProof(f.layout), nil)
				if control != nil {
					_ = control.Close()
				}
				if problem == nil || problem.Code != exit.Conflict {
					t.Fatal("generic idle control admitted held work", problem)
				}
				return
			}
			if mode != "succeeded" && mode != "paused" && mode != "blocked" && mode != "host" {
				problem := cli.HoldStoredDevelopmentWorker(ctx, f.cfg, f.rentalID, f.peer.bootID, io.Discard, nil)
				if problem == nil || problem.Code != exit.Conflict {
					t.Fatal("unverified retained work admitted", problem)
				}
				return
			}
			updates := make(chan cli.DevelopmentHoldResult, 2)
			done := make(chan *exit.Error, 1)
			go func() {
				done <- cli.HoldStoredDevelopmentWorker(ctx, f.cfg, f.rentalID, f.peer.bootID, io.Discard, func(row cli.DevelopmentHoldResult) { updates <- row })
			}()
			got := awaitDevelopmentState(t, updates, "holding")
			cancel()
			fatal(t, <-done)
			if got.SnapshotAcknowledged || f.peer.otherFrames.Load() != 0 {
				t.Fatal("maintenance opened dispatch or changed retained custody")
			}
			row, problem := f.store.RequestRow("retained")
			fatal(t, problem)
			attempt, problem := f.store.AttemptRow("retained", ordinal)
			fatal(t, problem)
			if row.State != state || !row.RetainWork || attempt.TerminalDigest != digest || attempt.State != "closed" {
				t.Fatal("maintenance rewrote retained execution")
			}
		})
	}
}
