package orchestrator

import (
	"bytes"
	"context"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
)

// IdleControl keeps an idle pod's existing owner-presence hold. It sends
// only Claim: no snapshot admission, desired
// state or attempt offer can originate from this connection.
type IdleControl struct {
	Context            context.Context
	ControlStreamEpoch uint64
	Host               CheckpointHost
	connection         *grpc.ClientConn
	cancel             context.CancelFunc
}

func (s *IdleControl) Close() error { s.cancel(); return s.connection.Close() }

func DialIdleControl(parent context.Context, remote *WorkerConnection, sign RentalClaimProofSource, retained *records.Store) (*IdleControl, *exit.Error) {
	if remote == nil || remote.CACert == "" || remote.WorkerBootID == "" || sign == nil {
		return nil, exit.New(exit.Credential, "operator control requires a pinned rental and Claim signer")
	}
	proof, problem := sign(remote, recordOwnerEpoch)
	if problem != nil {
		return nil, problem
	}
	conn, err := dialWorker(remote.Addr, remote)
	if err != nil {
		return nil, exit.Unavailablef("cannot open pinned operator control lane")
	}
	ctx, cancel := context.WithCancel(parent)
	control := &IdleControl{Context: ctx, connection: conn, cancel: cancel}
	ready := false
	defer func() {
		if !ready {
			_ = control.Close()
		}
	}()
	if problem := probeWorkerProtocol(ctx, conn, true); problem != nil {
		return nil, problem
	}
	stream, err := pb.NewWorkerControlClient(conn).Control(ctx)
	if err != nil {
		return nil, idleControlEnd(err)
	}
	claim := &pb.Claim{RecordOwnerEpoch: recordOwnerEpoch, RecordOwnerId: recordOwnerID, WorkerId: remote.WorkerID, WorkerBootId: remote.WorkerBootID, WireMinor: pb.WireMinor, Proof: proof}
	if err = stream.Send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_Claim{Claim: claim}}); err != nil {
		return nil, idleControlEnd(err)
	}
	var epoch uint64
	for {
		frame, err := stream.Recv()
		if err != nil {
			return nil, idleControlEnd(err)
		}
		if proto.Size(frame) > pb.MaxInlineControlBytes {
			return nil, exit.New(exit.Structural, "operator control frame exceeds its bound")
		}
		switch message := frame.Msg.(type) {
		case *pb.WorkerFrame_ClaimAck:
			ack := message.ClaimAck
			if !ack.Accepted || ack.RecordOwnerEpoch != recordOwnerEpoch || ack.WorkerId != remote.WorkerID || ack.WorkerBootId != remote.WorkerBootID || ack.ControlStreamEpoch == 0 || ack.WireMinor < pb.MinCompatibleWireMinor {
				return nil, exit.New(exit.Conflict, "operator control claim was refused or changed the pinned worker")
			}
			epoch = ack.ControlStreamEpoch
		case *pb.WorkerFrame_Snapshot:
			snap := message.Snapshot
			if epoch == 0 || snap.RecordOwnerEpoch != recordOwnerEpoch || snap.ControlStreamEpoch != epoch || snap.WorkerBootId != remote.WorkerBootID {
				return nil, exit.New(exit.Conflict, "operator control snapshot changed its claimed envelope")
			}
			if problem := validateIdleSnapshot(snap, remote.RentalID, retained); problem != nil {
				return nil, problem
			}
			control.ControlStreamEpoch = epoch
			control.Host = checkpointAdapter(pb.NewPodHostClient(conn), claim)
			// A lost control stream cancels the mover too: unary byte progress is not an
			// owner-presence hold and may not outlive this accepted control connection.
			go func() {
				defer cancel()
				for {
					frame, err := stream.Recv()
					if err != nil || proto.Size(frame) > pb.MaxInlineControlBytes {
						return
					}
				}
			}()
			ready = true
			return control, nil
		case *pb.WorkerFrame_BootFailure:
			return nil, exit.New(exit.Failed, "operator control worker reports a boot failure")
		default:
			return nil, exit.New(exit.Structural, "operator control expected ClaimAck and WorkerSnapshot")
		}
	}
}

func idleControlEnd(err error) *exit.Error {
	if classifyPrepareEnd(err).err != nil {
		return exit.Unavailablef("operator control is unavailable before its idle snapshot")
	}
	return exit.New(exit.Conflict, "operator control refused the fixed rental Claim")
}

func validateIdleSnapshot(snap *pb.WorkerSnapshot, rentalID string, retained *records.Store) *exit.Error {
	if snap.SnapshotId == "" || !bytes.Equal(canonical.Digest(snap.SnapshotCanonicalBytes), snap.SnapshotDigest) {
		return exit.New(exit.Structural, "operator control snapshot digest mismatch")
	}
	doc, err := canonical.Read(snap.SnapshotCanonicalBytes, &pb.WorkerSnapshotBody{})
	if err != nil {
		return exit.New(exit.Structural, "operator control snapshot is not canonical")
	}
	set, _ := canonical.Spell(canonical.Digest(snap.AcceptedPlacementSetCanonicalBytes))
	if declared := doc.Str("accepted_placement_set_digest"); declared != "" && declared != set {
		return exit.New(exit.Structural, "operator control accepted placement bytes changed")
	}
	if len(snap.HostSnapshotCanonicalBytes) == 0 || !bytes.Equal(canonical.Digest(snap.HostSnapshotCanonicalBytes), snap.HostSnapshotDigest) {
		return exit.New(exit.Structural, "operator control Host snapshot digest mismatch")
	}
	host, err := canonical.Read(snap.HostSnapshotCanonicalBytes, &pb.HostSnapshotBody{})
	if err != nil {
		return exit.New(exit.Structural, "operator control Host snapshot is not canonical")
	}
	seen := map[string]map[int64]bool{}
	for _, held := range append(doc.List("held_attempts"), host.List("held_outcomes")...) {
		request, ordinal := held.Str("request_id"), held.Int("attempt_ordinal")
		if seen[request] == nil {
			seen[request] = map[int64]bool{}
		}
		if seen[request][ordinal] {
			return exit.New(exit.Conflict, "operator control snapshot repeats a held attempt")
		}
		seen[request][ordinal] = true
		if problem := verifySettledRetainedAttempt(retained, rentalID, held); problem != nil {
			return problem
		}
	}
	return nil
}

// Only the development holder supplies its locked local authority. Generic idle
// and source-custody lanes still require an empty held-attempt census. Retention
// keeps completed outcomes in the workspace; accepting their exact settled
// identity neither releases bytes nor opens the snapshot dispatch barrier.
// Both snapshot censuses carry HeldAttempt outcome ID/digest; those must match
// the closed local terminal whose canonical body is independently verified here.
func verifySettledRetainedAttempt(store *records.Store, rentalID string, held canonical.Doc) *exit.Error {
	refuse := func() *exit.Error {
		return exit.New(exit.Conflict, "operator control cannot take over an unverified held attempt")
	}
	if store == nil || rentalID == "" || held.Int("state") != int64(pb.AttemptState_ATTEMPT_STATE_OUTCOME_PENDING_ACK) || held.Int("attempt_ordinal") <= 0 {
		return refuse()
	}
	request, problem := store.RequestRow(held.Str("request_id"))
	if problem != nil {
		return problem
	}
	if request == nil || request.Worker != rentalID || !request.RetainWork {
		return refuse()
	}
	switch request.State {
	case "paused", "blocked", "succeeded", "failed", "canceled", "refused", "abandoned":
	default:
		return refuse()
	}
	attempt, problem := store.AttemptRow(request.ID, held.Int("attempt_ordinal"))
	if problem != nil {
		return problem
	}
	if attempt == nil {
		return refuse()
	}
	terminalDigest, _ := canonical.Spell(canonical.Digest(attempt.TerminalBody))
	instance := (WorkerLaunchSpec{Connection: &WorkerConnection{RentalID: rentalID}}).InstanceID()
	if attempt.State != "closed" || attempt.InstanceID != instance || attempt.TerminalID == "" ||
		attempt.InvocationDigest != held.Str("invocation_spec_digest") || terminalDigest != attempt.TerminalDigest ||
		held.Str("outcome_id") != attempt.TerminalID || held.Str("outcome_digest") != attempt.TerminalDigest {
		return refuse()
	}
	body, err := canonical.Read(attempt.TerminalBody, &pb.AttemptOutcomeBody{})
	if err != nil || body.Str("request_id") != request.ID || body.Int("attempt_ordinal") != attempt.Attempt ||
		body.Str("invocation_spec_digest") != attempt.InvocationDigest {
		return refuse()
	}
	return nil
}
