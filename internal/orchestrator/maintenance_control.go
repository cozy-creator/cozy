package orchestrator

import (
	"bytes"
	"context"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// MaintenanceControl holds one rental's signed Claim while its Runtime is updated. It
// sends only Claim: no snapshot admission, desired state or attempt offer can originate
// from this connection. Idle release is the Host's alone.
type MaintenanceControl struct {
	Context            context.Context
	ControlStreamEpoch uint64
	Host               CheckpointHost
	connection         *grpc.ClientConn
	cancel             context.CancelFunc
}

func (s *MaintenanceControl) Close() error { s.cancel(); return s.connection.Close() }

// DialMaintenanceControl opens a Claim-only lane for Runtime repair.
func DialMaintenanceControl(parent context.Context, remote *WorkerConnection, sign RentalClaimProofSource, retained *records.Store) (*MaintenanceControl, *exit.Error) {
	control, problem := dialMaintenanceControl(parent, remote, sign, retained)
	if problem != nil && problem.ErrName() == "rental.control_reconnect" {
		// A cold Host transfers its workspace ledger to Runtime once and then
		// requires a new Claim. Never accept that first, unrecovered snapshot.
		return dialMaintenanceControl(parent, remote, sign, retained)
	}
	return control, problem
}

func dialMaintenanceControl(parent context.Context, remote *WorkerConnection, sign RentalClaimProofSource, retained *records.Store) (*MaintenanceControl, *exit.Error) {
	if remote == nil || remote.CACert == "" || remote.WorkerBootID == "" || sign == nil {
		return nil, exit.New(exit.Credential, "maintenance control requires a pinned rental and Claim signer")
	}
	proof, problem := sign(remote, recordOwnerEpoch)
	if problem != nil {
		return nil, problem
	}
	conn, err := dialWorker(remote.Addr, remote)
	if err != nil {
		return nil, exit.Unavailablef("cannot open pinned maintenance control lane")
	}
	ctx, cancel := context.WithCancel(parent)
	control := &MaintenanceControl{Context: ctx, connection: conn, cancel: cancel}
	ready := false
	defer func() {
		if !ready {
			_ = control.Close()
		}
	}()
	// Maintenance sends only Claim and reads snapshots, which no wire minor refuses; it
	// negotiates the Claim minor from the peer's range.
	probeContext, probeCancel := context.WithTimeout(ctx, hub.Timeout)
	info, err := pb.NewPodHostClient(conn).ProtocolInfo(probeContext, &pb.ProtocolInfoRequest{})
	probeCancel()
	if err != nil {
		return nil, maintenanceControlEnd(err)
	}
	wireMinor := min(info.WireMinor, pb.WireMinor)
	stream, err := pb.NewWorkerControlClient(conn).Control(ctx)
	if err != nil {
		return nil, maintenanceControlEnd(err)
	}
	claim := &pb.Claim{RecordOwnerEpoch: recordOwnerEpoch, RecordOwnerId: recordOwnerID, WorkerId: remote.WorkerID, WorkerBootId: remote.WorkerBootID, WireMinor: wireMinor, Proof: proof}
	if err = stream.Send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_Claim{Claim: claim}}); err != nil {
		return nil, maintenanceControlEnd(err)
	}
	var epoch uint64
	for {
		frame, err := stream.Recv()
		if err != nil {
			return nil, maintenanceControlEnd(err)
		}
		if proto.Size(frame) > pb.MaxInlineControlBytes {
			return nil, exit.New(exit.Structural, "maintenance control frame exceeds its bound")
		}
		switch message := frame.Msg.(type) {
		case *pb.WorkerFrame_ClaimAck:
			ack := message.ClaimAck
			if !ack.Accepted || ack.RecordOwnerEpoch != recordOwnerEpoch || ack.WorkerId != remote.WorkerID || ack.WorkerBootId != remote.WorkerBootID || ack.ControlStreamEpoch == 0 {
				return nil, exit.New(exit.Conflict, "maintenance control claim was refused or changed the pinned worker")
			}
			epoch = ack.ControlStreamEpoch
		case *pb.WorkerFrame_Snapshot:
			snap := message.Snapshot
			if epoch == 0 || snap.RecordOwnerEpoch != recordOwnerEpoch || snap.ControlStreamEpoch != epoch || snap.WorkerBootId != remote.WorkerBootID {
				return nil, exit.New(exit.Conflict, "maintenance control snapshot changed its claimed envelope")
			}
			if problem := validateMaintenanceSnapshot(snap, remote.RentalID, retained); problem != nil {
				return nil, problem
			}
			control.ControlStreamEpoch = epoch
			control.Host = checkpointAdapter(pb.NewPodHostClient(conn), claim)
			// A lost control stream cancels the maintenance too; nothing may outlive this
			// accepted control connection.
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
			return nil, exit.New(exit.Failed, "maintenance control worker reports a boot failure")
		default:
			return nil, exit.New(exit.Structural, "maintenance control expected ClaimAck and WorkerSnapshot")
		}
	}
}

func maintenanceControlEnd(err error) *exit.Error {
	if status.Code(err) == codes.Unavailable && status.Convert(err).Message() == "workspace authority transferred; reconnect for the recovered snapshot" {
		return exit.Named(exit.Unavailable, "rental.control_reconnect", "the worker transferred workspace authority; reconnecting for its recovered snapshot")
	}
	switch status.Code(err) {
	case codes.Unavailable, codes.DeadlineExceeded, codes.Canceled,
		codes.ResourceExhausted, codes.Aborted, codes.Internal, codes.Unknown:
		// A condition of the moment: the pod is down, busy, restarting, or the transport
		// broke. Anything else is the Host's word on this Claim.
		return exit.Unavailablef("maintenance control is unavailable before its snapshot: %s", err)
	}
	return exit.New(exit.Conflict, "maintenance control refused the fixed rental Claim")
}

func validateMaintenanceSnapshot(snap *pb.WorkerSnapshot, rentalID string, retained *records.Store) *exit.Error {
	if snap.SnapshotId == "" || !bytes.Equal(canonical.Digest(snap.SnapshotCanonicalBytes), snap.SnapshotDigest) {
		return exit.New(exit.Structural, "maintenance control snapshot digest mismatch")
	}
	doc, err := canonical.Read(snap.SnapshotCanonicalBytes, &pb.WorkerSnapshotBody{})
	if err != nil {
		return exit.New(exit.Structural, "maintenance control snapshot is not canonical")
	}
	set, _ := canonical.Spell(canonical.Digest(snap.AcceptedPlacementSetCanonicalBytes))
	if declared := doc.Str("accepted_placement_set_digest"); declared != "" && declared != set {
		return exit.New(exit.Structural, "maintenance control accepted placement bytes changed")
	}
	if len(snap.HostSnapshotCanonicalBytes) == 0 || !bytes.Equal(canonical.Digest(snap.HostSnapshotCanonicalBytes), snap.HostSnapshotDigest) {
		return exit.New(exit.Structural, "maintenance control Host snapshot digest mismatch")
	}
	host, err := canonical.Read(snap.HostSnapshotCanonicalBytes, &pb.HostSnapshotBody{})
	if err != nil {
		return exit.New(exit.Structural, "maintenance control Host snapshot is not canonical")
	}
	seen := map[string]map[int64]bool{}
	for _, held := range append(doc.List("held_attempts"), host.List("held_outcomes")...) {
		request, ordinal := held.Str("request_id"), held.Int("attempt_ordinal")
		if seen[request] == nil {
			seen[request] = map[int64]bool{}
		}
		if seen[request][ordinal] {
			return exit.New(exit.Conflict, "maintenance control snapshot repeats a held attempt")
		}
		seen[request][ordinal] = true
		if problem := verifySettledRetainedAttempt(retained, rentalID, held); problem != nil {
			return problem
		}
	}
	return nil
}

// Runtime maintenance supplies its local records; without them the lane requires an
// empty held-attempt census. Retention
// keeps completed outcomes in the workspace; accepting their exact settled
// identity neither releases bytes nor opens the snapshot dispatch barrier.
// Both snapshot censuses carry HeldAttempt outcome ID/digest; those must match
// the closed local terminal whose canonical body is independently verified here.
func verifySettledRetainedAttempt(store *records.Store, rentalID string, held canonical.Doc) *exit.Error {
	refuse := func() *exit.Error {
		return exit.New(exit.Conflict, "maintenance control cannot take over an unverified held attempt")
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
