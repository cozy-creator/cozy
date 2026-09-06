package orchestrator

import (
	"bytes"
	"context"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
)

// SourceCustodyControl keeps the pod's existing owner-presence hold while its
// source bytes are banked. It sends only Claim: no snapshot admission, desired
// state or attempt offer can originate from this connection.
type SourceCustodyControl struct {
	Context    context.Context
	Host       CheckpointHost
	connection *grpc.ClientConn
	cancel     context.CancelFunc
}

func (s *SourceCustodyControl) Close() error { s.cancel(); return s.connection.Close() }

func DialCheckpointHost(parent context.Context, remote *WorkerConnection, sign RentalClaimProofSource) (*SourceCustodyControl, *exit.Error) {
	if remote == nil || remote.CACert == "" || remote.WorkerBootID == "" || sign == nil {
		return nil, exit.New(exit.Credential, "source custody requires a pinned rental and Claim signer")
	}
	proof, problem := sign(remote, recordOwnerEpoch)
	if problem != nil {
		return nil, problem
	}
	conn, err := dialWorker(remote.Addr, remote)
	if err != nil {
		return nil, exit.Unavailablef("cannot open pinned source custody lane")
	}
	ctx, cancel := context.WithCancel(parent)
	control := &SourceCustodyControl{Context: ctx, connection: conn, cancel: cancel}
	ready := false
	defer func() {
		if !ready {
			_ = control.Close()
		}
	}()
	stream, err := pb.NewWorkerControlClient(conn).Control(ctx)
	if err != nil {
		return nil, exit.Unavailablef("source custody control stream is unavailable")
	}
	claim := &pb.Claim{RecordOwnerEpoch: recordOwnerEpoch, RecordOwnerId: recordOwnerID, WorkerId: remote.WorkerID, WorkerBootId: remote.WorkerBootID, WireMinor: pb.WireMinor, Proof: proof}
	if err = stream.Send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_Claim{Claim: claim}}); err != nil {
		return nil, exit.Unavailablef("source custody claim could not be sent")
	}
	var epoch uint64
	for {
		frame, err := stream.Recv()
		if err != nil {
			return nil, exit.Unavailablef("source custody control ended before its snapshot")
		}
		if proto.Size(frame) > pb.MaxInlineControlBytes {
			return nil, exit.New(exit.Structural, "source custody control frame exceeds its bound")
		}
		switch message := frame.Msg.(type) {
		case *pb.WorkerFrame_ClaimAck:
			ack := message.ClaimAck
			if !ack.Accepted || ack.RecordOwnerEpoch != recordOwnerEpoch || ack.WorkerId != remote.WorkerID || ack.WorkerBootId != remote.WorkerBootID || ack.ControlStreamEpoch == 0 || ack.WireMinor < pb.WireMinor {
				return nil, exit.New(exit.Conflict, "source custody claim was refused or changed the pinned worker")
			}
			epoch = ack.ControlStreamEpoch
		case *pb.WorkerFrame_Snapshot:
			snap := message.Snapshot
			if epoch == 0 || snap.RecordOwnerEpoch != recordOwnerEpoch || snap.ControlStreamEpoch != epoch || snap.WorkerBootId != remote.WorkerBootID {
				return nil, exit.New(exit.Conflict, "source custody snapshot changed its claimed envelope")
			}
			if problem := validateCustodySnapshot(snap); problem != nil {
				return nil, problem
			}
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
			return nil, exit.New(exit.Failed, "source custody worker reports a boot failure")
		default:
			return nil, exit.New(exit.Structural, "source custody expected ClaimAck and WorkerSnapshot")
		}
	}
}

func validateCustodySnapshot(snap *pb.WorkerSnapshot) *exit.Error {
	if snap.SnapshotId == "" || !bytes.Equal(canonical.Digest(snap.SnapshotCanonicalBytes), snap.SnapshotDigest) {
		return exit.New(exit.Structural, "source custody snapshot digest mismatch")
	}
	doc, err := canonical.Read(snap.SnapshotCanonicalBytes, &pb.WorkerSnapshotBody{})
	if err != nil {
		return exit.New(exit.Structural, "source custody snapshot is not canonical")
	}
	set, _ := canonical.Spell(canonical.Digest(snap.AcceptedPlacementSetCanonicalBytes))
	if declared := doc.Str("accepted_placement_set_digest"); declared != "" && declared != set {
		return exit.New(exit.Structural, "source custody accepted placement bytes changed")
	}
	if len(doc.List("held_attempts")) != 0 {
		return exit.New(exit.Conflict, "source custody cannot take over a worker holding attempts")
	}
	if len(snap.HostSnapshotCanonicalBytes) == 0 || !bytes.Equal(canonical.Digest(snap.HostSnapshotCanonicalBytes), snap.HostSnapshotDigest) {
		return exit.New(exit.Structural, "source custody Host snapshot digest mismatch")
	}
	host, err := canonical.Read(snap.HostSnapshotCanonicalBytes, &pb.HostSnapshotBody{})
	if err != nil {
		return exit.New(exit.Structural, "source custody Host snapshot is not canonical")
	}
	if len(host.List("held_outcomes")) != 0 {
		return exit.New(exit.Conflict, "source custody cannot ignore held producer outcomes")
	}
	return nil
}
