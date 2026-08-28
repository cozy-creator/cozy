package orchestrator

import (
	"context"
	"testing"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

func TestReleasePinAcceptsDynamicAttachedSilence(t *testing.T) {
	attached := &worker{spec: WorkerLaunchSpec{
		Placement:  DesiredPlacement{ReleaseID: "cozy/probe@cold"},
		Connection: &WorkerConnection{RentalID: "rnt-test"},
	}}
	if err := releasePin(attached, ""); err != nil {
		t.Fatalf("dynamic attached ClaimAck silence was refused: %v", err)
	}
	if err := releasePin(attached, "cozy/probe@warm"); err == nil || err.Name != "release_mismatch" {
		t.Fatalf("stale attached release was not refused: %#v", err)
	}
	if err := releasePin(attached, "cozy/probe@cold"); err != nil {
		t.Fatalf("matching declared attached release was refused: %v", err)
	}
}

func TestPrivateBootFailureRelaysClosedWorkerFrameAlternative(t *testing.T) {
	var got RentalSessionEvidence
	c := &Orchestrator{opt: Options{RelayRentalSession: func(_ context.Context,
		connection *WorkerConnection, evidence RentalSessionEvidence) *exit.Error {
		if connection.RentalID != "rental-1" {
			t.Fatalf("rental = %s", connection.RentalID)
		}
		got = evidence
		return nil
	}}}
	w := newWorker("slot", WorkerLaunchSpec{Connection: &WorkerConnection{RentalID: "rental-1"}})
	s := &session{ctx: context.Background(), instanceID: "slot"}
	failure := &pb.BootFailure{RecordOwnerEpoch: recordOwnerEpoch,
		ControlStreamGeneration: 3, WorkerBootId: "boot-1", WorkerInstanceId: "instance-1",
		Reason:               pb.BootFailureReason_BOOT_FAILURE_REASON_DRIVER_FAULT,
		ControlRuntimeDigest: "sha256:1111111111111111111111111111111111111111111111111111111111111111"}
	frame := &pb.WorkerFrame{Msg: &pb.WorkerFrame_BootFailure{BootFailure: failure}}
	c.relayBootFailure(s, w, frame, failure)
	if len(got.BootFailure) == 0 || len(got.ClaimAck) != 0 || len(got.Snapshot) != 0 ||
		len(got.ObservedState) != 0 || got.DesiredRevision != 0 {
		t.Fatalf("boot failure relay = %#v", got)
	}
}
