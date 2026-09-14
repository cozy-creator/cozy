package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestMachineInputReceiptsFenceLateRepliesWithoutExecutionRows(t *testing.T) {
	store, request, _ := machineObserverFixture(t)
	asset := records.AssetBinding{FieldPath: "files", Snapshot: &records.ByteInputSnapshot{Manifest: records.ArtifactObjectRef{Digest: childDigest("4"), Length: 128}, ContentBytes: 1024}}
	input, problem := store.BeginMachineInput(request.ID, asset)
	fatal(t, problem)
	if owed, problem := store.MachineExecutionOwesWork(request.ID); problem != nil || !owed {
		t.Fatal("ambiguous native intake lost its machine obligation")
	}
	result := &pb.NativeByteRetentionResult{Source: &pb.NativeByteTreeRef{ProducerRootId: childDigest("5"), ReceiptDigest: make([]byte, 32), Manifest: &pb.Ref{Digest: make([]byte, 32), Length: 128}, ContentBytes: 1024}, RetentionId: childDigest("6")}
	fatal(t, store.RecordMachineInput(request.ID, input, result))
	result.Released = true
	fatal(t, store.RecordMachineInput(request.ID, input, result))
	result.Released = false
	if store.RecordMachineInput(request.ID, input, result) == nil {
		t.Fatal("late intake reply reopened released custody")
	}
	rows, problem := store.MachineInputs(request.ID)
	fatal(t, problem)
	if len(rows) != 1 || rows[0].State != "released" {
		t.Fatal("input receipt did not remain released")
	}
	attempts, problem := store.Attempts(request.ID)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatal("input custody invented Creator execution rows")
	}
	changed := asset
	different := *asset.Snapshot
	different.ContentBytes++
	changed.Snapshot = &different
	if _, problem := store.BeginMachineInput(request.ID, changed); problem == nil {
		t.Fatal("input replay changed its manifest capacity")
	}
}
