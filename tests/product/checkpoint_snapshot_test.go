package producttest

import (
	"bytes"
	"github.com/cozy-creator/cozy/internal/canonical"
	"strings"
	"testing"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func TestHostCheckpointSnapshotUsesCanonicalDigestDecoder(t *testing.T) {
	original := &pb.HostSnapshotBody{WeightsTransactions: []*pb.WeightsTransactionStatus{{WeightsTransactionId: "sha256:" + strings.Repeat("a", 64), RequestId: "request", AttemptOrdinal: 2, InvocationSpecDigest: "sha256:" + strings.Repeat("b", 64), OutputSlot: "model", WriterEpoch: 3, State: pb.WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_INTENT,
		TensorfsDeclarationDigest: bytes.Repeat([]byte{0xc}, 32), Checkpoint: &pb.CheckpointRef{Head: &pb.Ref{Digest: bytes.Repeat([]byte{0xd}, 32), Length: 123}, PlanDigest: bytes.Repeat([]byte{0xe}, 32), Index: 1, Bytes: 4096}}}}
	raw, _, err := canonical.Identity(original)
	if err != nil {
		t.Fatal(err)
	}
	restored := new(pb.HostSnapshotBody)
	if err := canonical.Unmarshal(raw, restored); err != nil {
		t.Fatal(err)
	}
	if !proto.Equal(original, restored) {
		t.Fatal("canonical digest fields changed during typed snapshot read")
	}
	bad := bytes.Replace(raw, []byte(`"checkpoint":{`), []byte(`"checkpoint":{"foreign":1,`), 1)
	if err := canonical.Unmarshal(bad, new(pb.HostSnapshotBody)); err == nil {
		t.Fatal("unknown nested field accepted")
	}
}
