package producttest

import (
	"bytes"
	"os"
	"testing"

	"github.com/cozy-creator/cozy/internal/archive"
	"google.golang.org/protobuf/encoding/protowire"
)

// These bytes were emitted by the original generated binding producer before retirement.
func TestArchivedMachineDataReadsWithoutWorkerBindings(t *testing.T) {
	receiptBytes, err := os.ReadFile("testdata/record-archive/receipt.bin")
	must(t, err)
	receipt, err := archive.ReadReceipt(receiptBytes)
	must(t, err)
	if receipt.RequestID != "archive-run" || receipt.WorkerID != "archive-machine" || receipt.Number != 42 || receipt.AcceptedMS != 1000 {
		t.Fatalf("retained receipt changed: %+v", receipt)
	}
	added := append([]byte(nil), receiptBytes...)
	for field := protowire.Number(100); field < 1100; field++ {
		added = protowire.AppendTag(added, field, protowire.BytesType)
		added = protowire.AppendBytes(added, []byte("later metadata"))
	}
	added = protowire.AppendTag(added, 1200, protowire.VarintType)
	added = protowire.AppendVarint(added, 123)
	withAdditions, err := archive.ReadReceipt(added)
	must(t, err)
	if withAdditions != receipt {
		t.Fatal("additive archived fields changed the receipt")
	}
	outcomeBytes, err := os.ReadFile("testdata/record-archive/outcome.bin")
	must(t, err)
	outcome, err := archive.ReadOutcome(outcomeBytes)
	must(t, err)
	if outcome.Attempt != 1 || outcome.Status != 1 || outcome.RuntimeMS != 19 || !bytes.Equal(outcome.Result, []byte(`{"value":42}`)) {
		t.Fatalf("retained outcome changed: %+v", outcome)
	}
	if _, err := archive.ReadOutcome(outcomeBytes[:len(outcomeBytes)-1]); err == nil {
		t.Fatal("truncated archived outcome was read as a valid result")
	}
}
