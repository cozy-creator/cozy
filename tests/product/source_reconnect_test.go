package producttest

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// A restarted owner resumes durable checkpoint work before its rental has
// reattached. This exercises the real owner and store up to that connection gate;
// no checkpoint bytes, upload verdicts or replacement worker are fabricated.
func TestSourceCheckpointResumeWaitsForWorkerReattachment(t *testing.T) {
	o := hostOwner(t, "source-checkpoint-reconnect", func(options *orchestrator.Options) {
		options.Cfg.HubURL = "http://127.0.0.1:1"
		options.ModelTransfers = cli.NewModelTransferOwner(options.Cfg, options.Store, options.Log, nil)
	})
	requestID := records.NewID("job")
	const rental = "rental-awaiting-reattachment"
	const boot = "previous-worker-boot"
	header := []byte(`{"weight_map":{"x":"shard.safetensors"}}`)
	intent := &records.ModelTransferIntent{
		Kind: "model-upload", Destination: "proof/model",
		Source:          "hf://proof/source@" + strings.Repeat("4", 40),
		SourceSelection: "sha256:" + strings.Repeat("2", 64),
		SourceProfiles:  map[string]string{"shared": "hf/minimax-h3/shared-bf16/1"},
		SourceFiles: []records.ModelTransferSourceFile{{Member: "model.safetensors.index.json",
			SHA256: strings.Repeat("1", 64), Length: int64(len(header)), Header: header}},
		Outputs: []records.ModelTransferOutput{{Name: "model"}},
	}
	_, fresh, problem := o.store.Submit(records.Request{
		ID: requestID, IdemKey: requestID, BodyDigest: "sha256:" + strings.Repeat("c", 64),
		Package: "proof/producer", Entrypoint: "produce", Kind: "job",
		Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]", Worker: rental,
		Rental: true, ModelTransfer: intent,
	})
	fatal(t, problem)
	if !fresh {
		t.Fatal("source operation was not newly recorded")
	}
	fatal(t, o.store.BeginModelTransferMaterialization(requestID))
	checkpoint := records.ModelSourceCheckpoint{Slot: "shared", HeadID: "sha256:" + strings.Repeat("3", 64),
		HeadLength: 500, PlanDigest: "sha256:" + strings.Repeat("4", 64), Index: 7, Bytes: 4096}
	fatal(t, o.store.ObserveModelSourceCheckpoints(requestID, intent.SourceSelection, boot,
		[]records.ModelSourceCheckpoint{checkpoint}))
	before, problem := o.store.ModelTransferOf(requestID)
	fatal(t, problem)
	beforeProgress, problem := o.store.ModelSourceProgress(requestID)
	fatal(t, problem)

	fatal(t, o.c.ResumeModelTransfers())
	line, observed := waitEvent(o, "source checkpoint custody", 5*time.Second)
	if !observed {
		t.Fatal("resumed source custody did not report its missing worker")
	}
	after, problem := o.store.ModelTransferOf(requestID)
	fatal(t, problem)
	if after == nil || after.State != "materializing" {
		t.Fatalf("worker reconnect gap terminalized the source operation: %+v; %s", after, line)
	}
	if !strings.Contains(line, "source checkpoint custody pending (unavailable)") ||
		!strings.Contains(line, "rental "+rental+" has no attached worker") {
		t.Fatalf("missing worker was not exposed as retryable unavailability: %s", line)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatal("worker reconnect gap changed the durable source intent or materialization state")
	}
	afterProgress, problem := o.store.ModelSourceProgress(requestID)
	fatal(t, problem)
	if !reflect.DeepEqual(beforeProgress, afterProgress) || len(afterProgress) != 1 ||
		afterProgress[0].Observed != checkpoint || afterProgress[0].Acknowledged != nil {
		t.Fatal("worker reconnect gap lost or falsely acknowledged the observed source checkpoint")
	}
	request, problem := o.store.RequestRow(requestID)
	fatal(t, problem)
	if request == nil || request.State != "submitted" || request.Worker != rental || request.Ordinal != 0 {
		t.Fatal("worker reconnect gap changed the submitted request or created an attempt")
	}
}
