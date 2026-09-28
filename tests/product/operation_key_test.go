package producttest

import (
	"encoding/json"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestOperationKeyExcludesHistoryAndPreservesObservableInputs(t *testing.T) {
	artifact := records.ModelArtifact{ProducerRequestID: "producer-a", OutputSlot: "weights", Manifest: records.ArtifactObjectRef{Digest: childDigest("1"), Length: 161}, TensorFSReceiptDigest: childDigest("2")}
	payload := func(handle records.ModelArtifact, ordinary string) []byte {
		raw, err := json.Marshal(map[string]any{"source": handle, "producer_request_id": ordinary, "factor": 2})
		must(t, err)
		return raw
	}
	request := records.Request{ID: "first", ParentRequestID: "parent1", ReuseScope: "scope1", RetryOf: "old", ChildTargetDigest: childDigest("3"), Payload: payload(artifact, "observable-a"), Models: []records.ModelRef{{Slot: "source", Manifest: artifact.Manifest.Digest, ManifestLength: 161}}}
	key, problem := records.OperationKey(request)
	fatal(t, problem)
	request.ID, request.ParentRequestID, request.ReuseScope, request.RetryOf = "fresh", "unrelated-parent", "new-scope", ""
	artifact.ProducerRequestID, artifact.TensorFSReceiptDigest = "producer-b", childDigest("4")
	request.Payload = payload(artifact, "observable-a")
	reused, problem := records.OperationKey(request)
	fatal(t, problem)
	if key != reused {
		t.Fatal("run history or injected-model receipt provenance changed computation identity")
	}
	request.Payload = payload(artifact, "observable-b")
	changed, problem := records.OperationKey(request)
	fatal(t, problem)
	if changed == key {
		t.Fatal("lookalike ordinary argument was stripped from computation identity")
	}
	request.Payload = payload(artifact, "observable-a")
	request.ChildTargetDigest = childDigest("5")
	changed, problem = records.OperationKey(request)
	fatal(t, problem)
	if changed == key {
		t.Fatal("changed implementation/numerical environment reused computation")
	}
	request.Models[0].Manifest = childDigest("6")
	if _, problem := records.OperationKey(request); problem == nil {
		t.Fatal("forged model override passed its resolved input binding")
	}
}

func TestOperationCaptureOptionsParticipateInMemoIdentity(t *testing.T) {
	request := records.Request{ChildTargetDigest: childDigest("a"), Payload: []byte(`{"value":7}`)}
	before, problem := records.OperationKey(request)
	fatal(t, problem)
	request.Capture = `{"components":["dit"],"steps":[]}`
	captured, problem := records.OperationKey(request)
	fatal(t, problem)
	request.Capture = `{"components":["dit"],"steps":[0,2]}`
	selected, problem := records.OperationKey(request)
	fatal(t, problem)
	if before == captured || captured == selected || before == selected {
		t.Fatal("capture request reused incompatible observation")
	}
}
