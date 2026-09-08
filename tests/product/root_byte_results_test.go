package producttest

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRetainedFileAndTreeFieldsPreserveOrdinaryMediaPublication(t *testing.T) {
	entry := &launch.Entrypoint{Result: launch.Struct{Fields: []launch.Field{
		{Name: "report", Type: json.RawMessage(`{"asset":"file"}`)},
		{Name: "files", Type: json.RawMessage(`{"input":"tree"}`)},
		{Name: "image", Type: json.RawMessage(`{"asset":"image"}`)},
	}}}
	fields := launch.RetainedAssetPaths(entry)
	if len(fields) != 2 || !fields["report"] || !fields["files"] || fields["image"] {
		t.Fatalf("native result selection changed media semantics: %v", fields)
	}
}

func TestRootByteResultRetainsFinalCustodyWithoutInventingPublication(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	request, _, problem := store.Submit(records.Request{ID: "req-root-native", IdemKey: "root-native", BodyDigest: childDigest("3"), Package: "local/native", Entrypoint: "main", Kind: "job", Org: "local", PlanID: childDigest("4"), LocalPackageDigest: childDigest("5"), Payload: []byte(`{}`), RetainWork: true, ChildArtifacts: true})
	fatal(t, problem)
	request = offerChildParent(t, store, request)
	output := records.ByteOutput{RequestID: request.ID, Attempt: 1, OutputID: "value", Digest: childDigest("6"), Length: 100, MimeType: "application/vnd.cozy.tree-manifest", ProducerRootID: childDigest("7"), ReceiptDigest: childDigest("8"), ManifestID: childDigest("6"), ManifestLength: 100, ContentBytes: 10000}
	if _, problem := store.ReserveByteResult(request.ID, output); problem == nil {
		t.Fatal("uncompleted parent invented final custody")
	}
	_, problem = store.AcceptTerminal(records.Terminal{RequestID: request.ID, Attempt: 1, SessionID: "private-boot", InvocationDigest: childDigest("1"), TerminalID: "native-done", TerminalDigest: childDigest("a"), Status: "SUCCEEDED", RequestState: "succeeded", Body: []byte(`{}`), ByteOutputs: []records.ByteOutput{output}})
	fatal(t, problem)
	hold, problem := store.ReserveByteResult(request.ID, output)
	fatal(t, problem)
	if hold.ParentRequestID != request.ID || hold.ConsumerID != request.ID || hold.State != "pending" {
		t.Fatal("root result has no independent native recipient")
	}
	fatal(t, store.ConfirmNativeArtifact(hold.RetentionID, "private-worker", "private-boot"))
	fatal(t, store.Closed(request.ID, 1))
	fatal(t, store.ReleaseCompletedNativeBytes(request.ID))
	kept, problem := store.NativeArtifactRetentions(request.ID)
	fatal(t, problem)
	if len(kept) != 1 || kept[0].State != "held" {
		t.Fatal("early-result cleanup released explicit final result")
	}
	pub, problem := store.PublicationOf(request.ID)
	fatal(t, problem)
	if pub != nil {
		t.Fatal("native Tree invented a file publication")
	}
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	retaining, problem := store.RequestRetaining(*row)
	fatal(t, problem)
	if !retaining {
		t.Fatal("completed native result lost its retained workspace")
	}
	fatal(t, store.RequestRetainedCancellation(request.ID, "owner"))
	if _, problem := store.ReserveByteResult(request.ID, output); problem == nil {
		t.Fatal("canceling parent reacquired final bytes")
	}
	fatal(t, store.BeginNativeArtifactRelease(request.ID, false))
	if store.ConfirmNativeArtifact(hold.RetentionID, "private-worker", "private-boot") == nil {
		t.Fatal("late final recipient ACK reversed cancellation")
	}
}
