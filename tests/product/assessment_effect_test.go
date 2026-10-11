package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/publication"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestAssessmentFileReservationsKeepOriginalCompositionAndIndependentCustody(t *testing.T) {
	store, producer, _, workloads, _ := observedAssessment(t, true)
	parent, problem := store.RequestRow(producer)
	fatal(t, problem)
	report := []byte(`{"association_fixture":true}`)
	var outputs []records.ByteOutput
	for _, file := range []struct {
		name string
		body []byte
	}{{"report", report}, {"workloads", workloads}} {
		outputs = append(outputs, records.ByteOutput{RequestID: producer, Attempt: 1, OutputID: file.name, Digest: assessmentDigest(file.body), Length: int64(len(file.body)), MimeType: "application/json", ProducerRootID: assessmentDigest([]byte(file.name + "-root")), ReceiptDigest: assessmentDigest([]byte(file.name + "-receipt")), ManifestID: assessmentDigest([]byte(file.name + "-manifest")), ManifestLength: 100, ContentBytes: int64(len(file.body))})
	}
	_, problem = store.AcceptTerminal(records.Terminal{RequestID: producer, Attempt: 1, SessionID: "private-boot", InvocationDigest: childDigest("1"), TerminalID: "report-done", TerminalDigest: childDigest("b"), Status: "SUCCEEDED", RequestState: "succeeded", Body: []byte(`{}`), ByteOutputs: outputs})
	fatal(t, problem)
	fatal(t, store.Closed(producer, 1))
	for _, output := range outputs {
		hold, problem := store.ReserveByteResult(producer, output)
		fatal(t, problem)
		fatal(t, store.ConfirmNativeArtifact(hold.RetentionID, "private-worker", "private-boot"))
	}
	request := publication.AssessmentRequest{Checkpoint: publication.CheckpointRef{Destination: "alice/model", Checkpoint: childDigest("a"), Manifest: records.ArtifactObjectRef{Digest: childDigest("a"), Length: 100}, Publication: "upload"}, Report: assessmentDigest(report), Workloads: assessmentDigest(workloads)}
	call, _, problem := store.AcceptNativeCall(records.NativeCall{ID: "attach-fixture", ParentRequestID: parent.ParentRequestID, CallIndex: 1, Kind: "effect", Operation: "attach_assessment", IntentDigest: childDigest("c"), Request: assessmentJSON(t, request)}, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	holds, owner, problem := store.ReserveAssessmentBytes(call.ID, call.ParentRequestID, request.Report, request.Workloads)
	fatal(t, problem)
	if owner != producer || len(holds) != 2 {
		t.Fatal("report changed original render-composition owner")
	}
	for _, hold := range holds {
		if hold.Kind != "effect" || hold.State != "pending" {
			t.Fatal("effect did not reserve independent custody")
		}
		fatal(t, store.ConfirmNativeArtifact(hold.RetentionID, "private-worker", "private-boot"))
	}
	if _, _, problem := store.ReserveAssessmentBytes(call.ID, call.ParentRequestID, request.Report, childDigest("d")); problem == nil {
		t.Fatal("accepted effect changed workload identity")
	}
	fatal(t, store.BeginNativeArtifactRelease(producer, false))
	replay, owner, problem := store.ReserveAssessmentBytes(call.ID, call.ParentRequestID, request.Report, request.Workloads)
	fatal(t, problem)
	if owner != producer || len(replay) != 2 {
		t.Fatal("original producer cleanup erased effect-owned artifacts")
	}
	fatal(t, store.BeginNativeArtifactRelease(call.ID, false))
	for _, hold := range holds {
		if store.ConfirmNativeArtifact(hold.RetentionID, "private-worker", "private-boot") == nil {
			t.Fatal("late native acknowledgement resurrected canceled input")
		}
	}
	if _, _, problem := store.ReserveAssessmentBytes(call.ID, call.ParentRequestID, request.Report, request.Workloads); problem == nil {
		t.Fatal("released effect inputs were reacquired")
	}
}
