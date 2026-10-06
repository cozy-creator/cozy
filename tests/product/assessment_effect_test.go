package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/publication"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestAssessmentPublicationReconcilesExactBytesAndFencesCanceledWrites(t *testing.T) {
	report, err := os.ReadFile("testdata/assessment-v5.json")
	must(t, err)
	for _, mode := range []string{"normal", "lost-reply", "canceled", "wrong-readback", "changed-account", "wrong-ack-verdict"} {
		t.Run(mode, func(t *testing.T) {
			digest := assessmentDigest(report)
			checkpoint := childDigest("a")
			writes := 0
			var stored []byte
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodGet {
					if stored == nil {
						http.NotFound(w, r)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write(stored)
					return
				}
				writes++
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				if !bytes.Equal(body, report) {
					t.Error("changed canonical report bytes")
				}
				stored = append([]byte{}, body...)
				if mode == "lost-reply" {
					connection, _, err := w.(http.Hijacker).Hijack()
					if err != nil {
						t.Error(err)
					} else {
						_ = connection.Close()
					}
					return
				}
				w.Header().Set("Content-Type", "application/json")
				verdict := "pass"
				if mode == "wrong-ack-verdict" {
					verdict = "fail"
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"assessment": map[string]any{"checkpoint_id": checkpoint, "scope": "publisher_assessment", "report": map[string]any{"digest": digest, "length": len(report)}, "actor": "alice", "created_at": "fixture", "verdict": verdict}, "publisher_reported_verdict": "pass"})
			}))
			defer server.Close()
			client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("fixture")}, "")
			request := publication.AssessmentRequest{Checkpoint: publication.CheckpointRef{Destination: "alice/model", Checkpoint: checkpoint, Manifest: records.ArtifactObjectRef{Digest: checkpoint, Length: 100}, Publication: "upload", Observation: "acknowledged"}, Report: digest, Workloads: childDigest("b")}
			intent := publication.AssessmentIntent{Request: request, Account: "alice", Report: records.ArtifactObjectRef{Digest: digest, Length: int64(len(report))}, Workloads: records.ArtifactObjectRef{Digest: request.Workloads, Length: 10}, Verdict: "pass", RenderOwner: "producer"}
			calls := 0
			canceled := mode == "canceled"
			before := func() *exit.Error {
				calls++
				if canceled {
					return exit.Named(exit.Canceled, "publication.call_canceled", "canceled")
				}
				return nil
			}
			account := "alice"
			if mode == "changed-account" {
				account = "other"
			}
			if mode == "wrong-readback" {
				stored = []byte(`{"changed":true}`)
			}
			result, problem := publication.ApplyAssessment(context.Background(), client, intent, account, report, before)
			switch mode {
			case "normal":
				fatal(t, problem)
				if result.Observation != "acknowledged" || writes != 1 || calls != 1 {
					t.Fatal("new assessment did not use one fenced PUT")
				}
			case "lost-reply":
				if problem == nil || writes != 1 {
					t.Fatal("lost write reply was not uncertain")
				}
				canceled = true
				result, problem = publication.ApplyAssessment(context.Background(), client, intent, account, nil, before)
				fatal(t, problem)
				if result.Observation != "observed_convergence" || writes != 1 || calls != 1 {
					t.Fatal("canceled retry issued another write instead of exact readback")
				}
			case "wrong-ack-verdict":
				if problem == nil || writes != 1 {
					t.Fatal("a conflicting stored verdict was accepted as an acknowledgement")
				}
			default:
				if problem == nil || writes != 0 {
					t.Fatal("canceled, foreign-account or changed assessment was written")
				}
			}
		})
	}
}

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
