package orchestrator

import (
	"io"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestHostPrepareTransportEndDefersTheDesiredRevision(t *testing.T) {
	c := &Orchestrator{opt: Options{Log: io.Discard}}
	w := &worker{hostPrepareSeq: 4, revision: 9}
	if c.settleHostPrepare(nil, w, 4, "package proof/model", hostPrepareResult{
		err: status.Error(codes.Unavailable, "Hub is restarting"),
	}) {
		t.Fatal("a transport end without a verdict was accepted as prepared")
	}
	if w.desiredRefusal == nil || w.desiredRefusal.Code != exit.Unavailable {
		t.Fatalf("transport end did not release the waiter as retryable: %#v", w.desiredRefusal)
	}
	if got := w.desiredRefusal.Message; got == "" {
		t.Fatal("retryable preparation result has no diagnostic")
	}
}

func TestHostPrepareDocumentFaultRemainsFinal(t *testing.T) {
	c := &Orchestrator{opt: Options{Log: io.Discard}}
	w := &worker{hostPrepareSeq: 2, revision: 7}
	want := exit.Named(exit.Structural, "worker.prepare_document_invalid", "invalid placement")
	if c.settleHostPrepare(nil, w, 2, "package proof/model", hostPrepareResult{fault: want}) {
		t.Fatal("an invalid prepared document was accepted")
	}
	if w.desiredRefusal != want {
		t.Fatalf("document fault did not remain the final verdict: %#v", w.desiredRefusal)
	}
}
