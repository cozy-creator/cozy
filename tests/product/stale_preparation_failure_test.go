package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// Aborted dispatch history is not an open execution. A preparation captured after
// that abort can still fail the request, while the preceding selection cannot.
func TestQueuedPreparationFailureAfterAbortedDispatch(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const id = "job-preparation-after-abort"
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "worker-preparation-after-abort",
		Package: "proof/producer", WorkerID: "worker", Devices: []string{"cpu"}}))
	original, _, problem := store.Submit(records.Request{ID: id, IdemKey: id,
		Package: "proof/producer", Entrypoint: "quantize", Kind: "job", Payload: []byte("{}"),
		BodyDigest: "sha256:" + strings.Repeat("a", 64), ModelTransfer: outputPublicationSubmission().ModelTransfer})
	fatal(t, problem)
	ordinal, problem := store.Dispatch(records.Attempt{RequestID: id, SessionID: podBootID,
		InstanceID: "worker-preparation-after-abort", InvocationDigest: "sha256:" + strings.Repeat("b", 64),
		InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, store.AbortDispatch(id, ordinal, podBootID, "no offer crossed"))
	current, problem := store.RequestRow(id)
	fatal(t, problem)
	payload := map[string]any{"error_type": "placement_config_refused", "error": "current preparation refused"}
	applied, problem := store.FailQueuedPreparation(original, payload)
	fatal(t, problem)
	if applied {
		t.Fatal("pre-abort preparation settled the newer ordinal")
	}
	applied, problem = store.FailQueuedPreparation(*current, payload)
	fatal(t, problem)
	if !applied {
		t.Fatal("current preparation could not settle an aborted dispatch")
	}
	row, problem := store.RequestRow(id)
	fatal(t, problem)
	transfer, problem := store.ModelTransferOf(id)
	fatal(t, problem)
	events, problem := store.EventsAfter(id, 0, 100)
	fatal(t, problem)
	if row.State != "failed" || row.Ordinal != ordinal || transfer.State != "failed" || transfer.ErrorCode != "placement_config_refused" || len(events) != 1 || events[0].Type != "request.failed" {
		t.Fatalf("current failure not atomic: request=%+v transfer=%+v events=%+v", row, transfer, events)
	}
}

func TestQueuedFailureKeepsOrdinaryFinalization(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const id = "job-finalizing-zero-open-attempts"
	_, _, problem = store.Submit(records.Request{ID: id, IdemKey: id, Package: "proof/producer", Entrypoint: "quantize",
		Payload: []byte("{}"), BodyDigest: "sha256:" + strings.Repeat("a", 64), Kind: "job"})
	fatal(t, problem)
	fatal(t, store.SettleRequest(id, "finalizing"))
	applied, problem := store.FailQueuedRequest(id, map[string]any{"error": "late preparation"})
	fatal(t, problem)
	if applied {
		t.Fatal("zero open attempts let queued failure overwrite finalizing request")
	}
}

// A failed event encoding must roll back both lifecycle rows, so status and watch
// cannot disagree after the database transaction returns an error.
func TestQueuedFailureRollsBackRequestAndTransfer(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const id = "job-preparation-rollback"
	_, _, problem = store.Submit(records.Request{ID: id, IdemKey: id, Package: "proof/producer", Entrypoint: "quantize",
		Payload: []byte("{}"), BodyDigest: "sha256:" + strings.Repeat("a", 64), Kind: "job",
		ModelTransfer: outputPublicationSubmission().ModelTransfer})
	fatal(t, problem)
	payload := map[string]any{"error_type": "preparation_refused", "error": "typed detail", "unencodable": func() {}}
	applied, problem := store.FailQueuedRequest(id, payload)
	if problem == nil || applied {
		t.Fatal("unencodable terminal unexpectedly committed")
	}
	row, problem := store.RequestRow(id)
	fatal(t, problem)
	transfer, problem := store.ModelTransferOf(id)
	fatal(t, problem)
	events, problem := store.EventsAfter(id, 0, 100)
	fatal(t, problem)
	if row.State != "submitted" || transfer.State != "pending" || transfer.ErrorCode != "" || len(events) != 0 {
		t.Fatalf("partial failure committed: request=%s transfer=%+v events=%v", row.State, transfer, events)
	}
}
