package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

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
