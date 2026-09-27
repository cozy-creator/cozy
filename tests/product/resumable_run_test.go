package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// TestResumableRunWalksToTheRunToResume: a re-run of the same work retries the stopped run
// that retains its work, follows a canceled one with a fresh run, and replays a live run.
func TestResumableRunWalksToTheRunToResume(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const base = "model-upload-proof"
	submit := func(id, key, retryOf string) {
		t.Helper()
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: key, Kind: "job",
			Package: "local/example", Entrypoint: "main", Payload: []byte(`{}`),
			BodyDigest: "sha256:" + strings.Repeat("1", 64), RetainWork: true, RetryOf: retryOf})
		fatal(t, problem)
	}
	walk := func(wantKey, wantRetry string) {
		t.Helper()
		key, retryOf, problem := store.ResumableRun(base)
		fatal(t, problem)
		if key != wantKey || retryOf != wantRetry {
			t.Fatalf("ResumableRun = %q retrying %q, want %q retrying %q", key, retryOf, wantKey, wantRetry)
		}
	}
	walk(base, "")
	submit("job-first", base, "")
	walk(base, "")
	if changed, problem := store.BlockRetainedWork("job-first", "source_failed", "pod lost the network"); problem != nil || !changed {
		t.Fatalf("the first run did not stop with retained work: %v", problem)
	}
	walk(base+"/retry-of/job-first", "job-first")
	submit("job-retry", base+"/retry-of/job-first", "job-first")
	walk(base+"/retry-of/job-first", "job-first")
	fatal(t, store.SettleRequest("job-retry", "canceled"))
	walk(base+"/after/job-retry", "")
}
