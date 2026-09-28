package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// TestResumableRunWalksToTheRunToResume: a re-run of the same work retries the stopped run
// that retains its work on the same machine, starts fresh elsewhere or after a cancel, and
// replays a live run from any machine and, when asked to, a completed one.
func TestResumableRunWalksToTheRunToResume(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	submit := func(id, key, retryOf string) {
		t.Helper()
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: key, Kind: "job",
			Package: "local/example", Entrypoint: "main", Payload: []byte(`{}`),
			BodyDigest: "sha256:" + strings.Repeat("1", 64), RetainWork: true, RetryOf: retryOf})
		fatal(t, problem)
	}
	walk := func(base, machine, wantKey, wantRetry string) {
		t.Helper()
		key, retryOf, problem := store.ResumableRun(base, machine, true)
		fatal(t, problem)
		if key != wantKey || retryOf != wantRetry {
			t.Fatalf("ResumableRun(%s) = %q retrying %q, want %q retrying %q", machine, key, retryOf, wantKey, wantRetry)
		}
	}
	const base = "model-upload-proof"
	walk(base, "", base, "")
	submit("job-first", base, "")
	walk(base, "", base, "")
	if changed, problem := store.BlockRetainedWork("job-first", "source_failed", "pod lost the network"); problem != nil || !changed {
		t.Fatalf("the first run did not stop with retained work: %v", problem)
	}
	walk(base, "", base+"/retry-of/job-first", "job-first")
	walk(base, "pr-elsewhere", base+"/after/job-first", "")
	submit("job-retry", base+"/retry-of/job-first", "job-first")
	walk(base, "", base+"/retry-of/job-first", "job-first")
	fatal(t, store.SettleRequest("job-retry", "canceled"))
	walk(base, "", base+"/after/job-retry", "")

	const done = "model-upload-done"
	submit("job-done", done, "")
	fatal(t, store.SettleRequest("job-done", "succeeded"))
	walk(done, "pr-elsewhere", done, "")
	// Work whose executing Runtime decides the result runs again rather than replaying.
	key, retryOf, problem := store.ResumableRun(done, "pr-elsewhere", false)
	fatal(t, problem)
	if key != done+"/after/job-done" || retryOf != "" {
		t.Fatalf("a completed ingest replayed: %q retrying %q", key, retryOf)
	}
}
