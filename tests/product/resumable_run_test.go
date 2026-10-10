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
		key, retryOf, problem := store.ResumableRun(base, "https://hub-a.example", machine, true)
		fatal(t, problem)
		if key != wantKey || retryOf != wantRetry {
			t.Fatalf("ResumableRun(%s) = %q retrying %q, want %q retrying %q", machine, key, retryOf, wantKey, wantRetry)
		}
	}
	const base = "model-upload-proof"
	scoped, _, problem := store.ResumableRun(base, "https://hub-a.example", "", true)
	fatal(t, problem)
	walk(base, "", scoped, "")
	submit("job-first", scoped, "")
	walk(base, "", scoped, "")
	if changed, problem := store.FailQueuedRequest("job-first", retainedFailure("source_failed", "pod lost the network")); problem != nil || !changed {
		t.Fatalf("the first run did not stop with retained work: %v", problem)
	}
	walk(base, "", scoped+"/retry-of/job-first", "job-first")
	walk(base, "pr-elsewhere", scoped+"/after/job-first", "")
	submit("job-retry", scoped+"/retry-of/job-first", "job-first")
	walk(base, "", scoped+"/retry-of/job-first", "job-first")
	fatal(t, store.SettleRequest("job-retry", "canceled"))
	walk(base, "", scoped+"/after/job-retry", "")

	const done = "model-upload-done"
	doneScoped, _, problem := store.ResumableRun(done, "https://hub-a.example", "", true)
	fatal(t, problem)
	submit("job-done", doneScoped, "")
	fatal(t, store.SettleRequest("job-done", "succeeded"))
	walk(done, "pr-elsewhere", doneScoped, "")
	// Work whose executing Runtime decides the result runs again rather than replaying.
	key, retryOf, problem := store.ResumableRun(done, "https://hub-a.example", "pr-elsewhere", false)
	fatal(t, problem)
	if key != doneScoped+"/after/job-done" || retryOf != "" {
		t.Fatalf("a completed ingest replayed: %q retrying %q", key, retryOf)
	}
}

func TestAutomaticResumeStaysOnItsSelectedHub(t *testing.T) {
	for _, prefix := range []string{"conversion-", "model-upload-"} {
		t.Run(prefix, func(t *testing.T) {
			store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			base := prefix + strings.Repeat("a", 64)
			keyA, retry, problem := store.ResumableRun(base, "https://hub-a.example", "pr-shared", true)
			fatal(t, problem)
			if retry != "" {
				t.Fatal("new command already names a retry")
			}
			fatal(t, store.RecordRental(records.Rental{ID: "pr-shared", MachineName: "shared", SKU: "cpu", State: "ready", Hub: "https://hub-a.example", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1}))
			_, _, problem = store.Submit(records.Request{ID: "job-a", IdemKey: keyA, Hub: "https://hub-a.example", Kind: "job",
				Package: "proof/example", Entrypoint: "main", Payload: []byte(`{}`), Worker: "pr-shared",
				BodyDigest: "sha256:" + strings.Repeat("1", 64), RetainWork: true})
			fatal(t, problem)
			if changed, problem := store.FailQueuedRequest("job-a", retainedFailure("source_failed", "retryable source failure")); problem != nil || !changed {
				t.Fatalf("could not retain the prior operation: %v", problem)
			}
			key, retry, problem := store.ResumableRun(base, "https://hub-a.example/", "pr-shared", true)
			fatal(t, problem)
			if retry != "job-a" || !strings.HasPrefix(key, keyA+"/retry-of/") || len(key) > 200 {
				t.Fatalf("same-source command lost its bounded retry identity: %q %q", key, retry)
			}
			keyB, retry, problem := store.ResumableRun(base, "https://hub-b.example", "pr-shared", true)
			fatal(t, problem)
			if retry != "" || keyB == keyA || strings.Contains(keyB, "job-a") {
				t.Fatalf("a new command on B resumed A's retained work: %q %q", keyB, retry)
			}
			// Explicit resume still addresses job-a by ID, with its original source.
			prior, problem := store.RequestRow("job-a")
			fatal(t, problem)
			if prior == nil || prior.Hub != "https://hub-a.example" {
				t.Fatalf("new-source lookup mutated accepted work: %+v", prior)
			}
		})
	}
}
