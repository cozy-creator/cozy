package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A run blocked before its machine took it may be retried, and the retry leaves the blocked
// run's history as it was. One possibly sent to its machine (cozy.machine.v1's dispatch mark
// without acceptance) or canceled is not retried: that could run the work twice or revive it.
func TestMachineRetryAfterBlockedPreparationKeepsUnsentHistory(t *testing.T) {
	for _, state := range []string{"unsent", "sent", "canceled"} {
		t.Run(state, func(t *testing.T) {
			store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			prior, _, problem := store.Submit(records.Request{ID: "job-blocked", IdemKey: "blocked", Kind: "job", Package: "local/example", Entrypoint: "main",
				Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("1", 64), RetainWork: true, MachineExecutionObserver: true})
			fatal(t, problem)
			fatal(t, store.LinkMachineExecution(prior.ID, "local"))
			if state == "sent" {
				sent, problem := store.MarkRunV1Sent(prior.ID)
				fatal(t, problem)
				if !sent {
					t.Fatal("the fixture's dispatch was not marked")
				}
			}
			changed, problem := store.BlockRetainedWork(prior.ID, "worker.protocol_incompatible", "upgrade the idle Runtime")
			fatal(t, problem)
			if !changed {
				t.Fatal("preparation did not become blocked")
			}
			if state == "canceled" {
				_, problem = store.RequestMachineCancellation(prior.ID, "")
				fatal(t, problem)
			}
			retry, fresh, problem := store.Submit(records.Request{ID: "job-retry", IdemKey: "retry", Kind: "job", Package: prior.Package, Entrypoint: "main",
				Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("2", 64), RetainWork: true, RetryOf: prior.ID, MachineExecutionObserver: true})
			if state != "unsent" {
				if problem == nil {
					t.Fatalf("a retry of %s work was accepted: it could run twice or revive canceled work", state)
				}
				t.Logf("refused: %s", problem.Message)
				return
			}
			fatal(t, problem)
			if !fresh || retry.RetryOf != prior.ID || retry.ReuseScope != prior.ReuseScope {
				t.Fatal("retry lost retained predecessor lineage")
			}
			old, problem := store.RequestRow(prior.ID)
			fatal(t, problem)
			link, problem := store.MachineExecution(prior.ID)
			fatal(t, problem)
			sent, problem := store.RunV1Marked(prior.ID, records.RunV1Sent)
			fatal(t, problem)
			if old.State != "blocked" || sent || len(link.Receipt) != 0 {
				t.Fatal("retry changed old history or invented machine dispatch")
			}
		})
	}
}
