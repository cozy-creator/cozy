package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A cancel issued the moment `run watch` reports a run completed answers that run's own
// terminal, unchanged, however soon it lands after completion: the result's collection may
// still be in flight, and cancelling a finished call changes nothing.
func TestCancelRightAfterCompletionChangesNothing(t *testing.T) {
	root, err := os.MkdirTemp(os.TempDir(), "cozy-cancel-")
	must(t, err)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		_ = os.RemoveAll(root)
	})
	if code, out := runCozy(t, root, "package", "install", weightlessProject(t)); code != 0 {
		t.Fatalf("local directory package install [exit %d]\n%s", code, out)
	}
	for round := range 5 {
		key := "cancel-after-completion-" + strconv.Itoa(round)
		if code, out := runCozy(t, root, "run", localWeightlessRef+"/tile", "size=32", "seed="+strconv.Itoa(round),
			"--idempotency-key", key, "--json"); code != 0 {
			t.Fatalf("round %d: run was not accepted [exit %d]\n%s", round, code, out)
		}
		store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
		fatal(t, problem)
		request, problem := store.RequestByIdempotencyKey(key)
		store.Close()
		fatal(t, problem)
		if request == nil {
			t.Fatalf("round %d: run was not recorded", round)
		}
		number := request.ID
		if code, out := runCozy(t, root, "run", "watch", number, "--json"); code != 0 {
			t.Fatalf("round %d: run did not complete [exit %d]\n%s", round, code, out)
		}
		for _, attempt := range []string{"first", "second"} {
			code, out := runCozy(t, root, "run", "cancel", number, "--json")
			var answer struct {
				Changed bool   `json:"changed"`
				Status  string `json:"status"`
			}
			if code != 0 || json.Unmarshal([]byte(out), &answer) != nil || answer.Changed || answer.Status != "completed" {
				t.Fatalf("round %d: %s cancel of a completed run changed it [exit %d]\n%s", round, attempt, code, out)
			}
		}
	}
}
