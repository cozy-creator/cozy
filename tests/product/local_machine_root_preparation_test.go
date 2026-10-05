package producttest

import (
	"os"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// An unchanged unpublished package is prepared once per machine store: a warm run, and a run
// after the machine restarts, go straight to the code the machine holds, and the run shows no
// package preparation for them.
func TestAnUnpublishedRootIsPreparedOncePerMachine(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: this computer's machine runs the call")
	}
	root, err := os.MkdirTemp("", "czr")
	must(t, err)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if !t.Failed() {
			_ = removeAllForce(root)
		}
	})
	provisionMachine(t, root)
	if code, out := runCozy(t, root, "package", "install", weightlessProject(t), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	prepared := func(key string) bool {
		t.Helper()
		if code, out := runCozy(t, root, "run", localWeightlessRef+"/tile", "size=8", "--await", "--json", "--idempotency-key", key); code != 0 {
			t.Fatalf("%s run [exit %d]\n%s", key, code, out)
		}
		store, problem := records.Open(layout.DB)
		fatal(t, problem)
		defer store.Close()
		request, problem := store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		events, problem := store.EventsAfter(request.ID, 0, 1000)
		fatal(t, problem)
		stages := 0
		for _, event := range events {
			if event.Type == "request.preparing" && event.Payload["stage"] == "package" {
				stages++
			}
		}
		return stages == 1
	}
	if !prepared("first") {
		t.Fatal("the first run did not prepare its package on the machine")
	}
	if prepared("warm") {
		t.Fatal("a warm run prepared the package its machine already holds")
	}

	if code, out := runCozy(t, root, "machine", "stop"); code != 0 {
		t.Fatalf("machine stop [exit %d]\n%s", code, out)
	}
	// The machine's store outlives it: a new lifetime reopens the code it holds.
	if prepared("rebooted") {
		t.Fatal("a restarted machine's run prepared the package its store already holds")
	}
}
