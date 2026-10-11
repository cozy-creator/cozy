package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// The daemon keeps one connection per machine: a warm run rides the connection the first
// run opened, with no new TLS handshake, and a restarted machine is connected to anew.
func TestTheDaemonKeepsOneConnectionPerMachine(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: this computer's machine runs the call")
	}
	root, err := os.MkdirTemp("", "czk")
	must(t, err)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if !t.Failed() {
			_ = removeAllForce(root)
		}
	})
	provisionMachine(t, root)
	if code, out := runCozy(t, root, "package", "install", weightlessProject(t)); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	connect := func(key string) any {
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
		for _, event := range events {
			if event.Type == "request.preparing" && event.Payload["stage"] == "connect" {
				return event.Payload["ms"]
			}
		}
		t.Fatalf("the %s run shows no machine connection", key)
		return nil
	}
	connections := func() int {
		t.Helper()
		log, err := os.ReadFile(filepath.Join(root, "daemon.log"))
		must(t, err)
		return strings.Count(string(log), "machine local: connecting to ")
	}
	first, warm := connect("first"), connect("warm")
	if n := connections(); n != 1 {
		t.Fatalf("two runs on one machine opened %d connections", n)
	}
	if code, out := runCozy(t, root, "machine", "stop"); code != 0 {
		t.Fatalf("machine stop [exit %d]\n%s", code, out)
	}
	rebooted := connect("rebooted")
	if n := connections(); n != 2 {
		t.Fatalf("a restarted machine was reached over %d connections in all, want a second one", n)
	}
	t.Logf("machine connection stage: first %v ms, warm %v ms, after a restart %v ms", first, warm, rebooted)
}
