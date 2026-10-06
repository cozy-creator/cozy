//go:build linux

package producttest

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/calcifer"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

func requireCalciferIdentity(t *testing.T, root string) int {
	t.Helper()
	state := calcifer.Probe(config.Config{Home: root})
	if !state.Up || state.PID == 0 {
		t.Fatalf("Calcifer is not running: %+v", state)
	}
	proc := "/proc/" + strconv.Itoa(state.PID)
	comm, err := os.ReadFile(proc + "/comm")
	must(t, err)
	argv, err := os.ReadFile(proc + "/cmdline")
	must(t, err)
	if strings.TrimSpace(string(comm)) != "calcifer" || filepath.Base(strings.SplitN(string(argv), "\x00", 2)[0]) != "calcifer" {
		t.Fatalf("controller pid%d has comm%q argv0%q; want calcifer", state.PID, comm, strings.SplitN(string(argv), "\x00", 2)[0])
	}
	t.Logf("controller pid%d: OS name calcifer, argv0 calcifer", state.PID)
	return state.PID
}

func TestCalciferStartsAndRestartsWithoutChangingHistory(t *testing.T) {
	root := t.TempDir()
	t.Cleanup(func() { _, _ = runCozy(t, root, "down") })
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	request, _, problem := store.Submit(records.Request{ID: "calcifer-history", IdemKey: "calcifer-history",
		Package: "proof/retained", Entrypoint: "generate", Payload: []byte(`{"value":42}`), BodyDigest: childDigest("c"),
		MachineExecutionObserver: true})
	fatal(t, problem)
	_, problem = store.CancelQueuedRequest(request.ID, map[string]any{"canceled_by": "proof owner"})
	fatal(t, problem)
	before, problem := store.EventsAfter(request.ID, 0, 100)
	fatal(t, problem)
	store.Close()
	for launch := range 2 {
		if code, out := runCozy(t, root, "up", "--json", "--full"); code != 0 || !strings.Contains(out, `"daemon":"running"`) {
			t.Fatalf("cozy up launch%d [exit%d]: %s", launch, code, out)
		}
		pid := requireCalciferIdentity(t, root)
		if code, out := runCozy(t, root, "up", "--json", "--full"); code != 0 || !strings.Contains(out, `"changed":false`) ||
			calcifer.Probe(config.Config{Home: root}).PID != pid {
			t.Fatalf("repeated cozy up replaced the controller [exit%d]: %s", code, out)
		}
		if code, out := runCozy(t, root, "run", "list", "--json", "--full"); code != 0 || !strings.Contains(out, request.ID) {
			t.Fatalf("history missing on launch%d [exit%d]: %s", launch, code, out)
		}
		if code, out := runCozy(t, root, "down", "--json", "--full"); code != 0 || !strings.Contains(out, `"daemon":"stopped"`) ||
			calcifer.Probe(config.Config{Home: root}).Up {
			t.Fatalf("cozy down launch%d [exit%d]: %s", launch, code, out)
		}
	}
	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	after, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	events, problem := store.EventsAfter(request.ID, 0, 100)
	fatal(t, problem)
	if after.State != "canceled" || !bytes.Equal(after.Payload, request.Payload) || len(events) != len(before) {
		t.Fatalf("controller restarts changed retained history: %+v, events%d->%d", after, len(before), len(events))
	}
}

func TestCalciferKeepsAnExistingControllerUntilExplicitRestart(t *testing.T) {
	if *olderCozy == "" {
		t.Skip("requires -older-cozy: a real controller binary from before the process rename")
	}
	root := t.TempDir()
	t.Cleanup(func() { _, _ = runCozy(t, root, "down") })
	command := exec.Command(*olderCozy, "up", "--json", "--full")
	command.Env = childEnv(t, root)
	if out, err := command.CombinedOutput(); err != nil {
		t.Fatalf("existing controller startup: %v\n%s", err, out)
	}
	before := calcifer.Probe(config.Config{Home: root})
	if !before.Up || before.PID == 0 {
		t.Fatalf("existing controller is not running: %+v", before)
	}
	if code, out := runCozy(t, root, "up", "--json", "--full"); code != 0 || !strings.Contains(out, `"changed":false`) {
		t.Fatalf("new CLI could not attach to the existing controller [exit%d]: %s", code, out)
	}
	if after := calcifer.Probe(config.Config{Home: root}); after.PID != before.PID {
		t.Fatalf("new CLI replaced existing controller pid%d with pid%d", before.PID, after.PID)
	}
	if _, err := os.Stat(filepath.Join(root, "calcifer")); !os.IsNotExist(err) {
		t.Fatal("attachment installed a new controller executable before explicit restart")
	}
	if code, out := runCozy(t, root, "down", "--json", "--full"); code != 0 {
		t.Fatalf("explicit stop of existing controller [exit%d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "up", "--json", "--full"); code != 0 {
		t.Fatalf("new controller startup after explicit stop [exit%d]: %s", code, out)
	}
	if pid := requireCalciferIdentity(t, root); pid == before.PID {
		t.Fatal("explicit restart kept the old process")
	}
}
