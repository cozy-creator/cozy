package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
)

// A run whose Run was sent to this computer's machine, which then stopped before answering,
// may be on that machine. `cozy run cancel --await` starts the machine as a local run would,
// the cancel reaches it, and the run ends as the machine says: canceled, by its actor.
func TestCancelOnTheStoppedLocalMachineStartsItAndDelivers(t *testing.T) {
	root, store, _ := nativeLifecycleHome(t)
	if code, out := runCozy(t, root, "machine", "stop", "--json"); code != 0 {
		t.Fatalf("machine stop [%d]: %s", code, out)
	}
	host := machines.NewHost(filepath.Join(root, "machine"), "", nil)
	if status, problem := host.Status(); problem != nil || status.Running {
		t.Fatalf("the machine still runs after stop: %+v %v", status, problem)
	}
	payload, err := json.Marshal(map[string]any{"gate": filepath.Join(root, "stopped-cancel-gate"), "size": 64})
	must(t, err)
	request, _, problem := store.Submit(records.Request{ID: "native-stopped-cancel", IdemKey: "native-stopped-cancel",
		Package: "local/native-lifecycle", Entrypoint: "make", Kind: "job", Payload: payload,
		BodyDigest: childDigest("e"), MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, machines.Local))
	// The marker the native transport writes before sending Run; the machine stopped before it answered.
	fatal(t, store.AppendEvent(request.ID, records.RunV1Sent, 0, map[string]any{"machine": machines.Local}))
	code, out := cozyWithin(t, root, 3*time.Minute, "run", "cancel", request.ID, "--await", "--json")
	if code != 0 || !strings.Contains(out, `"status":"canceled"`) || !strings.Contains(out, `"canceled_by":"cozy run cancel"`) {
		t.Fatalf("the awaited cancel did not end canceled by its actor [exit %d]: %s\n%s", code, out, tail(filepath.Join(root, "daemon.log")))
	}
	if status, problem := host.Status(); problem != nil || !status.Running {
		t.Fatalf("the cancel did not start this computer's machine: %+v %v", status, problem)
	}
	machine, problem := (&machines.Resolver{Host: host}).DialV1(machines.AttachOnly(t.Context()), machines.Local, "stopped cancel test")
	fatal(t, problem)
	defer machine.Close()
	stream, err := machine.Run(t.Context(), request.ID, 0, nil)
	must(t, err)
	for {
		event, err := stream.Recv()
		must(t, err)
		if outcome := event.GetOutcome(); outcome != nil {
			if outcome.Status != "canceled" {
				t.Fatalf("the machine's run ended %s", outcome.Status)
			}
			break
		}
	}
	events, problem := store.EventsAfter(request.ID, 0, 1000)
	fatal(t, problem)
	for _, event := range events {
		if event.Payload["scope"] == "before_machine_submission" || event.Payload["scope"] == "machine_stopped" {
			t.Fatalf("sent work was settled without its machine: %s %v", event.Type, event.Payload)
		}
	}
}

// When this computer's machine cannot start, the cancel cannot reach the run: it stays
// canceling, and `cozy run cancel --await` says so with the next step instead of waiting.
func TestCancelOnALocalMachineThatCannotStartSaysSo(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the current native machine fixture")
	}
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName),
		[]byte("tensorhub_url: http://127.0.0.1:1\ntensorhub_token: unreachable\n"), 0o600))
	provisionMachine(t, root)
	// A machine that answers what it is but fails at boot, as one with a broken store does.
	binary := filepath.Join(root, "machine", "root", "usr", "local", "bin", "cozy-machine")
	must(t, os.Remove(binary))
	must(t, os.WriteFile(binary, []byte("#!/bin/sh\nif [ \"$1\" = version ]; then exec '"+*machineHostBinary+
		"' \"$@\"; fi\necho 'cozy-machine: cannot open its store: permission denied' >&2\nexit 1\n"), 0o755))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	request, _, problem := store.Submit(records.Request{ID: "req-unstartable-cancel", IdemKey: "unstartable-cancel", Package: "proof/stuck",
		Entrypoint: "generate", Payload: []byte(`{}`), BodyDigest: childDigest("d"), MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, machines.Local))
	fatal(t, store.AppendEvent(request.ID, records.RunV1Sent, 0, map[string]any{"machine": machines.Local}))
	store.Close()
	startDaemonProcess(t, root)
	code, out := cozyWithin(t, root, 3*time.Minute, "run", "cancel", request.ID, "--await")
	if code == 0 || !strings.Contains(out, "stays canceling") || !strings.Contains(out, "cozy machine start") ||
		!strings.Contains(out, "cannot open its store") {
		t.Fatalf("an unstartable machine's cancel did not say so with its next step [exit %d]: %s", code, out)
	}
	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if row.State != "canceling" || !link.CancelRequested {
		t.Fatalf("the undelivered cancel left the run %s (intent kept %v)", row.State, link.CancelRequested)
	}
}
