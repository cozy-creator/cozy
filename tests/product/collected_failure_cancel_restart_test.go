package producttest

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// The machine preserves the collected failure and releases its custody only after
// an explicit, generation-checked cancellation. Its authenticated RPC events are
// the observer's evidence that native release completed.
type collectedFailureMachine struct {
	*durableCancelMachine
}

func (m *collectedFailureMachine) ControlMachineExecution(_ context.Context, command *pb.MachineExecutionControl) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.commands[command.CommandId] {
		if command.Action != pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL || command.ExpectedGeneration != m.state.Generation {
			return nil, status.Error(codes.FailedPrecondition, "unexpected failed-root control")
		}
		m.commands[command.CommandId] = true
		m.state.Generation++
		m.state.State = "canceled"
		m.record("control", []byte(`{"action":"cancel","generation":1}`))
		m.record("retention_released", []byte(`{"collected":true}`))
	}
	return proto.Clone(m.state).(*pb.MachineExecutionState), nil
}

func collectedFailureFixture(t *testing.T) (string, *records.Store, *collectedFailureMachine, string) {
	t.Helper()
	h := newLadderHub(t)
	h.bind(goodLadder())
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "runtime-bundle.json"))
	must(t, err)
	m := &collectedFailureMachine{durableCancelMachine: &durableCancelMachine{
		heldMachine: &heldMachine{runtimeMachine: &runtimeMachine{triage: bundle, failure: "retained failure"},
			commands: map[string]bool{}, canceled: make(chan struct{})}, readGate: newCancelGate(),
	}}
	root := startRentedFixture(t, h, func(blocker string) machineExecutionPeer { m.blocker = blocker; return m }, "")
	if code, out := runCozy(t, root, "run", ladderPackage+"/long_form", "--rental=tessa", "--json"); code != 0 {
		t.Fatalf("submit retained failure [%d]: %s", code, out)
	}
	store, problem := records.Open(home.Paths(root).DB)
	fatal(t, problem)
	t.Cleanup(store.Close)
	var id string
	waitFor(t, root, "accepted failure fixture", func() bool {
		submitted := m.submitted()
		if submitted == nil {
			return false
		}
		id = submitted.Offer.RequestId
		link, _ := store.MachineExecution(id)
		return link != nil && len(link.Receipt) > 0
	})
	return root, store, m, id
}

func assertCollectedFailureReleased(t *testing.T, root string, store *records.Store, m *collectedFailureMachine, id string, original *records.MachineExecution) {
	t.Helper()
	waitFor(t, root, "explicit failed-root cancellation and native release", func() bool {
		link, _ := store.MachineExecution(id)
		events, _ := store.EventsAfter(id, 0, 1000)
		released := false
		for _, event := range events {
			released = released || event.Type == "machine.retention_released"
		}
		m.mu.Lock()
		accepted := len(m.commands) == 1 && m.state.Generation == 2 && m.state.State == "canceled"
		m.mu.Unlock()
		return accepted && released && link != nil && !link.CancelRequested && len(link.PendingControl) == 0
	})
	link, problem := store.MachineExecution(id)
	fatal(t, problem)
	if !bytes.Equal(link.Receipt, original.Receipt) || !bytes.Equal(link.Outcome, original.Outcome) {
		t.Fatal("cancel replaced the accepted execution or its collected failure")
	}
	m.mu.Lock()
	submissions := len(m.submissions)
	m.mu.Unlock()
	if submissions != 1 {
		t.Fatalf("cancel resubmitted the failed run %d times", submissions)
	}
}

func TestCollectedFailureCancelSurvivesDaemonRestart(t *testing.T) {
	root, store, m, id := collectedFailureFixture(t)
	m.finish()
	waitFor(t, root, "collected failed outcome", func() bool {
		row, _ := store.RequestRow(id)
		link, _ := store.MachineExecution(id)
		return row != nil && row.State == "failed" && link != nil && link.Collected
	})
	original, problem := store.MachineExecution(id)
	fatal(t, problem)
	if code, out := runCozy(t, root, "down", "--json"); code != 0 {
		t.Fatalf("observer down [%d]: %s", code, out)
	}
	// This is the durable API commit boundary with no observer present to send it.
	_, problem = store.RequestMachineCancellation(id, "cozy job cancel")
	fatal(t, problem)
	link, problem := store.MachineExecution(id)
	fatal(t, problem)
	if !link.Collected || !link.CancelRequested || len(link.PendingControl) != 0 {
		t.Fatal("fixture did not preserve an unsent cancel on a collected failure")
	}
	startDaemonProcess(t, root)
	assertCollectedFailureReleased(t, root, store, m, id, original)
}

// Standard Go build overlays hold the test daemon after its final local snapshot.
// The normal production binary has no hook. Verify the source seam so a refactor
// cannot quietly remove the deterministic interleaving this regression exercises.
func collectedFailureExitBinary(t *testing.T, gate string) string {
	t.Helper()
	root, err := filepath.Abs("../..")
	must(t, err)
	source := filepath.Join(root, "internal", "cli", "machine_runs.go")
	raw, err := os.ReadFile(source)
	must(t, err)
	needle := "observed, e := m.store.MachineExecution(request.ID)"
	if strings.Count(string(raw), needle) != 1 {
		t.Fatal("observer exit snapshot seam changed")
	}
	barrier := fmt.Sprintf(`
					if observed != nil && observed.Collected && !observed.CancelRequested {
						wanted, _ := os.ReadFile(%q)
						if string(wanted) == request.ID {
							_ = os.WriteFile(%q, nil, 0600)
							for {
								if _, err := os.Stat(%q); err == nil { break }
								select { case <-m.ctx.Done(): return; case <-time.After(5*time.Millisecond): }
							}
						}
					}
`, filepath.Join(gate, "request"), filepath.Join(gate, "entered"), filepath.Join(gate, "release"))
	patched := strings.Replace(string(raw), `"io"`, "\"io\"\n\t\"os\"", 1)
	patched = strings.Replace(patched, needle, needle+barrier, 1)
	dir := t.TempDir()
	replacement := filepath.Join(dir, "machine_runs.go")
	must(t, os.WriteFile(replacement, []byte(patched), 0600))
	overlay, err := json.Marshal(map[string]map[string]string{"Replace": {source: replacement}})
	must(t, err)
	manifest := filepath.Join(dir, "overlay.json")
	must(t, os.WriteFile(manifest, overlay, 0600))
	binary := filepath.Join(dir, "cozy")
	build := exec.Command("go", "build", "-overlay", manifest, "-o", binary, ".")
	build.Dir = root
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building the observer exit fixture: %v\n%s", err, output)
	}
	return binary
}

func TestCollectedFailureCancelDuringObserverExitIsDelivered(t *testing.T) {
	root, store, m, id := collectedFailureFixture(t)
	if code, out := runCozy(t, root, "down", "--json"); code != 0 {
		t.Fatalf("observer down [%d]: %s", code, out)
	}
	gate := t.TempDir()
	must(t, os.WriteFile(filepath.Join(gate, "request"), []byte(id), 0600))
	t.Cleanup(func() { _ = os.WriteFile(filepath.Join(gate, "release"), nil, 0600) })
	startDaemonBinary(t, collectedFailureExitBinary(t, gate), root)
	m.finish()
	waitFor(t, root, "observer paused after collected/cancel-false snapshot", func() bool {
		_, err := os.Stat(filepath.Join(gate, "entered"))
		return err == nil
	})
	original, problem := store.MachineExecution(id)
	fatal(t, problem)
	if !original.Collected || original.CancelRequested || len(original.PendingControl) != 0 {
		t.Fatal("fixture did not hold the final collected snapshot before cancellation")
	}
	if code, out := runCozy(t, root, "run", "cancel", id, "--json"); code != 0 {
		t.Fatalf("cancel at observer exit [%d]: %s", code, out)
	}
	link, problem := store.MachineExecution(id)
	fatal(t, problem)
	if !link.CancelRequested || len(link.PendingControl) != 0 {
		t.Fatal("cancel did not persist before the active observer exited")
	}
	must(t, os.WriteFile(filepath.Join(gate, "release"), nil, 0600))
	assertCollectedFailureReleased(t, root, store, m, id, original)
}

// The released native Runtime owns the failed attempt and its cleanup. Persist
// cancellation at the authored API commit boundary while its observer is down,
// then use normal CLI startup to deliver it through the standalone machine agent.
func TestNativeCollectedFailureCancelSurvivesDaemonRestart(t *testing.T) {
	integration(t)
	if *machineHostBinary == "" || *machineRuntimeWheel == "" || *machineTensorFSWheel == "" {
		if *requireMachineHost {
			t.Fatal("native failed-root qualification requires its agent and paired wheels")
		}
		t.Skip("requires an exact standalone agent and Runtime/TensorFS wheels")
	}
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "czcf-")
	must(t, err)
	claimScratch(root)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		if t.Failed() {
			t.Logf("native collected-failure evidence retained at %s", root)
		} else {
			_ = removeAllForce(root)
		}
	})
	script := filepath.Join(t.TempDir(), "failure.py")
	code := fmt.Sprintf(`# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=%s"]
# [tool.uv.sources]
# cozy-runtime = {path = %q}
# tensorfs = {path = %q}
# ///
async def main():
    raise ValueError("intentional retained failure")
`, runtimeFixtureVersion(t, *machineRuntimeWheel), *machineRuntimeWheel, *machineTensorFSWheel)
	must(t, os.WriteFile(script, []byte(code), 0600))
	if status, out := runCozy(t, root, "run", script, "--await", "--json"); status == 0 || !strings.Contains(out, "intentional retained failure") {
		t.Fatalf("native authored failure was not observed [%d]: %s", status, out)
	}
	store, problem := records.Open(home.Paths(root).DB)
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByReference("1")
	fatal(t, problem)
	if request == nil || request.State != "failed" || !request.RetainWork {
		t.Fatalf("failure lost resumable ownership: %+v", request)
	}
	waitFor(t, root, "native failed-result collection", func() bool {
		link, _ := store.MachineExecution(request.ID)
		return link != nil && link.Collected
	})
	original, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	journal, err := sql.Open("sqlite", "file:"+machineJournal(root)+"?mode=ro")
	must(t, err)
	defer journal.Close()
	journal.SetMaxOpenConns(1)
	var retained bool
	must(t, journal.QueryRow("SELECT retention_waived=0 AND desired='run' FROM executions WHERE request=?", request.ID).Scan(&retained))
	if !retained || original.CancelRequested || len(original.PendingControl) != 0 {
		t.Fatal("failed work was abandoned before its owner canceled it")
	}
	if status, out := runCozy(t, root, "down", "--json"); status != 0 {
		t.Fatalf("native observer down [%d]: %s", status, out)
	}
	_, problem = store.RequestMachineCancellation(request.ID, "cozy job cancel")
	fatal(t, problem)
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if !link.Collected || !link.CancelRequested || len(link.PendingControl) != 0 {
		t.Fatal("native restart boundary lost the unsent cancellation")
	}
	if status, out := runCozy(t, root, "up", "--json"); status != 0 {
		t.Fatalf("native observer up [%d]: %s", status, out)
	}
	waitFor(t, root, "native cancellation and retention release", func() bool {
		var released bool
		must(t, journal.QueryRow("SELECT retention_waived=1 AND desired='cancel' FROM executions WHERE request=?", request.ID).Scan(&released))
		link, _ := store.MachineExecution(request.ID)
		events, _ := store.EventsAfter(request.ID, 0, 1000)
		observed := false
		for _, event := range events {
			observed = observed || event.Type == "machine.retention_released"
		}
		return released && observed && link != nil && !link.CancelRequested && len(link.PendingControl) == 0
	})
	var commands int
	must(t, journal.QueryRow("SELECT count(*) FROM execution_commands WHERE request=? AND json_extract(CAST(intent AS TEXT),'$.action')='cancel'", request.ID).Scan(&commands))
	if commands != 1 {
		t.Fatalf("native Runtime accepted %d cancellation commands", commands)
	}
	link, problem = store.MachineExecution(request.ID)
	fatal(t, problem)
	if !bytes.Equal(original.Receipt, link.Receipt) || !bytes.Equal(original.Outcome, link.Outcome) {
		t.Fatal("native cleanup replaced the accepted execution or its failure")
	}
}
