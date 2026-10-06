package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRunCancelAbandonCLIIsLocalAndSurvivesDaemonRestart(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	request := pendingNativeRequest(t, store)
	_, problem = store.RequestMachineCancellation(request.ID, "")
	fatal(t, problem)
	before, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	store.Close()
	svc := startDaemonProcess(t, root)
	refusal := svc.call(t, http.MethodPost, "/v1/local/requests/"+request.ID+"/abandon", map[string]any{})
	if refusal.Status != http.StatusBadRequest {
		t.Fatalf("unnamed abandonment: %s", refusal.brief())
	}
	// A separate ordinary cancel observer is already waiting on unknown remote
	// acceptance. Explicit local abandonment must wake it without inventing stop.
	waiter := exec.Command(cozyBin, "run", "cancel", request.ID, "--await", "--json")
	waiter.Env = childEnv(t, root)
	var waitingOutput bytes.Buffer
	waiter.Stdout, waiter.Stderr = &waitingOutput, &waitingOutput
	setProcessGroup(waiter)
	must(t, waiter.Start())
	t.Cleanup(func() { _ = killGroup(waiter.Process.Pid) })
	waited := make(chan error, 1)
	go func() { waited <- waiter.Wait() }()
	reader, problem := records.OpenReadOnly(layout.DB)
	fatal(t, problem)
	observationBudget := time.After(10 * time.Second)
	for {
		actor, _, _, problem := reader.CancelAttribution(request.ID)
		fatal(t, problem)
		if actor != "" {
			break
		}
		select {
		case <-waited:
			t.Fatal("unknown acceptance completed cancel before abandonment")
		case <-observationBudget:
			t.Fatal("cancel observer did not record its intent")
		case <-time.After(10 * time.Millisecond):
		}
	}
	reader.Close()
	code, out := runCozy(t, root, "run", "cancel", request.ID, "--abandon", "--json")
	if code != 0 {
		t.Fatalf("ordinary abandon [%d]: %s", code, out)
	}
	var result struct {
		AbandonedLocally    bool `json:"abandoned_locally"`
		RemoteStopConfirmed bool `json:"remote_stop_confirmed"`
		RentalReleased      bool `json:"rental_released"`
	}
	must(t, json.Unmarshal([]byte(out), &result))
	if !result.AbandonedLocally || result.RemoteStopConfirmed || result.RentalReleased {
		t.Fatalf("CLI claimed a remote side effect: %s", out)
	}
	select {
	case <-waited:
		if strings.Contains(waitingOutput.String(), `"status":"canceled"`) {
			t.Fatal("local abandonment reported confirmed remote cancellation")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("explicit abandonment left the cancel observer waiting")
	}
	if code, out := runCozy(t, root, "run", "cancel", request.ID, "--abandon", "--await"); code == 0 || !strings.Contains(out, "cannot be combined") {
		t.Fatalf("abandon awaited remote cancellation [%d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("stop isolated observer [%d]: %s", code, out)
	}
	<-svc.exited
	svc = startDaemonProcess(t, root)
	row := svc.call(t, http.MethodGet, "/v1/requests/"+request.ID+"?recorded=1", nil)
	if row.Status != http.StatusOK || !strings.Contains(string(row.Body), `"abandoned_locally":true`) {
		t.Fatalf("restart forgot abandonment: %s", row.brief())
	}
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("stop restarted observer [%d]: %s", code, out)
	}
	<-svc.exited
	store, problem = records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	after, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if after == nil || !after.Abandoned || !bytes.Equal(after.Submission, before.Submission) || len(after.Receipt) != 0 || after.SubmissionClosed || !after.CancelRequested {
		t.Fatal("ordinary CLI/restart lost unknown acceptance evidence")
	}
}
