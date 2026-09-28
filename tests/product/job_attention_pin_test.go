package producttest

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A job's --attention-kernel pin reaches the machine that runs it, and the machine's warning
// that no call applied the pin shows in `cozy run show`, human and --json.
func TestJobAttentionPinReachesItsMachineAndItsWarningShows(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := &runtimeMachine{blocker: "none"}
	pod := &fakePod{machine: machine, deviceCount: 4}
	root, _ := rentedLadderMachine(t, h, pod, nil)
	if code, out := runCozy(t, root, "run", ladderPackage+"/long_form", "--attention-kernel=sageattention",
		"--rental=tessa", "--json", "--idempotency-key", "job-pin"); code != 0 {
		t.Fatalf("a job's attention pin was refused [%d]: %s", code, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByIdempotencyKey("job-pin")
	fatal(t, problem)
	if request == nil || request.Kind != "job" || request.AttentionKernel != "sageattention" {
		t.Fatalf("the job did not record its pin: %+v", request)
	}
	waitFor(t, root, "the job reaches its machine", func() bool { return machine.submitted() != nil })
	if root := machine.submitted().ReleaseRoot; root == nil || root.AttentionKernel != "sageattention" {
		t.Fatalf("the machine was not handed the pin: %+v", machine.submitted())
	}

	machine.mu.Lock()
	machine.record("warning", []byte(`{"code":"attention_pin_unapplied","message":"attention pin sageattention was never applied"}`))
	machine.mu.Unlock()
	waitFor(t, root, "the machine's warning is recorded", func() bool {
		events, problem := store.EventsAfter(request.ID, 0, 1000)
		if problem != nil {
			return false
		}
		for _, event := range events {
			if event.Type == "machine.warning" {
				return true
			}
		}
		return false
	})
	if code, out, errOut := runCozyStreams(t, root, "run", "show", request.ID); code != 0 ||
		!strings.Contains(out, "warning attention_pin_unapplied: attention pin sageattention was never applied") {
		t.Fatalf("run show did not render the warning [%d]:\n%s%s", code, out, errOut)
	}
	code, out := runCozy(t, root, "run", "show", request.ID, "--json")
	var report struct {
		Warnings []struct{ Code, Message string }
	}
	if code != 0 || json.Unmarshal([]byte(out), &report) != nil || len(report.Warnings) != 1 ||
		report.Warnings[0].Code != "attention_pin_unapplied" || !strings.Contains(report.Warnings[0].Message, "never applied") {
		t.Fatalf("run show --json lost the warning [%d]: %s", code, out)
	}
}
