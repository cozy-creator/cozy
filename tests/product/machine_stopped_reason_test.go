package producttest

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// A run whose machine stopped mid-run says so plainly and says it was not retried: in `run list`
// and in `run watch`, whether its machine sent the reason's code or (before tensord 0.1.1) only
// its internal words.
func TestARunWhoseMachineStoppedSaysSoPlainly(t *testing.T) {
	for name, reason := range map[string]*v1.Reason{
		"coded":  {Code: records.MachineStopped, Message: "machine_stopped: the machine running this run stopped mid-run", Origin: "machine"},
		"legacy": {Code: "failed", Message: "owner lost before durable result custody; exact executor birth has ended; started work will not be replayed", Origin: "machine"},
	} {
		t.Run(name, func(t *testing.T) {
			o := hostOwner(t, "machine-stopped-"+name)
			id := "req-machine-stopped"
			request, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: "idem-" + id,
				BodyDigest: "sha256:" + sixtyFour("7"), Package: "proof/example", Entrypoint: "generate",
				Payload: []byte("{}"), MachineExecutionObserver: true})
			fatal(t, problem)
			fatal(t, o.store.LinkMachineExecution(request.ID, "local"))
			if send, problem := o.store.MarkRunV1Sent(request.ID); problem != nil || !send {
				t.Fatalf("not sent: %v %v", send, problem)
			}
			fatal(t, o.store.AcceptRunV1(request.ID, "local", &v1.RunState{Id: id, Number: 1, State: "running", Attempt: 1}))
			fatal(t, o.store.RecordRunOutcomeV1(request.ID, records.RunEndV1{Outcome: &v1.Outcome{Status: "failed", Reason: reason}}))
			defer publicationControlAPI(t, o)()

			const plain = "the machine running it stopped mid-run, and the run was not retried"
			code, out := runCozy(t, o.root, "run", "list", "--json")
			var listed struct {
				Invocations []struct {
					Number    int64  `json:"number"`
					ErrorType string `json:"error_type"`
					Error     string `json:"error"`
				} `json:"invocations"`
			}
			if code != 0 || json.Unmarshal([]byte(out), &listed) != nil || len(listed.Invocations) != 1 ||
				listed.Invocations[0].ErrorType != records.MachineStopped || listed.Invocations[0].Error != plain {
				t.Fatalf("run list does not say the machine stopped [%d]: %s", code, out)
			}
			code, out = runCozy(t, o.root, "run", "watch", id, "--json")
			var watched struct {
				Error struct {
					Code    string `json:"code"`
					Message string `json:"message"`
				} `json:"error"`
			}
			if code == 0 || json.Unmarshal([]byte(out), &watched) != nil || watched.Error.Code != records.MachineStopped ||
				!strings.Contains(watched.Error.Message, plain) || strings.Contains(watched.Error.Message, "custody") {
				t.Fatalf("run watch does not say plainly what happened [%d]: %s", code, out)
			}
		})
	}
}
