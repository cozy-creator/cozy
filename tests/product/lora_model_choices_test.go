package producttest

import (
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Real argv -> daemon persistence -> authenticated machine submission. The
// machine fixture observes the transport; Runtime's own tests execute adapters.
func TestLoRAChoicesReachRootAndCapturedServingSlots(t *testing.T) {
	for _, function := range []string{"generate", "long_form"} {
		t.Run(function, func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(goodLadder())
			h.workflow = []byte(strings.Replace(string(workflowInterface),
				`"models":[{"class":"Source","component_use":{},"path":"long_form.models.source"}]`, `"models":[]`, 1))
			machine := newTerminalMachines(func(payload map[string]any) *pb.AttemptOutcomeBody {
				if _, present := payload["models"]; present {
					t.Error("model choices leaked into the authored payload")
				}
				return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
			})
			machine.modelOverrides = true
			root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine}, nil)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			slot := "model"
			args := []string{"steps=1"}
			if function == "long_form" {
				slot, args = "generate.models.model", nil
			}
			args = append(args, "--lora", slot+":fl2va_dit=proof/style-a@1.0.0/default,0",
				"--lora", slot+":fl2va_dit=proof/style-b@2.0.0/default,-0.25")
			row, output := rentedRun(t, root, store, "ordered-adapters", function, args...)
			if row.State != "succeeded" {
				t.Fatalf("adapter request did not complete: %+v\n%s", row, output)
			}
			submitted := machine.submitted()
			if len(submitted) != 1 || submitted[0].ReleaseRoot == nil {
				t.Fatalf("expected one release-root submission: %v", submitted)
			}
			choices := submitted[0].ReleaseRoot.Models
			if len(choices) != 1 || choices[0].Parameter != slot || choices[0].Repository != "" {
				t.Fatalf("adapter-only choice changed target/default base: %+v", choices)
			}
			stack := choices[0].Adapters
			if len(stack) != 2 || stack[0].Model != "proof/style-a" || stack[0].Scale != "0" ||
				stack[1].Model != "proof/style-b" || stack[1].Scale != "-0.25" || stack[1].SourceComponent != "adapter" {
				t.Fatalf("ordered adapter selection changed: %+v", stack)
			}
			// A retry with the same ordered stack reuses the request; reversing it
			// changes execution identity even though it selects the same two files.
			_, replay := runCozy(t, root, append([]string{"run", ladderPackage + "/" + function},
				append(args, "--rental=tessa", "--json", "--idempotency-key", "ordered-adapters")...)...)
			if strings.Contains(replay, "different body") || len(machine.submitted()) != 1 {
				t.Fatalf("unchanged stack did not replay: %s", replay)
			}
			reversed := append([]string(nil), args...)
			reversed[len(reversed)-3], reversed[len(reversed)-1] = reversed[len(reversed)-1], reversed[len(reversed)-3]
			code, conflict := runCozy(t, root, append([]string{"run", ladderPackage + "/" + function},
				append(reversed, "--rental=tessa", "--json", "--idempotency-key", "ordered-adapters")...)...)
			if code == 0 || !strings.Contains(conflict, "different body") || len(machine.submitted()) != 1 {
				t.Fatalf("changed stack reused an execution [exit %d]: %s", code, conflict)
			}
		})
	}
}
