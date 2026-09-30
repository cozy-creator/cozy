package producttest

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Provider credentials enter through the daemon's environment, then travel only
// in the transient submission. The selected source remains an ordinary capture
// fact; the credential must not become a replay, database, config or log fact.
func TestLoRAProviderCredentialStaysOutOfDurableClientState(t *testing.T) {
	const token = "fake-lora-source-token-9e3c7641"
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
		return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
	})
	machine.modelOverrides = true
	root, layout := rentedLadderHome(t, h, &fakePod{machine: machine}, nil)
	startDaemonProcess(t, root, "HF_TOKEN="+token)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	source := "hf://proof/style@" + strings.Repeat("c", 40)
	row, output := rentedRun(t, root, store, "provider-adapter", "generate", "steps=1",
		"--lora", "model:fl2va_dit="+source+",0.5")
	if row.State != "succeeded" {
		t.Fatalf("provider adapter request failed: %s\n%s", row.State, output)
	}
	submitted := machine.submitted()
	if len(submitted) != 1 || submitted[0].ReleaseRoot == nil {
		t.Fatal("provider adapter did not reach one machine submission")
	}
	want := []*pb.SourceCredential{{Provider: pb.NativeSourceOperation_NATIVE_SOURCE_OPERATION_HUGGINGFACE,
		Credential: "bearer " + token}}
	if !credentialsEqual(submitted[0].SourceCredentials, want) {
		t.Fatal("provider adapter did not carry the configured credential on the wire")
	}
	choice := submitted[0].ReleaseRoot.Models
	if len(choice) != 1 || len(choice[0].Adapters) != 1 || choice[0].Adapters[0].Source != source {
		t.Fatal("provider adapter changed the captured source selection")
	}
	link, problem := store.MachineExecution(row.ID)
	fatal(t, problem)
	if link == nil || bytes.Contains(link.Submission, []byte(token)) || strings.Contains(output, token) {
		t.Fatal("provider adapter credential became frozen submission or CLI output")
	}
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("daemon shutdown failed: %s", out)
	}
	must(t, filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || info.Mode()&os.ModeType != 0 {
			return err
		}
		body, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if bytes.Contains(body, []byte(token)) {
			t.Errorf("provider adapter credential was persisted in %s", path)
		}
		return nil
	}))
}

func TestYAMLModelOverrideCarriesBaseAndAdaptersOnlyToItsNamedChild(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	h.workflow = []byte(strings.Replace(string(workflowInterface),
		`"models":[{"class":"Source","component_use":{},"path":"long_form.models.source"}]`, `"models":[]`, 1))
	machine := newTerminalMachines(func(payload map[string]any) *pb.AttemptOutcomeBody {
		if len(payload) != 0 {
			t.Error("YAML model selection became an authored job argument")
		}
		return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
	})
	machine.modelOverrides = true
	root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine}, nil)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	input := filepath.Join(t.TempDir(), "scene.yaml")
	must(t, os.WriteFile(input, []byte(`models:
  generate.models.model:
    ref: proof/base@1.0.0/bf16
    lora:
      - ref: proof/style-a@1.0.0/default
        weight: 0.50
        component: fl2va_dit
      - ref: proof/style-b@2.0.0/default
        weight: -0.25
        component: fl2va_dit
`), 0600))
	row, output := rentedRun(t, root, store, "yaml-adapters", "long_form", "--input", input)
	if row.State != "succeeded" {
		t.Fatalf("YAML model selection failed: %s\n%s", row.State, output)
	}
	submitted := machine.submitted()
	if len(submitted) != 1 || submitted[0].ReleaseRoot == nil || len(submitted[0].ReleaseRoot.Models) != 1 {
		t.Fatalf("YAML created additional model targets: %+v", submitted)
	}
	choice := submitted[0].ReleaseRoot.Models[0]
	if choice.Parameter != "generate.models.model" || choice.Repository != "proof/base" || choice.Release != "1.0.0" || choice.Lane != "bf16" ||
		len(choice.Adapters) != 2 || choice.Adapters[0].Scale != "0.5" || choice.Adapters[1].Scale != "-0.25" {
		t.Fatalf("YAML changed the selected checkpoint or stack: %+v", choice)
	}
}

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
			// Mix equivalent exact and short spellings in one stack. Grouping by
			// the raw spelling would silently reverse these two adapters.
			args = append(args, "--lora", ladderPackage+"/generate.models.model:fl2va_dit=proof/style-a@1.0.0/default,0",
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
