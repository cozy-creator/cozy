package producttest

import (
	"context"
	"encoding/json"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// A machine names its independent agent release in --version, DescribeMachine, and its
// sealed readiness receipt. The personal controller's version is not the machine version.
func TestTheMachineReportsItsAgentRelease(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the machine reports its own release")
	}
	out, err := exec.Command(*machineHostBinary, "--version").Output()
	must(t, err)
	var version struct{ Version string }
	must(t, json.Unmarshal(out, &version))
	want := version.Version
	root := t.TempDir()
	provisionMachine(t, root)
	layout, problem := home.Open(root)
	fatal(t, problem)
	found := &machines.Resolver{Host: machines.NewHost(layout.Machine, "", nil), HubOrigin: testDefaultHub,
		UseRental: func(string, orchestrator.Holder) (func(), *exit.Error) { return func() {}, nil },
		RentalKey: func(string) (rental.CreatorIdentity, *exit.Error) { return rental.CreatorIdentity{}, nil }}
	machine, problem := found.Dial(context.Background(), machines.Local, orchestrator.Holder{What: "describing it"})
	fatal(t, problem)
	described, err := machine.Host.DescribeMachine(context.Background(), &pb.DescribeMachineQuery{Claim: machine.Claim})
	machine.Close()
	must(t, err)
	var envelope struct{ Payload []byte }
	var payload struct {
		MachineVersion string `json:"machine_version"`
	}
	raw, err := os.ReadFile(filepath.Join(layout.Machine, "root/run/cozy/bootstrap/readiness-envelope.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &envelope))
	must(t, json.Unmarshal(envelope.Payload, &payload))
	if want == "" || described.GetHost().GetVersion() != want || payload.MachineVersion != want {
		t.Fatalf("`--version` says %q; DescribeMachine says %q and the receipt %q", want,
			described.GetHost().GetVersion(), payload.MachineVersion)
	}
}
