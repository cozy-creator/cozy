package producttest

import (
	"bytes"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestCapturedSelfServingCallsGiveModelFreeParentItsCPULane(t *testing.T) {
	for _, arm := range []string{"composition", "callee", "no-calls", "model", "weights"} {
		t.Run(arm, func(t *testing.T) {
			code := bytes.Repeat([]byte{3}, 32)
			build, err := canonical.Spell(code)
			must(t, err)
			capture := &pb.MachineExecutionCapture{RootInstallationId: build,
				Bindings: []*pb.MachineCallableBinding{{CallerInstallationId: build, CalleeInstallationId: build, Module: "model_tools", Export: "segment", Entrypoint: "segment"}}}
			request := records.Request{ID: "root", IdemKey: "root", Kind: "job", Package: "local/tools", Entrypoint: "compose", PlanID: childDigest("9"), LocalInstallationID: build, Payload: []byte(`{}`), Org: "local"}
			plan := &orchestrator.JobPlan{Function: "compose", DescriptorID: request.PlanID, InstallationID: build, NeedsAccelerator: true, RSSCap: orchestrator.DefaultJobRSSCap}
			if arm == "callee" {
				request.Entrypoint = "segment"
			}
			if arm == "no-calls" {
				capture.Bindings = nil
			}
			if arm == "model" {
				request.Models = []records.ModelRef{{Slot: "model", Manifest: childDigest("5"), ManifestLength: 123}}
			}
			if arm == "weights" {
				plan.WeightsOutputs = []orchestrator.WeightsOutput{{OutputID: "weights", MaxBytes: 1024}}
			}
			raw, digest, err := canonical.Identity(capture)
			must(t, err)
			submitted, problem := orchestrator.MachineJobSubmission(request, localpackage.ExecutionCapture{Canonical: raw, Digest: digest}, plan, nil)
			fatal(t, problem)
			job := submitted.PreparedState.GetJob()
			cpu := arm == "composition"
			if job.Orchestration != cpu || (job.DeviceCount == 0) != cpu || job.ResourceCaps.DeviceRequired == cpu {
				t.Fatalf("%s has wrong execution lane: %+v", arm, job)
			}
			if cpu && job.ResourceCaps.MaxRssBytes != 0 {
				t.Fatal("composition retained the legacy arbitrary RSS cap")
			}
			if !plan.NeedsAccelerator {
				t.Fatal("submission changed the shared resolved plan")
			}
		})
	}
}
