package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

// An explicit model GPU count is exact on a larger rental: `cozy run` on a 4-GPU pod whose
// binding authors a 2-GPU group submits the root to Runtime as exactly 2 GPUs. A Runtime
// that cannot hold a call to an exact count is sent none when the count is the whole
// machine anyway, and refuses the run typed when it is narrower.
func TestExplicitGPUCountIsExactOnALargerRental(t *testing.T) {
	for _, arm := range []struct {
		name      string
		gpus      int
		exact     bool
		submitted uint32
	}{
		{"exact runtime", 2, true, 2},
		{"older runtime, whole machine", 4, false, 0},
		{"older runtime, narrower", 2, false, 0},
	} {
		t.Run(arm.name, func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(hub.PackageBindingRow{Slot: ladderSlot, Model: ladderModel, Release: "1.0.0-rc.1",
				Ladder: []hub.BindingRung{{GPU: "*", GPUs: arm.gpus, Lane: "bf16-full"}}, Revision: 3})
			machine := &runtimeMachine{blocker: "none", exactGPUs: arm.exact, devices: 4}
			root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine, deviceCount: 4}, nil, 2, 4)
			runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json",
				"--idempotency-key", "counted")
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			var row *records.Request
			waitFor(t, root, "the run to reach Runtime or fail", func() bool {
				row, problem = store.RequestByIdempotencyKey("counted")
				return problem == nil && row != nil && (machine.submitted() != nil || row.State == "failed")
			})
			if len(row.Models) != 1 || row.Models[0].GPUs != arm.gpus {
				t.Fatalf("the rental did not select the %d-GPU group: %+v", arm.gpus, row.Models)
			}
			if !arm.exact && arm.gpus < 4 {
				errType, _, message, problem := store.SettledFailure(row.ID)
				fatal(t, problem)
				if machine.submitted() != nil || errType != "machine_execution.worker_upgrade_required" {
					t.Fatalf("a narrower count reached a Runtime that cannot honour it: %s: %s", errType, message)
				}
				return
			}
			set := machine.submitted().PreparedState.GetPlacementSet()
			if set == nil || set.ExecutionGpus != arm.submitted || len(set.DevicePins) != 0 {
				t.Fatalf("submitted execution_gpus %d, pins %v; want exactly %d and no pin",
					set.GetExecutionGpus(), set.GetDevicePins(), arm.submitted)
			}
		})
	}
}
