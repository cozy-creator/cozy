package producttest

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Run 1223 (2026-09-27): a client script with Model inputs and weights outputs was placed
// as a device-bearing root, so the GPU child it called was refused ("managed child cannot
// coexist with an ancestor on the device lane"). A device-less root takes the machine's CPU
// slot when its Runtime reports `cpu_slot_model_inputs`; a Runtime without it keeps the root
// on its device lane, the only placement it admits for Models and weights. The root here is
// a CPU job of a torch package holding a Model input and a weights output, calling a
// published GPU callee, submitted through the real binary and daemon to a stand-in pod.
func TestDeviceLessRootHoldingModelsOrchestratesOnACapableRuntime(t *testing.T) {
	for _, arm := range []struct {
		name          string
		capable, want bool
	}{{"capable", true, true}, {"older runtime", false, false}} {
		t.Run(arm.name, func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(goodLadder())
			facts := publishCalleeReleaseOf(t, h, `{"application":"h3:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[{"accelerator":false,`+
				`"models":[{"class":"Source","component_use":{},"path":"long_form.models.source"}],"name":"long_form","publishes":false,"request":{"fields":[]},"result":{"fields":[]},`+
				`"weights_outputs":[{"max_bytes":1048576,"mime_type":"application/vnd.cozy.model-manifest","output_id":"adapters"}]}]}`)
			root := ladderRoot(t, h)
			layout, problem := home.Open(root)
			fatal(t, problem)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			identity, problem := rental.PendingCreatorIdentity(layout, "pass-through-root")
			fatal(t, problem)
			public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
			must(t, err)
			machine := &runtimeMachine{blocker: "none", cpuSlotModelInputs: arm.capable}
			pod := &fakePod{controlKey: public, machine: machine, deviceCount: 1,
				preparedPlacement: func(download []byte, pkg, release string) *pb.Placement {
					placement := podPlacement(download, pkg, release, "")
					placement.PackageInterface = facts[pkg].iface
					return placement
				}}
			connection, certPath := startFakePod(t, root, pod)
			cert, err := os.ReadFile(certPath)
			must(t, err)
			h.mu.Lock()
			h.rentals[podRental] = map[string]any{"rental_id": podRental, "name": "tessa", "state": "ready",
				"requested_accelerator_model": "fake-4090", "accelerator_count": 1, "hourly_rate_usd_micros": 1,
				"worker_address": connection.Addr, "media_address": connection.Media.Addr}
			h.mu.Unlock()
			fatal(t, rental.Attach(layout, store, records.Rental{ID: podRental, MachineName: "tessa", SKU: "fake-x", State: "ready",
				AcceleratorModel: "fake-4090", AcceleratorCount: 1, HourlyRateUSDMicros: 1, Hub: h.server.URL,
				Address: connection.Addr, MediaAddress: connection.Media.Addr,
				ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}, string(cert), connection.Media.Token, identity))
			store.Close()
			startDaemonProcess(t, root)

			const key = "pass-through-root"
			if code, out := runCozy(t, root, "run", ladderPackage+"/long_form", "--model.source="+ladderModel+"@"+ladderRelease+"/"+ladderLane,
				"--rental=tessa", "--json", "--idempotency-key", key); code != 0 {
				t.Fatalf("the rented job was refused [exit %d]: %s", code, out)
			}
			store, problem = records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			waitFor(t, root, "the Runtime submission or a terminal run", func() bool {
				row, _ := store.RequestByIdempotencyKey(key)
				return machine.submitted() != nil || row != nil && records.Settled(row.State)
			})
			if machine.submitted() == nil {
				_, show := runCozy(t, root, "run", "show", "1", "--json")
				t.Fatalf("the job ended before its submission:\n%s\n%s", show, tail(filepath.Join(root, "daemon.log")))
			}
			row, problem := store.RequestByIdempotencyKey(key)
			fatal(t, problem)
			if row.NeedsAccelerator || len(row.OwnModels()) != 1 {
				t.Fatalf("the root is not a CPU job holding its own Model: accelerator %v, models %+v", row.NeedsAccelerator, row.Models)
			}
			directive := machine.submitted().PreparedState.GetJob()
			if directive == nil || directive.DeviceCount != 0 || directive.ResourceCaps.GetDeviceRequired() {
				t.Fatalf("the root directive grants a device: %+v", directive)
			}
			// Runtime admits a device child only beneath a CPU-slot ancestor.
			if directive.Orchestration != arm.want {
				t.Fatalf("orchestration %v on a Runtime reporting cpu_slot_model_inputs=%v", directive.Orchestration, arm.capable)
			}
		})
	}
}
