package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// `cozy down --all` ends what it can. A rental whose hub is gone, or one that never
// finishes draining, is reported with its reason; the daemon still stops and the verb
// exits non-zero instead of retrying forever.
func TestDownAllStopsWhenRentalsCannotBeEnded(t *testing.T) {
	for _, mode := range []string{"hub-unreachable", "never-drains"} {
		t.Run(mode, func(t *testing.T) {
			var root, hubURL, id string
			if mode == "hub-unreachable" {
				hubURL = closedHub(t)
				root = hubHome(t, "down-all-unreachable", hubURL)
				id = "pr-downallunreachable0001"
			} else {
				var peer *fakeRentalHub
				root, hubURL, peer = rentalEndRoot(t, "down-all-draining")
				id = "pr-downalldraining00001"
				peer.add(id, "draining-machine")
				peer.setState(id, "release_requested", "")
			}
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1, ID: id, MachineName: "downall",
				SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000, State: "ready",
				Hub: hubURL, Address: "127.0.0.1:1", CertPath: filepath.Join(root, id+".pem")}))
			store.Close()
			if code, out := runCozy(t, root, "up"); code != 0 {
				t.Fatalf("up: [%d] %s", code, out)
			}
			code, out := runCozy(t, root, "down", "--all")
			if code == 0 {
				t.Fatalf("down --all exited 0 with a rental it could not end\n%s", out)
			}
			if !strings.Contains(out, id) || !strings.Contains(out, "NOT ended") {
				t.Fatalf("down --all did not name the unended rental and why\n%s", out)
			}
			if pids := daemonPidsOn(root); len(pids) != 0 {
				t.Fatalf("daemon still running after down --all: %v\n%s", pids, out)
			}
		})
	}
}
