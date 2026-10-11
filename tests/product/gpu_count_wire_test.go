package producttest

import "testing"

// Drive the real controller and its pinned-TLS machine transport. The peer records
// request intent only: this does not pretend to qualify GPU inference.
func TestRunGPUCountReachesNativeSpecOrRefusesBeforePreparation(t *testing.T) {
	for _, count := range []uint32{0, 1, 2} {
		for _, supported := range []bool{false, true} {
			peer := &observerLeasePeer{requestedGPUs: count, gpuCountCapable: supported, completeFirst: true}
			_, store, request, _ := observerLeaseFixture(t, peer)
			if count > 0 && !supported {
				observerEventually(t, func() bool { row, _ := store.RequestRow(request.ID); return row != nil && row.State == "failed" })
				if peer.specs.Load() != 0 {
					t.Fatal("unsupported count reached Run")
				}
				continue
			}
			observerEventually(t, func() bool { row, _ := store.MachineExecution(request.ID); return row != nil && row.Collected })
			peer.mu.Lock()
			observed := append([]uint32(nil), peer.observedGPUs...)
			statuses := len(peer.statusTimes)
			peer.mu.Unlock()
			if len(observed) != 1 || observed[0] != count {
				t.Fatalf("GPU count lost: expected%d, observed%v", count, observed)
			}
			if count == 0 && statuses != 0 {
				t.Fatalf("automatic request requires new peer capability: %d", statuses)
			}
		}
	}
}
