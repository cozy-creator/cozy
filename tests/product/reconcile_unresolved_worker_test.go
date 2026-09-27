package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A local worker row journaled before its OS birth identity no longer stops the daemon
// from starting: its devices stay reserved and every other device keeps working.
func TestUnresolvedLocalWorkerReservesItsDevicesWithoutBlockingBoot(t *testing.T) {
	o := hostOwner(t, "reconcile-unresolved")
	fatal(t, o.store.SpawnWorker(records.WorkerProcess{InstanceID: "orphan", Package: "local/test", WorkerID: "local", Devices: []string{"cuda:0"}}))
	_, _, problem := o.c.Reconcile()
	fatal(t, problem)
	if problem := o.store.SpawnWorker(records.WorkerProcess{InstanceID: "second", Package: "local/other", WorkerID: "local", Devices: []string{"cuda:0"}}); problem == nil {
		t.Fatal("the unresolved worker's device was released to another process")
	}
	fatal(t, o.store.SpawnWorker(records.WorkerProcess{InstanceID: "third", Package: "local/other", WorkerID: "local", Devices: []string{"cuda:1"}}))
}
