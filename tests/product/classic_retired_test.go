package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// Unfinished work that only the retired classic worker session could run ends once,
// named, at daemon start; machine executions are untouched.
func TestDaemonStartRetiresClassicWork(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	blocked := recordPrivateTransaction(t, store, "classic-blocked", "")
	_, problem = store.BlockRetainedWork(blocked.ID, "source.not_ready", "input is not available yet")
	fatal(t, problem)
	queued := recordPrivateTransaction(t, store, "classic-queued", "")
	canceling := recordPrivateTransaction(t, store, "classic-canceling", "")
	heldFinalization(t, store, canceling)
	machine := records.Request{ID: "req-machine-local", IdemKey: "idem-machine-local",
		BodyDigest: blocked.BodyDigest, Package: blocked.Package, Entrypoint: blocked.Entrypoint,
		Kind: "job", Org: "local", Release: blocked.Release, PlanID: blocked.PlanID,
		LocalInstallationID: blocked.LocalInstallationID, Payload: blocked.Payload,
		RetainWork: true, MachineExecutionObserver: true}
	_, _, problem = store.Submit(machine)
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(machine.ID, "local"))
	store.Close()

	store, problem = records.OpenForDaemon(path)
	fatal(t, problem)
	defer store.Close()
	retired, problem := store.RetireClassicWork()
	fatal(t, problem)
	if len(retired) != 3 {
		t.Fatalf("retired %v, want the three classic requests", retired)
	}
	pending, problem := store.PendingWeightsFinalizations(canceling.ID, 1)
	fatal(t, problem)
	if len(pending) != 0 {
		t.Fatalf("classic local custody stayed held: %+v", pending)
	}
	for id, want := range map[string]string{blocked.ID: "failed", queued.ID: "failed", canceling.ID: "canceled", machine.ID: ""} {
		row, problem := store.RequestRow(id)
		fatal(t, problem)
		if want == "" {
			if row == nil || row.State == "failed" || !row.RetainWork {
				t.Fatalf("machine work was retired: %+v", row)
			}
			continue
		}
		if row == nil || row.State != want || row.RetainWork {
			t.Fatalf("%s = %+v, want %s", id, row, want)
		}
	}
	events, problem := store.EventsAfter(blocked.ID, 0, 100)
	fatal(t, problem)
	last := events[len(events)-1]
	if last.Type != "request.failed" || last.Payload["error_type"] != records.ClassicRetiredCode {
		t.Fatalf("classic retirement was not named: %+v", last)
	}
	again, problem := store.RetireClassicWork()
	fatal(t, problem)
	if len(again) != 0 {
		t.Fatalf("retirement was not idempotent: %v", again)
	}
}

// Custody tied to a rental this host knows has ended is forgotten at daemon start; custody
// on a live rental stays held.
func TestDaemonStartForgetsEndedRentalCustody(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	defer store.Close()
	live := records.Rental{ID: "pr-live0000000000000000", MachineName: "alive", SKU: "h100-80", AcceleratorModel: "NVIDIA H100 80GB HBM3",
		AcceleratorCount: 1, HourlyRateUSDMicros: 1, State: "ready", Address: "127.0.0.1:1", CertPath: "/unused.pem", Hub: "http://127.0.0.1:1"}
	fatal(t, store.RecordRental(live))
	pending := func(label, rental string) records.Request {
		request := recordPrivateTransaction(t, store, label, rental)
		heldFinalization(t, store, request)
		return request
	}
	gone := pending("ended-rental", "pr-gone0000000000000000")
	kept := pending("live-rental", live.ID)
	fatal(t, store.ForgetEndedRentalCustody())
	for request, want := range map[string]int{gone.ID: 0, kept.ID: 1} {
		held, problem := store.PendingWeightsFinalizations(request, 1)
		fatal(t, problem)
		if len(held) != want {
			t.Fatalf("%s holds %d pending finalization(s), want %d", request, len(held), want)
		}
	}
}

// heldFinalization gives a request one closed attempt and a pending weights finalization,
// the custody a canceled run leaves until a store releases it.
func heldFinalization(t *testing.T, store *records.Store, request records.Request) {
	t.Helper()
	invocation := "sha256:" + strings.Repeat("e", 64)
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "ins-" + request.ID, Package: request.Package, WorkerID: "worker"}))
	ordinal, problem := store.Dispatch(records.Attempt{RequestID: request.ID, InstanceID: "ins-" + request.ID, SessionID: "boot",
		InvocationDigest: invocation, InvocationCanonical: []byte(`{}`)})
	fatal(t, problem)
	fatal(t, store.AbortDispatch(request.ID, ordinal, "boot", "fixture"))
	fatal(t, store.RequestRetainedCancellation(request.ID, "owner"))
	fatal(t, store.RecordRetainedFinalization(records.WeightsFinalization{RequestID: request.ID, Attempt: ordinal,
		InstanceID: "ins-" + request.ID, OwnerScope: "owner", InvocationDigest: invocation, OutputSlot: "weights"}))
	if held, problem := store.PendingWeightsFinalizations(request.ID, ordinal); problem != nil || len(held) != 1 {
		t.Fatalf("%s recorded no pending finalization: %+v %v", request.ID, held, problem)
	}
}
