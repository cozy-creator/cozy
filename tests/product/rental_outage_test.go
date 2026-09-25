package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

type restartLauncher struct {
	orchestrator.Launcher
	ready *atomic.Bool
	spec  orchestrator.WorkerLaunchSpec
	calls atomic.Int64
}

func TestPodPrepareTransportLossKeepsTheRentalRequestQueued(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, prepareUnavailable: 1}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	o := hostOwner(t, "pod-prepare-outage", rentalWiring(connection, private))
	requestID, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "pod-prepare-outage-1", Package: "acme/outage", Entrypoint: "tile",
		PlanID: podPlanID("acme/outage"), Release: "1.0.0",
		Payload: []byte(`{"size":16}`), Outputs: []string{"image"},
		Worker: podRental, Rental: true, RentalRequired: true,
	})
	fatal(t, problem)
	waitUntil(t, "the first unavailable PodHost preparation", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.prepares) == 1
	})
	if _, ok := waitEvent(o, "PARKED at queue position 1 (lanes none, overtaken 0): PodHost prepare", 5*time.Second); !ok {
		t.Fatal("the transport end did not release the request back to its queue")
	}
	if row, _ := o.store.RequestRow(requestID); row == nil || row.State == "failed" {
		t.Fatalf("transport loss settled the request: %#v", row)
	}

	o.c.WakeQueue()
	waitUntil(t, "the re-issued preparation to reach dispatch", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.prepares) >= 2 && len(pod.offers) == 1
	})
	attempts, problem := o.store.Attempts(requestID)
	fatal(t, problem)
	if len(attempts) != 1 || attempts[0].Attempt != 1 {
		t.Fatalf("preparation retry dispatched %d attempts, want exactly ordinal 1: %#v", len(attempts), attempts)
	}
}

func (l *restartLauncher) Resolve(string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	l.calls.Add(1)
	if !l.ready.Load() {
		return orchestrator.WorkerLaunchSpec{}, exit.Unavailablef("worker launch is held for restart")
	}
	return l.spec, nil
}

// A Hub outage is capacity weather, not the request's answer. The durable queue may be woken any
// number of times without settling or duplicating the request.
func TestRentalHubOutageKeepsTheRequestQueued(t *testing.T) {
	var observations atomic.Int64
	o := hostOwner(t, "rental-hub-outage", func(options *orchestrator.Options) {
		options.RentalFleet = func() (string, *exit.Error) {
			observations.Add(1)
			return "", exit.Unavailablef("Tensorhub is temporarily unreachable")
		}
		options.AcquireManagedRental = func(records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
			t.Fatal("rental acquisition ran without a readable fleet")
			return orchestrator.PlacementDecision{}, "", nil
		}
	})

	spec := fakeSpec("outage-recovery", "0")
	request := submission(planIDOf(t, spec), "fake/outage-recovery", "rental-hub-outage-1",
		map[string]any{"outage": true})
	request.Rental = true
	requestID, _, problem := o.c.Submit(request)
	fatal(t, problem)
	if row, _ := o.store.RequestRow(requestID); row == nil || row.State == "failed" || row.State == "canceled" || row.State == "completed" {
		t.Fatalf("Hub outage settled the durable request instead of queueing it: %#v", row)
	}

	o.c.WakeQueue()
	deadline := time.Now().Add(5 * time.Second)
	for observations.Load() < 2 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if observations.Load() < 2 {
		t.Fatal("a fleet observation did not re-ask the queued request")
	}
	if row, _ := o.store.RequestRow(requestID); row == nil || row.State == "failed" || row.State == "canceled" || row.State == "completed" {
		t.Fatalf("retrying the outage changed the request's durable state: %#v", row)
	}

	fatal(t, o.c.CancelQueued(requestID, "test cleanup"))
	attempts, problem := o.store.Attempts(requestID)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatalf("outage retries minted %d attempt(s) without capacity: %#v", len(attempts), attempts)
	}
}

// The queue itself is memory, but its authority is not. A daemon restart rebuilds the FIFO from
// the request row, and the ordinal gate permits exactly one dispatch when launch becomes possible.
func TestQueuedRequestSurvivesDaemonRestartAndDispatchesOnce(t *testing.T) {
	var ready atomic.Bool
	launcher := &restartLauncher{ready: &ready}
	o := hostOwner(t, "queued-daemon-restart")
	launcher.spec = fakeSpec("queued-daemon-restart", "0", "--arm", "output", "--cozy-home", o.root)
	o.close()

	// This is the crash seam: the submission transaction committed, while no live daemon had
	// yet projected it into its in-memory queue.
	before, problem := records.Open(o.l.DB)
	fatal(t, problem)
	requestID := "req-queued-daemon-restart"
	_, _, problem = before.Submit(records.Request{
		ID: requestID, IdemKey: "queued-daemon-restart-1",
		BodyDigest: "sha256:" + strings.Repeat("ab", 32),
		Package:    "fake/queued-daemon-restart", Entrypoint: "fake",
		PlanID: planIDOf(t, launcher.spec), Payload: []byte(`{"restart":true}`), Outputs: "image",
	})
	fatal(t, problem)
	before.Close()
	ready.Store(true)

	store, problem := records.Open(o.l.DB)
	fatal(t, problem)
	log, err := os.OpenFile(filepath.Join(o.root, "orchestrator-restart.log"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	must(t, err)
	restarted, problem := orchestrator.Open(orchestrator.Options{
		Cfg: o.cfg, Layout: o.l, Store: store, Yield: "smart", Log: log,
		Packages: launcher, MaxOutputMiB: 8,
	})
	fatal(t, problem)
	t.Cleanup(func() {
		restarted.Close(20 * time.Second)
		store.Close()
		log.Close()
	})
	if _, _, problem := restarted.Reconcile(); problem != nil {
		t.Fatal(problem.Message)
	}
	go func() { _ = restarted.Serve() }()
	if _, problem := restarted.AwaitSettled(requestID, 30*time.Second); problem != nil {
		t.Fatalf("restarted daemon did not settle the owed request: %s", briefly(problem))
	}
	attempts, problem := store.Attempts(requestID)
	fatal(t, problem)
	if len(attempts) != 1 || attempts[0].Attempt != 1 {
		t.Fatalf("restart dispatched %d attempts, want exactly ordinal 1: %#v", len(attempts), attempts)
	}
}
