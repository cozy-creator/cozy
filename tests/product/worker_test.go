package producttest

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	localclient "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/privatepackage"
	"github.com/cozy-creator/cozy/internal/records"
)

type fixedLauncher struct{ spec orchestrator.WorkerLaunchSpec }

func (l fixedLauncher) ResolvePlacement(string) (orchestrator.DesiredPlacement, *exit.Error) {
	return l.spec.Placement, nil
}
func (l fixedLauncher) Resolve(string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return l.spec, nil
}
func (l fixedLauncher) ResolveInstall(string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return l.spec, nil
}
func (l fixedLauncher) PrivateRevision(string, string) (privatepackage.Revision, *exit.Error) {
	return privatepackage.Revision{}, exit.Unavailablef("fixed local launcher has no private package")
}
func (l fixedLauncher) ResolveLogicalInstall(string, string) (orchestrator.LogicalPackage, *exit.Error) {
	return orchestrator.LogicalPackage{}, exit.Unavailablef("fixed local launcher has no remote package")
}
func (l fixedLauncher) ResolveJob(string, string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return l.spec, nil
}
func (l fixedLauncher) ResolveJobInstall(string, string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return l.spec, nil
}

type packageLauncher map[string]orchestrator.WorkerLaunchSpec

func (l packageLauncher) resolve(pkg string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	if spec, ok := l[pkg]; ok {
		return spec, nil
	}
	return orchestrator.WorkerLaunchSpec{}, exit.New(exit.NotFound, "no fake package %s", pkg)
}
func (l packageLauncher) ResolvePlacement(pkg string) (orchestrator.DesiredPlacement, *exit.Error) {
	spec, problem := l.resolve(pkg)
	return spec.Placement, problem
}
func (l packageLauncher) Resolve(pkg string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return l.resolve(pkg)
}
func (l packageLauncher) ResolveInstall(string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return orchestrator.WorkerLaunchSpec{}, exit.New(exit.NotFound, "no fake install")
}
func (l packageLauncher) ResolveLogicalInstall(string, string) (orchestrator.LogicalPackage, *exit.Error) {
	return orchestrator.LogicalPackage{}, exit.New(exit.NotFound, "no fake logical install")
}
func (l packageLauncher) ResolveJob(pkg, _ string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return l.resolve(pkg)
}
func (l packageLauncher) ResolveJobInstall(string, string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return orchestrator.WorkerLaunchSpec{}, exit.New(exit.NotFound, "no fake job install")
}

func waitRequest(t *testing.T, o *owner, requestID, state string, timeout time.Duration) *records.Request {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		row, problem := o.store.RequestRow(requestID)
		fatal(t, problem)
		if row != nil && row.State == state {
			return row
		}
		time.Sleep(10 * time.Millisecond)
	}
	row, _ := o.store.RequestRow(requestID)
	t.Fatalf("request %s did not reach %s: %#v", requestID, state, row)
	return nil
}

func TestProgressWatchOpensAfterSnapshotBarrier(t *testing.T) {
	o := hostOwner(t, "snapshot-watch-order")
	spec := fakeSpec("snapshot-watch-order", "0", "--arm", "snapshotbarrier")
	instance, _, problem := o.c.EnsureWorker(spec)
	fatal(t, problem)
	fatal(t, o.c.EnsurePlacementReady(instance, planIDOf(t, spec)))

	before := []byte("ARM: WatchProgress opened before WorkerSnapshot")
	after := []byte("ARM: WatchProgress opened after WorkerSnapshot")
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		data, err := os.ReadFile(o.c.WorkerLog(instance))
		if err == nil {
			if bytes.Contains(data, before) {
				t.Fatal("the lossy progress connection opened before the durable snapshot barrier")
			}
			if bytes.Contains(data, after) {
				return
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatal("the progress connection did not open after the snapshot barrier")
}

func TestDesiredStatePreconditionFailsWithoutReconnect(t *testing.T) {
	o := hostOwner(t, "desired-precondition")
	spec := fakeSpec("desired-precondition", "0", "--arm", "precondition")
	instance, _, problem := o.c.EnsureWorker(spec)
	fatal(t, problem)

	started := time.Now()
	problem = o.c.EnsurePlacementReady(instance, planIDOf(t, spec))
	if problem == nil || problem.ErrName() != "worker.desired_state_refused" ||
		!strings.Contains(problem.Message, "Runtime package preparation failed") {
		t.Fatalf("FailedPrecondition did not reach the desired-state waiter: %v", problem)
	}
	if elapsed := time.Since(started); elapsed > 3*time.Second {
		t.Fatalf("FailedPrecondition waited %s instead of failing immediately", elapsed)
	}
	time.Sleep(450 * time.Millisecond) // more than two former 200 ms reconnect periods
	if claims := countEvents(o, "ClaimAck boot="); claims != 1 {
		t.Fatalf("permanent desired-state refusal opened %d control sessions", claims)
	}
}

func TestPackageScopedUnloadPreservesUnrelatedWarmWorker(t *testing.T) {
	o := hostOwner(t, "package-scoped-unload")
	for _, id := range []string{"old-editable-install", "current-editable-install"} {
		_, problem := o.store.Activate(records.PackageInstall{
			ID: id, Package: "fake/editable-left", Major: 1, Version: "1.0.0",
			SourceKind: "local", SourceRef: t.TempDir(), SourceDigest: fakeRelease,
			Dir: t.TempDir(),
		})
		fatal(t, problem)
	}
	left, right := fakeSpec("editable-left", "0"), fakeSpec("warm-right", "1")
	left.Placement.InstallID = "old-editable-install"
	leftID, _, problem := o.c.EnsureWorker(left)
	fatal(t, problem)
	fatal(t, o.c.EnsurePlacementReady(leftID, planIDOf(t, left)))
	rightID, _, problem := o.c.EnsureWorker(right)
	fatal(t, problem)
	fatal(t, o.c.EnsurePlacementReady(rightID, planIDOf(t, right)))
	current := fakeSpec("editable-left", "2")
	current.Placement.InstallID = "current-editable-install"
	currentID, _, problem := o.c.EnsureWorker(current)
	fatal(t, problem)
	fatal(t, o.c.EnsurePlacementReady(currentID, planIDOf(t, current)))

	stopped, problem := o.c.UnloadIdleLocalPackage("fake/editable-left", "current-editable-install")
	fatal(t, problem)
	if len(stopped) != 1 || stopped[0].InstanceID != leftID || o.c.Worker(leftID) != nil {
		t.Fatalf("scoped unload did not retire only the edited package: %#v", stopped)
	}
	if worker := o.c.Worker(rightID); worker == nil || worker.Package != "fake/warm-right" {
		t.Fatalf("scoped unload evicted unrelated warm package: %#v", worker)
	}
	if worker := o.c.Worker(currentID); worker == nil || worker.Package != "fake/editable-left" {
		t.Fatalf("scoped unload evicted the current editable generation: %#v", worker)
	}
}

func TestUnloadClientReturnsAfterSuccessfulReclamation(t *testing.T) {
	spec := fakeSpec("unload-client-return", "0")
	acquisitionStarted := make(chan struct{}, 2)
	releaseAcquisition := make(chan struct{})
	o := hostOwnerConfigured(t, "unload-client-return", fixedLauncher{spec},
		func(options *orchestrator.Options) {
			options.RentalFleet = func() (string, *exit.Error) { return "test fleet", nil }
			options.AcquireManagedRental = func(records.Request) (string, string, *exit.Error) {
				acquisitionStarted <- struct{}{}
				<-releaseAcquisition
				return "", "", exit.Unavailablef("test rental acquisition released")
			}
		})
	instance, _, problem := o.c.EnsureWorker(spec)
	fatal(t, problem)
	fatal(t, o.c.EnsurePlacementReady(instance, planIDOf(t, spec)))
	rental := submission(planIDOf(t, spec), "fake/queued-rental",
		"unload-client-queued-rental", map[string]any{"n": 1})
	rental.Rental = true
	submitDone := make(chan *exit.Error, 1)
	go func() {
		_, _, problem := o.c.Submit(rental)
		submitDone <- problem
	}()
	select {
	case <-acquisitionStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("the queued rental did not enter its deliberately blocked acquisition")
	}

	creds, problem := api.Mint(o.l)
	fatal(t, problem)
	server := httptest.NewUnstartedServer(nil)
	localAPI := api.New(api.Options{
		Orchestrator: o.c, Cfg: o.cfg, Creds: creds,
		Addr: server.Listener.Addr().String(),
	})
	handler, problem := localAPI.Handler()
	fatal(t, problem)
	server.Config.Handler = handler
	server.Start()
	t.Cleanup(func() {
		close(releaseAcquisition)
		select {
		case <-submitDone:
		case <-time.After(5 * time.Second):
			t.Error("the test rental submission did not finish after its acquisition was released")
		}
		server.Close()
	})

	client, problem := localclient.Open(o.cfg, daemon.State{
		Up: true, Addr: server.Listener.Addr().String(),
	})
	fatal(t, problem)
	type answer struct {
		result  api.UnloadResult
		problem *exit.Error
	}
	done := make(chan answer, 1)
	go func() {
		result, problem := client.Unload()
		done <- answer{result: result, problem: problem}
	}()

	select {
	case got := <-done:
		fatal(t, got.problem)
		if got.result.Count != 1 || len(got.result.Stopped) != 1 ||
			got.result.Stopped[0].InstanceID != instance || o.c.Worker(instance) != nil {
			t.Fatalf("unload did not return the reclaimed worker: %#v", got.result)
		}
		select {
		case <-acquisitionStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("unload returned without reviving the queue after capacity changed")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("unload reclaimed its worker but synchronously joined unrelated rental acquisition")
	}
}

func TestDeviceEnvelopePressureWaitsThenReclaimsIdleHolder(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "device-envelope-wait")
	holder := fakeSpec("envelope-holder", "0", "--arm", "delayed-output", "--cozy-home", root)
	waiter := fakeSpec("envelope-waiter", "0", "--arm", "delayed-output", "--cozy-home", root)
	o := hostOwnerWithLauncher(t, "device-envelope-wait", packageLauncher{
		holder.Placement.Package: holder,
		waiter.Placement.Package: waiter,
	})

	holderInstance, _, problem := o.c.EnsureWorker(holder)
	fatal(t, problem)
	fatal(t, o.c.EnsurePlacementReady(holderInstance, planIDOf(t, holder)))
	holderRequest, holderAttempt, problem := o.c.Submit(submission(
		planIDOf(t, holder), holder.Placement.Package, "device-holder-active", map[string]any{"n": 1}))
	fatal(t, problem)
	fatal(t, o.c.AwaitAccepted(holderRequest, holderAttempt, 5*time.Second))

	waiterRequest, waiterAttempt, problem := o.c.Submit(submission(
		planIDOf(t, waiter), waiter.Placement.Package, "device-waiter-queued", map[string]any{"n": 2}))
	fatal(t, problem)
	if waiterAttempt != 0 {
		t.Fatalf("device waiter received attempt %d while the holder was active", waiterAttempt)
	}
	if _, ok := waitEvent(o, waiterRequest+" remains QUEUED for the local device envelope", 2*time.Second); !ok {
		t.Fatal("the device waiter did not remain queued behind the active holder")
	}
	if row := waitRequest(t, o, holderRequest, "dispatching", time.Second); row.Ordinal != 1 {
		t.Fatalf("active holder changed while the waiter queued: %#v", row)
	}
	if row := waitRequest(t, o, waiterRequest, "submitted", time.Second); row.Ordinal != 0 {
		t.Fatalf("device waiter minted work before capacity existed: %#v", row)
	}
	if current := o.c.Worker(holderInstance); current == nil || current.Package != holder.Placement.Package {
		t.Fatalf("active holder was preempted under pressure: %#v", current)
	}
	if attempts, readProblem := o.store.Attempts(waiterRequest); readProblem != nil || len(attempts) != 0 {
		t.Fatalf("queued waiter has attempts before capacity: %#v (%v)", attempts, readProblem)
	}
	// A later request for the holder's package owns only its FIFO position. It must not
	// make the holder look active after the current attempt ends and deadlock the waiter
	// ahead of it.
	followerRequest, followerAttempt, problem := o.c.Submit(submission(
		planIDOf(t, holder), holder.Placement.Package, "device-holder-follower", map[string]any{"n": 3}))
	fatal(t, problem)
	if followerAttempt != 0 {
		t.Fatalf("same-package follower overtook the envelope waiter as attempt %d", followerAttempt)
	}

	if result, problem := o.c.AwaitSettled(holderRequest, 10*time.Second); problem != nil || result.Status != "SUCCEEDED" {
		t.Fatalf("holder did not finish normally: %#v (%v)", result, problem)
	}
	if result, problem := o.c.AwaitSettled(waiterRequest, 10*time.Second); problem != nil || result.Status != "SUCCEEDED" {
		t.Fatalf("waiter did not run after the holder became idle: %#v (%v)", result, problem)
	}
	if result, problem := o.c.AwaitSettled(followerRequest, 10*time.Second); problem != nil || result.Status != "SUCCEEDED" {
		t.Fatalf("same-package follower did not run after the FIFO waiter: %#v (%v)", result, problem)
	}
	if countEvents(o, waiterRequest+" FAILED before any offer") != 0 {
		t.Fatal("device pressure was rendered as a terminal failure")
	}
}

func TestOneSlotWorkerQueuesSamePackageAndCancellation(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "one-slot-fifo")
	spec := fakeSpec("one-slot", "0", "--arm", "delayed-output", "--cozy-home", root)
	o := hostOwnerWithLauncher(t, "one-slot-fifo", packageLauncher{spec.Placement.Package: spec})
	instance, _, problem := o.c.EnsureWorker(spec)
	fatal(t, problem)
	fatal(t, o.c.EnsurePlacementReady(instance, planIDOf(t, spec)))

	first, attempt, problem := o.c.Submit(submission(
		planIDOf(t, spec), spec.Placement.Package, "one-slot-0", map[string]any{"n": 0}))
	fatal(t, problem)
	fatal(t, o.c.AwaitAccepted(first, attempt, 5*time.Second))
	waiting := make([]string, 3)
	for i := range waiting {
		waiting[i], attempt, problem = o.c.Submit(submission(
			planIDOf(t, spec), spec.Placement.Package, fmt.Sprintf("one-slot-%d", i+1),
			map[string]any{"n": i + 1}))
		fatal(t, problem)
		if attempt != 0 {
			t.Fatalf("queued request %d received attempt %d", i, attempt)
		}
		waitRequest(t, o, waiting[i], "submitted", time.Second)
	}
	if workers, readProblem := o.store.LiveWorkers(); readProblem != nil || len(workers) != 1 {
		t.Fatalf("same-package queue acquired duplicate workers: %#v (%v)", workers, readProblem)
	}
	for i, requestID := range waiting {
		position, depth := o.c.QueueState(requestID)
		if position != i+1 || depth != len(waiting) {
			t.Fatalf("queue snapshot for %s = %d/%d, want %d/%d",
				requestID, position, depth, i+1, len(waiting))
		}
	}

	// Cancel the FIFO head while it has no attempt. The following request must inherit
	// the queue head and still execute when this one-slot worker reports capacity again.
	fatal(t, o.c.CancelQueued(waiting[0]))
	waitRequest(t, o, waiting[0], "canceled", time.Second)
	if attempts, readProblem := o.store.Attempts(waiting[0]); readProblem != nil || len(attempts) != 0 {
		t.Fatalf("canceled queued request acquired an attempt: %#v (%v)", attempts, readProblem)
	}

	for _, requestID := range append([]string{first}, waiting[1:]...) {
		result, settleProblem := o.c.AwaitSettled(requestID, 10*time.Second)
		if settleProblem != nil || result.Status != "SUCCEEDED" || result.Attempt != 1 {
			t.Fatalf("%s did not execute exactly once: %#v (%v)", requestID, result, settleProblem)
		}
	}
}

func TestReconcilePreservesWorkerWithUnresolvedBirthIdentity(t *testing.T) {
	o := hostOwner(t, "unresolved-birth")
	defer o.close()

	pending := records.WorkerProcess{
		InstanceID: "ins-pending-birth", Package: "fake/pending", PackageRevisionDigest: fakeRelease,
		WorkerID: "local", Devices: []string{"7"},
	}
	fatal(t, o.store.SpawnWorker(pending))
	killed, forgotten, problem := o.c.Reconcile()
	if problem == nil || problem.ErrName() != "worker_birth_identity_unresolved" ||
		killed != 0 || forgotten != 0 {
		t.Fatalf("reconcile unresolved birth = killed %d forgotten %d problem %v",
			killed, forgotten, problem)
	}
	rows, readProblem := o.store.LiveWorkers()
	fatal(t, readProblem)
	if len(rows) != 1 || rows[0].InstanceID != pending.InstanceID ||
		rows[0].State != "spawned_without_birth" {
		t.Fatalf("unresolved worker row was not retained: %#v", rows)
	}
	if problem := o.store.SpawnWorker(records.WorkerProcess{
		InstanceID: "ins-rival", Package: "fake/rival", PackageRevisionDigest: fakeRelease,
		WorkerID: "local", Devices: []string{"7"},
	}); problem == nil || problem.Code != exit.Conflict {
		t.Fatalf("retained unresolved grant did not block a second worker: %v", problem)
	}
	fatal(t, o.store.CloseWorker(pending.InstanceID))
}

// TestWorkerRefusals is the interop proof. `tests/support/fakeworker` is a second, independent
// implementation of the worker protocol that the orchestrator was not co-developed
// against: it hosts WorkerControl, authors real canonical documents, and lies in exactly
// the ways a hostile or broken peer would. Every refusal below is the real orchestrator
// refusing a real process over the committed contract — the arms are observations, not
// plants inside the code under test.
func TestWorkerRefusals(t *testing.T) {
	o := hostOwner(t, "refusals")

	// THE DEVICE LEDGER: an envelope cannot be consumed twice.
	victim := fakeSpec("victim", "0", "--arm", "idle")
	instanceA, _, e := o.c.EnsureWorker(victim)
	fatal(t, e)
	if _, _, e := o.c.EnsureWorker(fakeSpec("rival", "0", "--arm", "idle")); e == nil || e.Code != exit.Conflict {
		t.Errorf("a second start against device 0 was not refused: %s", briefly(e))
	}
	// The same statement under real concurrency: two starts racing one envelope.
	var wg sync.WaitGroup
	results := make([]*exit.Error, 2)
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, _, results[i] = o.c.EnsureWorker(fakeSpec(fmt.Sprintf("racer%d", i), "9", "--arm", "idle"))
		}(i)
	}
	wg.Wait()
	won := 0
	for _, r := range results {
		if r == nil {
			won++
		}
	}
	if won != 1 {
		t.Errorf("two concurrent starts on one envelope: %d won (%s / %s)",
			won, briefly(results[0]), briefly(results[1]))
	}

	planA := planIDOf(t, victim)
	fatal(t, o.c.EnsurePlacementReady(instanceA, planA))
	factsA := o.c.Worker(instanceA)
	if factsA.BootID == "" {
		t.Fatal("worker A never claimed over raw protocol bytes")
	}

	// CLAIM-LEVEL REFUSALS (#436: the owner dials, so identity is checked on ClaimAck).
	for _, arm := range []struct {
		slot, name, device, want string
		args                     []string
	}{
		// Runtime mints its own incarnation, but an unnamed worker cannot own outcomes.
		{"ghost", "an undeclared instance identity", "7", "worker_instance_undeclared",
			[]string{"--arm", "noinstance"}},
		// The credential fence, flipped (#463): the owner presents the per-spawn bootstrap
		// credential as Claim.proof and the worker verifies it constant-time. This worker
		// refuses every proof, so the typed refusal must surface on the owner's side.
		{"badcred", "an unverifiable bootstrap credential", "8", "Claim REFUSED",
			[]string{"--arm", "badcred"}},
		// A worker reporting a release that is not the pinned one.
		{"badrelease", "an unpinned release", "6", "release_mismatch", []string{"--arm", "badrelease"}},
	} {
		_, _, _ = o.c.EnsureWorker(fakeSpec(arm.slot, arm.device, arm.args...))
		line, ok := waitEvent(o, arm.want, 15*time.Second)
		if !ok {
			t.Errorf("%s was not refused (no %q line)", arm.name, arm.want)
		} else {
			t.Logf("refused %s: %s", arm.name, trimLog(line))
		}
	}
	if now := o.c.Worker(instanceA); now == nil || now.BootID != factsA.BootID {
		t.Error("worker A's claimed boot was disturbed by the refusals")
	}

	// A HELD ATTEMPT, and a stranger's terminal over it.
	subA := submission(planA, "fake/victim", "idem-held", map[string]any{"held": true})
	requestA, attemptA, e := o.c.Submit(subA)
	fatal(t, e)
	fatal(t, o.c.AwaitAccepted(requestA, attemptA, 15*time.Second))
	rowA, _ := o.store.AttemptRow(requestA, int64(attemptA))
	specHex := ""
	if raw, err := canonical.Raw(rowA.InvocationDigest); err == nil {
		specHex = hex.EncodeToString(raw)
	}
	// A NEW ordinal over a live attempt: supersession is EXPLICIT, never implicit. The
	// wording is the orchestrator's (it names which live state it is in); the law is that
	// there is no second ordinal while the first is open.
	if _, ordErr := o.store.NextOrdinal(requestA); ordErr == nil {
		t.Error("a next ordinal over a LIVE attempt was minted")
	} else {
		t.Logf("refused a next ordinal over a live attempt: %s", ordErr.Message)
	}
	// A SECOND WRITER on the same attempt row.
	_, _, _ = o.c.EnsureWorker(fakeSpec("thief", "4", "--arm", "steal", "--request", requestA,
		"--attempt", fmt.Sprint(attemptA), "--spec", specHex))
	if line, ok := waitEvent(o, "does not own that attempt row", 20*time.Second); !ok {
		t.Error("a second writer on a held attempt row was not refused")
	} else {
		t.Logf("refused a second writer: %s", trimLog(line))
	}
	if row, _ := o.store.AttemptRow(requestA, int64(attemptA)); row.State != "accepted" || row.TerminalID != "" {
		t.Errorf("the held attempt was disturbed by the refusal: state %s", row.State)
	}

	// TERMINAL-LEVEL REFUSALS: the digest fence, the document fence, the replay. The peer
	// sends five outcomes for one attempt and only the fourth is admissible.
	badspec := fakeSpec("badterminal", "3", "--arm", "badterminal")
	instanceB, _, e := o.c.EnsureWorker(badspec)
	fatal(t, e)
	planB := planIDOf(t, badspec)
	fatal(t, o.c.EnsurePlacementReady(instanceB, planB))
	requestB, attemptB, e := o.c.Submit(submission(planB, "fake/badterminal", "idem-arms",
		map[string]any{"arms": true}))
	fatal(t, e)
	if _, e := o.c.Await(requestB, attemptB, 30*time.Second); e == nil ||
		!strings.Contains(e.Message, "no GPU") {
		t.Errorf("B's attempt did not close on its ONE admissible terminal: %s", briefly(e))
	}
	for _, want := range []string{
		"does not hash the",            // a planted terminal_digest, recomputed over the resident bytes
		"envelope/document divergence", // routing copies that disagree with the document
		"unknown field",                // a planted key in the closed terminal document
		"exact replay",                 // an exact replay is re-acked and applied ONCE
	} {
		if _, ok := waitEvent(o, want, 5*time.Second); !ok {
			t.Errorf("no refusal line matched %q", want)
		}
	}
	if n := countEvents(o, "applied in"); n != 1 {
		t.Errorf("%d terminals were applied for B, wanted exactly 1", n)
	}
	rowB, _ := o.store.AttemptRow(requestB, int64(attemptB))
	if rowB.TerminalStatus != "FAILED" || rowB.State != "closed" {
		t.Errorf("B's row carries %s/%s", rowB.TerminalStatus, rowB.State)
	}
	if outs, _ := o.store.VisibleOutputs(requestB); len(outs) != 0 {
		t.Errorf("a FAILED terminal published %d output(s)", len(outs))
	}
}

// TestDroppedOutcomeAck is the convergence half of the same contract. A worker whose
// AttemptOutcomeAck never arrived keeps replaying its journaled outcome byte for byte; the
// owner's obligation is to apply it ONCE, publish ONCE, and re-ack. It is the one lost-frame
// case with a local representation, and getting it wrong duplicates a user's result.
func TestDroppedOutcomeAck(t *testing.T) {
	o := hostOwner(t, "dropack")
	spec := fakeSpec("dropack", "7", "--arm", "dropack", "--cozy-home", o.root)
	instance, _, e := o.c.EnsureWorker(spec)
	fatal(t, e)
	planID := planIDOf(t, spec)
	fatal(t, o.c.EnsurePlacementReady(instance, planID))
	requestID, attempt, e := o.c.Submit(submission(planID, "fake/dropack", "dropack-1",
		map[string]any{"dropack": true}))
	fatal(t, e)
	if _, e := o.c.Await(requestID, attempt, 60*time.Second); e != nil {
		t.Fatalf("the attempt did not close on its journaled outcome: %s", briefly(e))
	}
	if line, ok := waitEvent(o, "exact replay of a closed outcome", 20*time.Second); !ok {
		t.Error("the replay after the dropped ack was not recognised")
	} else {
		t.Logf("%s", trimLog(line))
	}
	if n := countEvents(o, "applied in"); n != 1 {
		t.Errorf("%d outcomes applied across two identical arrivals, wanted 1", n)
	}
	outs, _ := o.store.VisibleOutputs(requestID)
	if len(outs) != 1 {
		t.Fatalf("%d visible output(s), wanted exactly 1", len(outs))
	}
	if info, err := stat(outs[0].Path); err != nil || info == 0 {
		t.Errorf("the published bytes are not on disk at %s", outs[0].Path)
	}
	row, _ := o.store.AttemptRow(requestID, int64(attempt))
	if row.State != "closed" || row.TerminalStatus != "SUCCEEDED" {
		t.Errorf("the attempt row is %s/%s", row.TerminalStatus, row.State)
	}
}

func TestLocalRuntimeDeathSettlesFromCreatorRecords(t *testing.T) {
	spec := fakeSpec("stateless-runtime-death", "2", "--arm", "idle")
	o := hostOwnerWithLauncher(t, "stateless-runtime-death", fixedLauncher{spec})
	instance, _, problem := o.c.EnsureWorker(spec)
	fatal(t, problem)
	planID := planIDOf(t, spec)
	fatal(t, o.c.EnsurePlacementReady(instance, planID))
	requestID, ordinal, problem := o.c.Submit(submission(planID,
		"fake/stateless-runtime-death", "stateless-runtime-death-1",
		map[string]any{"message": "marco"}))
	fatal(t, problem)
	fatal(t, o.c.AwaitAccepted(requestID, ordinal, 15*time.Second))

	worker := o.c.Worker(instance)
	if worker == nil || worker.PID <= 0 {
		t.Fatalf("local worker has no process identity: %#v", worker)
	}
	process, err := os.FindProcess(worker.PID)
	must(t, err)
	must(t, process.Kill())

	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		attempts, readProblem := o.store.Attempts(requestID)
		fatal(t, readProblem)
		request, readProblem := o.store.RequestRow(requestID)
		fatal(t, readProblem)
		if len(attempts) >= 2 && attempts[0].State == "closed" &&
			attempts[0].TerminalStatus == "ABANDONED" &&
			attempts[0].TerminalCause == "EXECUTOR_INVALIDATED" &&
			request != nil && request.Requeues == 1 && request.Ordinal >= 2 {
			if attempts[0].TerminalBody == nil {
				t.Fatal("Creator persisted no canonical ABANDONED body")
			}
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	attempts, _ := o.store.Attempts(requestID)
	request, _ := o.store.RequestRow(requestID)
	t.Fatalf("Creator did not settle and requeue after Runtime death: attempts=%#v request=%#v",
		attempts, request)
}

// TestLocalSucceededRequiresEveryGrantedOutput is the local half of the output-set fence.
// A local worker has no media peer, but it owes the exact same manifest completeness as a
// remote worker: SUCCEEDED may not silently publish and ack a subset of the granted set.
func TestLocalSucceededRequiresEveryGrantedOutput(t *testing.T) {
	o := hostOwner(t, "missing-granted-output")
	spec := fakeSpec("missing-output", "5", "--arm", "missing-output")
	instance, _, e := o.c.EnsureWorker(spec)
	fatal(t, e)
	planID := planIDOf(t, spec)
	fatal(t, o.c.EnsurePlacementReady(instance, planID))
	requestID, attempt, e := o.c.Submit(submission(planID, "fake/missing-output",
		"missing-output-1", map[string]any{"missing": true}))
	fatal(t, e)
	fatal(t, o.c.AwaitAccepted(requestID, attempt, 15*time.Second))

	if line, ok := waitEvent(o, "terminal omits granted output \"image\"", 15*time.Second); !ok {
		t.Fatal("a local SUCCEEDED outcome missing its granted output was not refused")
	} else {
		t.Logf("%s", trimLog(line))
	}
	row, e := o.store.AttemptRow(requestID, int64(attempt))
	fatal(t, e)
	if row == nil || row.State != "accepted" || row.TerminalID != "" {
		t.Fatalf("the refused terminal changed its attempt row: %#v", row)
	}
	if outputs, e := o.store.VisibleOutputs(requestID); e != nil || len(outputs) != 0 {
		t.Fatalf("the refused terminal exposed outputs: %d (%s)", len(outputs), briefly(e))
	}
}
