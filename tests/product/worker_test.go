package producttest

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
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
func (l fixedLauncher) ResolveLogicalInstall(string, string) (orchestrator.LogicalPackage, *exit.Error) {
	return orchestrator.LogicalPackage{}, exit.Unavailablef("fixed local launcher has no remote package")
}
func (l fixedLauncher) ResolveJob(string, string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return l.spec, nil
}
func (l fixedLauncher) ResolveJobInstall(string, string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return l.spec, nil
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
		// An instance identity this owner never spawned for that slot.
		{"ghost", "a foreign instance identity", "7", "REFUSING the claimed worker",
			[]string{"--arm", "idle", "--fake-instance", "ins-never-spawned"}},
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
