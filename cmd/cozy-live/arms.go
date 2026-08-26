package main

import (
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/coord"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/service"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// fakeSpec is a worker slot whose process is THIS binary speaking raw protocol. It is a
// real process dialing the real socket over the committed contract — the adversary, not
// a plant.
func fakeSpec(name string, device string, args ...string) coord.EndpointSpec {
	self, err := os.Executable()
	must("locating this binary", err)
	return coord.EndpointSpec{
		Endpoint:   "fake/" + name,
		ReleaseID:  release,
		Generation: "",
		Python:     self,
		Args:       append([]string{"fakeworker"}, args...),
		Devices:    []string{device},
		Bindings: []*coord.Binding{{
			Entrypoint: "fake",
			Record: map[string]any{
				"entrypoint": "fake", "release": release, "slot": name,
			},
		}},
	}
}

func waitEvent(lv *live, substr string, timeout time.Duration) (string, bool) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		for _, line := range lv.c.Events() {
			if strings.Contains(line, substr) {
				return line, true
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
	return "", false
}

func countEvents(lv *live, substr string) int {
	n := 0
	for _, line := range lv.c.Events() {
		if strings.Contains(line, substr) {
			n++
		}
	}
	return n
}

// sectionArms is the refusal matrix, every arm planted by a REAL peer or a REAL second
// process and observed against the real coordinator.
func sectionArms() {
	var e *exit.Error
	lv := hostCoordinator("arms", true)
	defer lv.close()

	head("the service claim: one LocalService per local root")
	first, e1 := service.Hold(lv.l, "127.0.0.1:2699", "sock")
	check("the first claim on this root succeeds", e1 == nil, briefly(e1))
	_, e2 := service.Hold(lv.l, "127.0.0.1:2699", "sock")
	check("a SECOND LocalService on the same root refuses", e2 != nil &&
		e2.Code == exit.Conflict, briefly(e2))
	first.Release()
	third, e3 := service.Hold(lv.l, "127.0.0.1:2699", "sock")
	check("releasing the claim frees the root again", e3 == nil, briefly(e3))
	third.Release()

	head("the device ledger: an envelope cannot be consumed twice")
	victim := fakeSpec("victim", "0", "--arm", "idle")
	instanceA, e := lv.c.StartWorker(victim)
	check("worker A holds device 0", e == nil, briefly(e))
	rival := fakeSpec("rival", "0", "--arm", "idle")
	_, e = lv.c.StartWorker(rival)
	check("a second start against device 0 REFUSES", e != nil && e.Code == exit.Conflict,
		briefly(e))

	// The same statement under real concurrency: two starts racing one envelope.
	var wg sync.WaitGroup
	results := make([]*exit.Error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, results[i] = lv.c.StartWorker(fakeSpec(fmt.Sprintf("racer%d", i), "9", "--arm", "idle"))
		}(i)
	}
	wg.Wait()
	won := 0
	for _, r := range results {
		if r == nil {
			won++
		}
	}
	check("two CONCURRENT starts on one envelope: exactly one wins", won == 1,
		fmt.Sprintf("%d won, refusals: %v", won, briefly(results[0])+" / "+briefly(results[1])))

	planA := planIDOf(victim, "fake")
	if e := lv.c.WaitReady(instanceA, planA); e != nil {
		check("worker A ready", false, e.Message)
		return
	}
	factsA := lv.c.Worker(instanceA)
	check("worker A claimed over raw protocol bytes", factsA.BootID != "",
		"boot "+factsA.BootID)

	head("claim-level refusals (#436: the owner dials, so identity is checked on ClaimAck)")
	// A worker whose ClaimAck reports an instance this owner never spawned for that slot:
	// the owner refuses to bind it and dispatches nothing.
	ghost := fakeSpec("ghost", "7", "--arm", "idle", "--fake-instance", "ins-never-spawned")
	_, _ = lv.c.StartWorker(ghost)
	line, ok := waitEvent(lv, "REFUSING the claimed worker", 10*time.Second)
	check("a foreign instance identity on ClaimAck is refused", ok, trimLog(line))

	// The credential fence, flipped (#463): the OWNER presents the per-spawn bootstrap
	// credential as Claim.proof and the WORKER verifies it constant-time. The badcred arm
	// is a worker that refuses every proof — the typed refusal must surface here.
	badcred := fakeSpec("badcred", "8", "--arm", "badcred")
	_, _ = lv.c.StartWorker(badcred)
	line, ok = waitEvent(lv, "Claim REFUSED", 10*time.Second)
	check("the bootstrap-credential fence refuses typed at Claim", ok, trimLog(line))
	check("worker A's claimed boot is untouched by the refusals",
		lv.c.Worker(instanceA) != nil && lv.c.Worker(instanceA).BootID == factsA.BootID,
		factsA.BootID)

	// A worker reporting a release that is not the pinned one.
	badrel := fakeSpec("badrelease", "6", "--arm", "badrelease")
	_, _ = lv.c.StartWorker(badrel)
	line, ok = waitEvent(lv, "RELEASE_ID_MISMATCH", 10*time.Second)
	check("a release that is not the pinned one is refused", ok, trimLog(line))

	// A second live worker presenting a boot id already bound to worker A.
	collide := fakeSpec("collide", "5", "--arm", "idle", "--session", factsA.BootID)
	_, _ = lv.c.StartWorker(collide)
	line, ok = waitEvent(lv, "boot binding for", 10*time.Second)
	check("two live workers cannot share a worker_boot_id", ok, trimLog(line))

	head("attempt-level refusals: a held attempt, and a stranger's terminal")
	bodyA := payload(map[string]any{"held": true})
	subA := submissionKey(planA, bodyA, "idem-held")
	subA.Endpoint, subA.Entrypoint = "fake/victim", "fake"
	requestA, attemptA, e := lv.c.Submit(subA)
	check("A accepts and HOLDS an attempt", e == nil, briefly(e))
	if e := lv.c.AwaitAccepted(requestA, attemptA, 15*time.Second); e != nil {
		check("A's acceptance journaled", false, e.Message)
		return
	}
	rowA, _ := lv.store.AttemptRow(requestA, int64(attemptA))
	specHex := ""
	if raw, err := canonical.Raw(rowA.InvocationDigest); err == nil {
		specHex = hex.EncodeToString(raw)
	}

	// A NEW ordinal over a live attempt: supersession is explicit, never implicit.
	_, ordErr := lv.store.NextOrdinal(requestA)
	check("a next ordinal over a LIVE attempt refuses", ordErr != nil &&
		strings.Contains(ordErr.Message, "supersession is explicit"), briefly(ordErr))
	// And a re-submit of the same key does not become one: it answers with the request
	// that is already running.
	sameID, sameAtt, e := lv.c.Submit(subA)
	check("re-submitting the same key starts nothing", e == nil && sameID == requestA &&
		sameAtt == attemptA, fmt.Sprintf("%s#%d", sameID, sameAtt))

	// A second session writing another session's attempt row.
	thief := fakeSpec("thief", "4", "--arm", "steal", "--request", requestA,
		"--attempt", fmt.Sprint(attemptA), "--spec", specHex)
	_, _ = lv.c.StartWorker(thief)
	line, ok = waitEvent(lv, "does not own that attempt row", 15*time.Second)
	check("a SECOND WRITER on the same attempt row refuses", ok, trimLog(line))
	rowA2, _ := lv.store.AttemptRow(requestA, int64(attemptA))
	check("the held attempt is untouched by the refusal", rowA2.State == "accepted" &&
		rowA2.TerminalID == "", "state "+rowA2.State)

	// Supersession is EXPLICIT: a cancel naming the full attempt triple, sent to the
	// session that holds it. The idle fake ignores it — which is the point of the
	// cooperative/forceful split on the worker side — but the coordinator's obligation
	// is to send a digest-fenced cancel and nothing else.
	must("cancel", errOf(lv.c.Cancel(requestA, attemptA, pb.CancelReason_CANCEL_REASON_SUPERSEDED, 1000)))
	line, ok = waitEvent(lv, "CancelAttempt "+requestA, 10*time.Second)
	check("supersession is an explicit digest-fenced CancelAttempt", ok, trimLog(line))
	_, ordErr2 := lv.store.NextOrdinal(requestA)
	check("and until its journaled terminal arrives, the ordinal STILL refuses",
		ordErr2 != nil, briefly(ordErr2))

	head("terminal-level refusals: the digest fence, the document fence, the replay")
	badspec := fakeSpec("badterminal", "3", "--arm", "badterminal")
	instanceB, e := lv.c.StartWorker(badspec)
	_ = instanceA
	check("worker B up", e == nil, briefly(e))
	planB := planIDOf(badspec, "fake")
	if e := lv.c.WaitReady(instanceB, planB); e != nil {
		check("worker B ready", false, e.Message)
		return
	}
	subB := submissionKey(planB, payload(map[string]any{"arms": true}), "idem-arms")
	subB.Endpoint, subB.Entrypoint = "fake/badterminal", "fake"
	requestB, attemptB, e := lv.c.Submit(subB)
	check("B is dispatched an attempt", e == nil, briefly(e))

	result, e := lv.c.Await(requestB, attemptB, 30*time.Second)
	_ = result
	check("B's attempt closes on its ONE admissible terminal", e != nil &&
		strings.Contains(e.Message, "no GPU"), briefly(e))

	line, ok = waitEvent(lv, "does not hash the", 5*time.Second)
	check("a planted terminal_digest refuses (recomputed over the resident bytes)", ok, trimLog(line))
	line, ok = waitEvent(lv, "envelope/document divergence", 5*time.Second)
	check("envelope routing copies that disagree with the document refuse", ok, trimLog(line))
	line, ok = waitEvent(lv, "unknown field", 5*time.Second)
	check("a planted key in the terminal document refuses", ok, trimLog(line))
	line, ok = waitEvent(lv, "exact replay", 5*time.Second)
	check("an exact replay is re-acked and applied ONCE", ok, trimLog(line))
	check("exactly one terminal was applied for B", countEvents(lv, "applied in") == 1,
		fmt.Sprintf("%d apply line(s)", countEvents(lv, "applied in")))

	rowB, _ := lv.store.AttemptRow(requestB, int64(attemptB))
	check("B's row carries the ONE terminal that was admissible",
		rowB.TerminalStatus == "FAILED" && rowB.State == "closed",
		fmt.Sprintf("%s/%s, %s", rowB.TerminalStatus, rowB.TerminalCause, rowB.State))
	outs, _ := lv.store.VisibleOutputs(requestB)
	check("a FAILED terminal published no output", len(outs) == 0,
		fmt.Sprintf("%d output(s)", len(outs)))

	head("teardown")
	// Close, not a StopWorker loop: Close sets `closing`, which is what keeps the
	// revive machinery from respawning a worker for the still-owed request while the
	// sweep runs. Stopping without it is a race the Windows runner actually lost —
	// a revived row landed between the last stop and the check (windows-proc run 2).
	lv.c.Close(10 * time.Second)
	rows, _ := lv.store.LiveWorkers()
	names := ""
	for _, row := range rows {
		names += " " + row.InstanceID + "(pid " + fmt.Sprint(row.PID) + ")"
	}
	check("every device grant is released", len(rows) == 0,
		fmt.Sprintf("%d live row(s)%s", len(rows), names))
}

func trimLog(line string) string {
	if i := strings.Index(line, "] "); i >= 0 {
		line = line[i+2:]
	}
	if len(line) > 150 {
		line = line[:150] + "…"
	}
	return line
}
