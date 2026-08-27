package main

import (
	"fmt"
	"strings"
	"time"
)

// sectionStall is cl-025's arm set: the timer-based kill authority is DELETED (decisions
// #613), and retirement happens on exactly three observed grounds. Each arm is a REAL
// adversary worker speaking raw protocol bytes at the real orchestrator.
//
//  1. A deliberately throttled MULTI-MINUTE fill (longer than the deleted 90 s
//     StallGrace) completes warm-up and serves — nothing kills it.
//  2. A wedged-but-heartbeating worker is retired via its OWN no-progress report.
//  3. A worker that stops heartbeating is retired via liveness.
func sectionStall() {
	lv := hostCoordinator("stall", true)
	defer lv.close()

	stallSub := func(name, planID string) orchestratorSubmission {
		return orchestratorSubmission{
			IdemKey: "idem-stall-" + name, Endpoint: "fake/" + name, Entrypoint: "fake",
			PlanID: planID, Payload: payload(map[string]any{"arm": name}),
		}
	}

	head("ground zero: a throttled multi-minute fill is NOT a stall (the deleted-timer arm)")
	fillMS := flag("fill-ms", "130000")
	slow := fakeSpec("slowfill", "0", "--arm", "slowfill", "--fill-ms", fillMS)
	planSlow := planIDOf(slow, "fake")
	// Submit FIRST, so the request parks at the head of the dispatch queue for the whole
	// fill — the exact shape the 90 s StallGrace used to kill on schedule.
	reqSlow, _, e := lv.c.Submit(stallSub("slowfill", planSlow))
	check("the cold request queues at the head", e == nil &&
		lv.c.QueuePosition(reqSlow) == 1, briefly(e))
	began := time.Now()
	_, _, e = lv.c.EnsureWorker(slow)
	check("the slow-filling worker is up", e == nil, briefly(e))
	fillIntMS := 130000
	fmt.Sscanf(fillMS, "%d", &fillIntMS)
	fillDur := time.Duration(fillIntMS) * time.Millisecond
	res, e := lv.c.AwaitSettled(reqSlow, fillDur+90*time.Second)
	took := time.Since(began)
	check("the request settles SUCCEEDED after the fill", e == nil && res != nil &&
		res.Status == "SUCCEEDED", briefly(e))
	check("the fill ran well past the deleted 90 s StallGrace", took > 100*time.Second,
		fmt.Sprintf("warm-up + serve took %s", took.Round(time.Second)))
	check("no retirement fired during the fill", countEvents(lv, "retiring it:") == 0,
		fmt.Sprintf("%d retirement line(s)", countEvents(lv, "retiring it:")))
	check("exactly one worker was spawned for it", countEvents(lv, "spawned pid") == 1,
		fmt.Sprintf("%d spawn line(s)", countEvents(lv, "spawned pid")))

	head("ground 3: a wedged-but-heartbeating worker retires on its OWN no-progress report")
	wedgedSpec := fakeSpec("wedged", "1", "--arm", "wedged")
	planWedged := planIDOf(wedgedSpec, "fake")
	reqWedged, _, e := lv.c.Submit(stallSub("wedged", planWedged))
	check("the request queues behind the wedged worker", e == nil, briefly(e))
	instWedged, _, e := lv.c.EnsureWorker(wedgedSpec)
	check("the wedged worker is up and heartbeating", e == nil, briefly(e))
	line, ok := waitEvent(lv, "declared itself WEDGED", 3*time.Minute)
	check("the orchestrator retires it on the worker-declared no-progress report", ok, trimLog(line))
	line, ok = waitEvent(lv, "worker "+instWedged+" stopped", time.Minute)
	check("the wedged worker's process is stopped and its grant released", ok, trimLog(line))
	must("canceling the wedged request", errOf(lv.c.CancelQueued(reqWedged)))

	head("ground 2: a worker that stops heartbeating retires via liveness")
	silentSpec := fakeSpec("silent", "2", "--arm", "silent")
	planSilent := planIDOf(silentSpec, "fake")
	reqSilent, _, e := lv.c.Submit(stallSub("silent", planSilent))
	check("the request queues behind the silent worker", e == nil, briefly(e))
	instSilent, _, e := lv.c.EnsureWorker(silentSpec)
	check("the soon-silent worker is up", e == nil, briefly(e))
	line, ok = waitEvent(lv, "missed reports", 2*time.Minute)
	check("the orchestrator retires it on missed heartbeats", ok &&
		strings.Contains(line, instSilent), trimLog(line))
	line, ok = waitEvent(lv, "worker "+instSilent+" stopped", time.Minute)
	check("the silent worker's process is stopped and its grant released", ok, trimLog(line))
	must("canceling the silent request", errOf(lv.c.CancelQueued(reqSilent)))

	head("teardown")
	lv.c.Close(10 * time.Second)
	rows, _ := lv.store.LiveWorkers()
	check("every device grant is released", len(rows) == 0, fmt.Sprintf("%d live row(s)", len(rows)))
}
