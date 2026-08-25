package main

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// THE FOUR M4 ARMS, from the product seat, on the FOUR-COMPONENT PIPELINE.
//
// M4's integrated proof is "weightless endpoint plus real sdxl-class load through
// cozy-creator; duplicate attempt, dropped TerminalAck, non-cooperative cancel, and
// restart recovery proved". cl-001 and cl-006 armed each of these at a lower layer against
// a single-component fixture. What this section adds is the seat and the subject: a real
// HTTP client and the real `cozy` binary, against an endpoint whose 6.46 GiB of weights do
// not fit beside their own activations, so every arm crosses the residency ladder too.
//
//	1  duplicate attempt      one key, two callers, a LIVE attempt -> one attempt, one output
//	2  dropped TerminalAck    the ack never lands -> the terminal replays -> applied ONCE
//	3  non-cooperative cancel  a handler that ignores cancel -> grace -> kill -> CANCELED
//	4  coordinator restart     kill -9 mid-attempt -> the recovered-attempts law -> one output

func sectionM4Arms() {
	idle := waitQuietGPU(45 * time.Minute)
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", "cl003-arms"))
	must("clearing the root", os.RemoveAll(root))
	must("creating the root", os.MkdirAll(root, 0o755))

	installPipeline(root, pipeRef, pipeRef+"@cr-008b", "plain-fp16", pipeSnapshots())
	port := freePort(2970)
	svc := startService(root, port, false)
	defer svc.stop()
	check("a real LocalService, in its own process", svc.alive(),
		fmt.Sprintf("pid %d on %s", svc.cmd.Process.Pid, svc.addr))

	armDuplicate(svc, root)
	armNonCooperativeCancel(svc, root, idle)
	armCoordinatorRestart(svc, root, port, idle)

	head("the card")
	// The service is still up here (its stop is deferred), so the worker has to be drained
	// EXPLICITLY before the card is read. Reading it first measures our own resident
	// worker and calls it a leak.
	code, out := cozyRun(root, "stop", "--all")
	check("stop --all drains every worker", code == 0, firstLine(out))
	used := gpuReleased(idle, 120*time.Second)
	check("the GPU is back at its baseline after four arms", used <= idle+40,
		fmt.Sprintf("%d MiB (idle was %d)", used, idle))
}

// ------------------------------------------------------------------ 1: duplicate attempt

func armDuplicate(svc *liveService, root string) {
	head("ARM 1 — a duplicate attempt: one key, two callers, a LIVE attempt")
	body := map[string]any{"steps": 24, "size": 1024, "seed": 33}
	key := "cl003-dup-" + itoa(int(time.Now().Unix()))
	first := svc.call("POST", "/v1/requests", map[string]any{
		"endpoint": pipeRef, "function": "generate", "input": body,
	}, "Idempotency-Key", key)
	check("the first submit is accepted", first.Status == http.StatusAccepted, first.brief())
	requestID, _ := first.json()["request_id"].(string)

	// The duplicate must land while the attempt is RUNNING. A second submit after the
	// request settled would prove idempotency, not "one attempt".
	live := svc.openUntilLive("/v1/requests/"+requestID+"/events", 300*time.Second)
	check("the attempt is accepted and running before the duplicate arrives",
		live.find("request.accepted") != nil && !live.stopped,
		strings.Join(live.types(), " · "))

	second := svc.call("POST", "/v1/requests", map[string]any{
		"endpoint": pipeRef, "function": "generate", "input": body,
	}, "Idempotency-Key", key)
	check("the SAME key + the SAME body answers with the SAME request, mid-flight",
		second.json()["request_id"] == requestID && second.json()["idempotent_replay"] == true,
		second.brief())
	// And through the product binary, which is the seat a person is in.
	code, doc, raw := cozyJSON(root, "run", pipeRef+"/v1/generate", "steps=24", "size=1024",
		"seed=33", "--idempotency-key", key)
	check("`cozy run` under the same key joins the running request rather than starting one",
		code == 0 && fmt.Sprint(doc["request"]) == requestID,
		fmt.Sprintf("exit %d, request %v", code, doc["request"]))
	if code != 0 {
		fmt.Println(indent(raw))
	}
	changed := svc.call("POST", "/v1/requests", map[string]any{
		"endpoint": pipeRef, "function": "generate", "input": map[string]any{"steps": 25},
	}, "Idempotency-Key", key)
	check("the same key with a CHANGED body refuses before any worker work",
		changed.Status == http.StatusConflict, changed.brief())

	final := svc.call("GET", "/v1/requests/"+requestID, nil)
	attempts := int(asFloat(final.json()["attempts"]))
	outs := asList(final.json()["outputs"])
	check("ONE attempt ran and ONE output became visible, for three callers of one key",
		attempts == 1 && len(outs) == 1,
		fmt.Sprintf("%d attempt(s), %d output(s), status %v", attempts, len(outs),
			final.json()["status"]))
}

// ------------------------------------------------------------------ 2: dropped TerminalAck

// sectionDropAck is ARM 2, and it stands alone because its instrument is the PROTOCOL
// PEER rather than the HTTP client.
//
// LOCALLY a worker cannot outlive its coordinator: `Coordinator.Reconcile` kills every
// orphan worker at boot, and the SO_PEERCRED slot fence refuses to adopt a process this
// coordinator did not start. So "the same worker replays its terminal to a restarted
// coordinator" has no local representation at all — on a pod it is the recovery path
// (cl-014/th-013), and locally the recovery path is the requeue projection (ARM 4).
//
// What a lost TerminalAck looks like to the COORDINATOR, though, is exactly one thing: a
// peer that keeps replaying its journaled terminal until it is acknowledged. That is what
// this fake worker does — real protocol bytes, a real output file under the real grant,
// the identical canonical document twice — and the obligation under test is the
// coordinator's: apply it once, publish once, re-ack.
func sectionDropAck() {
	lv := hostCoordinator("cl003-dropack", true)
	defer lv.close()

	head("ARM 2 — the TerminalAck is dropped: the terminal replays and applies ONCE")
	spec := fakeSpec("dropack", "7", "--arm", "dropack", "--cozy-home", lv.root)
	instance, e := lv.c.StartWorker(spec)
	if !check("the peer registers over the committed contract", e == nil, briefly(e)) {
		return
	}
	planID := planIDOf(spec, "fake")
	if e := lv.c.WaitReady(instance, planID, 30*time.Second); e != nil {
		check("the peer is ready", false, e.Message)
		return
	}
	sub := submissionKey(planID, payload(map[string]any{"dropack": true}), "cl003-dropack")
	sub.Endpoint, sub.Entrypoint = "fake/dropack", "fake"
	requestID, attempt, e := lv.c.Submit(sub)
	if !check("an attempt is dispatched", e == nil, briefly(e)) {
		return
	}
	_, e = lv.c.Await(requestID, attempt, 60*time.Second)
	check("the attempt closes on its journaled terminal", e == nil, briefly(e))

	line, ok := waitEvent(lv, "exact replay of a closed terminal", 20*time.Second)
	check("the REPLAY after the dropped ack is recognised and applied NOTHING twice", ok,
		trimLog(line))
	check("exactly ONE terminal was applied across two identical arrivals",
		countEvents(lv, "applied in") == 1,
		fmt.Sprintf("%d apply line(s), %d replay line(s)",
			countEvents(lv, "applied in"), countEvents(lv, "exact replay")))
	outs, _ := lv.store.VisibleOutputs(requestID)
	check("and the output the terminal carried is visible exactly ONCE",
		len(outs) == 1, fmt.Sprintf("%d visible output(s)", len(outs)))
	if len(outs) == 1 {
		info, err := os.Stat(outs[0].Path)
		check("the published bytes are on disk under the coordinator's own grant",
			err == nil && info.Size() > 0,
			fmt.Sprintf("%s (%s)", outs[0].MediaID, sizeOf(info)))
	}
	row, _ := lv.store.AttemptRow(requestID, int64(attempt))
	check("the attempt row is CLOSED on one terminal id",
		row.State == "closed" && row.TerminalStatus == "SUCCEEDED",
		fmt.Sprintf("%s/%s %s", row.TerminalStatus, row.TerminalCause, row.State))

	head("teardown")
	live, _ := lv.store.LiveWorkers()
	for _, w := range live {
		lv.c.StopWorker(w.InstanceID, 10*time.Second)
	}
	rows, _ := lv.store.LiveWorkers()
	check("every device grant is released", len(rows) == 0, fmt.Sprintf("%d live row(s)", len(rows)))
}

// ------------------------------------------------------------- 3: non-cooperative cancel

func armNonCooperativeCancel(svc *liveService, root string, idle int) {
	head("ARM 3 — an endpoint that IGNORES cancel: grace expiry, kill, typed CANCELED")
	// A FRESH worker first. ARM 1's 1024px attempt leaves the generation's residency
	// drifted (the cr-008b defect cl-003 reports), and an arm about CANCELLATION must not
	// be answered by an unrelated shortfall.
	code, _ := cozyRun(root, "stop", "--all")
	check("the worker from ARM 1 drains", code == 0, "")
	// `stubborn` runs UNet passes in a loop and never calls `ctx.raise_if_cancelled()`.
	// A cooperative handler converges on the next check; this one can only be ended by the
	// runtime, which is what the arm is about. The window comes from the PASS COST, not
	// from a big step count: the endpoint's own schema caps `steps` at 50 (and refuses 300
	// as `INVALID_REQUEST` before any GPU work — observed), so the arm buys its ~15 s from
	// 1024px passes at roughly 310 ms each. `stubborn` touches only the UNet, so there is
	// no 1024px decode in it to run out of room.
	body := map[string]any{"steps": 50, "size": 1024, "seed": 7}
	submitted := svc.call("POST", "/v1/requests", map[string]any{
		"endpoint": pipeRef, "function": "stubborn", "input": body,
	}, "Idempotency-Key", "cl003-stubborn-"+itoa(int(time.Now().Unix())))
	check("the stubborn request is accepted", submitted.Status == http.StatusAccepted,
		submitted.brief())
	requestID, _ := submitted.json()["request_id"].(string)
	live := svc.openUntilLive("/v1/requests/"+requestID+"/events", 300*time.Second)
	if !check("it is running on the card", live.find("request.accepted") != nil && !live.stopped,
		strings.Join(live.types(), " · ")) {
		fmt.Println(indent(string(svc.call("GET", "/v1/requests/"+requestID, nil).Body)))
		return
	}

	asked := time.Now()
	cancel := svc.call("POST", "/v1/requests/"+requestID+"/cancel", map[string]any{})
	check("the cancel is accepted", cancel.Status < 300, cancel.brief())
	settled := svc.openSSE(fmt.Sprintf("/v1/requests/%s/events?cursor=%d", requestID, live.cursor),
		180*time.Second)
	took := time.Since(asked)
	check("the request converges to a terminal", settled.stopped,
		strings.Join(settled.types(), " · "))
	final := svc.call("GET", "/v1/requests/"+requestID, nil)
	check("the terminal is typed CANCELED, not a failure and not a success",
		strings.Contains(strings.ToLower(fmt.Sprint(final.json()["status"])), "cancel"),
		fmt.Sprint(final.json()["status"]))
	check("and it published NOTHING", len(asList(final.json()["outputs"])) == 0,
		fmt.Sprintf("%d output(s)", len(asList(final.json()["outputs"]))))
	fmt.Printf("  cancel -> typed terminal: %s\n", ms(took))
	fmt.Println(indent(grepLog(filepath.Join(root, "driver-service.log"),
		"CancelAttempt", "grace", "CANCELED")))
	// THE DEVICE COMES BACK when the worker goes, and that is the honest claim here.
	// Cancelling a request does not stop a worker, and a resident worker still holding
	// 5.8 GiB afterwards is the product working. What this arm does NOT get to claim is
	// that the worker serves again: after a forceful cancel it goes on advertising READY
	// with the same `executor_incarnation` and every following attempt dies
	// `EXECUTOR_FAULT`, so the next request spends its whole requeue budget and fails
	// typed. NAMED for cr-007 — a device process that cannot serve must stop saying READY.
	code, out := cozyRun(root, "stop", pipeRef)
	check("stopping the worker returns the card the cancelled attempt was holding",
		code == 0, firstLine(out))
	used := gpuReleased(idle, 180*time.Second)
	check("the GPU is back at its baseline after a FORCEFUL cancel", used <= idle+40,
		fmt.Sprintf("%d MiB (idle was %d)", used, idle))
}

// ------------------------------------------------------------ 4: coordinator restart

func armCoordinatorRestart(svc *liveService, root string, port, idle int) {
	head("ARM 4 — kill -9 the coordinator mid-attempt, on the four-component pipeline")
	// A FRESH worker, for ARM 3's reason: each arm has to be answered by its own subject,
	// and a 1024px attempt on a generation another arm already drove is answered by the
	// residency defect instead of by the crash.
	if code, _ := cozyRun(root, "stop", "--all"); code != 0 {
		check("the worker from ARM 3 drains", false, "")
	}
	body := map[string]any{"steps": 24, "size": 1024, "seed": 44}
	key := "cl003-restart-" + itoa(int(time.Now().Unix()))
	submitted := svc.call("POST", "/v1/requests", map[string]any{
		"endpoint": pipeRef, "function": "generate", "input": body,
	}, "Idempotency-Key", key)
	check("submitted", submitted.Status == http.StatusAccepted, submitted.brief())
	requestID, _ := submitted.json()["request_id"].(string)
	pre := svc.openUntilLive("/v1/requests/"+requestID+"/events", 300*time.Second)
	cursor := pre.cursor
	check("the client holds a cursor from before the crash, on a RUNNING attempt",
		cursor > 0 && pre.find("request.accepted") != nil && !pre.stopped,
		fmt.Sprintf("cursor %d after %s", cursor, strings.Join(pre.types(), " · ")))

	svc.kill9()
	check("the service is gone", !svc.alive(), "")
	restarted := startService(root, port, false)
	*svc = *restarted
	check("the SAME root comes back up", svc.alive(), "")

	life := svc.call("GET", "/v1/requests/"+requestID, nil)
	check("nothing the dead coordinator had not committed is visible",
		len(asList(life.json()["outputs"])) == 0 && life.json()["status"] != "completed",
		fmt.Sprint(life.json()["status"]))
	replay := svc.call("POST", "/v1/requests", map[string]any{
		"endpoint": pipeRef, "function": "generate", "input": body,
	}, "Idempotency-Key", key)
	check("the idempotency key still names the same request across the crash",
		replay.json()["request_id"] == requestID, replay.brief())

	// The restart reaped the orphaned worker. Starting the slot again is what a user does
	// (or what the next `cozy run` does through select-or-start), and the recovered
	// attempt is an OPEN OBLIGATION until its own journaled terminal arrives.
	svc.call("POST", "/v1/local/workers", map[string]any{"endpoint": pipeRef})
	settled := svc.openSSE(fmt.Sprintf("/v1/requests/%s/events?cursor=%d", requestID, cursor),
		900*time.Second)
	check("the request reaches a terminal without ever being executed twice", settled.stopped,
		strings.Join(settled.types(), " · "))
	ended := settled.find("request.attempt_failed")
	check("the crashed attempt ended with a NON-terminal event, marked requeuing",
		ended != nil && ended.Payload["requeuing"] == true,
		fmt.Sprintf("%v/%v", payloadOf(ended, "status"), payloadOf(ended, "cause")))
	check("and a NEW ordinal was minted for it", settled.find("request.requeued") != nil, "")
	final := svc.call("GET", "/v1/requests/"+requestID, nil)
	attempts := int(asFloat(final.json()["attempts"]))
	outs := asList(final.json()["outputs"])
	check("no ordinal was minted while the recovered one was open",
		noOrdinalRace(settled), fmt.Sprintf("%d attempt(s)", attempts))
	check("exactly ONE attempt published output over the whole crash",
		len(outs) == 1, fmt.Sprintf("%d output(s) over %d attempt(s), status %v",
			len(outs), attempts, final.json()["status"]))
	if len(outs) == 1 {
		media := fmt.Sprint(outs[0].(map[string]any)["media_id"])
		got := svc.call("GET", "/v1/media/"+media, nil)
		check("and the recovered attempt's real 1024px PNG is fetchable by opaque id",
			got.Status == http.StatusOK && len(got.Body) > 1_000_000 &&
				string(got.Body[1:4]) == "PNG",
			fmt.Sprintf("%s, %d B", media, len(got.Body)))
	}
}

// grepLog returns the lines of a log naming any of the given substrings, at most ten.
func grepLog(path string, want ...string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	out := []string{}
	for _, line := range strings.Split(string(data), "\n") {
		for _, w := range want {
			if strings.Contains(line, w) {
				out = append(out, strings.TrimSpace(line))
				break
			}
		}
	}
	if len(out) > 10 {
		out = out[len(out)-10:]
	}
	return strings.Join(out, "\n")
}
