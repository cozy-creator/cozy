package main

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// THE COORDINATOR-KILL CRASH ARM.
//
// cl-001 armed the worker half of its crash-convergence matrix at two lifecycle points
// and named the COORDINATOR half as still owed, for one reason: the coordinator ran
// inside the verification driver, so killing it killed the observer. That reason is gone.
// This section drives a SEPARATE `cozy up` process over real HTTP, `kill -9`s it
// mid-attempt, and watches the whole recovery FROM THE API CONSUMER'S SEAT — which is
// the only seat from which the claim "a crash never produces a false success" means
// anything to a user.
//
// What must hold across the kill, and what this arm checks:
//
//  1. Nothing the dead coordinator had not COMMITTED is visible afterwards.
//  2. The request survives. Its id is still an id; its idempotency key still names it and
//     starting a second execution under that key is impossible.
//  3. The RECOVERED-ATTEMPTS LAW holds through a coordinator restart, not only a worker
//     one: the restarted supervisor reports its open attempt on Register, that attempt is
//     an OPEN OBLIGATION, and no next ordinal is minted until its own journaled terminal
//     arrives.
//  4. The event stream RESUMES from the cursor the client held before the kill. A stream
//     that dropped is not a verdict — reconnecting is always the right move — and the
//     durable rows written before the crash are still there, in the same order, with the
//     same ids.

func sectionAPICrash() {
	idle := requireFreeGPU()
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", "api-crash"))
	must("clearing the service root", os.RemoveAll(root))
	must("creating the service root", os.MkdirAll(root, 0o755))

	installEndpoint(root)
	port := freePort(2851)

	head("a REAL LocalService, in its own process")
	svc := startService(root, port, false)
	check("the API answers on "+svc.addr, svc.alive(), fmt.Sprintf("pid %d", svc.cmd.Process.Pid))

	start := svc.call("POST", "/v1/local/workers", map[string]any{"endpoint": "cozy/sdxl-unet"})
	instance, _ := start.json()["instance_id"].(string)
	if !waitReady(svc, 300*time.Second) {
		fmt.Println(tail(filepath.Join(root, "workers", instance, "worker.log"), 30))
		check("the worker reported READY", false, "see the log above")
		return
	}
	check("the REAL supervisor is READY", true, "instance "+instance)

	head("a long attempt, running, watched")
	// The longest run `denoise`'s own schema admits — 64 steps at latent 96 — which is
	// ten-odd seconds of real GPU work. The window has to be wide enough to crash INTO;
	// a request that finished before the kill would prove nothing.
	body := map[string]any{"steps": 64, "latent": 96, "seed": 61}
	submitted := svc.call("POST", "/v1/requests", map[string]any{
		"endpoint": "cozy/sdxl-unet", "function": "denoise", "input": body,
	}, "Idempotency-Key", "crash-key")
	check("submitted", submitted.Status == http.StatusAccepted, submitted.brief())
	requestID, _ := submitted.json()["request_id"].(string)

	// Hold the cursor the way a real client does: from the stream, before the crash. The
	// read stops at the FIRST live progress frame — the attempt is provably running and
	// has not finished, which is exactly the state the kill has to land in.
	preStream := svc.openUntilLive("/v1/requests/"+requestID+"/events", 120*time.Second)
	cursor := preStream.cursor
	check("the client holds a cursor from before the crash", cursor > 0,
		fmt.Sprintf("cursor %d after %s", cursor, strings.Join(preStream.types(), " · ")))
	check("and saw the attempt ACCEPTED and running", preStream.find("request.accepted") != nil &&
		preStream.count("request.progress") > 0 && !preStream.stopped,
		fmt.Sprintf("%d progress frame(s), not settled", preStream.count("request.progress")))
	preLife := svc.call("GET", "/v1/requests/"+requestID, nil)
	check("no output is visible while it runs",
		len(preLife.json()["outputs"].([]any)) == 0, preLife.json()["status"].(string))

	head("kill -9 the COORDINATOR, mid-attempt")
	killAt := time.Now()
	svc.kill9()
	check("the service is gone", !svc.alive(), ms(time.Since(killAt)))
	// From the consumer's seat, that is all a crash looks like: the connection refuses.
	dead := svc.call("GET", "/v1/requests/"+requestID, nil)
	check("every route now refuses at the transport — which is what a crash IS to a client",
		dead.Status == 0, strings.TrimSpace(string(dead.Body)))

	head("restart, and the law observed from the client's seat")
	restartAt := time.Now()
	// NOT fresh. The whole claim is that the authority and the supervisor's journal
	// survived; wiping the root would be arranging the answer.
	svc = startService(root, port, false)
	check("the SAME root comes back up", svc.alive(), ms(time.Since(restartAt)))

	// The request survived, and so did its history.
	life := svc.call("GET", "/v1/requests/"+requestID, nil)
	check("the request is still an id, with its endpoint and its attempt",
		life.Status == http.StatusOK && life.json()["request_id"] == requestID,
		fmt.Sprintf("status %v, attempt %v", life.json()["status"], life.json()["attempt"]))
	check("and NOTHING is visible that the dead coordinator had not committed",
		len(life.json()["outputs"].([]any)) == 0 && life.json()["status"] != "completed",
		fmt.Sprint(life.json()["status"]))

	// THE CURSOR. The durable rows written before the crash are still there.
	resumed := svc.openSSE(fmt.Sprintf("/v1/requests/%s/events?cursor=0", requestID), 20*time.Second)
	check("the stream replays its PRE-CRASH history from cursor 0", len(resumed.events) > 0 &&
		resumed.events[0].EventID == preStream.events[0].EventID,
		fmt.Sprintf("%d events, first id %d", len(resumed.events), resumed.events[0].EventID))
	check("with the same ids in the same order — a crash did not renumber the stream",
		sameIDs(preStream.events, resumed.events),
		fmt.Sprintf("pre %d · post %d", len(preStream.events), len(resumed.events)))

	// IDEMPOTENCY ACROSS THE CRASH: the key still names the same request, and cannot
	// start a second execution of it.
	replay := svc.call("POST", "/v1/requests", map[string]any{
		"endpoint": "cozy/sdxl-unet", "function": "denoise", "input": body,
	}, "Idempotency-Key", "crash-key")
	check("the idempotency key STILL names the same request after the crash",
		replay.json()["request_id"] == requestID && replay.json()["idempotent_replay"] == true,
		replay.brief())

	head("the recovered-attempts law, through a coordinator restart")
	// The restart reconciled the orphaned worker (it was killed with its launcher's
	// group, or reaped at boot). Starting the slot again replays the SUPERVISOR's own
	// journal, and the recovered attempt is an OPEN OBLIGATION.
	svc.call("POST", "/v1/local/workers", map[string]any{"endpoint": "cozy/sdxl-unet"})
	settled := svc.openSSE(fmt.Sprintf("/v1/requests/%s/events?cursor=%d", requestID, cursor),
		600*time.Second)
	check("the request reaches a terminal without ever being executed twice", settled.stopped,
		strings.Join(settled.types(), " · "))
	// The attempt that died in the crash ended; the REQUEST did not. A client that saw a
	// terminal there would have closed its stream on a request that goes on to succeed.
	attemptEnd := settled.find("request.attempt_failed")
	check("the crashed attempt ended with a NON-TERMINAL event",
		attemptEnd != nil && attemptEnd.Payload["requeuing"] == true,
		fmt.Sprintf("request.attempt_failed %v/%v", payloadOf(attemptEnd, "status"),
			payloadOf(attemptEnd, "cause")))
	check("and the coordinator minted a NEW ordinal for it",
		settled.find("request.requeued") != nil,
		fmt.Sprint(settled.find("request.requeued") != nil))
	check("attempt 2 ran and the request completed", settled.find("request.completed") != nil,
		strings.Join(settled.types(), " · "))

	// The whole history, read from the authority: every attempt, in order, with what
	// each one published. This is the claim in its strongest form.
	final := svc.call("GET", "/v1/requests/"+requestID, nil)
	attempts := int(asFloat(final.json()["attempts"]))
	fmt.Printf("    | request %s · status %v · %d attempt(s)\n",
		requestID, final.json()["status"], attempts)
	for _, e := range settled.events {
		if e.EventID > 0 {
			fmt.Printf("    | #%d %-22s attempt %d %v\n", e.EventID, e.Type, e.Attempt,
				brieflyPayload(e.Payload))
		}
	}
	check("no attempt ordinal was minted while the recovered one was open",
		noOrdinalRace(settled), fmt.Sprintf("%d attempt(s) total", attempts))
	outs, _ := final.json()["outputs"].([]any)
	check("exactly ONE attempt published output — the crashed one published nothing, ever",
		len(outs) == 1, fmt.Sprintf("%d visible output(s) over %d attempt(s)", len(outs), attempts))
	check("and the request never reported a false success",
		final.json()["status"] != "completed" || len(outs) == 1,
		fmt.Sprint(final.json()["status"]))
	// The bytes the SECOND attempt produced are fetchable by opaque id, from the client's
	// seat, after a crash killed the first one mid-flight.
	if len(outs) == 1 {
		media := fmt.Sprint(outs[0].(map[string]any)["media_id"])
		got := svc.call("GET", "/v1/media/"+media, nil)
		check("and its media is fetchable by opaque id after the crash",
			got.Status == http.StatusOK && len(got.Body) > 8 && string(got.Body[1:4]) == "PNG",
			fmt.Sprintf("%s, %d B PNG", media, len(got.Body)))
	}

	head("teardown")
	svc.stop()
	now := gpuReleased(idle, 60*time.Second)
	check("the GPU is back at its idle baseline", now <= idle+40,
		fmt.Sprintf("%d MiB now, %d MiB before", now, idle))
}

func sameIDs(a, b []sseEvent) bool {
	durable := func(in []sseEvent) []int64 {
		out := []int64{}
		for _, e := range in {
			if e.EventID > 0 {
				out = append(out, e.EventID)
			}
		}
		return out
	}
	x, y := durable(a), durable(b)
	if len(x) > len(y) {
		return false
	}
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

// noOrdinalRace checks the ordering the law promises, from the STREAM: a `request.
// dispatched` for ordinal N+1 may never appear before the terminal-shaped row that
// closed ordinal N. The client can check this because the coordinator publishes both.
func noOrdinalRace(st *sseStream) bool {
	open := map[uint64]bool{}
	for _, e := range st.events {
		if e.EventID == 0 {
			continue
		}
		switch e.Type {
		case "request.dispatched":
			for ordinal := range open {
				if ordinal < e.Attempt {
					return false // a new ordinal while an older one was still open
				}
			}
			open[e.Attempt] = true
		case "request.completed", "request.failed", "request.canceled",
			"request.requeued", "request.attempt_failed":
			delete(open, e.Attempt)
		}
	}
	return true
}

func brieflyPayload(p map[string]any) string {
	parts := []string{}
	for _, key := range []string{"status", "cause", "instance_id", "reason", "requeues", "plan"} {
		if v, ok := p[key]; ok && fmt.Sprint(v) != "" {
			parts = append(parts, fmt.Sprintf("%s=%v", key, v))
		}
	}
	joined := strings.Join(parts, " ")
	if len(joined) > 90 {
		joined = joined[:90] + "…"
	}
	return joined
}

func asFloat(v any) float64 {
	f, _ := v.(float64)
	return f
}

func payloadOf(e *sseEvent, key string) any {
	if e == nil {
		return nil
	}
	return e.Payload[key]
}
