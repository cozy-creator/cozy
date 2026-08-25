package main

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"syscall"
	"time"
)

// sectionRecovered proves the ONE hard obligation cr-007's execution record binds this
// issue to: the `recovered_attempts` a restarted supervisor reports on Register are
// closed BEFORE any next ordinal is minted for their request ids. A new session_id never
// manufactures absence.
//
// The kill is a real `kill -9` of the real supervisor mid-attempt. Nothing is planted.
func sectionRecovered() {
	idle := requireFreeGPU()
	lv := hostCoordinator("recovered", true)
	defer lv.close()

	spec := sdxlSpec("denoise")
	planID := planIDOf(spec, "denoise")

	head("a real attempt, killed mid-flight")
	instance, e := lv.c.StartWorker(spec)
	if e != nil {
		check("StartWorker", false, briefly(e))
		return
	}
	if e := lv.c.WaitReady(instance, planID, 300*time.Second); e != nil {
		check("READY", false, briefly(e))
		fmt.Println(tail(lv.c.WorkerLog(instance), 20))
		return
	}
	firstSession := lv.c.Worker(instance).SessionID
	check("the worker is serving", firstSession != "", "session "+firstSession)

	// 48 denoising steps: long enough that a kill lands INSIDE the attempt rather than
	// racing its end.
	body := payload(map[string]any{"steps": 48, "latent": 64, "seed": 7})
	sub := submissionKey(planID, body, "idem-recovered")
	requestID, attempt, e := lv.c.Submit(sub)
	if e != nil {
		check("submit", false, briefly(e))
		return
	}
	if e := lv.c.AwaitAccepted(requestID, attempt, 60*time.Second); e != nil {
		check("accepted", false, briefly(e))
		return
	}
	check("attempt 1 accepted and running", true, fmt.Sprintf("%s#%d", requestID, attempt))

	pid := lv.c.Worker(instance).PID
	time.Sleep(600 * time.Millisecond) // let the attempt get properly under way
	must("killing the supervisor", syscall.Kill(pid, syscall.SIGKILL))
	check("kill -9 the supervisor mid-attempt", true, fmt.Sprintf("pid %d", pid))

	waitFor(func() bool {
		rows, _ := lv.store.LiveWorkers()
		return len(rows) == 0
	}, 20*time.Second)
	rows, _ := lv.store.LiveWorkers()
	check("the dead worker's row (and its device grant) is released", len(rows) == 0,
		fmt.Sprintf("%d live row(s)", len(rows)))

	row, _ := lv.store.AttemptRow(requestID, int64(attempt))
	check("the attempt is still ACCEPTED with no terminal", row.State == "accepted" &&
		row.TerminalID == "", "state "+row.State)
	outs, _ := lv.store.VisibleOutputs(requestID)
	check("nothing became visible", len(outs) == 0, fmt.Sprintf("%d output(s)", len(outs)))

	head("the law: no next ordinal until the recovered attempt is closed")
	// A spinner asks the coordinator's OWN ordinal gate for the next ordinal, as fast as
	// it can, from BEFORE the restart. Every refusal it collects is the law being
	// enforced under contention rather than in a quiet moment chosen by the harness.
	var (
		mu       sync.Mutex
		refusals []string
		lawSeen  time.Time
		stop     = make(chan struct{})
	)
	go func() {
		for {
			select {
			case <-stop:
				return
			default:
			}
			if _, e := lv.store.NextOrdinal(requestID); e != nil {
				mu.Lock()
				refusals = append(refusals, e.Message)
				if strings.Contains(e.Message, "recovered attempt") && lawSeen.IsZero() {
					lawSeen = time.Now()
				}
				mu.Unlock()
			}
			// No sleep: the recovered window is milliseconds wide, and a poll that
			// pauses inside it observes nothing.
		}
	}()
	defer close(stop)

	restartAt := time.Now()
	if _, e := lv.c.StartWorker(spec); e != nil {
		check("restart the same worker slot", false, briefly(e))
		close(stop)
		return
	}
	check("the same slot restarts over the same journal", true,
		"instance "+instance+" (a slot keeps its instance_id across a supervisor restart)")

	line, ok := waitEvent(lv, "OPEN OBLIGATION", 120*time.Second)
	check("Register carried the recovered attempt", ok, trimLog(line))
	secondSession := lv.c.Worker(instance).SessionID
	check("it is a NEW session over the SAME instance", secondSession != firstSession &&
		secondSession != "", firstSession+" -> "+secondSession)

	// The recovered attempt is closed by its OWN journaled terminal — replayed
	// SUCCEEDED if the tail had already finished, ABANDONED if it had not. Either way
	// the coordinator never invents it.
	closeLine, ok := waitEvent(lv, "applied in", 180*time.Second)
	check("the recovered attempt was closed by a journaled terminal", ok, trimLog(closeLine))

	mu.Lock()
	sawLaw, tries, lawAt := !lawSeen.IsZero(), len(refusals), lawSeen
	mu.Unlock()
	when := "never"
	if sawLaw {
		when = ms(lawAt.Sub(restartAt)) + " after the restart"
	}
	check("a next ordinal was REFUSED while the recovered attempt was open", sawLaw,
		fmt.Sprintf("%d refusals of the ordinal gate, the recovered one first seen %s", tries, when))
	check("the coordinator's own log orders closure BEFORE the next dispatch",
		orderedBefore(lv, "applied in", fmt.Sprintf("StartAttempt %s#%d", requestID, attempt+1)),
		"the requeue projection dispatches attempt 2 only after attempt 1's terminal committed")

	head("the coordinator's own requeue projection runs attempt 2 to a visible output")
	result, e := lv.c.AwaitSettled(requestID, 240*time.Second)
	if e != nil {
		check("the request settled", false, briefly(e))
		fmt.Println(tail(lv.c.WorkerLog(instance), 20))
	} else {
		check("the request settled SUCCEEDED on its requeued ordinal", result.Status == "SUCCEEDED",
			fmt.Sprintf("attempt %d, %s/%s", result.Attempt, result.Status, result.Cause))
		check("its output is visible", len(result.Outputs) >= 1,
			fmt.Sprintf("%d output(s)", len(result.Outputs)))
	}
	all, _ := lv.store.Attempts(requestID)
	states := []string{}
	for _, a := range all {
		states = append(states, fmt.Sprintf("#%d %s/%s", a.Attempt, a.State, a.TerminalStatus))
	}
	check("the request's attempt history is exact: no duplicate, no silence",
		len(all) == 2, strings.Join(states, " · "))
	first, _ := lv.store.VisibleOutputsOf(requestID, int64(attempt))
	check("the killed attempt published nothing, ever", len(first) == 0,
		fmt.Sprintf("attempt %d: %d visible output(s)", attempt, len(first)))

	head("scenario B: an output written under the grant, with no accepted terminal")
	scenarioB(lv, instance, planID)

	head("teardown")
	live, _ := lv.store.LiveWorkers()
	for _, r := range live {
		lv.c.StopWorker(r.InstanceID, 20*time.Second)
	}
	time.Sleep(2 * time.Second)
	now := gpuUsedMiB()
	check("the GPU is back at its idle baseline", now <= idle+40,
		fmt.Sprintf("%d MiB now, %d MiB before", now, idle))
}

// scenarioB kills the supervisor the instant the runtime's tail writes the output file.
// Whichever side of the journal the kill lands on, the invariant is the same: bytes under
// a grant are not a result. They become visible only when the coordinator accepts a
// terminal, in the same transaction, or never.
func scenarioB(lv *live, instance, planID string) {
	body := payload(map[string]any{"steps": 6, "latent": 64, "seed": 11})
	sub := submissionKey(planID, body, "idem-written-not-visible")
	requestID, attempt, e := lv.c.Submit(sub)
	if e != nil {
		check("submit", false, briefly(e))
		return
	}
	if e := lv.c.AwaitAccepted(requestID, attempt, 60*time.Second); e != nil {
		check("accepted", false, briefly(e))
		return
	}
	path := lv.l.AttemptDir(requestID, attempt) + "/image"
	pid := lv.c.Worker(instance).PID
	deadline := time.Now().Add(120 * time.Second)
	killed := false
	for time.Now().Before(deadline) {
		if st, err := os.Stat(path); err == nil && st.Size() > 0 {
			_ = syscall.Kill(pid, syscall.SIGKILL)
			killed = true
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !killed {
		check("the output file appeared under the grant", false, path)
		return
	}
	size := int64(0)
	if st, err := os.Stat(path); err == nil {
		size = st.Size()
	}
	check("the runtime wrote bytes under the grant, and the supervisor was killed at once",
		size > 0, fmt.Sprintf("%d B at %s", size, path))
	outs, _ := lv.store.VisibleOutputsOf(requestID, int64(attempt))
	check("those bytes are NOT a visible output", len(outs) == 0,
		"the coordinator accepted no terminal for them")

	waitFor(func() bool {
		rows, _ := lv.store.LiveWorkers()
		return len(rows) == 0
	}, 20*time.Second)
	if _, e := lv.c.StartWorker(sdxlSpec("denoise")); e != nil {
		check("restart", false, briefly(e))
		return
	}
	// Convergence: the journal decides. A tail that had finished replays SUCCEEDED and
	// the SAME bytes become visible with their terminal; one that had not mints
	// ABANDONED and the bytes stay invisible forever.
	var status string
	waitFor(func() bool {
		row, _ := lv.store.AttemptRow(requestID, int64(attempt))
		if row != nil && row.TerminalStatus != "" {
			status = row.TerminalStatus
			return true
		}
		return false
	}, 240*time.Second)
	outs, _ = lv.store.VisibleOutputsOf(requestID, int64(attempt))
	switch status {
	case "SUCCEEDED":
		check("the journaled terminal replayed, and the bytes became visible WITH it",
			len(outs) == 1, "SUCCEEDED replayed across the restart")
	case "ABANDONED":
		check("the attempt converged as ABANDONED and the written bytes stayed invisible",
			len(outs) == 0, "ABANDONED/EXECUTOR_INVALIDATED")
	default:
		check("the attempt reached a terminal", false, "status "+status)
	}
}

func waitFor(cond func() bool, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(20 * time.Millisecond)
	}
	return false
}

// orderedBefore reads the coordinator's own event order: `first` must appear before
// `second` ever does.
func orderedBefore(lv *live, first, second string) bool {
	seenFirst := false
	for _, line := range lv.c.Events() {
		if strings.Contains(line, second) {
			return seenFirst
		}
		if strings.Contains(line, first) {
			seenFirst = true
		}
	}
	return seenFirst
}
