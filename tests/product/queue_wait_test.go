package producttest

import (
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

// TestQueueWaitCauses is cl-103 as behaviour: a waiting request's durable queued/parked
// events name WHAT the queue is doing in a stable `wait` cause beside the verbatim
// dispatcher diagnostic. The human client renders the cause as a calm stage line; the
// machine surfaces (--json/--stream, this payload) keep the raw `reason` unchanged.
func TestQueueWaitCauses(t *testing.T) {
	o := hostOwner(t, "queue-wait")

	// ARM 1 — worker_start: nothing serves the package when the request arrives, so the
	// honest answer is "a worker is being started", not "no way to place this".
	cold := fakeSpec("wait-cold", "13")
	coldID, _, e := o.c.Submit(submission(planIDOf(t, cold), "fake/wait-cold", "wait-cold-1",
		map[string]any{"n": 1}))
	fatal(t, e)
	queued := awaitDurable(t, o, coldID, "request.queued")
	if queued.Payload["wait"] != "worker_start" || queued.Payload["package"] != "fake/wait-cold" {
		t.Errorf("cold submission wait = %v/%v, want worker_start for fake/wait-cold",
			queued.Payload["wait"], queued.Payload["package"])
	}
	if reason, _ := queued.Payload["reason"].(string); reason == "" {
		t.Error("the verbatim diagnostic left the queued payload; --json must keep it")
	}
	// This host resolves no local packages, so the start itself fails and the queue
	// empties before the next arm.
	if _, e := o.c.AwaitSettled(coldID, 20*time.Second); e == nil {
		t.Fatal("the cold request did not settle on this package-less host")
	}

	// ARM 2 — slot_busy and queue_ahead, on the lanes worker whose placement draws ONE
	// seat: A holds it, B waits on the slot, C waits in line behind B.
	spec := fakeSpec("wait-lanes", "10,11", "--arm", "lanes")
	instance, _, e := o.c.EnsureWorker(spec)
	fatal(t, e)
	planID := planIDOf(t, spec)
	fatal(t, o.c.EnsurePlacementReady(instance, planID))
	requestA, attemptA, e := o.c.Submit(submission(planID, "fake/wait-lanes", "wait-lanes-a",
		map[string]any{"n": 1}))
	fatal(t, e)
	fatal(t, o.c.AwaitAccepted(requestA, attemptA, 15*time.Second))
	requestB, _, e := o.c.Submit(submission(planID, "fake/wait-lanes", "wait-lanes-b",
		map[string]any{"n": 2}))
	fatal(t, e)
	queuedB := awaitDurable(t, o, requestB, "request.queued")
	if queuedB.Payload["wait"] != "slot_busy" {
		t.Errorf("B's queued wait = %v, want slot_busy while A holds the only seat", queuedB.Payload["wait"])
	}
	requestC, _, e := o.c.Submit(submission(planID, "fake/wait-lanes", "wait-lanes-c",
		map[string]any{"n": 3}))
	fatal(t, e)
	queuedC := awaitDurable(t, o, requestC, "request.queued")
	if queuedC.Payload["wait"] != "queue_ahead" {
		t.Errorf("C's queued wait = %v, want queue_ahead behind B", queuedC.Payload["wait"])
	}
	if position, ok := queuedC.Payload["position"].(float64); !ok || position < 2 {
		t.Errorf("C's queued position = %v, want its real place in line", queuedC.Payload["position"])
	}
	// The drain's own skip record: the raw dispatcher spelling stays VERBATIM in the
	// durable payload — the machine surface — beside the same stable cause.
	parkedB := awaitDurable(t, o, requestB, "request.parked")
	if parkedB.Payload["wait"] != "slot_busy" {
		t.Errorf("B's parked wait = %v, want slot_busy", parkedB.Payload["wait"])
	}
	if reason, _ := parkedB.Payload["reason"].(string); !strings.Contains(reason, "DISPATCHABLE") {
		t.Errorf("B's parked reason lost the raw diagnostic: %q", reason)
	}
	// FIFO settles all three; the fake attempts end on their own "no GPU" outcome.
	for _, id := range []string{requestA, requestB, requestC} {
		if _, e := o.c.AwaitSettled(id, 30*time.Second); e == nil || !strings.Contains(e.Message, "no GPU") {
			t.Fatalf("%s did not settle on its own outcome: %s", id, briefly(e))
		}
	}

	// ARM 3 — worker_warming: a live worker occupies the slot but never reports its
	// placement dispatchable (a stand-in process that never dials home).
	warming := fakeSpec("wait-warming", "14")
	warming.Python = "/bin/sh"
	warming.Args = []string{"-c", "sleep 120", "wait-warming"}
	_, _, e = o.c.EnsureWorker(warming)
	fatal(t, e)
	warmID, _, e := o.c.Submit(submission(planIDOf(t, warming), "fake/wait-warming",
		"wait-warming-1", map[string]any{"n": 1}))
	fatal(t, e)
	queuedW := awaitDurable(t, o, warmID, "request.queued")
	if queuedW.Payload["wait"] != "worker_warming" {
		t.Errorf("warming wait = %v, want worker_warming while the worker loads", queuedW.Payload["wait"])
	}
	fatal(t, o.c.CancelQueued(warmID, "test client"))
}

// awaitDurable is the first durable event of one type for a request, from the store the
// events live in.
func awaitDurable(t *testing.T, o *owner, requestID, eventType string) records.Event {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		rows, e := o.store.EventsAfter(requestID, 0, 200)
		fatal(t, e)
		for _, row := range rows {
			if row.Type == eventType {
				return row
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("no %s event for %s in 15s", eventType, requestID)
	return records.Event{}
}
