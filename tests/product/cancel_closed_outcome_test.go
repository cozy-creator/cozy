package producttest

import (
	"encoding/json"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// Run 524: a job paused on 2026-09-12 holds one attempt whose CANCELED outcome closed long
// ago, and the local worker that ran it is gone. `cozy run cancel` settles it at once — no
// worker is started to reach the held attempt — and returns on the terminal event (it used
// to sit in `canceling` while the daemon started a workspace worker and waited on it).
func TestCancelOfAPausedRunWithAClosedOutcomeSettlesAtOnce(t *testing.T) {
	o := hostOwner(t, "paused-cancel")
	const id = "job-paused-closed-outcome"
	_, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: "idem-" + id, BodyDigest: "sha256:" + sixtyFour("9"),
		Package: "local/cozy-script", Entrypoint: "main", Kind: "job", Payload: []byte("{}"), RetainWork: true})
	fatal(t, problem)
	fatal(t, o.store.SpawnWorker(records.WorkerProcess{InstanceID: "ins-gone", Package: "local/cozy-script", WorkerID: "local", Devices: []string{"cpu"}}))
	session, digest := "session-gone", "sha256:"+sixtyFour("a")
	attempt, problem := o.store.Dispatch(records.Attempt{RequestID: id, SessionID: session, InstanceID: "ins-gone",
		InvocationDigest: digest, InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, o.store.OfferDispatch(id, attempt, session))
	fatal(t, o.store.Accepted(id, attempt, session))
	_, problem = o.store.RequestPause(id, "cozy run pause")
	fatal(t, problem)
	_, problem = o.store.AcceptTerminal(records.Terminal{RequestID: id, Attempt: attempt, SessionID: session,
		InvocationDigest: digest, TerminalID: "out-drain", TerminalDigest: "sha256:" + sixtyFour("f"),
		Status: "CANCELED", Cause: "DRAIN_CANCEL", EventType: "request.pausing",
		EventPayload: map[string]any{"status": "pausing"}, RequestState: "pausing"})
	fatal(t, problem)
	fatal(t, o.store.Closed(id, attempt))
	paused, problem := o.store.CompleteRequestPause(id)
	fatal(t, problem)
	fatal(t, o.store.CloseWorker("ins-gone"))
	if !paused {
		t.Fatal("the fixture did not pause")
	}
	defer publicationControlAPI(t, o)()

	code, out := runCozy(t, o.root, "run", "cancel", id, "--json")
	var shown struct {
		Changed bool `json:"changed"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &shown) != nil || !shown.Changed {
		t.Fatalf("cozy run cancel of a paused run [%d]: %s", code, out)
	}
	row, problem := o.store.RequestRow(id)
	fatal(t, problem)
	if row.State != "canceled" {
		t.Fatalf("the canceled run is %s", row.State)
	}
	workers, problem := o.store.LiveWorkers()
	fatal(t, problem)
	if len(workers) != 0 {
		t.Fatalf("the cancel started %d worker(s): %+v", len(workers), workers)
	}
}
