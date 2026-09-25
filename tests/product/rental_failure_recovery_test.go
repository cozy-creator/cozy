package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// TestRentalFailureRecovery is the owner's ruling as behaviour: "it's fine for cozy-daemon
// to assign work to specific pods, but when that pod fails it should recover those jobs and
// schedule them elsewhere if possible".
//
// The pin stays. What this proves is the arm that was missing: a rental that reaches a
// terminal failed state HANDS BACK the work pinned to it, and every request it was holding
// reaches a stated outcome instead of waiting on a machine that no longer exists.
//
// Observed live 2026-09-04 on the code this test was written against: rental
// pr-183abac284d1e16f5f0a (nitian) went `failed` with readiness.receipt_conflict while
// holding one in-flight and two queued anima requests. Nothing released them. The deadlock
// was mutual — the requests could not be routed because they named a dead rental, and the
// rental could not be released because `RentalRunCounts` still counted them — so the daemon
// bought a second pod, ran an identical request on it, and idle-released it while the
// stranded three still sat there. Request 284 read `in_progress 300.8s` against a pod whose
// provider resource had been destroyed four minutes earlier.
//
// The daemon here is the real process on a real root, deciding from its real records
// against a hub that answers the rental routes and can move a rental to `failed` the way
// Tensorhub does.
func TestRentalFailureRecovery(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "rental-failure-recovery")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	port := reservePort(t)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			"tensorhub_token: rental-idle-test\n"+
			""+
			"daemon:\n  idle_shutdown_s: 0\n"), 0o600))
	logPath := filepath.Join(root, "daemon.log")

	hub := newFakeRentalHub(t, port)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()

	// The pod's on-disk credentials exist, as they do for any rental this host actually
	// rented. Without them routing refuses the pin for a reason that has nothing to do with
	// the machine being dead, and the arm under test never runs.
	rentals := filepath.Join(root, "rentals")
	must(t, os.MkdirAll(rentals, 0o700))
	plant := func(id, machine string) {
		t.Helper()
		hub.add(id, machine)
		must(t, os.WriteFile(filepath.Join(rentals, id+".media-token"), []byte("media-"+id), 0o600))
		must(t, os.WriteFile(filepath.Join(rentals, id+".pem"), []byte("-----BEGIN CERTIFICATE-----\n"), 0o600))
		_, private, err := ed25519.GenerateKey(rand.Reader)
		must(t, err)
		key, err := x509.MarshalPKCS8PrivateKey(private)
		must(t, err)
		must(t, os.WriteFile(filepath.Join(rentals, id+".creator.pem"),
			pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: key}), 0o600))
		fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
			ID: id, MachineName: machine, SKU: "cpu", AcceleratorModel: "CPU",
			HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL,
			Address: "127.0.0.1:1", CertPath: filepath.Join(rentals, id+".pem"),
		}))
	}
	pin := func(requestID, rentalID, seed string) {
		t.Helper()
		if _, _, problem := store.Submit(records.Request{
			ID: requestID, IdemKey: "idem-" + requestID,
			BodyDigest: "sha256:" + strings.Repeat(seed, 32),
			Package:    "fake/lost", Entrypoint: "generate", Payload: []byte("{}"),
			Rental: true, Worker: rentalID,
		}); problem != nil {
			t.Fatal(problem.Message)
		}
	}

	// The rental holds three requests when it dies, and they are NOT the same case.
	// req-lost-queued-a and -b never reached a worker. req-lost-running was accepted by
	// one and may have partially executed.
	plant("rental-lost", "nitian")
	pin("req-lost-queued-a", "rental-lost", "a1")
	pin("req-lost-queued-b", "rental-lost", "b2")
	pin("req-lost-running", "rental-lost", "c3")
	session := "session-lost-running"
	fatal(t, store.SpawnWorker(records.WorkerProcess{
		InstanceID: "ins-lost", Package: "fake/lost", WorkerID: "remote", Devices: []string{"cpu"}}))
	attempt, problem := store.Dispatch(records.Attempt{
		RequestID: "req-lost-running", SessionID: session, InstanceID: "ins-lost",
		InvocationDigest: "sha256:" + strings.Repeat("d4", 32), InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, store.OfferDispatch("req-lost-running", attempt, session))
	fatal(t, store.Accepted("req-lost-running", attempt, session))

	// BEFORE: the rental is holding all three, and the listing says so. This is the state
	// the owner saw — and every part of it is true right up until the machine dies.
	queued, running, problem := store.RentalRunCounts("rental-lost")
	fatal(t, problem)
	if queued != 2 || running != 1 {
		t.Fatalf("the rental should be holding 2 queued and 1 running before it fails, got %d/%d",
			queued, running)
	}

	daemon := startDaemonProcess(t, root)
	_ = daemon

	// THE MACHINE DIES. Tensorhub reclaims the provider resource and serves the rental as
	// `failed`. That state — not an elapsed clock — is the whole trigger.
	hub.setState("rental-lost", "failed", "readiness.receipt_conflict")

	// AFTER: nothing is left pinned to it. Each request took the arm that fits it.
	deadline := time.Now().Add(60 * time.Second)
	for {
		queued, running, problem = store.RentalRunCounts("rental-lost")
		fatal(t, problem)
		recovered, problem := store.RequestRow("req-lost-running")
		fatal(t, problem)
		// Recovery commits the unpin before charging the retry budget. Zero
		// rental counts alone can observe the interval between those two facts.
		if queued == 0 && running == 0 && recovered != nil && recovered.Requeues > 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("rental recovery did not finish after 60s: %d queued, %d running, recovered request %+v\n%s",
				queued, running, recovered, tail(logPath))
		}
		time.Sleep(500 * time.Millisecond)
	}

	// (a) THE QUEUED ARM. Neither request ever reached a worker, so neither is charged a
	// requeue life and neither may still name the dead machine. `machine` is kept: which
	// pod a run waited on is history worth having (cl-107).
	for _, id := range []string{"req-lost-queued-a", "req-lost-queued-b"} {
		row, problem := store.RequestRow(id)
		fatal(t, problem)
		if row == nil {
			t.Fatalf("%s vanished", id)
		}
		if row.Worker == "rental-lost" {
			t.Fatalf("%s is still pinned to the dead rental (state %s)\n%s",
				id, row.State, tail(logPath))
		}
		if row.Requeues != 0 {
			t.Fatalf("%s was charged %d requeue(s) for an attempt it never made",
				id, row.Requeues)
		}
		if row.Machine != "nitian" {
			t.Fatalf("%s lost the machine word it ran against: %q", id, row.Machine)
		}
		// "If possible" needs an honest else-branch. This daemon's hub offers no SKU, so
		// nothing can serve the replan — and the request must SAY so, not wait forever.
		if !settled(row.State) && row.Worker != "" {
			t.Fatalf("%s is neither replanned nor settled: state %s, worker %q",
				id, row.State, row.Worker)
		}
	}

	// (b) THE IN-FLIGHT ARM is not the queued one. Its attempt was accepted by a worker and
	// may have partially executed, so it is closed as lost and re-offered through the
	// EXISTING requeue budget — a life is charged, and a request out of lives fails saying
	// so rather than being retried silently or stranded silently.
	inFlight := attemptRow(t, store, "req-lost-running", attempt)
	if inFlight.State != "closed" {
		t.Fatalf("the in-flight attempt is still %q; it owes a terminal no destroyed pod "+
			"can ever deliver\n%s", inFlight.State, tail(logPath))
	}
	if inFlight.TerminalStatus != "ABANDONED" || inFlight.TerminalCause != "EXECUTION_CONTEXT_LOST" {
		t.Fatalf("the in-flight attempt did not say WHY it ended: status %q cause %q",
			inFlight.TerminalStatus, inFlight.TerminalCause)
	}
	row, problem := store.RequestRow("req-lost-running")
	fatal(t, problem)
	if row.Worker == "rental-lost" {
		t.Fatalf("the re-offered request is still pinned to the dead rental; it would spend "+
			"its whole budget rediscovering that the machine is gone\n%s", tail(logPath))
	}
	if row.Requeues != 1 {
		t.Fatalf("an attempt that may have partially executed was re-offered without "+
			"charging the budget: requeues=%d", row.Requeues)
	}

	// A failed machine has left the current fleet; its retained diagnosis and
	// historical request association survive the inventory projection.
	if code, out := runCozy(t, root, "rental", "list", "--json", "--full"); code != 0 || strings.Contains(out, "rental-lost") {
		t.Fatalf("failed rental remained in the current fleet [exit %d]: %s", code, out)
	}
	stored, problem := store.RentalRow("rental-lost")
	fatal(t, problem)
	if stored == nil || stored.MachineName != "nitian" || stored.State != "failed" || stored.Failure.Code != "readiness.receipt_conflict" {
		t.Fatalf("failed rental history lost its diagnosis: %+v", stored)
	}

}

// TestRentalFailureKeepsARecordedTerminal is the arm that must NOT fire. An attempt whose
// worker already committed a terminal holds a real outcome that has not been acked yet.
// Recovery must leave it completely alone: replacing it with a synthetic failure would
// publish `request.failed` over a run that succeeded, which the dispatch code calls the
// worst failure class this system has.
func TestRentalFailureKeepsARecordedTerminal(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "rental-failure-terminal")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()

	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
		ID: "rental-terminal", MachineName: "gone", SKU: "cpu", AcceleratorModel: "CPU",
		HourlyRateUSDMicros: 100_000, State: "ready", Hub: "http://127.0.0.1:1",
		Address: "127.0.0.1:1", CertPath: filepath.Join(root, "rental-terminal.pem"),
	}))
	if _, _, problem := store.Submit(records.Request{
		ID: "req-terminal", IdemKey: "idem-req-terminal",
		BodyDigest: "sha256:" + strings.Repeat("e5", 32),
		Package:    "fake/lost", Entrypoint: "generate", Payload: []byte("{}"),
		Rental: true, Worker: "rental-terminal",
	}); problem != nil {
		t.Fatal(problem.Message)
	}
	session := "session-terminal"
	fatal(t, store.SpawnWorker(records.WorkerProcess{
		InstanceID: "ins-terminal", Package: "fake/lost", WorkerID: "remote", Devices: []string{"cpu"}}))
	digest := "sha256:" + strings.Repeat("f6", 32)
	attempt, problem := store.Dispatch(records.Attempt{
		RequestID: "req-terminal", SessionID: session, InstanceID: "ins-terminal",
		InvocationDigest: digest, InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, store.OfferDispatch("req-terminal", attempt, session))
	fatal(t, store.Accepted("req-terminal", attempt, session))
	applied, problem := store.AcceptTerminal(records.Terminal{
		RequestID: "req-terminal", Attempt: attempt, SessionID: session,
		InvocationDigest: digest, TerminalID: "out-req-terminal",
		TerminalDigest: "sha256:" + strings.Repeat("07", 32),
		Status:         "SUCCEEDED", Cause: "COMPLETED"})
	fatal(t, problem)
	if !applied {
		t.Fatal("the terminal was not recorded")
	}

	// The store refuses to strand it, whatever the caller believes about the machine.
	stranded, problem := store.AbandonLostAttempt("req-terminal", attempt, "the pod is gone",
		records.RequeueAfterLoss)
	fatal(t, problem)
	if stranded {
		t.Fatal("a recorded terminal was overwritten with a synthetic failure")
	}
	row := attemptRow(t, store, "req-terminal", attempt)
	if row.TerminalStatus != "SUCCEEDED" {
		t.Fatalf("the committed outcome changed to %q", row.TerminalStatus)
	}
}

func attemptRow(t *testing.T, store *records.Store, requestID string, attempt int64) records.Attempt {
	t.Helper()
	attempts, problem := store.Attempts(requestID)
	fatal(t, problem)
	for _, a := range attempts {
		if a.Attempt == attempt {
			return a
		}
	}
	t.Fatalf("%s#%d has no attempt row", requestID, attempt)
	return records.Attempt{}
}

func settled(state string) bool {
	switch state {
	case "succeeded", "failed", "canceled", "refused", "abandoned":
		return true
	}
	return false
}
