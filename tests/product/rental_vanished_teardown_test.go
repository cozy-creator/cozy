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

// plantLostAttempt puts one request in the exact state that wedged the live daemon:
// pinned to a rental, holding an `accepted` attempt, with the rental record GONE. That is
// what `cozy rental end` leaves behind when it destroys a pod that was still holding work.
func plantLostAttempt(t *testing.T, store *records.Store, root, requestID, rentalID string) int64 {
	t.Helper()
	if _, _, problem := store.Submit(records.Request{
		ID: requestID, IdemKey: "idem-" + requestID,
		BodyDigest: "sha256:" + strings.Repeat("a7", 32),
		Package:    "fake/lost", Entrypoint: "generate", Payload: []byte("{}"),
		Rental: true, Worker: rentalID,
	}); problem != nil {
		t.Fatal(problem.Message)
	}
	session := "session-" + requestID
	fatal(t, store.SpawnWorker(records.WorkerProcess{
		InstanceID: "ins-" + requestID, Package: "fake/lost", WorkerID: "remote",
		Devices: []string{"cpu"}}))
	attempt, problem := store.Dispatch(records.Attempt{
		RequestID: requestID, SessionID: session, InstanceID: "ins-" + requestID,
		InvocationDigest: "sha256:" + strings.Repeat("b8", 32), InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, store.OfferDispatch(requestID, attempt, session))
	fatal(t, store.Accepted(requestID, attempt, session))
	return attempt
}

// TestVanishedRentalReleasesItsAttempt is req-b2df33d17e663a8a1e047246's exact state.
//
// The first cut of rental recovery keyed on OBSERVING a rental go `failed`. Once the rental
// RECORD was gone — which is what `cozy rental end` does — there was no object left to
// observe and nothing ever looked at the work again. The daemon said "holds a remote
// attempt with no terminal; reconnecting to its supervisor ledger" at boot, about a pod
// destroyed hours earlier, then logged "could not be settled: … has an attempt and is not a
// queued failure" every two seconds, forever. `cozy run list` read `in_progress 2590s` and
// climbing against a container that did not exist.
//
// Reconnecting cannot work when the ledger died with the pod. The hub's own knowledge that
// the rental is gone is sufficient evidence the attempt is lost, and it is a LOCAL durable
// fact — which matters, because the thing that would otherwise have to answer is the thing
// that vanished.
func TestVanishedRentalReleasesItsAttempt(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "rental-vanished")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	port := reservePort(t)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			"tensorhub_token: rental-idle-test\n"+
			"rentals:\n  max_hourly_spend_usd: 1.00\n  idle_release_s: 600\n"+
			"daemon:\n  idle_shutdown_s: 0\n"), 0o600))
	logPath := filepath.Join(root, "daemon.log")
	newFakeRentalHub(t, port)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()

	// No rental row is ever written: the pod was ended and its record forgotten while this
	// attempt was still open.
	attempt := plantLostAttempt(t, store, root, "req-vanished", "rental-that-was-ended")

	daemon := startDaemonProcess(t, root)
	_ = daemon

	deadline := time.Now().Add(60 * time.Second)
	for {
		row := attemptRow(t, store, "req-vanished", attempt)
		if row.State == "closed" {
			if row.TerminalStatus != "ABANDONED" ||
				row.TerminalCause != "EXECUTION_CONTEXT_LOST" {
				t.Fatalf("the lost attempt closed without saying why: status %q cause %q",
					row.TerminalStatus, row.TerminalCause)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the attempt of a request pinned to a VANISHED rental is still %q after 60s; "+
				"nothing observes a rental that no longer exists\n%s", row.State, tail(logPath))
		}
		time.Sleep(500 * time.Millisecond)
	}

	// And the request reached a stated outcome rather than sitting in `dispatching`.
	req, problem := store.RequestRow("req-vanished")
	fatal(t, problem)
	if req.Worker == "rental-that-was-ended" {
		t.Fatalf("the request is still pinned to a rental that does not exist\n%s", tail(logPath))
	}
	if req.State == "dispatching" {
		t.Fatalf("the request is still in_progress on a destroyed machine\n%s", tail(logPath))
	}

}

// TestDownAllIsNotRefusable is the escape hatch the owner ruled on: "you should be able to
// cozy down --all or force it to shut down (cancel all requests, shutdown all rentals and
// close)".
//
// Measured live on the code this was written against, with one request whose pod had been
// destroyed:
//
//	$ cozy down       -> daemon shutdown refused: active req-b2df33d1…(dispatching)
//	$ cozy down --all -> the stream that holds req-b2df33d17e663a8a1e047246#1 is gone
//
// `--all` refused for the SAME reason plain `down` did, so the documented escape from a
// wedged request was itself blocked by the wedged request, and the daemon could only be
// stopped with a signal. Cancelling an attempt normally means telling its worker to stop;
// when the worker is gone there is nobody to tell, and teardown may not wait on it.
func TestDownAllIsNotRefusable(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "down-all-wedged")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	port := reservePort(t)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			"tensorhub_token: rental-idle-test\n"+
			"rentals:\n  max_hourly_spend_usd: 1.00\n  idle_release_s: 600\n"+
			"daemon:\n  idle_shutdown_s: 0\n"), 0o600))
	hub := newFakeRentalHub(t, port)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()

	// A wedged request AND a live rental: teardown has to handle both, and the rental must
	// still be ended rather than skipped because the request was awkward.
	rentals := filepath.Join(root, "rentals")
	must(t, os.MkdirAll(rentals, 0o700))
	hub.add("rental-down-all", "vanished")
	must(t, os.WriteFile(filepath.Join(rentals, "rental-down-all.media-token"), []byte("m"), 0o600))
	must(t, os.WriteFile(filepath.Join(rentals, "rental-down-all.pem"),
		[]byte("-----BEGIN CERTIFICATE-----\n"), 0o600))
	_, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	key, err := x509.MarshalPKCS8PrivateKey(private)
	must(t, err)
	must(t, os.WriteFile(filepath.Join(rentals, "rental-down-all.creator.pem"),
		pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: key}), 0o600))
	fatal(t, store.RecordRental(records.Rental{
		ID: "rental-down-all", MachineName: "vanished", SKU: "cpu", AcceleratorModel: "CPU",
		HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL,
		Address: "127.0.0.1:1", CertPath: filepath.Join(rentals, "rental-down-all.pem"),
	}))
	// PINNED TO THE HEALTHY-LOOKING RENTAL ON PURPOSE. If this named a vanished rental the
	// recovery sweep would settle it and the teardown would never be tested. Here the rental
	// row is `ready`, so recovery correctly leaves the attempt alone — and the only thing
	// standing between the operator and a stopped daemon is teardown's own handling of a
	// control stream this process does not hold. "Never refusable" is its own property and
	// gets its own arm.
	plantLostAttempt(t, store, root, "req-wedged", "rental-down-all")

	daemon := startDaemonProcess(t, root)

	// THE LOOP IS BOUNDED. `failQueued` is reached for this request at boot — routing
	// cannot attach to 127.0.0.1:1 — and the store permanently refuses a queued failure for
	// a request holding an open attempt. On the code this arm was written against that
	// refusal was rescheduled every 2 s forever: a permanent condition restated 1200 times
	// an hour, filling the log and never surfacing. It must be said once.
	time.Sleep(7 * time.Second)
	logPath := filepath.Join(root, "daemon.log")
	log, _ := os.ReadFile(logPath)
	repeats := strings.Count(string(log), "could not be settled: request req-wedged")
	repeats += strings.Count(string(log), "req-wedged cannot be failed as queued work")
	if repeats > 1 {
		t.Fatalf("a permanent refusal was restated %d times; it should be said once\n%s",
			repeats, tail(logPath))
	}

	// Plain `down` MAY refuse — that is its job, and it names what is holding it.
	// `--all` may not, and it must actually stop the process.
	code, out := runCozy(t, root, "down", "--all")
	if code != 0 {
		t.Fatalf("`cozy down --all` refused with a wedged request [exit %d]; the daemon "+
			"cannot be stopped by any documented means\n%s", code, out)
	}
	if code := awaitDaemonExit(t, daemon, 30*time.Second); code != 0 {
		t.Fatalf("`cozy down --all` reported success but the daemon exited %d\n%s", code, out)
	}
	if hub.releases("rental-down-all") != 1 {
		t.Fatalf("the rental was not ended by the teardown (%d releases)\n%s",
			hub.releases("rental-down-all"), out)
	}
	// It cancelled the owner's in-flight work by design, so it has to SAY so.
	if !strings.Contains(out, "canceled") {
		t.Fatalf("the teardown did not report what it destroyed\n%s", out)
	}
}
