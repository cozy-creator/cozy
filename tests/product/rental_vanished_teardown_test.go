package producttest

import (
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
	root := filepath.Join(scratchBase, "down-all-wedged")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	hub := newFakeRentalHub(t, 0)
	port := hub.port()
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			"tensorhub_token: rental-idle-test\n"+
			""+
			"daemon:\n  idle_shutdown_s: 0\n"), 0o600))
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
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
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

	logPath := filepath.Join(root, "daemon.log")
	// Plain `down` MAY refuse — that is its job, and it names what is holding it.
	// `--all` may not, and it must actually stop the process.
	code, out := runCozy(t, root, "down", "--all")
	if code != 0 {
		t.Fatalf("`cozy down --all` refused with a wedged request [exit %d]; the daemon "+
			"cannot be stopped by any documented means\n%s\n%s", code, out, tail(logPath))
	}
	if code := awaitDaemonExit(t, daemon, 30*time.Second); code != 0 {
		t.Fatalf("`cozy down --all` reported success but the daemon exited %d\n%s", code, out)
	}
	if hub.releases("rental-down-all") != 1 {
		t.Fatalf("the rental was not ended by the teardown (%d releases)\n%s",
			hub.releases("rental-down-all"), out)
	}
	// It cancelled the owner's in-flight work by design, so it has to SAY so.
	if !strings.Contains(out, "cancelled") {
		t.Fatalf("the teardown did not report what it destroyed\n%s", out)
	}
}
