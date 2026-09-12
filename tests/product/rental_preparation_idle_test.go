package producttest

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentalSettlementOrdersWholeAndFractionalSeconds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	defer store.Close()
	const rentalID, first, second = "rental-clock", "job-first", "job-second"
	submit := func(id string) {
		t.Helper()
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, Kind: "job",
			Package: "proof/clock", Entrypoint: "run", Payload: []byte("{}"),
			BodyDigest: "sha256:" + strings.Repeat("a", 64), Rental: true, Worker: rentalID})
		fatal(t, problem)
	}
	submit(first)
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "ins-clock", Package: "proof/clock", WorkerID: rentalID, Devices: []string{"cpu"}}))
	digest := "sha256:" + strings.Repeat("b", 64)
	ordinal, problem := store.Dispatch(records.Attempt{RequestID: first, SessionID: "boot-clock",
		InstanceID: "ins-clock", InvocationDigest: digest, InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, store.OfferDispatch(first, ordinal, "boot-clock"))
	fatal(t, store.Accepted(first, ordinal, "boot-clock"))
	_, problem = store.AcceptTerminal(records.Terminal{RequestID: first, Attempt: ordinal,
		SessionID: "boot-clock", InvocationDigest: digest, TerminalID: "out-clock",
		TerminalDigest: "sha256:" + strings.Repeat("c", 64), Status: "FAILED", Cause: "EXCEPTION",
		EventType: "request.failed"})
	fatal(t, problem)
	fatal(t, store.Closed(first, ordinal))
	// Replay controlled historical timestamps through real records. The producer's
	// RFC3339Nano formatter emits both spellings; SQL string MAX reverses them.
	db, err := sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	whole := "2026-09-09T05:24:32Z"
	fraction := "2026-09-09T05:24:32.456Z"
	_, err = db.Exec(`UPDATE attempts SET closed_at=? WHERE request_id=?`, whole, first)
	must(t, err)
	_, err = db.Exec(`UPDATE request_events SET at=? WHERE request_id=? AND type='request.failed'`, fraction, first)
	must(t, err)
	check := func(id, at string, attempted bool) {
		t.Helper()
		last, found, problem := store.RentalLastSettlement(rentalID)
		fatal(t, problem)
		if !found || last.RequestID != id || last.SettledAt.Format(time.RFC3339Nano) != at || last.ClosedAt.IsZero() == attempted {
			t.Fatalf("wrong chronological settlement or attempt classification: %+v", last)
		}
	}
	check(first, fraction, true)
	submit(second)
	_, problem = store.FailQueuedRequest(second, nil)
	fatal(t, problem)
	_, err = db.Exec(`UPDATE request_events SET at=? WHERE request_id=? AND type='request.failed'`, whole, second)
	must(t, err)
	check(first, fraction, true)
	later := "2026-09-09T05:24:32.789Z"
	_, err = db.Exec(`UPDATE request_events SET at=? WHERE request_id=? AND type='request.failed'`, later, second)
	must(t, err)
	check(second, later, false)
}

// Preparation can spend minutes on a paid pod before an attempt exists. The
// terminal request event ends that work even when the controller was restarted
// between observations. Run467/468 lost 105 GB because only attempt closure
// advanced the idle clock, making the ready timestamp the deadline again.
func TestRentalIdleGraceAfterPreparationSettlementAndRestart(t *testing.T) {
	for _, outcome := range []string{"failed", "canceled"} {
		t.Run(outcome, func(t *testing.T) {
			root := t.TempDir()
			defer func() {
				if t.Failed() {
					t.Log(tail(filepath.Join(root, "daemon.log")))
				}
			}()
			port := reservePort(t)
			origin := fmt.Sprintf("http://127.0.0.1:%d", port)
			const grace = 5 * time.Second
			must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
				"tensorhub_url: "+origin+"\ntensorhub_token: rental-idle-test\n"+
					"rentals:\n  max_hourly_spend_usd: 1\n  idle_release_s: 5\n"+
					"daemon:\n  idle_shutdown_s: 0\n"), 0600))
			peer := newFakeRentalHub(t, port)
			peer.publishListing()
			const rentalID, requestID = "rental-preparation-idle", "job-preparation-idle"
			peer.add(rentalID, "sherlock")
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			fatal(t, store.RecordRental(records.Rental{ID: rentalID, MachineName: "sherlock",
				SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, State: "ready",
				HourlyRateUSDMicros: 100000, Hub: origin, Address: "127.0.0.1:1",
				CertPath: filepath.Join(root, "unavailable.pem"),
				ReadyAt:  time.Now().Add(-time.Hour).UTC().Format(time.RFC3339Nano)}))
			_, _, problem = store.Submit(records.Request{ID: requestID, IdemKey: requestID,
				BodyDigest: "sha256:" + strings.Repeat("a", 64), Kind: "job",
				Package: "proof/preparing", Entrypoint: "prepare", Payload: []byte("{}"),
				Rental: true, RentalRequired: true, Worker: rentalID, RequestedRental: rentalID})
			fatal(t, problem)
			// Observe the queued setup before starting the real daemon below.
			// The listing command now starts it, allowing the intentional missing-
			// bearer refusal to settle before an initial queued-state assertion.
			queued, running, problem := store.RentalRunCounts(rentalID)
			fatal(t, problem)
			if queued != 1 || running != 0 || peer.releases(rentalID) != 0 {
				t.Fatalf("preparation setup lost queued work: queued=%d running=%d", queued, running)
			}
			if outcome == "failed" {
				// The real daemon rejects preparation: this recorded peer has no
				// media credential. No attempt exists, exactly as with run467.
				d := startDaemonProcess(t, root)
				until := time.Now().Add(10 * time.Second)
				for {
					r, problem := store.RequestRow(requestID)
					fatal(t, problem)
					if r.State == "failed" {
						break
					}
					if time.Now().After(until) {
						t.Fatalf("preparation never settled: %+v", r)
					}
					time.Sleep(20 * time.Millisecond)
				}
				must(t, d.cmd.Process.Signal(syscall.SIGTERM))
				select {
				case code := <-d.exited:
					if code != 0 {
						t.Fatalf("daemon exit=%d", code)
					}
				case <-time.After(10 * time.Second):
					t.Fatal("daemon did not stop")
				}
			} else {
				// No idle sweep can observe this transition; use its durable event.
				changed, problem := store.CancelQueuedRequest(requestID, nil)
				fatal(t, problem)
				if !changed {
					t.Fatal("preparation cancellation did not settle")
				}
			}
			events, problem := store.EventsAfter(requestID, 0, 100)
			fatal(t, problem)
			var settled time.Time
			for _, event := range events {
				if records.TerminalEvent(event.Type) {
					var err error
					settled, err = time.Parse(time.RFC3339Nano, event.At)
					must(t, err)
				}
			}
			if settled.IsZero() {
				t.Fatal("missing durable settlement event")
			}
			idle := listedRental(t, root, rentalID)
			if want := settled.Add(grace).UTC().Format(time.RFC3339); idle.ReleaseDue != want {
				t.Fatalf("idle deadline=%q, want terminal event + grace %q (ready was an hour ago)", idle.ReleaseDue, want)
			}
			startDaemonProcess(t, root)
			deadline := settled.Add(grace)
			for time.Now().Before(deadline) {
				if peer.releases(rentalID) != 0 {
					t.Fatal("restarted daemon released before the new idle deadline")
				}
				time.Sleep(20 * time.Millisecond)
			}
			awaitRentalGone(t, store, rentalID, 10*time.Second, filepath.Join(root, "daemon.log"))
			if peer.releases(rentalID) != 1 {
				t.Fatal("idle rental was not released exactly once")
			}
			attempts, problem := store.Attempts(requestID)
			fatal(t, problem)
			if len(attempts) != 0 {
				t.Fatal("preparation fixture crossed the attempt boundary")
			}
		})
	}
}
