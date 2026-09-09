package producttest

import (
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
			busy := listedRental(t, root, rentalID)
			if busy.Queued == nil || *busy.Queued != 1 || busy.IdleSeconds != nil || busy.ReleaseDue != "" || peer.releases(rentalID) != 0 {
				t.Fatalf("preparation did not hold the rental: %+v", busy)
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
