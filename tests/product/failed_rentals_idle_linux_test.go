package producttest

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// Rentals that ended stay on the books, and the daemon asks the Hub about them on its poll.
// An answer that says nothing new writes nothing: the owner's laptop rewrote fourteen failed
// rentals every two seconds, re-settling their runs each time and waking every watcher of
// the records, at ~10% CPU.
func TestEndedRentalsTheHubRepeatsLeaveTheDaemonIdle(t *testing.T) {
	h := newFakeRentalHub(t, 0)
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+"\ntensorhub_token: rental-idle-test\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	for i := range 30 {
		id, name := fmt.Sprintf("pr-ended-%04d", i), fmt.Sprintf("ended-%d", i)
		fatal(t, store.RecordRental(records.Rental{ID: id, MachineName: name, AcceleratorModel: "CPU", AcceleratorCount: 1,
			HourlyRateUSDMicros: 1, State: "failed", Hub: h.server.URL, Failure: records.RentalFailure{Code: "supervisor_never_started"}}))
		h.mu.Lock()
		h.rentals[id] = map[string]any{"rental_id": id, "name": name, "state": "failed", "requested_accelerator_model": "CPU",
			"accelerator_count": 1, "hourly_rate_usd_micros": 1, "failure_code": "supervisor_never_started"}
		h.mu.Unlock()
	}
	store.Close()

	daemon := startDaemonProcess(t, root)
	wal := filepath.Join(root, "creator.sqlite-wal")
	time.Sleep(8 * time.Second)
	h.mu.Lock()
	read := h.reads
	h.mu.Unlock()
	if read == 0 {
		t.Fatal("the daemon never asked the Hub about its rentals")
	}
	written, _ := os.Stat(wal)
	before, began := cpuTime(t, daemon.cmd.Process.Pid), time.Now()
	time.Sleep(5 * time.Second)
	busy, elapsed := cpuTime(t, daemon.cmd.Process.Pid)-before, time.Since(began)
	h.mu.Lock()
	asked := h.reads - read
	h.mu.Unlock()
	if after, _ := os.Stat(wal); written != nil && after != nil && !after.ModTime().Equal(written.ModTime()) {
		t.Errorf("the daemon wrote its records while the Hub said nothing new (%d rental reads)", asked)
	}
	if busy > elapsed/50 {
		t.Fatalf("the daemon used %s of CPU in %s polling %d ended rentals", busy, elapsed, 30)
	}
	t.Logf("the daemon used %s of CPU in %s and asked the Hub %d times", busy, elapsed, asked)
}
