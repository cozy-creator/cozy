package rental

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestUsedRentalKeepaliveAcknowledgesNoDeadline(t *testing.T) {
	row := records.Rental{ID: "used", MachineName: "used", State: "ready", AcceleratorModel: "CPU", AcceleratorCount: 1,
		HourlyRateUSDMicros: 1, ExpectedWorkerID: "worker", ExpectedWorkerBootID: "boot"}
	receipt, problem := acknowledged(&row, "worker", "boot", 0)
	if problem != nil || receipt.IdleDeadlineMS != 0 {
		t.Fatalf("used-rental acknowledgment: %+v %v", receipt, problem)
	}
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.sqlite"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	if problem := store.RecordRental(row); problem != nil {
		t.Fatal(problem)
	}
	if problem := store.RecordRentalKeepalive(row.ID, receipt, time.Now()); problem != nil {
		t.Fatal(problem)
	}
	for _, check := range []struct {
		worker, boot string
		deadline     int64
	}{{"other", "boot", 0}, {"worker", "other", 0}, {"worker", "boot", -1}, {"worker", "boot", 1}} {
		if _, problem := acknowledged(&row, check.worker, check.boot, check.deadline); problem == nil {
			t.Fatalf("invalid acknowledgment accepted: %+v", check)
		}
	}
}
