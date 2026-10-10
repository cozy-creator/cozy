package rental

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A machine that names no deadline (TensorD 0.5.2–0.5.5 after a job) is still an acknowledgment;
// another worker or boot, or a deadline already past, is not.
func TestKeepaliveAcceptsAMachineNamingNoDeadline(t *testing.T) {
	row := records.Rental{ID: "older", State: "ready", ExpectedWorkerID: "worker", ExpectedWorkerBootID: "boot"}
	if receipt, problem := acknowledged(&row, "worker", "boot", 0); problem != nil || receipt.IdleDeadlineMS != 0 {
		t.Fatalf("no-deadline acknowledgment: %+v %v", receipt, problem)
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
