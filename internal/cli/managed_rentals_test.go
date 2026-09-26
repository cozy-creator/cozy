package cli

import (
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestManagedRentalsWakeQueueAsync(t *testing.T) {
	woke := make(chan struct{}, 1)
	fleet := &managedRentals{wakeQueue: func() { woke <- struct{}{} }}
	fleet.wakeQueueAsync()
	select {
	case <-woke:
	case <-time.After(time.Second):
		t.Fatal("rental readiness did not wake queued machine execution")
	}
}

func TestLocalRentalAttachableRequiresPublishedControlTarget(t *testing.T) {
	base := records.Rental{State: "ready", Address: "worker:1", CertPath: "rental.pem"}
	if !localRentalAttachable(base) {
		t.Fatal("complete ready rental was not attachable")
	}
	for name, mutate := range map[string]func(*records.Rental){
		"acquiring":           func(row *records.Rental) { row.State = "acquiring" },
		"missing address":     func(row *records.Rental) { row.Address = "" },
		"missing certificate": func(row *records.Rental) { row.CertPath = "" },
	} {
		t.Run(name, func(t *testing.T) {
			row := base
			mutate(&row)
			if localRentalAttachable(row) {
				t.Fatal("incomplete rental was treated as attachable")
			}
		})
	}
}
