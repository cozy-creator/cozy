package records

import (
	"path/filepath"
	"testing"
)

func TestObserveRentalWorkerCPU(t *testing.T) {
	store, problem := Open(filepath.Join(t.TempDir(), "records.db"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	if problem := store.RecordRental(Rental{
		ID: "pr-cpu", EndpointRef: "cozy/marco-polo-cpu/v1/marco",
		AcceleratorModel: "CPU", State: "converging", Hub: "http://hub",
	}); problem != nil {
		t.Fatal(problem)
	}
	if problem := store.ObserveRentalWorker(
		"pr-cpu", "", "none", "", "", 0, "instance-cpu", "boot-1", 0,
	); problem != nil {
		t.Fatal(problem)
	}
	row, problem := store.RentalRow("pr-cpu")
	if problem != nil || row == nil || row.ObservedBackend != "none" ||
		row.ObservedAccelerator != "" || row.ObservedAcceleratorCount != 0 ||
		row.ObservedWorkerInstance != "instance-cpu" || row.ObservedAt == "" {
		t.Fatalf("CPU observation = %+v, %v", row, problem)
	}
	if problem := store.ObserveRentalWorker(
		"pr-cpu", "", "none", "", "", 0, "instance-cpu", "boot-2", 0,
	); problem != nil {
		t.Fatalf("same CPU worker restart refused: %v", problem)
	}
	if problem := store.ObserveRentalWorker(
		"pr-cpu", "GPU", "cuda", "580.1", "13.0", 1, "instance-cpu", "boot-3", 1,
	); problem == nil {
		t.Fatal("CPU rental accepted GPU Claim facts")
	}
}
