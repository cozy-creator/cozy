package records

import (
	"path/filepath"
	"testing"
)

func TestStableMachineNamePersistsWithoutRenamingRental(t *testing.T) {
	path := filepath.Join(t.TempDir(), "records.sqlite")
	s, problem := Open(path)
	if problem != nil {
		t.Fatal(problem)
	}
	row := Rental{ID: "pr-first", MachineName: "julian", SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, State: "acquiring"}
	if problem := s.RecordRental(row); problem != nil {
		t.Fatal(problem)
	}
	row.StableMachineName = "mikatan-356e23"
	if problem := s.RecordRental(row); problem != nil {
		t.Fatal(problem)
	}
	s.Close()
	s, problem = Open(path)
	if problem != nil {
		t.Fatal(problem)
	}
	defer s.Close()
	for _, name := range []string{"pr-first", "JULIAN", "MIKATAN-356E23"} {
		got, problem := s.RentalByMachine(name)
		if problem != nil || got == nil || got.ID != row.ID || got.MachineName != "julian" || got.DisplayName() != row.StableMachineName {
			t.Fatalf("lookup %q: %+v %v", name, got, problem)
		}
	}
	// An observation without a known physical host must fall back to the rental alias.
	row.StableMachineName = ""
	if problem := s.RecordRental(row); problem != nil {
		t.Fatal(problem)
	}
	got, problem := s.RentalByMachine("mikatan-356e23")
	if problem != nil || got != nil {
		t.Fatalf("unknown host retained a stale physical name: %+v %v", got, problem)
	}
	// A replan changes the host label while the operation alias stays fixed.
	row.StableMachineName = "elpeo-474f11"
	if problem := s.RecordRental(row); problem != nil {
		t.Fatal(problem)
	}
	if got, problem := s.RentalByMachine("mikatan-356e23"); problem != nil || got != nil {
		t.Fatalf("old host still names new placement: %+v %v", got, problem)
	}
	row.ID, row.MachineName = "pr-second", "egami"
	if problem := s.RecordRental(row); problem != nil {
		t.Fatal(problem)
	}
	if _, problem := s.RentalByMachine(row.StableMachineName); problem == nil || problem.ErrName() != "rental.machine_name_ambiguous" {
		t.Fatalf("same-host rentals resolved arbitrarily: %v", problem)
	}
	if got, problem := s.RentalByMachine("pr-first"); problem != nil || got == nil || got.ID != "pr-first" {
		t.Fatalf("ID stopped resolving on collision: %+v %v", got, problem)
	}
}
