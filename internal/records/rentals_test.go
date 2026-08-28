package records

import (
	"path/filepath"
	"testing"
)

func TestArtifactGrantRevisionSurvivesRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "records.db")
	store, problem := Open(path)
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := store.RecordRental(Rental{
		ID: "rental-1", EndpointRef: "cozy/endpoint/v1/generate", AcceleratorModel: "H100",
		State: "ready", Hub: "https://hub.invalid", PlacementRevision: 1,
	}); problem != nil {
		t.Fatal(problem)
	}
	first, problem := store.ReserveArtifactGrantRevision("rental-1")
	if problem != nil || first != 1 {
		t.Fatalf("first revision = %d, %v", first, problem)
	}
	store.Close()

	store, problem = Open(path)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	second, problem := store.ReserveArtifactGrantRevision("rental-1")
	if problem != nil || second != 2 {
		t.Fatalf("second revision after restart = %d, %v", second, problem)
	}
}

func TestRentalControlRevisionAdvancesAtomically(t *testing.T) {
	store, problem := Open(filepath.Join(t.TempDir(), "records.db"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	if problem := store.RecordRental(Rental{ID: "rental-1", EndpointRef: "cozy/a/v1/generate",
		AcceleratorModel: "H100", State: "ready", Hub: "https://hub.invalid",
		PlacementRevision: 1, ControlSnapshotDigest: "sha256:old", ControlSnapshotBytes: []byte("old")}); problem != nil {
		t.Fatal(problem)
	}
	if problem := store.ReviseRentalControl("rental-1", "cozy/b/v1/generate", 2,
		"sha256:new", []byte("new")); problem != nil {
		t.Fatal(problem)
	}
	row, problem := store.RentalRow("rental-1")
	if problem != nil || row == nil || row.PlacementRevision != 2 ||
		row.EndpointRef != "cozy/b/v1/generate" || string(row.ControlSnapshotBytes) != "new" {
		t.Fatalf("revised rental = %#v, %v", row, problem)
	}
	if problem := store.ReviseRentalControl("rental-1", "cozy/a/v1/generate", 1,
		"sha256:old", []byte("old")); problem == nil ||
		problem.ErrName() != "rental.placement_revision_regressed" {
		t.Fatalf("stale revision was not refused: %v", problem)
	}
}
