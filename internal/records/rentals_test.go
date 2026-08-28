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

func TestConvergingRentalProjectionIsWriteOnceUntilReady(t *testing.T) {
	store, problem := Open(filepath.Join(t.TempDir(), "records.db"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	first := Rental{ID: "rental-1", EndpointRef: "cozy/a/v1/generate",
		AcceleratorModel: "H100", State: "converging", Hub: "https://hub.invalid",
		Address: "worker:443", MediaAddress: "media:443", CertPath: "/pins/worker.pem"}
	if problem := store.RecordRental(first); problem != nil {
		t.Fatal(problem)
	}
	ready := first
	ready.State = "ready"
	if problem := store.RecordRental(ready); problem != nil {
		t.Fatal(problem)
	}
	row, problem := store.RentalRow(first.ID)
	if problem != nil || row == nil || row.State != "ready" {
		t.Fatalf("ready row = %#v, %v", row, problem)
	}
	changed := ready
	changed.Address = "stranger:443"
	if problem := store.RecordRental(changed); problem == nil ||
		problem.ErrName() != "rental.attach_projection_conflict" {
		t.Fatalf("changed attach projection was not refused: %v", problem)
	}
	row, problem = store.RentalRow(first.ID)
	if problem != nil || row.Address != first.Address || row.MediaAddress != first.MediaAddress ||
		row.CertPath != first.CertPath {
		t.Fatalf("stored projection changed = %#v, %v", row, problem)
	}
	// A resumed POST may answer only identity + state before the first GET. It must not
	// erase the already-attached projection while the same operation resumes convergence.
	if problem := store.RecordRental(Rental{ID: first.ID, EndpointRef: first.EndpointRef,
		AcceleratorModel: first.AcceleratorModel, State: "converging", Hub: first.Hub}); problem != nil {
		t.Fatal(problem)
	}
	row, problem = store.RentalRow(first.ID)
	if problem != nil || row.State != "ready" || row.Address != first.Address ||
		row.MediaAddress != first.MediaAddress || row.CertPath != first.CertPath {
		t.Fatalf("sparse resume erased attached projection = %#v, %v", row, problem)
	}
}
