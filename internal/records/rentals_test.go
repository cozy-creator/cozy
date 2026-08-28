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
		State: "ready", Hub: "https://hub.invalid",
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
