package records

import (
	"path/filepath"
	"testing"
)

func TestPlacementAcquisitionPersistsAndRefusesRegression(t *testing.T) {
	store, problem := Open(filepath.Join(t.TempDir(), "records.db"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	observation := PlacementAcquisition{
		InstanceID: "ins-1", WorkerBootID: "boot-1", PlacementID: "placement-1",
		PlacementSpecDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Endpoint:            AcquisitionLeg{StartedNS: 10, EndedNS: 30, DownloadedBytes: 100},
		Model:               AcquisitionLeg{StartedNS: 20, EndedNS: 40, DownloadedBytes: 200},
	}
	if problem := store.ObservePlacementAcquisition(observation); problem != nil {
		t.Fatal(problem)
	}
	got, problem := store.PlacementAcquisition(observation.InstanceID, observation.WorkerBootID,
		observation.PlacementID, observation.PlacementSpecDigest)
	if problem != nil || got == nil || got.Endpoint.DownloadedBytes != 100 || got.Model.EndedNS != 40 {
		t.Fatalf("persisted observation = %#v, %v", got, problem)
	}
	observation.Model.DownloadedBytes = 199
	if problem := store.ObservePlacementAcquisition(observation); problem == nil ||
		problem.ErrName() != "placement.acquisition_regressed" {
		t.Fatalf("regressed observation was not refused: %v", problem)
	}
}
