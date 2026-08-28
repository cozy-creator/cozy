package orchestrator

import "testing"

func TestReleasePinAcceptsDynamicAttachedSilence(t *testing.T) {
	attached := &worker{spec: WorkerLaunchSpec{
		Placement:  DesiredPlacement{ReleaseID: "cozy/probe@cold"},
		Connection: &WorkerConnection{RentalID: "rnt-test"},
	}}
	if err := releasePin(attached, ""); err != nil {
		t.Fatalf("dynamic attached ClaimAck silence was refused: %v", err)
	}
	if err := releasePin(attached, "cozy/probe@warm"); err == nil || err.Name != "release_mismatch" {
		t.Fatalf("stale attached release was not refused: %#v", err)
	}
	if err := releasePin(attached, "cozy/probe@cold"); err != nil {
		t.Fatalf("matching declared attached release was refused: %v", err)
	}
}
