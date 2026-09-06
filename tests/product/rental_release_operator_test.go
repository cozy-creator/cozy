package producttest

import (
	"os/exec"
	"path/filepath"
	"testing"
)

func TestOperatorRentalReleaseUsesHubWithoutStartingDaemon(t *testing.T) {
	binary := filepath.Join(t.TempDir(), "rentalrelease")
	build := exec.Command("go", "build", "-o", binary, "./tests/support/rentalrelease")
	build.Dir = "../.."
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build operator: %v\n%s", err, output)
	}
	proof := exec.Command("python3", "testdata/rental-release.py", binary)
	if output, err := proof.CombinedOutput(); err != nil {
		t.Fatalf("operator Hub boundary: %v\n%s", err, output)
	}
}
