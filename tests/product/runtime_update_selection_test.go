package producttest

import (
	"os/exec"
	"testing"
)

func TestRentalRuntimeUpdatePublishedSelection(t *testing.T) {
	command := exec.Command("uv", "run", "--isolated", "--no-project", "--no-config", "--with", "packaging==26.2", "python", "testdata/runtime_update_selection.py")
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("published Runtime update selection: %v\n%s", err, output)
	}
	t.Log(string(output))
}
