package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/devcomfy"
)

func TestInputDefaultsAndRefusals(t *testing.T) {
	input, err := devcomfy.ParseInput([]byte(`{"graph_json":"{\"1\":{}}","output_root":"/outputs"}`))
	if err != nil || input.Port != 8188 || input.TimeoutS != 3600 || input.ExpectedSteps != 8 {
		t.Fatalf("%+v %v", input, err)
	}
	// Any positive step count: a benchmark graph runs the steps its model takes.
	if input, err = devcomfy.ParseInput([]byte(`{"graph_json":"{\"1\":{}}","output_root":"/outputs","expected_steps":40}`)); err != nil || input.ExpectedSteps != 40 {
		t.Fatalf("40 steps refused: %+v %v", input, err)
	}
	for _, raw := range []string{`{"graph_json":"{}","output_root":"/out"}`, `{"graph_json":"{\"1\":{}}","output_root":"relative"}`, `{"graph_json":"{\"1\":{}}","output_root":"/out","port":0}`, `{"graph_json":"{\"1\":{}}","output_root":"/out","expected_steps":0}`, `{"graph_json":"{\"1\":{}}","output_root":"/out","network":true}`} {
		if _, err = devcomfy.ParseInput([]byte(raw)); err == nil {
			t.Fatal("accepted", raw)
		}
	}
}
