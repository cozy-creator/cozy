package producttest

import (
	"encoding/json"
	"testing"
)

type machineTiming struct {
	ID             string `json:"id"`
	AttemptWallMS  int64  `json:"attempt_wall_ms"`
	ExecutionMS    *int64 `json:"execution_ms"`
	ExecutionKnown bool   `json:"execution_known"`
}

func listedMachineTiming(t *testing.T, root, id string) machineTiming {
	t.Helper()
	code, output := runCozy(t, root, "run", "list", "--json", "--full")
	var document struct {
		Invocations []machineTiming `json:"invocations"`
	}
	if code != 0 || json.Unmarshal([]byte(output), &document) != nil {
		t.Fatalf("list retained timing [%d]: %s", code, output)
	}
	for _, row := range document.Invocations {
		if row.ID == id {
			return row
		}
	}
	t.Fatalf("run %s missing from timing readback: %s", id, output)
	return machineTiming{}
}
