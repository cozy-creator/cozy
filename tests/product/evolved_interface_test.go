package producttest

import (
	"encoding/json"
	"strings"
	"testing"
)

// A package interface written by a newer Runtime carries members, constraints and type
// grammar this CLI has never seen (h3a-054: one unknown asset_bound member refused every
// run of a published release). The real run path still reaches model acquisition, and
// the readable bounds are still enforced early.
func TestRunAcceptsInterfaceMembersFromANewerRuntime(t *testing.T) {
	var doc map[string]any
	must(t, json.Unmarshal(admissionInterface(t, false, false), &doc))
	doc["capabilities"] = []any{"streaming"}
	ep := doc["entrypoints"].([]any)[0].(map[string]any)
	ep["timeout_s"] = 600
	ep["models"].([]any)[0].(map[string]any)["stamps"] = map[string]any{}
	ep["assets"].(map[string]any)["kinds"].([]any)[0].(map[string]any)["max_frames"] = 1
	request := ep["request"].(map[string]any)
	fields := request["fields"].([]any)
	prompt := fields[0].(map[string]any)
	prompt["description"] = "what to draw"
	prompt["constraints"].(map[string]any)["pattern"] = "^.+$"
	request["fields"] = append(fields, map[string]any{
		"name": "mask", "wire": "optional", "type": map[string]any{"tensor": map[string]any{"dtype": "f16"}},
	})
	raw, err := json.Marshal(doc)
	must(t, err)
	root, path, _, probe := admissionRoot(t, raw, "NVIDIA B200", false)

	code, out := runAdmissionCLI(t, root, path, "run", ladderPackage+"/generate", "--json",
		"--idempotency-key", "evolved-short", "prompt=", `mask:={"shape":[1]}`)
	if code == 0 || !strings.Contains(out, "prompt") || strings.Contains(out, "proof stopped before model acquisition") {
		t.Fatalf("the readable min_length bound was not enforced before acquisition [exit %d]: %s", code, out)
	}
	code, out = runAdmissionCLI(t, root, path, "run", ladderPackage+"/generate", "--json",
		"--idempotency-key", "evolved", "prompt=hello", `mask:={"shape":[1]}`)
	if _, lanes := probe.snapshot(); code == 0 || !strings.Contains(out, "proof stopped before model acquisition") || len(lanes) != 1 {
		t.Fatalf("a newer Runtime's interface members refused the run [exit %d]: %s", code, out)
	}
	assertAdmissionDidNotSubmit(t, root, "evolved")
}
