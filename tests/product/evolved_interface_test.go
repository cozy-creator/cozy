package producttest

import (
	"encoding/json"
	"strings"
	"testing"
)

// A package interface written by a newer Runtime carries a newer format, members, constraints,
// type grammar and a callable this CLI has never seen (h3a-054: one unknown asset_bound member
// refused every run of a published release). The real run path still reaches the machine, the
// readable bounds are still enforced before anything is submitted, and only the callable this
// CLI cannot read refuses, saying why.
func TestRunAcceptsInterfaceMembersFromANewerRuntime(t *testing.T) {
	var doc map[string]any
	must(t, json.Unmarshal(admissionInterface(t, false, false), &doc))
	doc["format"] = "cozy.package.interface/2"
	doc["capabilities"] = []any{"streaming"}
	doc["entrypoints"] = append(doc["entrypoints"].([]any), map[string]any{"name": "tune",
		"models":  []any{map[string]any{"class": "H3", "path": "tune.weights.model", "component_use": map[string]any{}}},
		"request": map[string]any{"fields": []any{}}, "result": map[string]any{"fields": []any{}}})
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
	root, path, _, _ := admissionRoot(t, raw, "NVIDIA B200", false)

	code, out := runAdmissionCLI(t, root, path, "run", ladderPackage+"/generate", "--json",
		"--idempotency-key", "evolved-short", "prompt=", `mask:={"shape":[1]}`)
	if code == 0 || !strings.Contains(out, "prompt") || strings.Contains(out, "proof stopped before model acquisition") {
		t.Fatalf("the readable min_length bound was not enforced before acquisition [exit %d]: %s", code, out)
	}
	assertAdmissionDidNotSubmit(t, root, "evolved-short")
	code, out = runAdmissionCLI(t, root, path, "run", ladderPackage+"/generate", "--json",
		"--idempotency-key", "evolved", "prompt=hello", `mask:={"shape":[1]}`)
	if code == 0 || !strings.Contains(out, `"machine":"local"`) {
		t.Fatalf("a newer Runtime's interface members refused the run before it reached the machine [exit %d]: %s", code, out)
	}
	code, out = runAdmissionCLI(t, root, path, "run", ladderPackage+"/tune", "--json", "--idempotency-key", "evolved-tune")
	if code == 0 || !strings.Contains(out, `tune has invalid model path \"tune.weights.model\"`) {
		t.Fatalf("the callable this CLI cannot read did not refuse with its reason [exit %d]: %s", code, out)
	}
	assertAdmissionDidNotSubmit(t, root, "evolved-tune")
}
