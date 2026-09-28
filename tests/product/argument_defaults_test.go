package producttest

import (
	"encoding/json"
	"strings"
	"testing"
)

// Drive both help surfaces through a real CLI process using the retained interface.
// Values come from the package's existing metadata, including explicit null/false/zero;
// absent metadata must not be guessed from optionality or a type's zero value.
func TestArgumentHelpShowsDeclaredDefaults(t *testing.T) {
	var doc map[string]any
	must(t, json.Unmarshal(admissionInterface(t, false, false), &doc))
	ep := doc["entrypoints"].([]any)[0].(map[string]any)
	request := ep["request"].(map[string]any)
	request["fields"] = append(request["fields"].([]any),
		map[string]any{"name": "seed", "type": map[string]any{"union": []any{"int", "null"}}, "wire": "optional"},
		map[string]any{"name": "duration_s", "type": "int", "wire": "optional", "constraints": map[string]any{"ge": 5, "le": 15}},
		map[string]any{"name": "frames", "type": "int", "constraints": map[string]any{"ge": 1, "le": 8}},
		map[string]any{"name": "settings", "wire": "optional", "type": map[string]any{"fields": []any{
			map[string]any{"name": "enabled", "type": "bool", "wire": "optional"},
			map[string]any{"name": "count", "type": "int", "wire": "optional"},
			map[string]any{"name": "label", "type": "str", "wire": "optional"},
		}}},
	)
	ep["result"] = map[string]any{"fields": []any{
		map[string]any{"name": "warnings", "type": map[string]any{"list": "str"}, "wire": "optional"},
	}}
	ep["invocable"] = map[string]any{
		"context": "ctx", "module": "assets", "export": "generate",
		"parameters": []string{"prompt", "assets", "steps", "seed", "duration_s", "frames", "settings"},
		"defaults": map[string]any{
			"request/seed": nil, "request/duration_s": 5,
			"request/settings/enabled": false, "request/settings/count": 0,
			"request/settings/label": "", "result/warnings": []string{},
		},
		"type_names": map[string]string{}, "enum_members": map[string]any{},
		"memoize": false, "capabilities": []string{},
	}
	raw, err := json.Marshal(doc)
	must(t, err)
	root, path, _, _ := admissionRoot(t, raw, "unmatched GPU", true)
	for _, test := range []struct {
		name string
		args []string
		code int
	}{
		{name: "describe", args: []string{"--describe"}},
		{name: "missing arguments", code: 1},
		{name: "invalid value", args: []string{"prompt=hello", "duration_s=16"}, code: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			args := append([]string{"run", ladderPackage + "/generate"}, test.args...)
			code, out := runAdmissionCLI(t, root, path, args...)
			for _, want := range []string{
				"prompt: str (min length 1)",
				"steps: int (>=1, <=50) (optional)",
				"seed: int|null (default = null)",
				"duration_s: int (>=5, <=15) (default = 5)",
				"frames: int (>=1, <=8)\n",
				"enabled: bool (default = false)",
				"count: int (default = 0)",
				`label: str (default = "")`,
			} {
				if code != test.code || !strings.Contains(out, want) {
					t.Errorf("%s omitted %q [exit %d]: %s", test.name, want, code, out)
				}
			}
			if test.code == 0 && !strings.Contains(out, "warnings: list[str] (default = [])") {
				t.Errorf("output default was lost: %s", out)
			}
			if strings.Contains(out, "Resolving model") {
				t.Errorf("argument help resolved a model: %s", out)
			}
		})
	}
	// The refusal itself names each missing field's type and bounds, for --json and API
	// callers that never see the Arguments listing.
	code, out := runAdmissionCLI(t, root, path, "--json", "run", ladderPackage+"/generate", "prompt=hello")
	if code != 1 || !strings.Contains(out, "provide required arguments: [frames: int (>=1, <=8)]") {
		t.Errorf("the --json refusal omitted the missing field's bounds [exit %d]: %s", code, out)
	}
}
