package producttest

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
)

func TestAssetBundleHelpMatchesAutomaticEmptyInputs(t *testing.T) {
	for _, test := range []struct {
		name     string
		minimum  any
		bundle   bool
		optional bool
	}{
		{name: "implicit zero minimum", bundle: true, optional: true},
		{name: "explicit zero minimum", minimum: 0, bundle: true, optional: true},
		{name: "required references", minimum: 1, bundle: true},
		{name: "ordinary list", minimum: 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			var doc map[string]any
			must(t, json.Unmarshal(admissionInterface(t, false, false), &doc))
			ep := doc["entrypoints"].([]any)[0].(map[string]any)
			fields := ep["request"].(map[string]any)["fields"].([]any)
			assets := fields[1].(map[string]any)
			assets["name"] = "pictures"
			bounds := map[string]any{"max_length": 2}
			if test.minimum != nil {
				bounds["min_length"] = test.minimum
			}
			assets["constraints"] = bounds
			if test.bundle {
				ep["assets"].(map[string]any)["parameter"] = "pictures"
			} else {
				delete(ep, "assets")
				assets["type"] = map[string]any{"list": "str"}
			}
			// A same-named result list does not receive the input bundle default.
			ep["result"] = map[string]any{"fields": []any{map[string]any{
				"name": "pictures", "type": map[string]any{"list": "str"},
			}}}
			raw, err := json.Marshal(doc)
			must(t, err)
			parsed, problem := launch.DecodePackageInterface(raw)
			fatal(t, problem)
			entrypoint := &parsed.Entrypoints[0]
			for _, out := range []string{
				launch.DescribeContract("proof/generic/generate", entrypoint, nil),
				launch.DescribeArguments(entrypoint) + launch.UsageLine("proof/generic/generate", entrypoint),
			} {
				request := strings.Split(out, "  output:")[0]
				if strings.Contains(request, "(default = [])") != test.optional {
					t.Fatalf("asset optionality is wrong: %s", out)
				}
				want := "pictures:=<json>"
				if test.bundle {
					want = "[--asset <file>]..."
					if !test.optional {
						want = "--asset <file> " + want
					}
				}
				if !strings.Contains(out, want) || (test.bundle && strings.Contains(out, "pictures:=<json>")) {
					t.Fatalf("usage does not match asset input syntax: %s", out)
				}
				if output := strings.SplitN(out, "  output:", 2); len(output) == 2 && strings.Contains(output[1], "(default = [])") {
					t.Fatalf("input asset default leaked into output: %s", out)
				}
			}
		})
	}
}
