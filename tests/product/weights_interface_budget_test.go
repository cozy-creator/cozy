package producttest

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// Graft-only jobs write no new tensor payload. Their explicit zero budget must
// survive interface admission, while absent or malformed limits still refuse.
func TestWeightsInterfaceAcceptsExplicitZeroNewBytes(t *testing.T) {
	for _, arm := range []struct {
		name string
		cap  string
		ok   bool
	}{
		{"graft", `,"max_bytes":0`, true},
		{"one-byte", `,"max_bytes":1`, true},
		{"maximum", `,"max_bytes":9007199254740991`, true},
		{"absent", "", false},
		{"null", `,"max_bytes":null`, false},
		{"negative", `,"max_bytes":-1`, false},
		{"fraction", `,"max_bytes":0.5`, false},
		{"boolean", `,"max_bytes":false`, false},
		{"string", `,"max_bytes":"0"`, false},
		{"overflow", `,"max_bytes":9007199254740992`, false},
	} {
		t.Run(arm.name, func(t *testing.T) {
			row := fmt.Sprintf(`{"output_id":"model","mime_type":%q%s}`, orchestrator.WeightsManifestMime, arm.cap)
			body := fmt.Sprintf(`{"format":"cozy.package.interface/1","application":"graft:app","entrypoints":[],"jobs":[{"name":"project","publishes":false,"request":{"fields":[]},"result":{"fields":[]},"weights_outputs":[%s]}]}`, row)
			parsed, problem := launch.DecodePackageInterface([]byte(body))
			if (problem == nil) != arm.ok {
				t.Fatalf("admission = %v, want accepted=%v", problem, arm.ok)
			}
			if !arm.ok {
				return
			}
			var expected launch.WeightsOutput
			must(t, json.Unmarshal([]byte(row), &expected))
			if got := parsed.Jobs[0].WeightsOutputs[0]; got != expected {
				t.Fatalf("admission changed the output contract: %+v != %+v", got, expected)
			}
		})
	}
}
