package producttest

import (
	"os"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
)

func TestPublishedH3ToolsInterfaceAcceptsMaps(t *testing.T) {
	raw, err := os.ReadFile("testdata/h3-tools-2.9-package-interface.json")
	must(t, err)
	iface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	job, problem := iface.Function("retable")
	fatal(t, problem)
	if len(job.WeightsOutputs) != 4 || len(job.Result.Fields) != 3 {
		t.Fatalf("retable interface changed: %+v", job)
	}
	const valid = `"map":{"key":"str","value":"str"}`
	if !strings.Contains(string(raw), valid) {
		t.Fatal("published fixture no longer includes the map that blocked installation")
	}
	for _, invalid := range []string{
		`"map":{"key":"str"}`,
		`"map":{"value":"str"}`,
		`"map":{"key":"str","value":"str","extra":true}`,
		`"map":{"key":"str","value":{"unknown":"str"}}`,
		`"map":{"key":{"unknown":"str"},"value":"str"}`,
	} {
		if _, problem := launch.DecodePackageInterface([]byte(strings.Replace(string(raw), valid, invalid, 1))); problem == nil {
			t.Fatalf("malformed map schema accepted: %s", invalid)
		}
	}
}
