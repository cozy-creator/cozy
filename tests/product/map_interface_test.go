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
	// Runtime owns its type grammar; a map shape this host cannot read still loads and its
	// values are validated by Runtime at execution.
	for _, evolved := range []string{
		`"map":{"key":"str","value":"str","extra":true}`,
		`"map":{"key":"str","value":{"unknown":"str"}}`,
		`"map":{"key":{"unknown":"str"},"value":"str"}`,
	} {
		if _, problem := launch.DecodePackageInterface([]byte(strings.Replace(string(raw), valid, evolved, 1))); problem != nil {
			t.Fatalf("an evolved map schema refused the interface: %s: %s", evolved, problem.Message)
		}
	}
}
