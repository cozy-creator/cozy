package producttest

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
)

func multipleOfInterface(kind, multiple string) []byte {
	return []byte(fmt.Sprintf(`{"format":"cozy.package.interface/1","application":"example:app","entrypoints":[{"name":"generate","models":[],"request":{"fields":[{"name":"value","type":%q,"constraints":{"multiple_of":%s}}]},"result":{"fields":[]}}],"jobs":[]}`, kind, multiple))
}

func TestMultipleOfInterfaceAndExactPayload(t *testing.T) {
	for _, example := range []struct {
		kind, multiple, value string
		valid                 bool
	}{
		{"int", "32", "1024", true}, {"int", "32", "1025", false},
		{"int", "32", "0", true}, {"int", "32", "-64", true},
		{"float", "0.1", "0.3", true}, {"float", "0.1", "0.31", false},
		{"float", "1e-3", "0.003", true}, {"float", "0.1", "0.30000000000000004", false},
	} {
		t.Run(example.kind+"/"+example.multiple+"/"+example.value, func(t *testing.T) {
			surface, problem := launch.DecodePackageInterface(multipleOfInterface(example.kind, example.multiple))
			fatal(t, problem)
			entry, problem := surface.Function("generate")
			fatal(t, problem)
			if entry.Request.Fields[0].Constraints.MultipleOf == nil {
				t.Fatal("multiple_of was dropped")
			}
			problem = launch.ValidatePayload("paul/example", entry, json.RawMessage(`{"value":`+example.value+`}`))
			if (problem == nil) != example.valid {
				t.Fatalf("valid=%v problem=%v", example.valid, problem)
			}
			if problem != nil && !strings.Contains(problem.Message, "multiple of") {
				t.Fatal(problem)
			}
		})
	}
}

// A bound this host cannot evaluate is left to Runtime's execution-time validation; it
// neither refuses the package nor the request.
func TestMultipleOfUnreadableBoundIsLeftToRuntime(t *testing.T) {
	for _, value := range []string{"0", "-32", "null"} {
		surface, problem := launch.DecodePackageInterface(multipleOfInterface("int", value))
		if problem != nil {
			t.Fatalf("multiple_of %s refused the package interface: %s", value, problem.Message)
		}
		entry, problem := surface.Function("generate")
		fatal(t, problem)
		if problem := launch.ValidatePayload("paul/example", entry, json.RawMessage(`{"value":7}`)); problem != nil {
			t.Fatalf("multiple_of %s refused the request: %s", value, problem.Message)
		}
	}
}

var referenceImageInterface = flag.String("reference-image-interface", "", "current reference-image package-interface.json")

func TestMultipleOfActualReferenceImageInterface(t *testing.T) {
	path := *referenceImageInterface
	if path == "" {
		t.Skip("requires the current reference-image interface")
	}
	raw, err := os.ReadFile(path)
	must(t, err)
	surface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	entry, problem := surface.Function("generate")
	fatal(t, problem)
	for _, width := range []int{1024, 1025} {
		payload := json.RawMessage(fmt.Sprintf(`{"prompt":"A character on white","width":%d,"height":1024}`, width))
		problem = launch.ValidatePayload("paul/reference-image", entry, payload)
		if (problem == nil) != (width == 1024) {
			t.Fatalf("width=%d problem=%v", width, problem)
		}
	}
}
