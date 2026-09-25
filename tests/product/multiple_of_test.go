package producttest

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
)

func multipleOfInterface(kind, multiple string) []byte {
	return []byte(fmt.Sprintf(`{"application":"example:app","entrypoints":[{"name":"generate","models":[],"request":{"fields":[{"name":"value","type":%q,"constraints":{"multiple_of":%s}}]},"result":{"fields":[]}}],"jobs":[]}`, kind, multiple))
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

func TestMultipleOfInvalidSchemaRefusesEarly(t *testing.T) {
	for _, value := range []string{"0", "-32", "null", `"32"`, "1e400"} {
		if _, problem := launch.DecodePackageInterface(multipleOfInterface("int", value)); problem == nil {
			t.Fatalf("invalid multiple_of %s accepted", value)
		}
	}
}
