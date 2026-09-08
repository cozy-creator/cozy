package packagepublish

import (
	"encoding/json"
	"reflect"
	"testing"
)

// uv's extra nodes are separate roots even when inactive. Only the caller's
// dependency edges activate them; neither ambient installs nor all roots do.
func wheelGraphFixture(t *testing.T) []byte {
	t.Helper()
	return []byte(`{
"schema":{"version":"preview"},"roots":[{"id":"caller"},{"id":"caller-dev"}],
"resolution":{
 "caller":{"name":"caller","version":"1","kind":"package","dependencies":[{"id":"library-jobs"},{"id":"unrelated"}],"optional_dependencies":[{"name":"dev","id":"caller-dev"}]},
 "caller-dev":{"name":"caller","version":"1","kind":{"extra":"dev"},"dependencies":[{"id":"unused"}]},
 "library":{"name":"library","version":"2","kind":"package","dependencies":[{"id":"base"}]},
 "library-jobs":{"name":"library","version":"2","kind":{"extra":"jobs"},"dependencies":[{"id":"library"},{"id":"scoring"}]},
 "base":{"name":"base","version":"3","kind":"package","dependencies":[]},
 "scoring":{"name":"scoring","version":"4","kind":"package","dependencies":[{"id":"base"}]},
 "unrelated":{"name":"unrelated","version":"5","kind":"package","dependencies":[]},
 "unused":{"name":"unused","version":"6","kind":"package","dependencies":[]}
}}`)
}

func TestWheelGraphSelectsCalleeExtrasWithoutCallerOrUnrelatedDependencies(t *testing.T) {
	installed := "base==3\ncaller==1\nlibrary==2\nscoring==4\nunrelated==5\nunused==6"
	graph, problem := readWheelClosures(wheelGraphFixture(t), installed, "caller", "")
	if problem != nil {
		t.Fatal(problem)
	}
	want := map[string]string{"library": "2", "base": "3", "scoring": "4"}
	if !reflect.DeepEqual(graph["library"], want) {
		t.Fatalf("library closure = %#v", graph["library"])
	}
	if graph["unused"] != nil || graph["caller"] != nil {
		t.Fatalf("inactive root or caller became a child: %#v", graph)
	}
	withoutAmbient := "base==3\ncaller==1\nlibrary==2\nscoring==4\nunrelated==5"
	again, problem := readWheelClosures(wheelGraphFixture(t), withoutAmbient, "caller", "")
	if problem != nil || !reflect.DeepEqual(again["library"], want) {
		t.Fatalf("ambient dependency affected callee: %v %#v", problem, again)
	}
}

func TestWheelGraphRefusesInstalledDriftAndCallerCycles(t *testing.T) {
	for _, mode := range []string{"missing", "version", "schema", "cycle", "extra"} {
		t.Run(mode, func(t *testing.T) {
			var graph map[string]any
			if err := json.Unmarshal(wheelGraphFixture(t), &graph); err != nil {
				t.Fatal(err)
			}
			installed := "base==3\ncaller==1\nlibrary==2\nscoring==4\nunrelated==5"
			extra := ""
			switch mode {
			case "missing":
				installed = "base==3\ncaller==1\nlibrary==2\nunrelated==5"
			case "version":
				installed = "base==3\ncaller==1\nlibrary==9\nscoring==4\nunrelated==5"
			case "schema":
				graph["schema"] = map[string]any{"version": "different"}
			case "cycle":
				graph["resolution"].(map[string]any)["base"].(map[string]any)["dependencies"] = []any{map[string]any{"id": "caller"}}
			case "extra":
				extra = "unknown"
			}
			raw, _ := json.Marshal(graph)
			if _, problem := readWheelClosures(raw, installed, "caller", extra); problem == nil {
				t.Fatal("accepted inconsistent locked graph")
			}
		})
	}
}
