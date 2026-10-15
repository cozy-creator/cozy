package producttest

import (
	"fmt"
	"strings"
	"testing"
)

func TestResultCollectionsRequireFixedNativeOutputPaths(t *testing.T) {
	for _, arm := range []struct {
		name, result, refusedPath string
	}{
		{"file-list", `{"list":{"asset":"file"}}`, ""}, // a list output: one `value.*` grant covers its items
		{"tree-list", `{"list":{"input":"tree"}}`, "value[]"},
		{"nested-list", `{"fields":[{"name":"images","type":{"list":{"asset":"image"}}}]}`, ""},
		{"list-struct", `{"list":{"fields":[{"name":"file","type":{"asset":"file"}}]}}`, "value[].file"},
		{"file-map", `{"map":{"key":"str","value":{"asset":"file"}}}`, "value[value]"},
		{"optional-file", `{"union":[{"asset":"file"},"null"]}`, "value"},
		{"tagged-union", `{"union":[{"tag":"a","fields":[{"name":"file","type":{"asset":"file"}}]},{"tag":"b","fields":[]}],"tag_field":"kind"}`, "value.file"},
		{"scalar-list", `{"list":"str"}`, ""},
		{"nested-scalar-list", `{"list":{"list":"int"}}`, ""},
		{"scalar-map", `{"map":{"key":"str","value":{"list":"float"}}}`, ""},
		{"scalar-union", `{"union":["int","null"]}`, ""},
		{"file", `{"asset":"file"}`, ""},
		{"tree", `{"input":"tree"}`, ""},
		{"fixed-struct", `{"fields":[{"name":"file","type":{"asset":"file"}},{"name":"tree","type":{"input":"tree"}},{"name":"scores","type":{"list":"float"}}]}`, ""},
	} {
		t.Run(arm.name, func(t *testing.T) {
			body := fmt.Sprintf(`{"format":"cozy.package.interface/1","application":"control:app","entrypoints":[{"name":"main","models":[],"request":{"fields":[{"name":"inputs","type":{"list":{"asset":"file"}}}]},"result":{"fields":[{"name":"value","type":%s}]},"publishes":false}]}`, arm.result)
			_, problem := callableOf(t, []byte(body), "main")
			if arm.refusedPath == "" {
				fatal(t, problem)
				return
			}
			if problem == nil || problem.ErrName() != "output_collection_unsupported" ||
				!strings.Contains(problem.Error(), "result "+arm.refusedPath+" contains") ||
				!strings.Contains(problem.Remedy, "Tree") {
				t.Fatalf("result collection did not refuse with a path/remedy: %v", problem)
			}
		})
	}
}
