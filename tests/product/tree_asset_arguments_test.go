package producttest

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
)

func treeArgumentEntry(t *testing.T) *launch.Entrypoint {
	t.Helper()
	raw := []byte(`{"format":"cozy.package.interface/1","application":"fixture:app","entrypoints":[],"jobs":[{"name":"main","models":[],"publishes":false,"weights_outputs":[],"request":{"fields":[{"name":"resume_from","type":{"union":["null",{"input":"tree"}]},"wire":"optional","asset_bound":{"max_bytes":200000}},{"name":"nested","type":{"union":["null",{"fields":[{"name":"prefix","type":{"union":["null",{"input":"tree"}]},"wire":"optional"}]}]},"wire":"optional"},{"name":"report","type":{"asset":"file"},"wire":"optional"}]},"result":{"fields":[]}}]}`)
	parsed, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	return &parsed.Jobs[0]
}

func TestTypedTreeAssetArgumentsPreserveOptionalNestedAndFileInputs(t *testing.T) {
	entry := treeArgumentEntry(t)
	payload, files, trees, problem := launch.ParseTreeAssets(entry, []byte(`{"resume_from":null}`), []string{"RESUME_FROM=/prefix", "nested.prefix=/nested", "report=/report.json"})
	fatal(t, problem)
	if !reflect.DeepEqual(files, []string{"report=/report.json"}) || !reflect.DeepEqual(trees, []string{"asset:resume_from=/prefix", "asset:nested.prefix=/nested"}) {
		t.Fatalf("typed Tree input replaced ordinary file arguments: %v %v", files, trees)
	}
	refs, problem := launch.TreeInputRefs(entry, payload)
	fatal(t, problem)
	if !reflect.DeepEqual(refs, map[string]string{"resume_from": "asset:resume_from", "nested.prefix": "asset:nested.prefix"}) {
		t.Fatalf("nullable Tree fields were lost: %s %v", payload, refs)
	}
	for _, value := range []json.RawMessage{[]byte(`{}`), []byte(`{"resume_from":null,"nested":null}`)} {
		refs, problem := launch.TreeInputRefs(entry, value)
		fatal(t, problem)
		if len(refs) != 0 {
			t.Fatalf("absent optional input produced a Tree reference: %v", refs)
		}
	}
}

func TestTypedTreeAssetArgumentsRejectAmbiguousFields(t *testing.T) {
	entry := treeArgumentEntry(t)
	for _, specs := range [][]string{
		{"resume_from=/one", "RESUME_FROM=/two"},
		{"resume_from="},
		{"unknown=/one"},
		{"nested..prefix=/one"},
	} {
		payload, files, _, problem := launch.ParseTreeAssets(entry, []byte(`{}`), specs)
		if problem == nil {
			_, _, problem = launch.ParseAssets(entry, payload, files, nil, nil)
		}
		if problem == nil {
			t.Fatalf("invalid Tree inputs accepted: %v", specs)
		}
	}
	if _, _, _, problem := launch.ParseTreeAssets(entry, []byte(`{"resume_from":"existing"}`), []string{"resume_from=/other"}); problem == nil {
		t.Fatal("Tree asset silently overwrote an existing field reference")
	}
}
