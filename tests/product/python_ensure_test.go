package producttest

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
)

func TestEnsurePythonUsesSharedRuntimeAndPreservesExactPatch(t *testing.T) {
	root := t.TempDir()
	script := `#!/bin/sh
if [ "$2" = version ]; then
 printf '%s\n' '{"distribution":"` + hostruntime.ToolFloor + `","wire_protocol":"cozy.worker.v1+minor.54"}'
 exit 0
fi
if [ "$1" != --json ] || [ "$2" != python-ensure ] || [ "$3" != '>=3.13,<3.14' ] || [ "$4" != 3.13.7 ]; then exit 2; fi
printf '%s\n' '{"executable":"/managed/python3.13","version":"3.13.7","abi":"cp313"}'
`
	if err := os.WriteFile(filepath.Join(root, "cozy-runtime"), []byte(script), 0700); err != nil { //cozy:allow stand-in Runtime command used to verify the shared host boundary
		t.Fatal(err)
	}
	t.Setenv("PATH", root)
	selected, problem := hostruntime.EnsurePython(context.Background(), ">=3.13,<3.14", "3.13.7")
	if problem != nil || selected.Version != "3.13.7" {
		t.Fatalf("%+v %v", selected, problem)
	}
}

func TestEnsurePythonPreservesTypedProvisioningRefusal(t *testing.T) {
	root := t.TempDir()
	script := `#!/bin/sh
if [ "$2" = version ]; then
 printf '%s\n' '{"distribution":"` + hostruntime.ToolFloor + `","wire_protocol":"cozy.worker.v1+minor.54"}'
 exit 0
fi
printf '%s\n' 'Installing CPython 3.13.7' '{"error":{"name":"python_provision_failed","message":"download unavailable","remedy":"retry when online"}}' >&2
exit 9
`
	if err := os.WriteFile(filepath.Join(root, "cozy-runtime"), []byte(script), 0700); err != nil { //cozy:allow stand-in Runtime command used to verify the shared host boundary
		t.Fatal(err)
	}
	t.Setenv("PATH", root)
	_, problem := hostruntime.EnsurePython(context.Background(), ">=3.13", "3.13.7")
	if problem == nil || problem.Name != "python_provision_failed" || problem.Message != "download unavailable" || problem.Remedy != "retry when online" {
		t.Fatalf("%v", problem)
	}
}
