package hostruntime

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestAdmissionRequiresPythonEnsureAndAcceptsSourceDevWheel(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX stand-in tool")
	}
	for _, test := range []struct {
		version  string
		admitted bool
	}{
		{"0.18.0", false}, {"0.18.13", false}, {"0.18.14rc1", false},
		{"0.18.14", true}, {"0.18.14+dev.h687ee141", true}, {"0.18.15", true},
	} {
		t.Run(test.version, func(t *testing.T) {
			tool := filepath.Join(t.TempDir(), "runtime")
			script := fmt.Sprintf("#!/bin/sh\nprintf '%%s\\n' '{\"distribution\":\"%s\",\"wire_protocol\":\"cozy.worker.v1+minor.58\"}'\n", test.version)
			if err := os.WriteFile(tool, []byte(script), 0700); err != nil {
				t.Fatal(err)
			}
			problem := admitHostRuntime(tool, nil)
			if test.admitted {
				if problem != nil {
					t.Fatal(problem)
				}
				return
			}
			if problem == nil || problem.Name != "host_runtime_below_floor" || !strings.Contains(problem.Error(), "python-ensure") || !strings.Contains(problem.Remedy, "cozy-runtime[media,model-execution]>=0.18.14") {
				t.Fatalf("missing early upgrade refusal: %+v", problem)
			}
		})
	}
}
