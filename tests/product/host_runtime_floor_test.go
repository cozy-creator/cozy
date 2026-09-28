package producttest

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestAdmissionRequiresNativeIngestionRuntimeAndAcceptsSourceDevWheel(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX stand-in tool")
	}
	for _, test := range []struct {
		version  string
		admitted bool
	}{
		{"0.18.0", false}, {"0.18.41", false}, {"0.18.51", false}, {"0.18.66", false},
		{"0.18.67rc1", false}, {"0.18.67", true}, {"0.18.67+dev.h687ee141", true}, {"0.18.68", true},
	} {
		t.Run(test.version, func(t *testing.T) {
			tool := filepath.Join(t.TempDir(), "cozy-runtime") //cozy:allow stand-in Runtime command for host admission
			script := fmt.Sprintf("#!/bin/sh\nprintf '%%s\\n' '{\"distribution\":\"%s\",\"wire_protocol\":\"cozy.worker.v1+minor.%d\"}'\n", test.version, pb.WireMinor)
			if err := os.WriteFile(tool, []byte(script), 0700); err != nil { //cozy:allow stand-in Runtime command for host admission
				t.Fatal(err)
			}
			t.Setenv("PATH", filepath.Dir(tool))
			_, problem := hostruntime.Path(nil)
			if test.admitted {
				if problem != nil {
					t.Fatal(problem)
				}
				return
			}
			if problem == nil || problem.Name != "host_runtime_below_floor" || !strings.Contains(problem.Error(), "native model ingestion") || !strings.Contains(problem.Remedy, "cozy-runtime[media,model-execution]>="+hostruntime.ToolFloor) {
				t.Fatalf("missing early upgrade refusal: %+v", problem)
			}
		})
	}
}
