package producttest

import (
	"archive/zip"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/wheel"
)

func TestAdmissionRequiresNativeIngestionRuntimeAndAcceptsSourceDevWheel(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX stand-in tool")
	}
	for _, test := range []struct {
		version  string
		admitted bool
	}{
		{"0.18.0", false}, {"0.18.14", false}, {"0.18.20", false}, {"0.18.21rc1", false},
		{"0.18.21", false}, {"0.18.22", false}, {"0.18.23", false},
		{"0.18.24rc1", false}, {"0.18.24", true}, {"0.18.24+dev.h687ee141", true}, {"0.18.25", true},
	} {
		t.Run(test.version, func(t *testing.T) {
			tool := filepath.Join(t.TempDir(), "cozy-runtime") //cozy:allow stand-in Runtime command for host admission
			script := fmt.Sprintf("#!/bin/sh\nprintf '%%s\\n' '{\"distribution\":\"%s\",\"wire_protocol\":\"cozy.worker.v1+minor.58\"}'\n", test.version)
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
			if problem == nil || problem.Name != "host_runtime_below_floor" || !strings.Contains(problem.Error(), "native model ingestion") || !strings.Contains(problem.Remedy, "cozy-runtime[media,model-execution]>=0.18.24") {
				t.Fatalf("missing early upgrade refusal: %+v", problem)
			}
		})
	}
}

func TestNewControllerAcceptsExistingPackageSDKClosure(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX stand-in tool")
	}
	tool := filepath.Join(t.TempDir(), "cozy-runtime") //cozy:allow stand-in controller for package SDK compatibility
	must(t, os.WriteFile(tool, []byte(stubRuntime(t, hostruntime.ToolFloor, 58)), 0700))
	t.Setenv("PATH", filepath.Dir(tool))
	_, problem := hostruntime.Path(nil)
	fatal(t, problem)
	// A real captured wheel may pin SDK18.13 while proving the established
	// package API floor. Its requirements must survive a controller upgrade.
	path := filepath.Join(t.TempDir(), "existing-1.0-py3-none-any.whl")
	file, err := os.Create(path)
	must(t, err)
	archive := zip.NewWriter(file)
	metadata, err := archive.Create("existing-1.0.dist-info/METADATA")
	must(t, err)
	_, err = metadata.Write([]byte("Metadata-Version: 2.4\nName: existing\nVersion: 1.0\nRequires-Dist: cozy-runtime==0.18.13\nRequires-Dist: cozy-runtime>=0.18.0\n\n"))
	must(t, err)
	must(t, archive.Close())
	must(t, file.Close())
	captured, problem := packagepublish.CaptureDependency(path)
	fatal(t, problem)
	if captured.Path != path || captured.Name != "existing" || captured.Version != "1.0" {
		t.Fatal("existing SDK wheel changed during capture")
	}
	raw, problem := wheel.Metadata(captured.Path)
	fatal(t, problem)
	if !strings.Contains(string(raw), "Requires-Dist: cozy-runtime==0.18.13") {
		t.Fatal("controller upgrade rewrote package SDK requirements")
	}

}
