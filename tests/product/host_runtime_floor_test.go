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
	"github.com/cozy-creator/cozy/internal/localpackage"
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
			if problem == nil || problem.Name != "host_runtime_below_floor" || !strings.Contains(problem.Error(), "python-ensure") || !strings.Contains(problem.Remedy, "cozy-runtime[media,model-execution]>=0.18.14") {
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
	must(t, os.WriteFile(tool, []byte(stubRuntime(t, "0.18.14", 58)), 0700))
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
	captured := localpackage.Revision{Files: []localpackage.File{{Kind: "project", Path: path}}}
	fatal(t, localpackage.RequireRuntimeFloor(captured, hostruntime.PackageFloor))
	if problem := localpackage.RequireRuntimeFloor(captured, hostruntime.ToolFloor); problem == nil {
		t.Fatal("control: this old package must not claim the new controller API")
	}
}
