package producttest

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// TestSlowRuntimeMetadataIsAnswered is xs-007 row 10 as behaviour.
//
// `cozy-runtime describe` used to run under a 5-second deadline, and a runtime that had
// not answered by then was `runtime_query_stalled` — which refused the job dispatch. Five
// seconds is a guess about Python cold start, not a fact about a metadata verb: on a loaded
// host an interpreter importing its entry module exceeds it while doing exactly what it was
// asked. The child's own exit is the answer.
//
// The runtime here takes six seconds — comfortably past the deleted bound — and its answer
// is read.
func TestSlowRuntimeMetadataIsAnswered(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the stand-in runtime is a POSIX shell script")
	}
	root := t.TempDir()
	slowRuntime := filepath.Join(root, "cozy-runtime") //cozy:allow a stand-in runtime, not this host's
	version := strings.TrimPrefix(stubRuntime(t, hostruntime.Floor, pb.WireMinor), "#!/bin/sh\n")
	must(t, os.WriteFile(slowRuntime, []byte("#!/bin/sh\nfor arg in \"$@\"; do\n"+
		"if [ \"$arg\" = version ]; then\n"+version+"exit 0\nfi\ndone\nsleep 6\n"+
		`printf '{"job_descriptor_id":"jd-slow"}\n'`+"\n"), 0o700))
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))

	facts := &launch.Facts{
		Install:          records.PackageInstall{Package: "acme/slow"},
		PackageInterface: &launch.PackageInterface{Jobs: []launch.Entrypoint{{Name: "render"}}},
		RuntimeCLI:       launch.RuntimeCLI{Bin: slowRuntime, Dir: root, Home: root}, //cozy:allow drives the metadata verb directly
	}

	started := time.Now()
	job, e := facts.Job("render")
	if e != nil {
		t.Fatalf("a runtime answering in %s was refused: %s", time.Since(started), briefly(e))
	}
	if job.DescriptorID != "jd-slow" {
		t.Fatalf("the runtime's own derivation was not read: %+v", job.DescriptorID)
	}
	if elapsed := time.Since(started); elapsed < 5*time.Second {
		t.Fatalf("the stand-in runtime answered in %s; this arm only means something "+
			"past the deleted five-second bound", elapsed)
	}
}
