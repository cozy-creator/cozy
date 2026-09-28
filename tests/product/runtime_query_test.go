package producttest

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/launch"
)

// TestSlowRuntimeMetadataIsAnswered is xs-007 row 10 as behaviour.
//
// Runtime metadata verbs used to run under a 5-second deadline, and a runtime that had
// not answered by then was `runtime_query_stalled`. Five seconds is a guess about Python
// cold start, not a fact about a metadata verb: on a loaded host an interpreter importing
// its entry module exceeds it while doing exactly what it was asked. The child's own exit
// is the answer.
//
// The runtime here takes six seconds — comfortably past the deleted bound — and its answer
// is read through a metadata verb Creator still asks the Runtime.
func TestSlowRuntimeMetadataIsAnswered(t *testing.T) {
	fullRun(t, "outwaits the deleted five-second metadata deadline")
	if runtime.GOOS == "windows" {
		t.Skip("the stand-in runtime is a POSIX shell script")
	}
	root := t.TempDir()
	slowRuntime := filepath.Join(root, "cozy-runtime") //cozy:allow a stand-in runtime, not this host's
	must(t, os.WriteFile(slowRuntime, []byte("#!/bin/sh\n/bin/sleep 6\n"+
		`printf '{"recipe":{"name":"slow"}}\n'`+"\n"), 0o700))
	tool := launch.RuntimeCLI{Bin: slowRuntime, Dir: root, Home: root} //cozy:allow drives the metadata verb directly

	started := time.Now()
	recipe, e := tool.ModelIngestionPlan(context.Background(), "acme/slow", "main")
	if e != nil {
		t.Fatalf("a runtime answering in %s was refused: %s", time.Since(started), briefly(e))
	}
	if recipe == nil || recipe.Name != "slow" {
		t.Fatalf("the runtime's own answer was not read: %+v", recipe)
	}
	if elapsed := time.Since(started); elapsed < 5*time.Second {
		t.Fatalf("the stand-in runtime answered in %s; this arm only means something "+
			"past the deleted five-second bound", elapsed)
	}
}
